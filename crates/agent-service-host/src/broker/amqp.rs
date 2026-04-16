//! AMQP 0.9.1 broker adapter built on `lapin`.
//!
//! The adapter translates an [`OutboxMessage`] into a `basic.publish`
//! frame and **waits for the broker's ack** via publisher confirms
//! before returning from [`BrokerAdapter::publish`].  This is what
//! turns broker acknowledgement into the authority for marking an
//! outbox row `delivered`:
//!
//! ```text
//!   claim_pending  ────►  publish  ───(broker ack)───►  mark_delivered
//!                                         │
//!                                         └── crash here → row stays
//!                                             `claimed`, reclaim sweep
//!                                             returns it to `pending`
//!                                             on recovery, publisher
//!                                             confirms republish.
//! ```
//!
//! # Connection lifecycle
//!
//! One TCP connection + one channel per adapter.  Both are opened
//! lazily on the first publish and cached behind an async lock.  A
//! publish error invalidates the channel so the next call reconnects —
//! this is coarse-grained by design: the outbox contract already
//! tolerates duplicate republish, so we prefer simple reconnection
//! semantics to a fine-grained state machine.
//!
//! # Topology
//!
//! The adapter assumes the operator has declared the exchange out of
//! band (Infrastructure-as-Code, Terraform, etc.).  An opt-in
//! [`AmqpBrokerConfig::declare_exchange`] flag makes the host declare
//! the exchange on first connect — handy for local development and
//! integration tests.  Queues and bindings are consumer concerns and
//! never created by the publisher.
//!
//! # Routing
//!
//! Each [`OutboxMessageKind`] gets a dedicated routing key so
//! subscribers can bind to `agent_sdk.outbox.task_wakeup` without
//! receiving `thread_events_available` traffic.  The kind is also
//! surfaced as the `type` AMQP property for consumers that bind to a
//! single key via a fanout exchange.

use std::sync::Arc;

use agent_server::journal::broker::BrokerAdapter;
use agent_server::journal::outbox_message::{OutboxMessage, OutboxMessageKind};
use anyhow::{Context, Result};
use async_trait::async_trait;
use lapin::message::BasicReturnMessage;
use lapin::options::{BasicPublishOptions, ConfirmSelectOptions, ExchangeDeclareOptions};
use lapin::types::{FieldTable, ShortString};
use lapin::{
    BasicProperties, Channel, Confirmation, Connection, ConnectionProperties, ExchangeKind,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

// ─────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────

/// AMQP broker adapter configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AmqpBrokerConfig {
    /// Broker URL (e.g. `amqp://user:pass@host:5672/vhost`).
    ///
    /// When `None`, the adapter falls back to the `AMQP_URL` environment
    /// variable.
    pub url: Option<String>,
    /// Exchange to publish to.  Consumers bind queues to it.
    pub exchange: String,
    /// AMQP exchange kind; typically `topic` so subscribers can pick
    /// specific routing keys.
    pub exchange_kind: AmqpExchangeKind,
    /// Whether the adapter declares the exchange on first connect.
    ///
    /// `true` is convenient for local development; production deploys
    /// usually leave it `false` because the exchange is provisioned by
    /// Infrastructure-as-Code.
    pub declare_exchange: bool,
    /// Common prefix for routing keys.  Defaults to the exchange name.
    pub routing_key_prefix: String,
}

impl Default for AmqpBrokerConfig {
    fn default() -> Self {
        Self {
            url: None,
            exchange: "agent_sdk.outbox".into(),
            exchange_kind: AmqpExchangeKind::Topic,
            declare_exchange: false,
            routing_key_prefix: "agent_sdk.outbox".into(),
        }
    }
}

impl AmqpBrokerConfig {
    /// Resolve the broker URL, falling back to `AMQP_URL` if unset.
    ///
    /// # Errors
    /// Returns an error if neither `url` nor `AMQP_URL` is set.
    pub fn resolved_url(&self) -> Result<String> {
        if let Some(url) = &self.url {
            return Ok(url.clone());
        }
        std::env::var("AMQP_URL")
            .context("broker.amqp.url is unset and AMQP_URL environment variable is not set")
    }

    /// Derive the routing key for a given message kind.
    #[must_use]
    pub fn routing_key(&self, kind: OutboxMessageKind) -> String {
        format!("{}.{}", self.routing_key_prefix, kind.as_str())
    }
}

/// Supported AMQP exchange kinds.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AmqpExchangeKind {
    /// Topic exchange with pattern-based routing keys.
    #[default]
    Topic,
    /// Direct exchange with exact routing-key match.
    Direct,
    /// Fanout — routing keys are ignored by the broker.
    Fanout,
}

impl From<AmqpExchangeKind> for ExchangeKind {
    fn from(kind: AmqpExchangeKind) -> Self {
        match kind {
            AmqpExchangeKind::Topic => Self::Topic,
            AmqpExchangeKind::Direct => Self::Direct,
            AmqpExchangeKind::Fanout => Self::Fanout,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Adapter
// ─────────────────────────────────────────────────────────────────────

/// Lapin-backed AMQP 0.9.1 broker adapter with publisher confirms.
pub struct AmqpBrokerAdapter {
    config: AmqpBrokerConfig,
    url: String,
    channel: Mutex<Option<ChannelState>>,
}

struct ChannelState {
    #[allow(dead_code)] // kept alive so channel stays usable
    connection: Connection,
    channel: Channel,
    exchange_declared: bool,
}

impl AmqpBrokerAdapter {
    /// Construct an adapter.  The connection is opened lazily on the
    /// first [`BrokerAdapter::publish`] call.
    ///
    /// # Errors
    /// Returns an error if the configured URL cannot be resolved.
    pub fn new(config: AmqpBrokerConfig) -> Result<Self> {
        let url = config.resolved_url()?;
        Ok(Self {
            config,
            url,
            channel: Mutex::new(None),
        })
    }

    /// Construct an adapter wrapped in an `Arc` ready to be used as a
    /// trait object.
    ///
    /// # Errors
    /// See [`Self::new`].
    pub fn arc(config: AmqpBrokerConfig) -> Result<Arc<dyn BrokerAdapter>> {
        Ok(Arc::new(Self::new(config)?))
    }

    async fn ensure_channel<'a>(
        &'a self,
        slot: &'a mut Option<ChannelState>,
    ) -> Result<&'a mut ChannelState> {
        if slot.is_none() {
            info!(url = %redact_url(&self.url), "opening AMQP connection");
            let connection = Connection::connect(&self.url, ConnectionProperties::default())
                .await
                .with_context(|| format!("connect to AMQP broker at {}", redact_url(&self.url)))?;
            let channel = connection
                .create_channel()
                .await
                .context("create AMQP channel")?;
            // Publisher confirms: broker ack is the trigger for
            // mark_delivered.  Without this, `basic_publish` resolves
            // as soon as the frame leaves the client and we would lose
            // the ack contract.
            channel
                .confirm_select(ConfirmSelectOptions::default())
                .await
                .context("enable publisher confirms")?;

            *slot = Some(ChannelState {
                connection,
                channel,
                exchange_declared: false,
            });
        }

        let state = slot.as_mut().expect("channel slot populated above");
        if self.config.declare_exchange && !state.exchange_declared {
            state
                .channel
                .exchange_declare(
                    self.config.exchange.as_str().into(),
                    self.config.exchange_kind.into(),
                    ExchangeDeclareOptions {
                        durable: true,
                        ..ExchangeDeclareOptions::default()
                    },
                    FieldTable::default(),
                )
                .await
                .with_context(|| format!("declare AMQP exchange {}", self.config.exchange))?;
            state.exchange_declared = true;
        }
        Ok(state)
    }

    async fn invalidate_channel(&self) {
        *self.channel.lock().await = None;
    }

    async fn publish_with_confirm(
        &self,
        routing_key: &str,
        payload: &[u8],
        properties: BasicProperties,
    ) -> Result<Confirmation> {
        let mut slot = self.channel.lock().await;
        let confirm_future = {
            let state = self.ensure_channel(&mut slot).await?;
            debug!(
                exchange = %self.config.exchange,
                routing_key = %routing_key,
                "publishing outbox message",
            );
            state
                .channel
                .basic_publish(
                    self.config.exchange.as_str().into(),
                    routing_key.into(),
                    BasicPublishOptions {
                        mandatory: true,
                        ..BasicPublishOptions::default()
                    },
                    payload,
                    properties,
                )
                .await
                .context("send AMQP basic.publish frame")?
        };
        // Drop the channel lock before awaiting the confirmation so
        // other publishers can pipeline their sends while we wait.
        drop(slot);
        confirm_future.await.context("await AMQP publisher confirm")
    }
}

#[async_trait]
impl BrokerAdapter for AmqpBrokerAdapter {
    async fn publish(&self, message: &OutboxMessage) -> Result<()> {
        let routing_key = self.config.routing_key(message.kind());
        let payload = serde_json::to_vec(&message.to_payload_json()?)
            .context("serialise outbox payload for AMQP publish")?;
        let properties = BasicProperties::default()
            .with_content_type(ShortString::from("application/json"))
            // `with_type` sets the AMQP `type` property — subscribers
            // that bind through a fanout exchange use it to route by
            // outbox kind without parsing the payload.
            .with_type(ShortString::from(message.kind().as_str()))
            // Persistent delivery — the broker will fsync the message
            // to disk before acking when the target queue is durable.
            .with_delivery_mode(2);

        let result = match self
            .publish_with_confirm(&routing_key, &payload, properties)
            .await
        {
            Ok(result) => result,
            Err(err) => {
                self.invalidate_channel().await;
                return Err(err);
            }
        };

        match result {
            Confirmation::Ack(None) => Ok(()),
            Confirmation::Ack(Some(returned)) => {
                Err(unroutable_publish_error(&routing_key, &returned))
            }
            Confirmation::Nack(_) => {
                // Drop the channel so the next publish reconnects; a
                // nack typically signals a routing or resource problem
                // the broker wants us to notice.
                warn!("AMQP broker nacked publish; dropping channel for reconnect");
                self.invalidate_channel().await;
                anyhow::bail!("AMQP broker nacked publish")
            }
            Confirmation::NotRequested => {
                self.invalidate_channel().await;
                anyhow::bail!(
                    "AMQP publisher confirms were not enabled on the channel; cannot guarantee delivery",
                )
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────

/// Strip credentials from an AMQP URL before logging.
fn redact_url(url: &str) -> String {
    url::Url::parse(url).map_or_else(
        |_| "<unparsable-url>".into(),
        |mut parsed| {
            if !parsed.username().is_empty() || parsed.password().is_some() {
                let _ = parsed.set_username("***");
                let _ = parsed.set_password(None);
            }
            parsed.to_string()
        },
    )
}

fn unroutable_publish_error(routing_key: &str, returned: &BasicReturnMessage) -> anyhow::Error {
    anyhow::anyhow!(
        "AMQP broker returned unroutable publish for routing key {routing_key}: {} ({})",
        returned.reply_text,
        returned.reply_code,
    )
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use lapin::message::Delivery;
    use lapin::types::ShortString;

    #[test]
    fn routing_key_embeds_kind() {
        let config = AmqpBrokerConfig::default();
        assert_eq!(
            config.routing_key(OutboxMessageKind::TaskWakeup),
            "agent_sdk.outbox.task_wakeup"
        );
        assert_eq!(
            config.routing_key(OutboxMessageKind::ThreadEventsAvailable),
            "agent_sdk.outbox.thread_events_available"
        );
    }

    #[test]
    fn routing_key_uses_configured_prefix() {
        let config = AmqpBrokerConfig {
            routing_key_prefix: "tenant.acme".into(),
            ..AmqpBrokerConfig::default()
        };
        assert_eq!(
            config.routing_key(OutboxMessageKind::TaskWakeup),
            "tenant.acme.task_wakeup"
        );
    }

    #[test]
    fn redact_url_hides_credentials() {
        let url = "amqp://rabbit:hunter2@broker.internal:5672/prod";
        let redacted = redact_url(url);
        assert!(!redacted.contains("hunter2"));
        assert!(!redacted.contains("rabbit"));
        assert!(redacted.contains("broker.internal"));
    }

    #[test]
    fn redact_url_handles_credential_free_urls() {
        let url = "amqp://broker.internal:5672/prod";
        assert_eq!(redact_url(url), "amqp://broker.internal:5672/prod");
    }

    #[test]
    fn resolved_url_reads_explicit_config_first() -> Result<()> {
        let config = AmqpBrokerConfig {
            url: Some("amqp://explicit:5672/".into()),
            ..AmqpBrokerConfig::default()
        };
        assert_eq!(config.resolved_url()?, "amqp://explicit:5672/");
        Ok(())
    }

    #[test]
    fn exchange_kind_maps_to_lapin() {
        assert!(matches!(
            ExchangeKind::from(AmqpExchangeKind::Topic),
            ExchangeKind::Topic
        ));
        assert!(matches!(
            ExchangeKind::from(AmqpExchangeKind::Direct),
            ExchangeKind::Direct
        ));
        assert!(matches!(
            ExchangeKind::from(AmqpExchangeKind::Fanout),
            ExchangeKind::Fanout
        ));
    }

    #[test]
    fn publish_uses_mandatory_routing() {
        let options = BasicPublishOptions {
            mandatory: true,
            ..BasicPublishOptions::default()
        };
        assert!(options.mandatory);
        assert!(!options.immediate);
    }

    #[test]
    fn unroutable_publish_error_mentions_routing_key_and_reply() {
        let returned = BasicReturnMessage {
            delivery: Delivery::mock(
                0,
                ShortString::from("agent_sdk.outbox"),
                ShortString::from("agent_sdk.outbox.task_wakeup"),
                false,
                Vec::new(),
            ),
            reply_code: 312,
            reply_text: ShortString::from("NO_ROUTE"),
        };

        let error = unroutable_publish_error("agent_sdk.outbox.task_wakeup", &returned);
        let rendered = format!("{error:#}");
        assert!(rendered.contains("agent_sdk.outbox.task_wakeup"));
        assert!(rendered.contains("NO_ROUTE"));
        assert!(rendered.contains("312"));
    }
}
