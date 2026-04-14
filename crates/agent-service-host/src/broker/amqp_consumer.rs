//! AMQP 0.9.1 consumer for `task_wakeup` advisory deliveries.
//!
//! Phase 8.3 builds this consumer to **nudge** the worker pool without
//! ever letting a broker payload drive execution.  The consumer does
//! four things on every delivery:
//!
//! 1. Deserialises the JSON payload into
//!    [`agent_server::journal::outbox_message::TaskWakeupPayload`].
//! 2. Calls a [`TaskWakeupHandler`] to re-check durable state.  That
//!    handler is the single place where the journal is consulted.
//! 3. Acks (or nacks) the broker message based on the result.
//! 4. Logs the outcome so operators can reason about delivery flow.
//!
//! Execution itself is always driven by the worker pool in
//! [`crate::host`], which consumes the nudge via the
//! shared [`agent_server::journal::WakeupSignal`] and
//! then runs `AgentTaskStore::acquire_next_runnable`.  Two duplicate
//! deliveries therefore resolve into a single execution: one CAS
//! winner, plus any number of losers that observe `Ok(None)` and go
//! back to idle.
//!
//! # Topology
//!
//! The consumer declares (idempotently) a durable queue and binds it
//! to the configured exchange with the `task_wakeup` routing key
//! produced by the Phase 8.2 publisher:
//!
//! ```text
//!    producer → exchange (topic) ─── routing_key_prefix.task_wakeup ───▶ queue
//! ```
//!
//! Queues are named deterministically so multiple pods on the same
//! service share a single queue and see **competing-consumer**
//! semantics: each delivery goes to one pod, not every pod.  Two pods
//! can therefore run simultaneously without double-processing a
//! wakeup.  The broker handles the deduplication for us.
//!
//! # Shutdown
//!
//! The consumer's `run` future is driven until a
//! [`CancellationToken`] fires.  When it fires the consumer stops
//! pulling new deliveries, cancels the broker-side consumer
//! subscription, and closes the channel.  The biased `select!` only
//! observes cancellation between loop iterations, so a
//! `handle_delivery` call that is already in flight runs to
//! completion with its normal outcome: success → `ack`, handler error
//! → `nack` with requeue.  Any deliveries that were buffered in the
//! consumer stream but never pulled are requeued automatically when
//! the channel is closed, so the journal-backed contract still holds:
//! the broker redelivers and the next consumer re-checks the journal.

use std::sync::Arc;

use agent_server::journal::{TaskWakeupHandler, outbox_message::TaskWakeupPayload};
use anyhow::{Context, Result};
use futures::StreamExt;
use lapin::options::{
    BasicAckOptions, BasicConsumeOptions, BasicNackOptions, ConfirmSelectOptions,
    ExchangeDeclareOptions, QueueBindOptions, QueueDeclareOptions,
};
use lapin::types::FieldTable;
use lapin::{Channel, Connection, ConnectionProperties, Consumer};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use super::amqp::{AmqpBrokerConfig, AmqpExchangeKind};

// ─────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────

/// Tunables for an AMQP task-wakeup consumer.
///
/// Defaults line up with the Phase 8.2 publisher defaults so a single
/// shared `AmqpBrokerConfig` (or a pair of them pointing at the same
/// broker) works out of the box for local development.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AmqpTaskWakeupConsumerConfig {
    /// Broker connection settings.  Re-uses the publisher's config so
    /// deployments can share routing-key prefixes, exchange names, and
    /// credentials.
    pub broker: AmqpBrokerConfig,
    /// Durable queue name.  Multiple pods that share this name compete
    /// for deliveries; different names produce a fan-out topology
    /// (useful for shadow deployments).
    pub queue: String,
    /// Consumer tag prefix.  Suffixed with a random nonce so multiple
    /// consumers on the same pod do not collide.
    pub consumer_tag_prefix: String,
    /// Whether the consumer declares the queue on first connect.
    ///
    /// `true` is convenient for local development and CI.  Production
    /// deploys usually leave it `false` because the queue is
    /// provisioned by Infrastructure-as-Code.
    pub declare_queue: bool,
    /// Whether the consumer binds the queue to the exchange on first
    /// connect.  Same trade-off as [`Self::declare_queue`].
    pub bind_queue: bool,
}

impl Default for AmqpTaskWakeupConsumerConfig {
    fn default() -> Self {
        Self {
            broker: AmqpBrokerConfig::default(),
            queue: "agent_sdk.wakeup".into(),
            consumer_tag_prefix: "agent-sdk-wakeup".into(),
            declare_queue: false,
            bind_queue: false,
        }
    }
}

impl AmqpTaskWakeupConsumerConfig {
    /// Routing key the consumer binds to; always the publisher's
    /// `task_wakeup` key so the two halves stay in sync.
    #[must_use]
    pub fn routing_key(&self) -> String {
        self.broker
            .routing_key(agent_server::journal::outbox_message::OutboxMessageKind::TaskWakeup)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Consumer
// ─────────────────────────────────────────────────────────────────────

/// AMQP consumer that pulls `task_wakeup` deliveries and forwards them
/// to a [`TaskWakeupHandler`].
///
/// Generic over the handler so tests can swap the default journal
/// handler for a capturing double without touching the broker code.
pub struct AmqpTaskWakeupConsumer {
    config: AmqpTaskWakeupConsumerConfig,
    handler: Arc<dyn TaskWakeupHandler>,
    connection: Mutex<Option<ConnectionState>>,
}

struct ConnectionState {
    // Kept alive so the channel stays usable.
    #[allow(dead_code)]
    connection: Connection,
    channel: Channel,
}

impl AmqpTaskWakeupConsumer {
    /// Construct a consumer bound to `handler`.
    ///
    /// The broker connection is opened lazily on [`Self::run`] so
    /// cheap construction works from tests and config validators.
    #[must_use]
    pub fn new(config: AmqpTaskWakeupConsumerConfig, handler: Arc<dyn TaskWakeupHandler>) -> Self {
        Self {
            config,
            handler,
            connection: Mutex::new(None),
        }
    }

    /// Run the consumer until `cancel` fires.
    ///
    /// The method establishes a connection + channel, declares /
    /// binds the queue if configured, opens a `basic.consume`
    /// subscription, and loops over deliveries.  On `cancel` it
    /// flushes the in-flight delivery, cancels the subscription, and
    /// closes the channel.
    ///
    /// # Errors
    /// Returns an error only if a **structural** step fails: the
    /// broker connection, the exchange / queue declaration, or the
    /// subscription itself.  Per-delivery errors are logged and the
    /// delivery is nacked (with requeue) so the broker can redeliver.
    pub async fn run(&self, cancel: CancellationToken) -> Result<()> {
        info!(
            queue = %self.config.queue,
            routing_key = %self.config.routing_key(),
            "task wakeup consumer starting",
        );

        // Race the connection handshake against cancellation so a
        // shutdown that fires while the broker is unreachable does not
        // stall the drain behind the OS TCP SYN timeout.
        let (channel, mut consumer) = tokio::select! {
            biased;
            () = cancel.cancelled() => {
                info!("task wakeup consumer cancelled during connection establishment");
                return Ok(());
            }
            result = self.open_consumer() => result?,
        };
        let consumer_tag = consumer.tag().to_string();

        loop {
            tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    info!(%consumer_tag, "task wakeup consumer shutting down");
                    let _ = channel
                        .basic_cancel(
                            &consumer_tag,
                            lapin::options::BasicCancelOptions::default(),
                        )
                        .await;
                    self.invalidate_connection().await;
                    return Ok(());
                }
                delivery = consumer.next() => {
                    match delivery {
                        Some(Ok(delivery)) => {
                            self.handle_delivery(delivery).await;
                        }
                        Some(Err(err)) => {
                            warn!(error = %err, "task wakeup consumer received broker error");
                            self.invalidate_connection().await;
                            return Err(anyhow::Error::new(err)
                                .context("AMQP task wakeup consumer stream error"));
                        }
                        None => {
                            info!("task wakeup consumer stream closed by broker; exiting so the supervisor can restart with a fresh connection");
                            self.invalidate_connection().await;
                            return Ok(());
                        }
                    }
                }
            }
        }
    }

    async fn open_consumer(&self) -> Result<(Channel, Consumer)> {
        let channel = {
            let mut slot = self.connection.lock().await;
            if slot.is_none() {
                *slot = Some(self.open_connection().await?);
            }
            slot.as_ref()
                .context("connection state populated above")?
                .channel
                .clone()
        };

        if self.config.declare_queue {
            channel
                .queue_declare(
                    &self.config.queue,
                    QueueDeclareOptions {
                        durable: true,
                        ..QueueDeclareOptions::default()
                    },
                    FieldTable::default(),
                )
                .await
                .with_context(|| format!("declare queue {}", self.config.queue))?;
        }

        if self.config.bind_queue {
            channel
                .queue_bind(
                    &self.config.queue,
                    &self.config.broker.exchange,
                    &self.config.routing_key(),
                    QueueBindOptions::default(),
                    FieldTable::default(),
                )
                .await
                .with_context(|| {
                    format!(
                        "bind queue {} to exchange {} with routing key {}",
                        self.config.queue,
                        self.config.broker.exchange,
                        self.config.routing_key(),
                    )
                })?;
        }

        let consumer_tag = format!(
            "{}-{}",
            self.config.consumer_tag_prefix,
            &uuid::Uuid::new_v4().simple().to_string()[..12],
        );
        let consumer = channel
            .basic_consume(
                &self.config.queue,
                &consumer_tag,
                BasicConsumeOptions::default(),
                FieldTable::default(),
            )
            .await
            .with_context(|| {
                format!(
                    "open basic.consume subscription on queue {}",
                    self.config.queue
                )
            })?;
        Ok((channel, consumer))
    }

    async fn open_connection(&self) -> Result<ConnectionState> {
        let url = self
            .config
            .broker
            .resolved_url()
            .context("resolve AMQP URL for wakeup consumer")?;

        info!("opening AMQP connection for task wakeup consumer");
        let connection = Connection::connect(&url, ConnectionProperties::default())
            .await
            .context("connect to AMQP broker for wakeup consumer")?;
        let channel = connection
            .create_channel()
            .await
            .context("open AMQP channel for wakeup consumer")?;
        // Publisher confirms are not strictly needed for a consumer
        // but cost nothing and let the channel be re-used if we ever
        // want to publish from the consumer loop.
        channel
            .confirm_select(ConfirmSelectOptions::default())
            .await
            .context("enable publisher confirms on consumer channel")?;

        if self.config.broker.declare_exchange {
            channel
                .exchange_declare(
                    &self.config.broker.exchange,
                    match self.config.broker.exchange_kind {
                        AmqpExchangeKind::Topic => lapin::ExchangeKind::Topic,
                        AmqpExchangeKind::Direct => lapin::ExchangeKind::Direct,
                        AmqpExchangeKind::Fanout => lapin::ExchangeKind::Fanout,
                    },
                    ExchangeDeclareOptions {
                        durable: true,
                        ..ExchangeDeclareOptions::default()
                    },
                    FieldTable::default(),
                )
                .await
                .with_context(|| {
                    format!(
                        "declare AMQP exchange {} from wakeup consumer",
                        self.config.broker.exchange
                    )
                })?;
        }

        Ok(ConnectionState {
            connection,
            channel,
        })
    }

    async fn invalidate_connection(&self) {
        *self.connection.lock().await = None;
    }

    async fn handle_delivery(&self, delivery: lapin::message::Delivery) {
        let routing_key = delivery.routing_key.as_str().to_owned();
        let delivery_tag = delivery.delivery_tag;

        let payload = match serde_json::from_slice::<TaskWakeupPayload>(&delivery.data) {
            Ok(payload) => payload,
            Err(err) => {
                warn!(
                    routing_key,
                    error = %err,
                    "wakeup delivery payload did not decode; rejecting without requeue",
                );
                // Payload corruption is not transient — requeuing
                // would loop forever.  The broker-side dead letter
                // exchange (if any) catches it.
                if let Err(err) = delivery
                    .acker
                    .nack(BasicNackOptions {
                        multiple: false,
                        requeue: false,
                    })
                    .await
                {
                    warn!(error = %err, "failed to nack undecodeable wakeup delivery");
                }
                return;
            }
        };

        let now = OffsetDateTime::now_utc();
        let outcome = self.handler.handle_payload(&payload, now).await;

        match outcome {
            Ok(outcome) => {
                debug!(
                    routing_key,
                    delivery_tag,
                    task_id = %payload.task_id,
                    thread_id = %payload.thread_id,
                    nudged = outcome.nudged(),
                    status = ?outcome.observed_status(),
                    "wakeup delivery handled",
                );
                if let Err(err) = delivery.acker.ack(BasicAckOptions::default()).await {
                    warn!(
                        routing_key,
                        delivery_tag,
                        error = %err,
                        "failed to ack wakeup delivery after re-check"
                    );
                }
            }
            Err(err) => {
                warn!(
                    routing_key,
                    delivery_tag,
                    task_id = %payload.task_id,
                    thread_id = %payload.thread_id,
                    error = %err,
                    "wakeup handler failed; nacking for broker redelivery",
                );
                if let Err(nack_err) = delivery
                    .acker
                    .nack(BasicNackOptions {
                        multiple: false,
                        requeue: true,
                    })
                    .await
                {
                    warn!(
                        routing_key,
                        delivery_tag,
                        error = %nack_err,
                        "failed to nack wakeup delivery; broker may redeliver on reconnect",
                    );
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests (payload + config; broker integration tests are in wakeup.rs)
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_match_publisher_routing_scheme() {
        let config = AmqpTaskWakeupConsumerConfig::default();
        assert_eq!(config.routing_key(), "agent_sdk.outbox.task_wakeup");
    }

    #[test]
    fn config_routing_key_follows_explicit_prefix() {
        let mut config = AmqpTaskWakeupConsumerConfig::default();
        config.broker.routing_key_prefix = "tenant.acme".into();
        assert_eq!(config.routing_key(), "tenant.acme.task_wakeup");
    }

    #[test]
    fn config_round_trips_through_yaml() -> Result<()> {
        let config = AmqpTaskWakeupConsumerConfig {
            queue: "agent_sdk.wakeup.test".into(),
            consumer_tag_prefix: "pod-one".into(),
            declare_queue: true,
            bind_queue: true,
            ..AmqpTaskWakeupConsumerConfig::default()
        };
        let yaml = serde_yaml::to_string(&config).context("serialise")?;
        let parsed: AmqpTaskWakeupConsumerConfig =
            serde_yaml::from_str(&yaml).context("round trip")?;
        assert_eq!(parsed.queue, config.queue);
        assert_eq!(parsed.consumer_tag_prefix, config.consumer_tag_prefix);
        assert!(parsed.declare_queue);
        assert!(parsed.bind_queue);
        Ok(())
    }

    #[test]
    fn default_config_does_not_declare_queue_by_default() {
        // Production deploys usually have the queue provisioned out of
        // band — the default must match that so a misconfigured pod
        // does not silently start a topology change.
        let config = AmqpTaskWakeupConsumerConfig::default();
        assert!(!config.declare_queue);
        assert!(!config.bind_queue);
    }
}
