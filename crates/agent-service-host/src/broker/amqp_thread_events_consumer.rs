//! AMQP 0.9.1 consumer for `thread_events_available` advisory
//! deliveries.
//!
//! Phase 8.4 wires this consumer to turn cross-instance advisories
//! into **local notifier nudges** without ever treating the broker
//! payload as the authoritative event stream.  On every delivery the
//! consumer:
//!
//! 1. Deserialises the JSON payload into
//!    [`agent_server::journal::outbox_message::ThreadEventsAvailablePayload`].
//! 2. Calls a [`ThreadEventsWatchHandler`] to re-check the durable
//!    event repository and replay any missing suffix through the
//!    local [`agent_server::journal::EventNotifier`].
//! 3. Acks the broker message on success (including benign duplicates
//!    and unknown-thread advisories).
//! 4. Nacks with requeue on transient handler failures so the broker
//!    redelivers and the journal stays the authority for recovery.
//!
//! # Topology
//!
//! The consumer declares (idempotently) a durable queue and binds it
//! to the configured exchange with the `thread_events_available`
//! routing key produced by the Phase 8.2 publisher:
//!
//! ```text
//!    producer → exchange (topic) ─── routing_key_prefix.thread_events_available ──▶ queue
//! ```
//!
//! Queues are named deterministically; pods on the same service that
//! share a queue name see **competing-consumer** semantics so each
//! advisory goes to exactly one pod.  Different queue names produce a
//! fan-out topology (every pod receives every advisory).  Phase 8.4
//! prefers fan-out per instance because an advisory's only effect is
//! to nudge the **local** subscriber hub — the work is not
//! deduplicatable across instances.  Operators that want that should
//! run distinct queue names per pod.
//!
//! # Shutdown
//!
//! Same contract as the task-wakeup consumer: the `run` future is
//! driven until a [`CancellationToken`] fires.  Cancellation cancels
//! the broker-side subscription and closes the channel.  In-flight
//! deliveries run to completion; buffered-but-unpulled deliveries are
//! requeued automatically.

use std::sync::Arc;

use agent_server::journal::ThreadEventsWatchHandler;
use agent_server::journal::outbox_message::{OutboxMessageKind, ThreadEventsAvailablePayload};
use anyhow::{Context, Result};
use futures::StreamExt;
use lapin::options::{
    BasicAckOptions, BasicConsumeOptions, BasicNackOptions, ConfirmSelectOptions,
    ExchangeDeclareOptions, QueueBindOptions, QueueDeclareOptions,
};
use lapin::types::{FieldTable, ShortString};
use lapin::{Channel, Connection, ConnectionProperties, Consumer};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use super::amqp::{AmqpBrokerConfig, AmqpExchangeKind};

// ─────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────

/// Tunables for an AMQP `thread_events_available` consumer.
///
/// The defaults line up with the Phase 8.2 publisher defaults so a
/// single shared `AmqpBrokerConfig` (or a pair of them pointing at
/// the same broker) works out of the box for local development.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AmqpThreadEventsConsumerConfig {
    /// Broker connection settings.  Re-uses the publisher's config so
    /// deployments can share routing-key prefixes, exchange names,
    /// and credentials.
    pub broker: AmqpBrokerConfig,
    /// Durable queue name.  Each pod should usually bind its **own**
    /// queue name because advisories are per-instance nudges — every
    /// subscriber on a given pod only benefits from the advisories
    /// that pod receives.  Pods that share a queue name see
    /// competing-consumer semantics and will miss their peers'
    /// nudges.
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

impl Default for AmqpThreadEventsConsumerConfig {
    fn default() -> Self {
        Self {
            broker: AmqpBrokerConfig::default(),
            queue: "agent_sdk.thread_events".into(),
            consumer_tag_prefix: "agent-sdk-thread-events".into(),
            declare_queue: false,
            bind_queue: false,
        }
    }
}

impl AmqpThreadEventsConsumerConfig {
    /// Routing key the consumer binds to; always the publisher's
    /// `thread_events_available` key so the two halves stay in sync.
    #[must_use]
    pub fn routing_key(&self) -> String {
        self.broker
            .routing_key(OutboxMessageKind::ThreadEventsAvailable)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Consumer
// ─────────────────────────────────────────────────────────────────────

/// AMQP consumer that pulls `thread_events_available` deliveries and
/// forwards them to a [`ThreadEventsWatchHandler`].
///
/// Generic over the handler so tests can swap the default
/// notifier-backed handler for a capturing double without touching the
/// broker code.
pub struct AmqpThreadEventsConsumer {
    config: AmqpThreadEventsConsumerConfig,
    handler: Arc<dyn ThreadEventsWatchHandler>,
    connection: Mutex<Option<ConnectionState>>,
}

struct ConnectionState {
    // Kept alive so the channel stays usable.
    #[allow(dead_code)]
    connection: Connection,
    channel: Channel,
}

impl AmqpThreadEventsConsumer {
    /// Construct a consumer bound to `handler`.
    ///
    /// The broker connection is opened lazily on [`Self::run`] so
    /// cheap construction works from tests and config validators.
    #[must_use]
    pub fn new(
        config: AmqpThreadEventsConsumerConfig,
        handler: Arc<dyn ThreadEventsWatchHandler>,
    ) -> Self {
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
            "thread events watch consumer starting",
        );

        let (channel, mut consumer) = tokio::select! {
            biased;
            () = cancel.cancelled() => {
                info!("thread events watch consumer cancelled during connection establishment");
                return Ok(());
            }
            result = self.open_consumer() => result?,
        };
        let consumer_tag = consumer.tag().to_string();

        loop {
            tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    info!(%consumer_tag, "thread events watch consumer shutting down");
                    let _ = channel
                        .basic_cancel(
                            ShortString::from(consumer_tag.clone()),
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
                            warn!(error = %err, "thread events watch consumer received broker error");
                            self.invalidate_connection().await;
                            return Err(anyhow::Error::new(err)
                                .context("AMQP thread events watch consumer stream error"));
                        }
                        None => {
                            info!("thread events watch consumer stream closed by broker; exiting so the supervisor can restart with a fresh connection");
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
                    self.config.queue.as_str().into(),
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
            let routing_key = self.config.routing_key();
            channel
                .queue_bind(
                    self.config.queue.as_str().into(),
                    self.config.broker.exchange.as_str().into(),
                    routing_key.as_str().into(),
                    QueueBindOptions::default(),
                    FieldTable::default(),
                )
                .await
                .with_context(|| {
                    format!(
                        "bind queue {} to exchange {} with routing key {}",
                        self.config.queue, self.config.broker.exchange, routing_key,
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
                self.config.queue.as_str().into(),
                consumer_tag.as_str().into(),
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
            .context("resolve AMQP URL for thread events watch consumer")?;

        info!("opening AMQP connection for thread events watch consumer");
        let connection = Connection::connect(&url, ConnectionProperties::default())
            .await
            .context("connect to AMQP broker for thread events watch consumer")?;
        let channel = connection
            .create_channel()
            .await
            .context("open AMQP channel for thread events watch consumer")?;
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
                    self.config.broker.exchange.as_str().into(),
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
                        "declare AMQP exchange {} from thread events watch consumer",
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

        // Phase 9 · B5: time delivery → ack/nack so dashboards can
        // see watch-side latency the same way they see wakeup-side.
        #[cfg(feature = "otel")]
        let started_at = std::time::Instant::now();

        let payload = match serde_json::from_slice::<ThreadEventsAvailablePayload>(&delivery.data) {
            Ok(payload) => payload,
            Err(err) => {
                warn!(
                    routing_key,
                    error = %err,
                    "thread events watch delivery payload did not decode; rejecting without requeue",
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
                    warn!(
                        error = %err,
                        "failed to nack undecodeable thread events watch delivery",
                    );
                }
                #[cfg(feature = "otel")]
                crate::observability::HostMetrics::global().record_amqp_consume(
                    &routing_key,
                    started_at.elapsed().as_secs_f64(),
                    false,
                );
                return;
            }
        };

        let outcome = self.handler.handle_payload(&payload).await;

        match outcome {
            Ok(outcome) => {
                debug!(
                    routing_key,
                    delivery_tag,
                    thread_id = %payload.thread_id,
                    last_sequence = payload.last_sequence,
                    forwarded = outcome.forwarded(),
                    emitted_count = outcome.emitted_count(),
                    "thread events watch delivery handled",
                );
                if let Err(err) = delivery.acker.ack(BasicAckOptions::default()).await {
                    warn!(
                        routing_key,
                        delivery_tag,
                        error = %err,
                        "failed to ack thread events watch delivery after re-check",
                    );
                }
                #[cfg(feature = "otel")]
                crate::observability::HostMetrics::global().record_amqp_consume(
                    &routing_key,
                    started_at.elapsed().as_secs_f64(),
                    true,
                );
            }
            Err(err) => {
                warn!(
                    routing_key,
                    delivery_tag,
                    thread_id = %payload.thread_id,
                    last_sequence = payload.last_sequence,
                    error = %err,
                    "thread events watch handler failed; nacking for broker redelivery",
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
                        "failed to nack thread events watch delivery; broker may redeliver on reconnect",
                    );
                }
                #[cfg(feature = "otel")]
                crate::observability::HostMetrics::global().record_amqp_consume(
                    &routing_key,
                    started_at.elapsed().as_secs_f64(),
                    false,
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests (payload + config; broker integration tests are in watch.rs)
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_match_publisher_routing_scheme() {
        let config = AmqpThreadEventsConsumerConfig::default();
        assert_eq!(
            config.routing_key(),
            "agent_sdk.outbox.thread_events_available",
        );
    }

    #[test]
    fn config_routing_key_follows_explicit_prefix() {
        let mut config = AmqpThreadEventsConsumerConfig::default();
        config.broker.routing_key_prefix = "tenant.acme".into();
        assert_eq!(config.routing_key(), "tenant.acme.thread_events_available");
    }

    #[test]
    fn config_round_trips_through_yaml() -> Result<()> {
        let config = AmqpThreadEventsConsumerConfig {
            queue: "agent_sdk.thread_events.pod-one".into(),
            consumer_tag_prefix: "pod-one-watch".into(),
            declare_queue: true,
            bind_queue: true,
            ..AmqpThreadEventsConsumerConfig::default()
        };
        let yaml = serde_yaml::to_string(&config).context("serialise")?;
        let parsed: AmqpThreadEventsConsumerConfig =
            serde_yaml::from_str(&yaml).context("round trip")?;
        assert_eq!(parsed.queue, config.queue);
        assert_eq!(parsed.consumer_tag_prefix, config.consumer_tag_prefix);
        assert!(parsed.declare_queue);
        assert!(parsed.bind_queue);
        Ok(())
    }

    #[test]
    fn default_config_does_not_declare_queue_by_default() {
        // Production deploys usually have the queue provisioned out
        // of band — the default must match that so a misconfigured
        // pod does not silently start a topology change.
        let config = AmqpThreadEventsConsumerConfig::default();
        assert!(!config.declare_queue);
        assert!(!config.bind_queue);
    }

    #[test]
    fn default_queue_name_is_distinct_from_task_wakeup_queue() {
        // Same broker, different queues — the two consumer classes
        // must not compete for each other's deliveries even when
        // deployed with default names.
        let events = AmqpThreadEventsConsumerConfig::default();
        assert_ne!(events.queue, "agent_sdk.wakeup");
        assert_ne!(events.routing_key(), "agent_sdk.outbox.task_wakeup");
    }
}
