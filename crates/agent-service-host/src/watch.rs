//! Cross-instance thread-event watch lifecycle.
//!
//! Phase 8.4 (ENG-7968) closes the "broker as nudge, journal as
//! authority" contract for committed events.  An instance that does
//! **not** commit the event batch learns about it through an advisory
//! `thread_events_available` delivery, which this module turns into a
//! local
//! [`agent_server::journal::EventNotifier`](agent_server::journal::EventNotifier)
//! nudge after a durable replay from the event repository.
//!
//! The broker payload never drives state on its own:
//!
//! 1. Payload carries `{ thread_id, last_sequence }` — no event
//!    bodies, no auth context, no rendered content.
//! 2. Handler reads the authoritative events from
//!    [`agent_server::journal::EventRepository`].
//! 3. Handler forwards those events to the local notifier so any
//!    active [`agent_server::journal::EventStream`] wakes up and
//!    delivers the tail.
//! 4. Duplicate and out-of-order advisories short-circuit on the
//!    per-thread high-water map inside the handler.
//!
//! Execution never touches the broker payload; every correctness
//! property is owned by the durable log.
//!
//! # Fan-out semantics
//!
//! Every advisory must reach **every** instance serving the thread
//! because the nudge is inherently instance-scoped: it wakes local
//! subscribers, not remote ones.  Operators achieve that either by
//! running distinct queue names per pod (native fan-out) or by
//! pointing all pods at a fanout exchange and giving each pod an
//! anonymous queue.  Competing-consumer semantics on a shared queue
//! are not appropriate here — a peer that "wins" the delivery cannot
//! nudge anyone else's subscribers.

use std::sync::Arc;

use agent_server::journal::ThreadEventsWatchHandler;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

#[cfg(feature = "amqp")]
use super::broker::amqp_thread_events_consumer::{
    AmqpThreadEventsConsumer, AmqpThreadEventsConsumerConfig,
};

// ─────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────

/// Tunables for the cross-instance watch scheduler.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ThreadEventsWatchConfig {
    /// Whether the scheduler runs.
    ///
    /// Disabled by default so hosts that have not yet provisioned the
    /// broker-side queue bindings keep the journal-only fallback:
    /// clients that reconnect always replay from the durable event
    /// log, regardless of whether cross-instance fanout is wired.
    pub enabled: bool,
    /// AMQP consumer settings.  Ignored when
    /// `AmqpThreadEventsConsumerSection::enabled == false`.
    #[cfg(feature = "amqp")]
    pub amqp_consumer: AmqpThreadEventsConsumerSection,
}

/// AMQP-consumer-specific section of the watch config.
#[cfg(feature = "amqp")]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AmqpThreadEventsConsumerSection {
    /// Whether the AMQP consumer runs.
    pub enabled: bool,
    /// Consumer topology + broker settings.
    pub config: AmqpThreadEventsConsumerConfig,
}

// ─────────────────────────────────────────────────────────────────────
// Scheduler
// ─────────────────────────────────────────────────────────────────────

/// Composition root for the cross-instance thread-event watch
/// scheduler.
///
/// Spawns the AMQP consumer (when enabled) behind a single
/// [`CancellationToken`] and supervises it with bounded backoff so a
/// transient broker disconnect recovers on its own.
pub struct ThreadEventsWatchScheduler {
    config: ThreadEventsWatchConfig,
    handler: Arc<dyn ThreadEventsWatchHandler>,
}

impl ThreadEventsWatchScheduler {
    /// Construct a scheduler that forwards every delivered advisory
    /// to `handler`.
    #[must_use]
    pub fn new(
        config: ThreadEventsWatchConfig,
        handler: Arc<dyn ThreadEventsWatchHandler>,
    ) -> Self {
        Self { config, handler }
    }

    /// Spawn the consumer task.  Returns a handle that owns its
    /// lifetime; drop (or call
    /// [`ThreadEventsWatchSchedulerHandle::shutdown`]) to drain.
    pub fn spawn(self, cancel: CancellationToken) -> ThreadEventsWatchSchedulerHandle {
        let Self { config, handler } = self;

        #[cfg(feature = "amqp")]
        let consumer_handle = spawn_amqp_consumer_if_enabled(
            &config.amqp_consumer,
            Arc::clone(&handler),
            cancel.clone(),
        );

        #[cfg(not(feature = "amqp"))]
        let consumer_handle: Option<JoinHandle<Result<()>>> = {
            let _ = handler;
            let _ = &config;
            None
        };

        info!(
            amqp_consumer_enabled = consumer_handle.is_some(),
            "thread events watch scheduler started",
        );

        ThreadEventsWatchSchedulerHandle {
            cancel,
            consumer: consumer_handle,
        }
    }
}

#[cfg(feature = "amqp")]
fn spawn_amqp_consumer_if_enabled(
    section: &AmqpThreadEventsConsumerSection,
    handler: Arc<dyn ThreadEventsWatchHandler>,
    cancel: CancellationToken,
) -> Option<JoinHandle<Result<()>>> {
    if !section.enabled {
        return None;
    }
    let config = section.config.clone();
    Some(tokio::spawn(async move {
        supervise_amqp_consumer(config, handler, cancel).await
    }))
}

/// Supervise the AMQP consumer, restarting it after broker-side
/// stream close or transient error with a bounded backoff.
///
/// The consumer is purely a latency optimisation — the durable event
/// log keeps cross-instance delivery correct while the consumer is
/// down.  This supervisor exists only so the fast path recovers on
/// its own after a broker restart, network blip, or channel recycle.
#[cfg(feature = "amqp")]
async fn supervise_amqp_consumer(
    config: AmqpThreadEventsConsumerConfig,
    handler: Arc<dyn ThreadEventsWatchHandler>,
    cancel: CancellationToken,
) -> Result<()> {
    const INITIAL_BACKOFF: std::time::Duration = std::time::Duration::from_millis(250);
    const MAX_BACKOFF: std::time::Duration = std::time::Duration::from_secs(30);
    /// Runs that last at least this long are considered "stable" and
    /// reset the backoff so a later disconnect reconnects fast.
    const STABILITY_THRESHOLD: std::time::Duration = std::time::Duration::from_secs(60);

    let mut backoff = INITIAL_BACKOFF;
    loop {
        if cancel.is_cancelled() {
            return Ok(());
        }
        let consumer = AmqpThreadEventsConsumer::new(config.clone(), Arc::clone(&handler));
        let started = std::time::Instant::now();
        match consumer.run(cancel.clone()).await {
            Ok(()) if cancel.is_cancelled() => return Ok(()),
            Ok(()) => {
                warn!(
                    backoff_ms = u64::try_from(backoff.as_millis()).unwrap_or(u64::MAX),
                    "thread events watch consumer exited cleanly; restarting after backoff",
                );
            }
            Err(err) => {
                warn!(
                    error = %err,
                    backoff_ms = u64::try_from(backoff.as_millis()).unwrap_or(u64::MAX),
                    "thread events watch consumer exited with error; restarting after backoff",
                );
            }
        }

        if started.elapsed() >= STABILITY_THRESHOLD {
            backoff = INITIAL_BACKOFF;
        }

        tokio::select! {
            biased;
            () = cancel.cancelled() => return Ok(()),
            () = tokio::time::sleep(backoff) => {}
        }
        backoff = (backoff * 2).min(MAX_BACKOFF);
    }
}

// ─────────────────────────────────────────────────────────────────────
// Handle
// ─────────────────────────────────────────────────────────────────────

/// Handle returned by [`ThreadEventsWatchScheduler::spawn`].
///
/// Dropping the handle fires the shared [`CancellationToken`] so the
/// scheduler's loops exit on their own, but does **not** await their
/// drain — the tasks are detached.  Prefer an explicit
/// [`Self::shutdown`] so the caller can await drain and surface any
/// background-task errors.
pub struct ThreadEventsWatchSchedulerHandle {
    cancel: CancellationToken,
    consumer: Option<JoinHandle<Result<()>>>,
}

impl Drop for ThreadEventsWatchSchedulerHandle {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

impl ThreadEventsWatchSchedulerHandle {
    /// Cancellation token the scheduler observes.
    #[must_use]
    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    /// Cancel the scheduler and await its drain.
    ///
    /// Errors from the supervised consumer are logged at `warn!` and
    /// swallowed.  The consumer is a latency optimisation; shutdown
    /// must never fail the host's top-level drain.
    ///
    /// # Errors
    /// Currently never returns `Err`.  Callers should still use `?`
    /// to stay forward-compatible if a future change propagates
    /// errors instead of logging them.
    pub async fn shutdown(mut self) -> Result<()> {
        self.cancel.cancel();
        if let Some(handle) = self.consumer.take() {
            match handle.await {
                Ok(Ok(())) => {}
                Ok(Err(err)) => {
                    warn!(error = %err, "thread events watch consumer exited with error");
                }
                Err(err) => {
                    warn!(error = %err, "thread events watch consumer task panicked");
                }
            }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use agent_server::journal::CapturingThreadEventsWatchHandler;
    use anyhow::Context;

    #[test]
    fn default_watch_config_is_disabled() {
        let config = ThreadEventsWatchConfig::default();
        assert!(!config.enabled);
        #[cfg(feature = "amqp")]
        {
            assert!(!config.amqp_consumer.enabled);
        }
    }

    #[test]
    fn watch_config_round_trips_through_yaml() -> Result<()> {
        let yaml = r"
enabled: true
amqp_consumer:
  enabled: true
  config:
    queue: 'agent_sdk.thread_events.pod-a'
    consumer_tag_prefix: 'pod-a-watch'
    declare_queue: true
    bind_queue: true
";
        let config: ThreadEventsWatchConfig =
            serde_yaml::from_str(yaml).context("parse watch config yaml")?;
        assert!(config.enabled);
        #[cfg(feature = "amqp")]
        {
            assert!(config.amqp_consumer.enabled);
            assert_eq!(
                config.amqp_consumer.config.queue,
                "agent_sdk.thread_events.pod-a",
            );
            assert!(config.amqp_consumer.config.declare_queue);
            assert!(config.amqp_consumer.config.bind_queue);
        }
        Ok(())
    }

    #[tokio::test]
    async fn handle_shutdown_is_idempotent() -> Result<()> {
        // Scheduler with the consumer disabled — shutdown must still
        // drain cleanly and return Ok.
        let handler: Arc<dyn ThreadEventsWatchHandler> =
            Arc::new(CapturingThreadEventsWatchHandler::new());
        let scheduler = ThreadEventsWatchScheduler::new(
            ThreadEventsWatchConfig {
                enabled: true,
                #[cfg(feature = "amqp")]
                amqp_consumer: AmqpThreadEventsConsumerSection::default(),
            },
            handler,
        );
        let cancel = CancellationToken::new();
        let handle = scheduler.spawn(cancel.clone());
        handle.shutdown().await.context("clean shutdown")?;
        assert!(cancel.is_cancelled());
        Ok(())
    }
}
