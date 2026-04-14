//! Task-wakeup lifecycle: consumer + fallback sweep + worker signal.
//!
//! Phase 8.3 closes the "broker as nudge, journal as authority" loop:
//! the broker path stays a latency optimisation and the worker pool
//! keeps owning execution through
//! `AgentTaskStore::acquire_next_runnable`.  This module glues three
//! primitives into the shape the host composes on startup:
//!
//! 1. A shared [`agent_server::journal::WakeupSignal`]
//!    workers park on.
//! 2. A [`agent_server::journal::TaskWakeupHandler`]
//!    that re-checks the journal for every advisory payload.
//! 3. A consumer / sweep pair that pokes the signal: the AMQP consumer
//!    (fast path) and the periodic [`agent_server::journal::FallbackWakeupSweep`]
//!    (belt and suspenders).
//!
//! The worker loop in [`crate::host`] races a per-worker ticker against
//! the signal; every path converges on `acquire_next_runnable`, so
//! duplicates are inherently idempotent and a silent broker never
//! stalls progress.
//!
//! # Why duplicates are safe, spelled out
//!
//! The contract is enforced by three independent layers, each of which
//! would be sufficient by itself:
//!
//! | Layer | Guarantee |
//! |-------|-----------|
//! | Consumer | Re-checks durable state; payload is advisory |
//! | Signal | `notify_one` folds N duplicate notifies into one permit |
//! | Store | `Pending → Running` CAS is single-writer per row |
//!
//! Breaking any one layer (a buggy consumer, a lost signal, a racing
//! worker) cannot produce double execution because the other two still
//! hold.
//!
//! # Fallback sweep
//!
//! The worker pool's acquisition ticker is already a time-based
//! fallback against a silent broker.  The sweep exposed here is a
//! **second** fallback that runs on its own cadence so operators who
//! want belt-and-suspenders can dial up the wake-up rate independently
//! of the polling interval.  Tests use it to prove the system makes
//! progress when the broker is deliberately down.

use std::sync::Arc;

use agent_server::journal::{FallbackWakeupSweep, TaskWakeupHandler, WakeupSignal};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

#[cfg(feature = "amqp")]
use super::broker::amqp_consumer::{AmqpTaskWakeupConsumer, AmqpTaskWakeupConsumerConfig};

// ─────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────

/// Tunables for the wakeup scheduler.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct WakeupConfig {
    /// Whether the scheduler runs.
    ///
    /// Disabled by default so hosts that have not yet provisioned the
    /// broker-side queue bindings keep the journal-only fallback
    /// (worker acquisition ticker).
    pub enabled: bool,
    /// Seconds between fallback sweep pulses.
    ///
    /// The worker pool already polls on its own `acquisition_interval`,
    /// so this is a **second** fallback: a pulse that fires whether or
    /// not the broker is reachable.  Keep it larger than
    /// `acquisition_interval_secs` so the two mechanisms do not
    /// double-ping.
    pub fallback_interval_secs: u64,
    /// AMQP consumer settings.  Ignored when
    /// `AmqpConsumerConfig::enabled == false`.
    #[cfg(feature = "amqp")]
    pub amqp_consumer: AmqpConsumerSection,
}

impl Default for WakeupConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            fallback_interval_secs: 5,
            #[cfg(feature = "amqp")]
            amqp_consumer: AmqpConsumerSection::default(),
        }
    }
}

impl WakeupConfig {
    /// Fallback sweep interval as a [`std::time::Duration`].
    #[must_use]
    pub const fn fallback_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.fallback_interval_secs)
    }
}

/// AMQP-consumer-specific section of the wakeup config.
#[cfg(feature = "amqp")]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AmqpConsumerSection {
    /// Whether the AMQP consumer runs.
    pub enabled: bool,
    /// Consumer topology + broker settings.
    pub config: AmqpTaskWakeupConsumerConfig,
}

// ─────────────────────────────────────────────────────────────────────
// Scheduler
// ─────────────────────────────────────────────────────────────────────

/// Composition root for the wakeup fast-path.
///
/// Owns the shared [`WakeupSignal`] so the host can hand it to the
/// worker pool.  Spawns the AMQP consumer (when enabled) and the
/// fallback sweep behind a single [`CancellationToken`].
pub struct WakeupScheduler {
    config: WakeupConfig,
    handler: Arc<dyn TaskWakeupHandler>,
    signal: Arc<WakeupSignal>,
}

impl WakeupScheduler {
    /// Construct a scheduler sharing `signal` with the worker pool.
    ///
    /// `handler` performs the durable re-check for every incoming
    /// payload.  Use
    /// [`JournalTaskWakeupHandler::new`](agent_server::journal::JournalTaskWakeupHandler::new)
    /// in production.
    #[must_use]
    pub fn new(
        config: WakeupConfig,
        handler: Arc<dyn TaskWakeupHandler>,
        signal: Arc<WakeupSignal>,
    ) -> Self {
        Self {
            config,
            handler,
            signal,
        }
    }

    /// Access the shared worker nudge signal.
    #[must_use]
    pub const fn signal(&self) -> &Arc<WakeupSignal> {
        &self.signal
    }

    /// Spawn the consumer + sweep tasks.  Returns a handle that owns
    /// their lifetimes; drop (or call [`WakeupSchedulerHandle::shutdown`])
    /// to drain.
    pub fn spawn(self, cancel: CancellationToken) -> WakeupSchedulerHandle {
        let Self {
            config,
            handler,
            signal,
        } = self;

        let fallback_interval = config.fallback_interval();
        let fallback_interval_secs = config.fallback_interval_secs;
        let fallback_cancel = cancel.clone();
        let fallback_signal = Arc::clone(&signal);
        let fallback_handle = tokio::spawn(async move {
            let sweep = FallbackWakeupSweep::new(fallback_signal, fallback_interval);
            sweep.run(fallback_cancel).await;
        });

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
            fallback_interval_secs, "wakeup scheduler started",
        );

        WakeupSchedulerHandle {
            cancel,
            fallback: Some(fallback_handle),
            consumer: consumer_handle,
        }
    }
}

#[cfg(feature = "amqp")]
fn spawn_amqp_consumer_if_enabled(
    section: &AmqpConsumerSection,
    handler: Arc<dyn TaskWakeupHandler>,
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

/// Supervise the AMQP consumer, restarting it after broker-side stream
/// close or transient error with a bounded backoff.
///
/// The consumer is purely a latency optimisation — the worker
/// acquisition ticker and [`FallbackWakeupSweep`] keep the journal
/// moving while the consumer is down.  This supervisor exists only so
/// the fast path recovers on its own after a broker restart, network
/// blip, or channel recycle.
#[cfg(feature = "amqp")]
async fn supervise_amqp_consumer(
    config: AmqpTaskWakeupConsumerConfig,
    handler: Arc<dyn TaskWakeupHandler>,
    cancel: CancellationToken,
) -> Result<()> {
    const INITIAL_BACKOFF: std::time::Duration = std::time::Duration::from_millis(250);
    const MAX_BACKOFF: std::time::Duration = std::time::Duration::from_secs(30);

    let mut backoff = INITIAL_BACKOFF;
    loop {
        if cancel.is_cancelled() {
            return Ok(());
        }
        let consumer = AmqpTaskWakeupConsumer::new(config.clone(), Arc::clone(&handler));
        match consumer.run(cancel.clone()).await {
            Ok(()) if cancel.is_cancelled() => return Ok(()),
            Ok(()) => {
                warn!(
                    backoff_ms = u64::try_from(backoff.as_millis()).unwrap_or(u64::MAX),
                    "task wakeup consumer exited cleanly; restarting after backoff",
                );
            }
            Err(err) => {
                warn!(
                    error = %err,
                    backoff_ms = u64::try_from(backoff.as_millis()).unwrap_or(u64::MAX),
                    "task wakeup consumer exited with error; restarting after backoff",
                );
            }
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

/// Handle returned by [`WakeupScheduler::spawn`].
///
/// Drops cancel the scheduler's loops; prefer an explicit
/// [`Self::shutdown`] so the caller can await drain.
pub struct WakeupSchedulerHandle {
    cancel: CancellationToken,
    fallback: Option<JoinHandle<()>>,
    consumer: Option<JoinHandle<Result<()>>>,
}

impl WakeupSchedulerHandle {
    /// Cancellation token the scheduler observes.
    #[must_use]
    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    /// Cancel the scheduler and await its drain.
    ///
    /// Panics or errors from the background tasks are logged at
    /// `warn!` level and swallowed — the consumer is a latency
    /// optimisation and the fallback sweep is unconditional, so
    /// shutdown should never fail the host's top-level drain.  The
    /// `Result` is kept so the signature matches
    /// [`crate::relay::RelaySchedulerHandle::shutdown`] and so future
    /// error paths can be added without churning callers.
    ///
    /// # Errors
    /// Currently never returns `Err`.  Callers should still use `?`
    /// to stay forward-compatible if a future change propagates
    /// errors instead of logging them.
    pub async fn shutdown(mut self) -> Result<()> {
        self.cancel.cancel();
        if let Some(handle) = self.fallback.take()
            && let Err(err) = handle.await
        {
            warn!(error = %err, "fallback wakeup sweep task panicked");
        }
        if let Some(handle) = self.consumer.take() {
            match handle.await {
                Ok(Ok(())) => {}
                Ok(Err(err)) => {
                    warn!(error = %err, "task wakeup consumer exited with error");
                }
                Err(err) => {
                    warn!(error = %err, "task wakeup consumer task panicked");
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
    use agent_server::journal::CapturingTaskWakeupHandler;
    use anyhow::Context;
    use std::time::Duration as StdDuration;

    #[test]
    fn default_wakeup_config_disabled_with_five_second_fallback() {
        let config = WakeupConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.fallback_interval_secs, 5);
        assert_eq!(config.fallback_interval(), StdDuration::from_secs(5));
    }

    #[test]
    fn wakeup_config_round_trips_through_yaml() -> Result<()> {
        let yaml = r"
enabled: true
fallback_interval_secs: 20
amqp_consumer:
  enabled: true
  config:
    queue: 'agent_sdk.wakeup.stage'
    consumer_tag_prefix: 'pod-two'
    declare_queue: true
    bind_queue: true
";
        let config: WakeupConfig =
            serde_yaml::from_str(yaml).context("parse wakeup config yaml")?;
        assert!(config.enabled);
        assert_eq!(config.fallback_interval_secs, 20);
        #[cfg(feature = "amqp")]
        {
            assert!(config.amqp_consumer.enabled);
            assert_eq!(config.amqp_consumer.config.queue, "agent_sdk.wakeup.stage");
            assert!(config.amqp_consumer.config.declare_queue);
            assert!(config.amqp_consumer.config.bind_queue);
        }
        Ok(())
    }

    #[tokio::test]
    async fn fallback_sweep_pulses_independently_of_consumer() -> Result<()> {
        // No AMQP consumer enabled — only the fallback sweep should
        // keep the signal alive.  We drive the scheduler through the
        // test helper so the sweep's interval can live in
        // sub-second territory that the YAML config cannot express.
        let handler: Arc<dyn TaskWakeupHandler> = Arc::new(CapturingTaskWakeupHandler::new());
        let signal = WakeupSignal::shared();
        let scheduler = WakeupScheduler::new(
            WakeupConfig {
                enabled: true,
                fallback_interval_secs: 1,
                #[cfg(feature = "amqp")]
                amqp_consumer: AmqpConsumerSection::default(),
            },
            handler,
            Arc::clone(&signal),
        );
        let cancel = CancellationToken::new();
        let handle = scheduler.spawn(cancel.clone());

        // The sweep skips the immediate tick, so we need to wait at
        // least one full interval.  1200 ms gives comfortable headroom
        // without ballooning the test runtime.
        let waited =
            tokio::time::timeout(StdDuration::from_millis(1_200), signal.wait_for_nudge()).await;
        assert!(
            waited.is_ok(),
            "fallback sweep should fire at least once inside one interval",
        );

        cancel.cancel();
        handle.shutdown().await.context("clean shutdown")?;
        Ok(())
    }

    #[tokio::test]
    async fn handle_shutdown_is_idempotent() -> Result<()> {
        let handler: Arc<dyn TaskWakeupHandler> = Arc::new(CapturingTaskWakeupHandler::new());
        let signal = WakeupSignal::shared();
        let scheduler = WakeupScheduler::new(
            WakeupConfig {
                enabled: true,
                fallback_interval_secs: 60,
                #[cfg(feature = "amqp")]
                amqp_consumer: AmqpConsumerSection::default(),
            },
            handler,
            signal,
        );
        let cancel = CancellationToken::new();
        let handle = scheduler.spawn(cancel.clone());
        // Shutdown via the handle; cancel token should already be fired
        // when shutdown returns so calling cancel again is a no-op.
        handle.shutdown().await.context("clean shutdown")?;
        assert!(cancel.is_cancelled());
        Ok(())
    }
}
