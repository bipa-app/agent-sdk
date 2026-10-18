//! Health and readiness surface for the service host.
//!
//! The health model splits into two independent dimensions:
//!
//! | Dimension | Represents | Failure meaning |
//! |-----------|------------|-----------------|
//! | **Core** (`CoreHealth`) | Journal store reachable, sweep loop alive, worker pool alive | Service is broken — stop routing |
//! | **Latency layer** (`LatencyLayerHealth`) | Optional relay/broker degraded | Extra latency but correctness intact |
//!
//! The top-level [`ServiceHealth`] composes both into a single
//! [`HealthStatus`] for Kubernetes-style probes:
//!
//! - **readiness** — `true` when the core is `Healthy`.  Latency-layer
//!   degradation does **not** remove the pod from the load balancer.
//! - **liveness** — `true` when `status` is not `Unhealthy`.  A pod
//!   that reports `Unhealthy` should be restarted.
//!
//! # Thread safety
//!
//! [`HealthSurface`] is the shared, lock-free handle that background
//! tasks write to (via `set_*` methods) and probe endpoints read from
//! (via [`HealthSurface::snapshot`]).  All state is stored in atomics
//! so reads never block writers and vice-versa.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use anyhow::Context;
use serde::Serialize;

// ─────────────────────────────────────────────────────────────────────
// Status enums
// ─────────────────────────────────────────────────────────────────────

/// Aggregate health status used by Kubernetes-style probes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    /// Everything is operating normally.
    Healthy,
    /// The service is functional but something optional is degraded.
    ///
    /// Core correctness is intact — the latency layer (relay, broker)
    /// is the source of degradation.
    Degraded,
    /// The service cannot process work.  Restart is advised.
    Unhealthy,
}

/// Health of the durable core (journal + workers).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreHealth {
    /// Store is reachable, sweep loop is alive, workers are alive.
    Healthy,
    /// One or more core subsystems have failed.
    Unhealthy,
}

/// Health of the optional latency-optimisation layer (relay, broker).
///
/// This is modelled as a separate axis because latency-layer outages
/// do not affect correctness — the journal guarantees delivery.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LatencyLayerHealth {
    /// Relay/broker is operating normally.
    Healthy,
    /// Relay/broker is unreachable or falling behind.
    Degraded,
    /// No latency layer is configured.
    NotConfigured,
}

/// Whether this host leads its store, under
/// [`EpochRecovery::PostgresLeader`](crate::config::EpochRecovery::PostgresLeader).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Leadership {
    /// The host takes no leader lock.
    #[default]
    NotApplicable,
    /// Another host leads; this one waits for the lock without workers.
    /// It reports ready so a rolling deploy (`maxUnavailable: 0`) can
    /// route to it and stop the old host, which frees the lock. It serves
    /// nothing else until it leads: the host binary and `LocalDaemon`
    /// open their gRPC port only after `ServiceHost::begin_epoch`, so a
    /// standby pod is ready with that port still closed.
    Standby,
    /// This host holds the store's leader lock.
    Leader,
    /// The lock's connection died; the host is shutting down.
    Lost,
}

// ─────────────────────────────────────────────────────────────────────
// Snapshot
// ─────────────────────────────────────────────────────────────────────

/// A point-in-time snapshot of service health.
///
/// Cheap to construct (no heap allocation) and suitable for
/// serialization into a probe response body.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub struct ServiceHealth {
    /// Aggregate status.
    pub status: HealthStatus,
    /// Core subsystem health.
    pub core: CoreHealth,
    /// Latency-layer subsystem health.
    pub latency_layer: LatencyLayerHealth,
    /// Whether the sweep loop background task is alive.
    pub sweep_loop_alive: bool,
    /// Whether the worker pool background tasks are alive.
    pub worker_pool_alive: bool,
    /// Whether this host leads its store.
    pub leadership: Leadership,
}

impl ServiceHealth {
    /// Returns `true` when the service should accept traffic.
    ///
    /// Readiness is gated on core health only.  Latency-layer
    /// degradation does not pull the pod from the load balancer
    /// because the journal still guarantees correctness.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.core == CoreHealth::Healthy
    }

    /// Returns `true` when the service is alive and should not be
    /// restarted.
    ///
    /// `Degraded` is still live — only `Unhealthy` triggers a restart.
    #[must_use]
    pub fn is_live(&self) -> bool {
        self.status != HealthStatus::Unhealthy
    }

    /// Serialize this snapshot to a JSON byte vector.
    ///
    /// Used by the HTTP health endpoint to build the response body.
    ///
    /// # Errors
    /// Returns an error if serialization fails (should not happen for
    /// this type, but errors are propagated per project convention).
    pub fn to_json_bytes(&self) -> anyhow::Result<Vec<u8>> {
        serde_json::to_vec(self).context("serializing ServiceHealth to JSON")
    }
}

// ─────────────────────────────────────────────────────────────────────
// Shared surface (lock-free)
// ─────────────────────────────────────────────────────────────────────

/// Shared, lock-free health surface.
///
/// Background tasks write component health via `set_*` methods.
/// Probe endpoints read via [`HealthSurface::snapshot`].
///
/// Internal representation uses `AtomicU8` for enum states and
/// `AtomicBool` for boolean flags.  All operations use `Relaxed`
/// ordering — health probes tolerate stale reads by one iteration
/// because the sweep and worker loops update on every cycle.
#[derive(Debug)]
pub struct HealthSurface {
    /// 0 = Healthy, 1 = Unhealthy
    core: AtomicU8,
    /// 0 = Healthy, 1 = Degraded, 2 = `NotConfigured`
    latency_layer: AtomicU8,
    sweep_alive: AtomicBool,
    workers_alive: AtomicBool,
    /// 0 = `NotApplicable`, 1 = `Standby`, 2 = `Leader`, 3 = `Lost`
    leadership: AtomicU8,
}

impl Default for HealthSurface {
    /// New surface starts as healthy with no latency layer.
    fn default() -> Self {
        Self {
            core: AtomicU8::new(0),
            latency_layer: AtomicU8::new(2), // NotConfigured
            sweep_alive: AtomicBool::new(false),
            workers_alive: AtomicBool::new(false),
            leadership: AtomicU8::new(0),
        }
    }
}

impl HealthSurface {
    /// Create a new health surface wrapped in an `Arc`.
    #[must_use]
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::default())
    }

    // ── Writers (called by background tasks) ─────────────────────

    /// Mark the sweep loop as alive or dead.
    pub fn set_sweep_alive(&self, alive: bool) {
        self.sweep_alive.store(alive, Ordering::Relaxed);
    }

    /// Mark the worker pool as alive or dead.
    pub fn set_workers_alive(&self, alive: bool) {
        self.workers_alive.store(alive, Ordering::Relaxed);
    }

    /// Update core health.
    pub fn set_core(&self, health: CoreHealth) {
        let val = match health {
            CoreHealth::Healthy => 0,
            CoreHealth::Unhealthy => 1,
        };
        self.core.store(val, Ordering::Relaxed);
    }

    /// Update whether this host leads its store.
    pub fn set_leadership(&self, leadership: Leadership) {
        let val = match leadership {
            Leadership::NotApplicable => 0,
            Leadership::Standby => 1,
            Leadership::Leader => 2,
            Leadership::Lost => 3,
        };
        self.leadership.store(val, Ordering::Relaxed);
    }

    /// Update latency-layer health.
    pub fn set_latency_layer(&self, health: LatencyLayerHealth) {
        let val = match health {
            LatencyLayerHealth::Healthy => 0,
            LatencyLayerHealth::Degraded => 1,
            LatencyLayerHealth::NotConfigured => 2,
        };
        self.latency_layer.store(val, Ordering::Relaxed);
    }

    // ── Readers (called by probe endpoints) ──────────────────────

    /// Take a point-in-time snapshot of service health.
    #[must_use]
    pub fn snapshot(&self) -> ServiceHealth {
        let core = match self.core.load(Ordering::Relaxed) {
            0 => CoreHealth::Healthy,
            _ => CoreHealth::Unhealthy,
        };
        let latency_layer = match self.latency_layer.load(Ordering::Relaxed) {
            0 => LatencyLayerHealth::Healthy,
            1 => LatencyLayerHealth::Degraded,
            _ => LatencyLayerHealth::NotConfigured,
        };
        let sweep_loop_alive = self.sweep_alive.load(Ordering::Relaxed);
        let workers_alive = self.workers_alive.load(Ordering::Relaxed);
        let leadership = match self.leadership.load(Ordering::Relaxed) {
            1 => Leadership::Standby,
            2 => Leadership::Leader,
            3 => Leadership::Lost,
            _ => Leadership::NotApplicable,
        };

        // Derive core health from component liveness. A standby host runs
        // no sweep and no workers by design, and stays ready.
        let components_alive =
            leadership == Leadership::Standby || (sweep_loop_alive && workers_alive);
        let effective_core =
            if core == CoreHealth::Healthy && components_alive && leadership != Leadership::Lost {
                CoreHealth::Healthy
            } else {
                CoreHealth::Unhealthy
            };

        let status = match (effective_core, latency_layer) {
            (CoreHealth::Unhealthy, _) => HealthStatus::Unhealthy,
            (CoreHealth::Healthy, LatencyLayerHealth::Degraded) => HealthStatus::Degraded,
            (CoreHealth::Healthy, _) => HealthStatus::Healthy,
        };

        ServiceHealth {
            status,
            core: effective_core,
            latency_layer,
            sweep_loop_alive,
            worker_pool_alive: workers_alive,
            leadership,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_surface_is_unhealthy_until_loops_start() {
        let surface = HealthSurface::default();
        let snap = surface.snapshot();

        // Workers and sweep haven't started yet.
        assert_eq!(snap.status, HealthStatus::Unhealthy);
        assert_eq!(snap.core, CoreHealth::Unhealthy);
        assert!(!snap.is_ready());
        assert!(!snap.is_live());
    }

    #[test]
    fn healthy_when_all_components_alive() {
        let surface = HealthSurface::default();
        surface.set_sweep_alive(true);
        surface.set_workers_alive(true);
        surface.set_core(CoreHealth::Healthy);

        let snap = surface.snapshot();
        assert_eq!(snap.status, HealthStatus::Healthy);
        assert!(snap.is_ready());
        assert!(snap.is_live());
    }

    #[test]
    fn degraded_when_latency_layer_down() {
        let surface = HealthSurface::default();
        surface.set_sweep_alive(true);
        surface.set_workers_alive(true);
        surface.set_core(CoreHealth::Healthy);
        surface.set_latency_layer(LatencyLayerHealth::Degraded);

        let snap = surface.snapshot();
        assert_eq!(snap.status, HealthStatus::Degraded);
        // Still ready — core is healthy.
        assert!(snap.is_ready());
        assert!(snap.is_live());
    }

    #[test]
    fn unhealthy_when_sweep_dies() {
        let surface = HealthSurface::default();
        surface.set_sweep_alive(false);
        surface.set_workers_alive(true);
        surface.set_core(CoreHealth::Healthy);

        let snap = surface.snapshot();
        assert_eq!(snap.status, HealthStatus::Unhealthy);
        assert!(!snap.is_ready());
    }

    #[test]
    fn unhealthy_when_workers_die() {
        let surface = HealthSurface::default();
        surface.set_sweep_alive(true);
        surface.set_workers_alive(false);
        surface.set_core(CoreHealth::Healthy);

        let snap = surface.snapshot();
        assert_eq!(snap.status, HealthStatus::Unhealthy);
        assert!(!snap.is_ready());
    }

    #[test]
    fn unhealthy_when_core_explicitly_set() {
        let surface = HealthSurface::default();
        surface.set_sweep_alive(true);
        surface.set_workers_alive(true);
        surface.set_core(CoreHealth::Unhealthy);

        let snap = surface.snapshot();
        assert_eq!(snap.status, HealthStatus::Unhealthy);
        assert!(!snap.is_ready());
    }

    #[test]
    fn latency_layer_not_configured_does_not_degrade() {
        let surface = HealthSurface::default();
        surface.set_sweep_alive(true);
        surface.set_workers_alive(true);
        surface.set_core(CoreHealth::Healthy);
        surface.set_latency_layer(LatencyLayerHealth::NotConfigured);

        let snap = surface.snapshot();
        assert_eq!(snap.status, HealthStatus::Healthy);
        assert_eq!(snap.latency_layer, LatencyLayerHealth::NotConfigured);
    }

    #[test]
    fn shared_returns_arc() {
        let surface = HealthSurface::shared();
        surface.set_sweep_alive(true);
        let cloned = Arc::clone(&surface);
        assert!(cloned.snapshot().sweep_loop_alive);
    }

    #[test]
    fn service_health_serializes_to_expected_json() -> anyhow::Result<()> {
        let health = ServiceHealth {
            status: HealthStatus::Degraded,
            core: CoreHealth::Healthy,
            latency_layer: LatencyLayerHealth::Degraded,
            sweep_loop_alive: true,
            worker_pool_alive: true,
            leadership: Leadership::Standby,
        };
        let json: serde_json::Value = serde_json::from_slice(&health.to_json_bytes()?)?;
        assert_eq!(json["leadership"], "standby");
        assert_eq!(json["status"], "degraded");
        assert_eq!(json["core"], "healthy");
        assert_eq!(json["latency_layer"], "degraded");
        assert_eq!(json["sweep_loop_alive"], true);
        assert_eq!(json["worker_pool_alive"], true);
        Ok(())
    }

    #[test]
    fn not_configured_latency_layer_serializes_correctly() -> anyhow::Result<()> {
        let health = ServiceHealth {
            status: HealthStatus::Healthy,
            core: CoreHealth::Healthy,
            latency_layer: LatencyLayerHealth::NotConfigured,
            sweep_loop_alive: true,
            worker_pool_alive: true,
            leadership: Leadership::NotApplicable,
        };
        let json: serde_json::Value = serde_json::from_slice(&health.to_json_bytes()?)?;
        assert_eq!(json["latency_layer"], "not_configured");
        Ok(())
    }

    /// A rolling deploy routes to the new host before the old one stops, so
    /// a host waiting for the leader lock (no sweep, no workers) is ready.
    #[test]
    fn a_standby_host_is_ready_and_a_host_that_lost_its_lock_is_not() {
        let surface = HealthSurface::default();
        assert!(
            !surface.snapshot().is_ready(),
            "no workers and no leader lock"
        );

        surface.set_leadership(Leadership::Standby);
        let standby = surface.snapshot();
        assert!(standby.is_ready() && standby.is_live(), "{standby:?}");
        assert_eq!(standby.leadership, Leadership::Standby);

        surface.set_leadership(Leadership::Leader);
        assert!(!surface.snapshot().is_ready(), "a leader needs its workers");
        surface.set_sweep_alive(true);
        surface.set_workers_alive(true);
        assert!(surface.snapshot().is_ready());

        surface.set_leadership(Leadership::Lost);
        let lost = surface.snapshot();
        assert!(!lost.is_ready() && !lost.is_live(), "{lost:?}");
    }
}
