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

// ─────────────────────────────────────────────────────────────────────
// Status enums
// ─────────────────────────────────────────────────────────────────────

/// Aggregate health status used by Kubernetes-style probes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatencyLayerHealth {
    /// Relay/broker is operating normally.
    Healthy,
    /// Relay/broker is unreachable or falling behind.
    Degraded,
    /// No latency layer is configured.
    NotConfigured,
}

// ─────────────────────────────────────────────────────────────────────
// Snapshot
// ─────────────────────────────────────────────────────────────────────

/// A point-in-time snapshot of service health.
///
/// Cheap to construct (no heap allocation) and suitable for
/// serialization into a probe response body.
#[derive(Clone, Debug, PartialEq, Eq)]
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
}

impl Default for HealthSurface {
    /// New surface starts as healthy with no latency layer.
    fn default() -> Self {
        Self {
            core: AtomicU8::new(0),
            latency_layer: AtomicU8::new(2), // NotConfigured
            sweep_alive: AtomicBool::new(false),
            workers_alive: AtomicBool::new(false),
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

        // Derive core health from component liveness.
        let effective_core = if core == CoreHealth::Healthy && sweep_loop_alive && workers_alive {
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
}
