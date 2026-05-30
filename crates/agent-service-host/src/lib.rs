//! # Agent Service Host
//!
//! Process-level composition layer for the agent server.
//!
//! This crate owns the concerns that sit **above** the domain logic in
//! [`agent_server`] and **below** any specific transport (gRPC, HTTP,
//! AMQP):
//!
//! - **Configuration model** ([`config`]) — typed, serde-stable settings
//!   for storage backend, worker pool sizing, retention policy, and
//!   transport enablement.
//! - **Postgres durable contract + runtime** ([`postgres`]) —
//!   reviewable schema, migration, repository-boundary definitions,
//!   and the current durable-core SQL implementation for task, thread,
//!   message, attempt, and checkpoint state.
//! - **Store registry** ([`stores`]) — constructs and owns the full set
//!   of durable store trait-objects that the journal and worker layers
//!   consume.
//! - **Health surface** ([`health`]) — lock-free health and readiness
//!   reporting that separates core correctness from latency-layer
//!   degradation.
//! - **Service lifecycle** ([`host`]) — startup / background-task /
//!   graceful-shutdown orchestration via a single [`ServiceHost`] struct.
//!   Boots a worker pool, runs sweep loops, and reports health.
//!
//! ## Crate boundary
//!
//! ```text
//! agent-sdk-core  (leaf types)
//!       ↑              ↑
//! agent-sdk-tools   agent-sdk-providers
//!       ↑      ↑         ↑        ↑
//! agent-sdk   agent-server  (domain logic)
//!                  ↑
//!         agent-service-host  ← you are here
//! ```
//!
//! `agent-service-host` depends on `agent-server` (and transitively on
//! core / tools / providers).  It does **not** depend on `agent-sdk`
//! (the local-agent umbrella), keeping the server deploy surface free
//! of local-agent conveniences.
//!
//! ## Transport integration
//!
//! Transport layers (gRPC, HTTP) are **not** owned by this crate.  They
//! compose on top by accepting a [`StoreRegistry`] reference:
//!
//! ```ignore
//! let host = ServiceHost::new(config, registry)?;
//! let grpc = GrpcTransport::new(host.stores());
//! tokio::select! {
//!     res = host.run() => res?,
//!     res = grpc.serve() => res?,
//! }
//! ```
//!
//! This keeps the service-host crate focused on composition and
//! lifecycle while transports remain pluggable. The reviewable gRPC
//! contract for the desktop/service path lives under
//! `crates/agent-service-host/proto/agent/service/v1`.
//!
//! [`ServiceHost`]: host::ServiceHost
//! [`StoreRegistry`]: stores::StoreRegistry

#![forbid(unsafe_code)]

pub mod broker;
pub mod collector_event_store;
pub mod config;
pub mod grpc;
pub mod health;
pub mod host;
pub mod http_health;
pub mod metrics;
#[cfg(feature = "otel")]
pub mod observability;
#[cfg(feature = "postgres")]
pub mod postgres;
pub mod proto;
pub mod registry_tool_executor;
pub mod relay;
pub mod runtime;
#[cfg(feature = "sqlite")]
pub mod sqlite;
pub mod stores;
pub mod wakeup;
pub mod watch;

/// Re-export of the `fail-rs` registry, available only under the
/// `failpoints` feature (which forwards to `agent-server/failpoints`).
///
/// Durability tests in this crate arm the `commit.before_event_commit`
/// failpoint — the one the durable completed-turn committers fire just
/// before their atomic `tx.commit()` — through this re-export, so they do
/// not need a direct `fail` dependency. Like `agent-server`'s mirror, the
/// symbol simply does not exist in a default build, keeping production
/// builds free of `fail-rs`.
#[cfg(feature = "failpoints")]
pub use agent_server::__fail_reexport as fail;

#[cfg(test)]
mod conformance;
#[cfg(test)]
mod durability_suite;
#[cfg(test)]
mod ga_regression;
#[cfg(test)]
mod journal_conformance;
