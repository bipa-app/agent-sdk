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
//! - **Postgres durable contract** ([`postgres`]) — reviewable schema,
//!   migration, and repository-boundary definitions for the current
//!   durable core, ready for a future SQL implementation.
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

pub mod config;
pub mod health;
pub mod host;
pub mod postgres;
pub mod stores;
