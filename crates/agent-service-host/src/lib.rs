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
//! - **Store registry** ([`stores`]) — constructs and owns the full set
//!   of durable store trait-objects that the journal and worker layers
//!   consume.
//! - **Service lifecycle** ([`host`]) — startup / background-task /
//!   graceful-shutdown orchestration via a single [`ServiceHost`] struct.
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
//! lifecycle while transports remain pluggable.
//!
//! [`ServiceHost`]: host::ServiceHost
//! [`StoreRegistry`]: stores::StoreRegistry

#![forbid(unsafe_code)]

pub mod config;
pub mod host;
pub mod stores;
