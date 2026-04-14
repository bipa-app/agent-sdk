//! Broker adapter implementations composable into the outbox relay.
//!
//! Phase 8.2 lands the first concrete [`BrokerAdapter`] that talks to a
//! real broker — AMQP 0.9.1 via `lapin`.  The trait itself lives in
//! [`agent_server::journal::broker`]; this module ships the host-side
//! implementations so transport libraries and connection pools stay
//! outside the domain crate.
//!
//! # Layering
//!
//! ```text
//!    OutboxStore  ─►  RelayWorker  ─►  Publisher  ─►  BrokerAdapter
//!                                                       ▲
//!                                                       │
//!                                                       └── Amqp / InMemory
//! ```
//!
//! # Implementations
//!
//! - [`amqp::AmqpBrokerAdapter`] — production AMQP adapter with
//!   publisher confirms.  Gated on the `amqp` cargo feature.
//! - [`InMemoryBrokerAdapter`] is re-exported from
//!   [`agent_server::journal::broker`] for consumers that want it
//!   alongside the real adapter.

#[cfg(feature = "amqp")]
pub mod amqp;

pub use agent_server::journal::broker::{BrokerAdapter, InMemoryBrokerAdapter};
