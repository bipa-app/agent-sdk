//! # agent-acp
//!
//! ACP (Agent Client Protocol) stdio transport: expose an agent as a
//! newline-delimited JSON-RPC server, as spoken by the
//! [buzz-acp](https://github.com/block/buzz) harness.
//!
//! A client (harness) spawns the agent process and drives it over stdio:
//! `initialize` → `session/new` → `session/prompt` (blocking, resolves with
//! a `stopReason`), consuming streamed `session/update` notifications along
//! the way, and cancelling via the `session/cancel` notification. This crate
//! owns the wire and the dispatch loop; what a prompt *does* is supplied
//! through the [`PromptHandler`] seam.
//!
//! ```no_run
//! use agent_acp::{AcpServer, PromptHandler, PromptRequest, PromptError, StopReason, UpdateSink};
//! use tokio_util::sync::CancellationToken;
//!
//! struct Echo;
//!
//! #[async_trait::async_trait]
//! impl PromptHandler for Echo {
//!     async fn prompt(
//!         &self,
//!         request: PromptRequest,
//!         updates: UpdateSink,
//!         _cancel: CancellationToken,
//!     ) -> Result<StopReason, PromptError> {
//!         let text = request.blocks.join("\n");
//!         updates
//!             .agent_message_chunk(&text)
//!             .await
//!             .map_err(|e| PromptError::new(e.to_string()))?;
//!         Ok(StopReason::EndTurn)
//!     }
//! }
//!
//! # async fn run() -> std::io::Result<()> {
//! AcpServer::new(Echo).serve_stdio().await
//! # }
//! ```
//!
//! ## Protocol version stance
//!
//! This server implements **"ACP v2 as spoken by buzz-acp at block/buzz
//! `7e34bee`"** — the harness requests `protocolVersion: 2` ahead of the
//! upstream ACP RFD, with hand-rolled wire shapes. The compatibility
//! contract is the recorded fixture set in `tests/fixtures/`, captured
//! verbatim from that revision's source.
//!
//! **Decision (ENG-9395):** the published `agent-client-protocol` crate
//! (v2.0.0, Zed) is **not** used for the wire types. The harness's "v2" is a
//! pre-RFD squat whose shapes are pinned only by the revision we deploy
//! against; a typed third-party contract that drifts from those fixtures
//! would fail at runtime, not at compile time. Envelopes here are small,
//! serde-direct, and fixture-tested instead. Revisit when the upstream ACP
//! RFD merges and block/buzz tracks a released protocol version.
//!
//! ## Liveness
//!
//! buzz-acp kills turns on an idle timeout keyed to stdout activity.
//! Handlers must stream [`UpdateSink`] updates while working; the durable
//! backend layered on this crate in later milestones emits deltas, tool
//! progress, and keepalives for exactly that reason.

#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod backend;
mod mapper;
pub mod run;
pub mod server;
pub mod session;
pub mod wire;

pub use backend::{
    AcpBackend, AcpRunHandle, BackendError, BackendPromptHandler, BackendTaskStatus, EventStream,
    RunEvent, RunStreamItem,
};
pub use server::{
    AcpServer, AgentInfo, PromptError, PromptHandler, PromptRequest, UpdateSink, UpdateSinkClosed,
};
pub use session::NewSessionParams;
pub use wire::{PROTOCOL_VERSION, StopReason};
