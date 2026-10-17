//! The session view projection. The types and the pure [`fold`] live in
//! `agent-sdk-foundation`, where the match over [`AgentEvent`] can be
//! exhaustive; stores and hosts reach them here.
//!
//! [`AgentEvent`]: agent_sdk_foundation::events::AgentEvent

pub use agent_sdk_foundation::view::*;
