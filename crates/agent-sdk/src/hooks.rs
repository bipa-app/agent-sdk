//! Agent lifecycle hooks for customization.
//!
//! Re-exported from [`agent_sdk_tools::hooks`] and [`agent_sdk_tools::audit`].

pub use agent_sdk_tools::audit::{NoopAuditSink, ToolAuditSink};
pub use agent_sdk_tools::hooks::*;
