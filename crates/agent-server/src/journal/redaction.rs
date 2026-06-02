//! Compatibility shim — the canonical redaction types now live in
//! [`agent_sdk_foundation::privacy::redaction`].
//!
//! This module re-exports them so existing call sites in
//! `agent-server` (and downstream tests via
//! `agent_server::journal::redaction::*`) keep compiling unchanged.
//!
//! New code should reach for the SDK path directly:
//!
//! ```
//! use agent_sdk_foundation::privacy::{RedactionPolicy, redact_value};
//! let policy = RedactionPolicy::baseline();
//! let _ = redact_value(&serde_json::json!({"api_key": "sk-x"}), &policy);
//! ```

pub use agent_sdk_foundation::privacy::redaction::{
    REDACTED_MARKER, RedactionLevel, RedactionPolicy, redact_error, redact_for_observability,
    redact_string, redact_value,
};
