//! The session view projection. The types and the pure [`fold`] live in
//! `agent-sdk-foundation`, where the match over [`AgentEvent`] can be
//! exhaustive; stores and hosts reach them here.
//!
//! [`AgentEvent`]: agent_sdk_foundation::events::AgentEvent

pub use agent_sdk_foundation::view::*;

/// The signals of the read-time view, recorded through the server metrics
/// when the `otel` feature is on and dropped otherwise.
pub mod signals {
    /// Stream steps for [`stream`].
    pub const OPENED: &str = "opened";
    pub const RESNAPSHOT: &str = "resnapshot";
    pub const CLOSED: &str = "closed";
    pub const REFUSED: &str = "refused";
    pub const NOT_FOUND: &str = "not_found";

    /// One fold of a thread's events into a view: how many and how long.
    pub fn folded(events: usize, seconds: f64) {
        #[cfg(feature = "otel")]
        crate::observability::ServerMetrics::global().record_view_fold(events, seconds);
        #[cfg(not(feature = "otel"))]
        let _ = (events, seconds);
    }

    /// One view stream lifecycle step, or a refused view RPC.
    pub fn stream(event: &'static str) {
        #[cfg(feature = "otel")]
        crate::observability::ServerMetrics::global().record_view_stream(event);
        #[cfg(not(feature = "otel"))]
        let _ = event;
    }
}
