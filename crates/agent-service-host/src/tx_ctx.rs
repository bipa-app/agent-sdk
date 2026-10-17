//! Store settings that a write transaction's static helpers need.

use agent_server::journal::view::ViewConfig;

/// Borrowed from the store for one transaction and passed down to the
/// helpers that write events.
pub struct TxCtx<'a> {
    pub view: &'a ViewConfig,
}
