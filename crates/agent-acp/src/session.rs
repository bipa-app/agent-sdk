//! Session registry for the ACP server.
//!
//! A session is created by `session/new` and identified by a
//! server-generated UUID. Sessions are connection-scoped handles; durable
//! identity (threads) is a backend concern layered on top in later
//! milestones. At most one prompt may be in flight per session — the
//! buzz-acp harness serializes prompts per agent process, so an overlapping
//! prompt on the same session is a protocol error, not a queueing request.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde_json::Value;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

/// Parameters captured from `session/new`.
///
/// All fields are optional-tolerant: the harness always sends `cwd` and
/// `mcpServers`, but the server does not fail sessions over their absence —
/// what a backend requires is the backend's decision.
#[derive(Debug, Default, Clone)]
pub struct NewSessionParams {
    /// Working directory for the session (absolute path, per ACP).
    pub cwd: Option<String>,
    /// MCP server descriptors passed through verbatim.
    pub mcp_servers: Vec<Value>,
    /// System prompt supplied by the client (the buzz persona arrives here).
    pub system_prompt: Option<String>,
}

impl NewSessionParams {
    /// Extract session parameters from raw `session/new` params.
    #[must_use]
    pub fn from_params(params: &Value) -> Self {
        Self {
            cwd: params.get("cwd").and_then(Value::as_str).map(str::to_owned),
            mcp_servers: params
                .get("mcpServers")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            system_prompt: params
                .get("systemPrompt")
                .and_then(Value::as_str)
                .map(str::to_owned),
        }
    }
}

struct SessionState {
    params: Arc<NewSessionParams>,
    in_flight: Option<CancellationToken>,
}

/// Why a prompt could not begin on a session.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum BeginPromptError {
    /// The session id was never created by `session/new`.
    UnknownSession,
    /// A prompt is already running on this session.
    PromptInFlight,
}

/// Thread-safe registry of live sessions.
///
/// Uses a std [`Mutex`]: every critical section is a short map operation
/// with no `.await` inside.
#[derive(Default)]
pub(crate) struct SessionStore {
    inner: Mutex<HashMap<String, SessionState>>,
}

impl SessionStore {
    /// Create a session and return its generated id.
    pub fn create(&self, params: NewSessionParams) -> String {
        let id = Uuid::new_v4().to_string();
        let state = SessionState {
            params: Arc::new(params),
            in_flight: None,
        };
        self.lock().insert(id.clone(), state);
        id
    }

    /// Mark a prompt as in flight and return the session params plus a fresh
    /// cancellation token for the turn.
    pub fn begin_prompt(
        &self,
        session_id: &str,
    ) -> Result<(Arc<NewSessionParams>, CancellationToken), BeginPromptError> {
        let mut inner = self.lock();
        let state = inner
            .get_mut(session_id)
            .ok_or(BeginPromptError::UnknownSession)?;
        if state.in_flight.is_some() {
            return Err(BeginPromptError::PromptInFlight);
        }
        let token = CancellationToken::new();
        state.in_flight = Some(token.clone());
        let params = Arc::clone(&state.params);
        drop(inner);
        Ok((params, token))
    }

    /// Clear the in-flight marker after a prompt resolves.
    pub fn end_prompt(&self, session_id: &str) {
        if let Some(state) = self.lock().get_mut(session_id) {
            state.in_flight = None;
        }
    }

    /// Cancel the in-flight prompt on a session, if any.
    ///
    /// Returns `true` when a running prompt was signalled. Cancelling a
    /// session with no in-flight prompt (or an unknown session) is a no-op —
    /// `session/cancel` is a notification and has no failure channel.
    pub fn cancel(&self, session_id: &str) -> bool {
        self.lock()
            .get(session_id)
            .and_then(|state| state.in_flight.as_ref())
            .is_some_and(|token| {
                token.cancel();
                true
            })
    }

    /// Cancel every in-flight prompt (used at shutdown/EOF).
    pub fn cancel_all(&self) {
        for state in self.lock().values() {
            if let Some(token) = &state.in_flight {
                token.cancel();
            }
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, SessionState>> {
        // A poisoned lock means a panic while holding a short map-op section;
        // the map itself cannot be left mid-invariant by any of the
        // operations above, so continuing with the inner value is safe.
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn from_params_reads_harness_shapes_and_tolerates_absence() {
        let full = NewSessionParams::from_params(&json!({
            "cwd": "/data/x",
            "mcpServers": [{"name": "dev"}],
            "systemPrompt": "persona",
        }));
        assert_eq!(full.cwd.as_deref(), Some("/data/x"));
        assert_eq!(full.mcp_servers.len(), 1);
        assert_eq!(full.system_prompt.as_deref(), Some("persona"));

        let empty = NewSessionParams::from_params(&json!({}));
        assert!(empty.cwd.is_none());
        assert!(empty.mcp_servers.is_empty());
        assert!(empty.system_prompt.is_none());
    }

    #[test]
    fn prompt_lifecycle_unknown_overlap_and_cancel() {
        let store = SessionStore::default();
        assert_eq!(
            store.begin_prompt("nope").unwrap_err(),
            BeginPromptError::UnknownSession
        );

        let id = store.create(NewSessionParams::default());
        let (_params, token) = store.begin_prompt(&id).expect("first prompt begins");
        assert_eq!(
            store.begin_prompt(&id).unwrap_err(),
            BeginPromptError::PromptInFlight
        );

        assert!(store.cancel(&id));
        assert!(token.is_cancelled());

        store.end_prompt(&id);
        assert!(!store.cancel(&id), "no in-flight prompt after end_prompt");
        store.begin_prompt(&id).expect("prompt can begin again");
    }
}
