//! JSON-RPC 2.0 wire types for the ACP stdio transport.
//!
//! The protocol is newline-delimited JSON (one message per line) over
//! stdin/stdout. This module owns envelope construction and incoming-line
//! classification; it deliberately keeps params/results as
//! [`serde_json::Value`] — the ACP surface the harness exercises is small
//! and evolving (see the crate docs for the protocol-version stance), and
//! fixtures pin the shapes that matter.

use serde_json::{Value, json};

/// The JSON-RPC protocol version string carried on every message.
pub const JSONRPC_VERSION: &str = "2.0";

/// The ACP protocol version this server implements.
///
/// This is "v2 as spoken by the buzz-acp harness at block/buzz `7e34bee`",
/// which requests `protocolVersion: 2` ahead of the upstream ACP RFD. The
/// recorded fixtures in `tests/fixtures/` are the compatibility contract.
pub const PROTOCOL_VERSION: u64 = 2;

/// JSON-RPC error codes used by this transport.
pub mod error_codes {
    /// The method does not exist / is not available.
    pub const METHOD_NOT_FOUND: i64 = -32601;
    /// Invalid method parameter(s).
    pub const INVALID_PARAMS: i64 = -32602;
    /// Internal JSON-RPC error (handler failure).
    pub const INTERNAL_ERROR: i64 = -32603;
}

/// A single incoming line, classified.
///
/// JSON-RPC 2.0 distinguishes requests (`id` + `method`), notifications
/// (`method`, no `id`), and responses (`id`, no `method`). The server only
/// initiates notifications in this milestone, so incoming responses are
/// ignored by the dispatch loop — but they still classify cleanly here.
#[derive(Debug)]
pub enum Incoming {
    /// A request that expects a response with the same `id`.
    Request {
        /// Request id. JSON-RPC 2.0 permits both numeric and string ids, so
        /// it is kept as a raw [`Value`] and echoed verbatim.
        id: Value,
        /// Method name, e.g. `session/prompt`.
        method: String,
        /// Request params (`Value::Null` when absent).
        params: Value,
    },
    /// A notification: no `id`, no response expected.
    Notification {
        /// Method name, e.g. `session/cancel`.
        method: String,
        /// Notification params (`Value::Null` when absent).
        params: Value,
    },
    /// A response to a server-initiated request (none exist in this
    /// milestone; classified so the dispatch loop can ignore it explicitly).
    Response {
        /// The id of the request this responds to.
        id: Value,
    },
    /// An object with neither `method` nor `id` — not valid JSON-RPC.
    Malformed,
}

impl Incoming {
    /// Classify a parsed JSON value into a JSON-RPC message shape.
    #[must_use]
    pub fn classify(value: Value) -> Self {
        let Value::Object(mut map) = value else {
            return Self::Malformed;
        };
        let id = map.remove("id");
        let method = map
            .remove("method")
            .and_then(|m| m.as_str().map(str::to_owned));
        let params = map.remove("params").unwrap_or(Value::Null);
        match (id, method) {
            (Some(id), Some(method)) => Self::Request { id, method, params },
            (None, Some(method)) => Self::Notification { method, params },
            (Some(id), None) => Self::Response { id },
            (None, None) => Self::Malformed,
        }
    }
}

/// Build a success response envelope.
#[must_use]
pub fn response(id: &Value, result: Value) -> Value {
    let mut map = serde_json::Map::with_capacity(3);
    map.insert(
        "jsonrpc".to_owned(),
        Value::String(JSONRPC_VERSION.to_owned()),
    );
    map.insert("id".to_owned(), id.clone());
    map.insert("result".to_owned(), result);
    Value::Object(map)
}

/// Build an error response envelope.
#[must_use]
pub fn error_response(id: &Value, code: i64, message: &str) -> Value {
    let mut map = serde_json::Map::with_capacity(3);
    map.insert(
        "jsonrpc".to_owned(),
        Value::String(JSONRPC_VERSION.to_owned()),
    );
    map.insert("id".to_owned(), id.clone());
    map.insert(
        "error".to_owned(),
        json!({ "code": code, "message": message }),
    );
    Value::Object(map)
}

/// Build a notification envelope.
#[must_use]
pub fn notification(method: &str, params: Value) -> Value {
    let mut map = serde_json::Map::with_capacity(3);
    map.insert(
        "jsonrpc".to_owned(),
        Value::String(JSONRPC_VERSION.to_owned()),
    );
    map.insert("method".to_owned(), Value::String(method.to_owned()));
    map.insert("params".to_owned(), params);
    Value::Object(map)
}

/// Why a prompt (turn) ended, in ACP wire vocabulary.
///
/// String forms match what the buzz-acp harness parses (case-insensitively)
/// at the pinned rev: `end_turn`, `cancelled`, `max_tokens`,
/// `max_turn_requests`, `refusal`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The agent completed the turn normally.
    EndTurn,
    /// The turn was cancelled via `session/cancel`.
    Cancelled,
    /// The agent hit a token limit.
    MaxTokens,
    /// The agent hit a per-turn request limit.
    MaxTurnRequests,
    /// The agent refused the prompt.
    Refusal,
}

impl StopReason {
    /// The ACP wire string for this stop reason.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EndTurn => "end_turn",
            Self::Cancelled => "cancelled",
            Self::MaxTokens => "max_tokens",
            Self::MaxTurnRequests => "max_turn_requests",
            Self::Refusal => "refusal",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_request_notification_response_malformed() {
        let req = Incoming::classify(json!({"id": 1, "method": "initialize", "params": {}}));
        assert!(matches!(req, Incoming::Request { .. }));

        let notif = Incoming::classify(json!({"method": "session/cancel"}));
        assert!(matches!(notif, Incoming::Notification { .. }));

        let resp = Incoming::classify(json!({"id": 7, "result": {}}));
        assert!(matches!(resp, Incoming::Response { .. }));

        assert!(matches!(
            Incoming::classify(json!({"result": {}})),
            Incoming::Malformed
        ));
        assert!(matches!(
            Incoming::classify(json!([1, 2])),
            Incoming::Malformed
        ));
    }

    #[test]
    fn string_ids_are_preserved_verbatim() {
        let Incoming::Request { id, .. } =
            Incoming::classify(json!({"id": "abc-1", "method": "m"}))
        else {
            panic!("expected request");
        };
        assert_eq!(id, json!("abc-1"));
        let resp = response(&id, json!({"ok": true}));
        assert_eq!(resp["id"], json!("abc-1"));
    }

    #[test]
    fn stop_reason_wire_strings_match_harness_vocabulary() {
        assert_eq!(StopReason::EndTurn.as_str(), "end_turn");
        assert_eq!(StopReason::Cancelled.as_str(), "cancelled");
        assert_eq!(StopReason::MaxTokens.as_str(), "max_tokens");
        assert_eq!(StopReason::MaxTurnRequests.as_str(), "max_turn_requests");
        assert_eq!(StopReason::Refusal.as_str(), "refusal");
    }
}
