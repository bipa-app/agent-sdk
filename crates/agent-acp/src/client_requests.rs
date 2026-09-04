//! Server-initiated JSON-RPC requests to the client, and their answers.
//!
//! ACP is bidirectional for one thing this transport needs:
//! `session/request_permission`, which the agent SENDS and the client
//! answers. The read loop in [`crate::server`] classifies every incoming
//! response and hands it here; the waiter registered under that id wakes
//! with the payload. Ids are the server's own counter — the client's
//! request ids live in a separate space, so the two never collide.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;
use tokio::sync::oneshot;

/// The client's answer to one of our requests: its `result`, or its
/// JSON-RPC `error` object.
pub type ClientAnswer = Result<Value, Value>;

/// In-flight server→client requests keyed by id.
#[derive(Default)]
pub struct ClientRequests {
    next_id: AtomicU64,
    pending: Mutex<HashMap<u64, oneshot::Sender<ClientAnswer>>>,
}

impl ClientRequests {
    /// Reserve an id and the channel its answer arrives on.
    pub fn register(&self) -> (u64, oneshot::Receiver<ClientAnswer>) {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id, tx);
        (id, rx)
    }

    /// Deliver a response. Returns `false` when no waiter is registered
    /// under `id` — a late answer after the waiter gave up, or a client
    /// bug — so the caller can log it and move on.
    pub fn resolve(&self, id: &Value, answer: ClientAnswer) -> bool {
        let Some(id) = id.as_u64() else {
            return false;
        };
        let waiter = self
            .pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&id);
        waiter.is_some_and(|tx| tx.send(answer).is_ok())
    }

    /// Drop every waiter (transport EOF): each pending request resolves
    /// with a closed channel.
    pub fn fail_all(&self) {
        self.pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn answers_route_by_id_and_unknown_ids_are_reported() {
        let requests = ClientRequests::default();
        let (first, first_rx) = requests.register();
        let (second, second_rx) = requests.register();
        assert_ne!(first, second);

        assert!(requests.resolve(&json!(second), Ok(json!({"n": 2}))));
        assert!(requests.resolve(&json!(first), Err(json!({"code": -1}))));
        assert!(
            !requests.resolve(&json!(first), Ok(json!(null))),
            "answered twice"
        );
        assert!(!requests.resolve(&json!("not-ours"), Ok(json!(null))));

        assert_eq!(second_rx.await.expect("answered"), Ok(json!({"n": 2})));
        assert_eq!(first_rx.await.expect("answered"), Err(json!({"code": -1})));
    }

    #[tokio::test]
    async fn fail_all_wakes_every_waiter_with_a_closed_channel() {
        let requests = ClientRequests::default();
        let (_, rx) = requests.register();
        requests.fail_all();
        assert!(rx.await.is_err());
    }
}
