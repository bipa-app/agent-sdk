//! End-to-end tests for the backend seam: the wire server + the SDK run
//! loop driving a scripted [`AcpBackend`], asserting the card's contracts —
//! typed cursor (C-c), duplicate suppression, terminal mapping, and the
//! cancel-forwarding drain.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_acp::{
    AcpBackend, AcpRunHandle, AcpServer, BackendError, BackendPromptHandler, EventStream,
    NewSessionParams, RunEvent, RunStreamItem,
};
use agent_sdk_foundation::{AgentEvent, ThreadId, TokenUsage};
use futures::StreamExt;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream, ReadHalf, WriteHalf};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

const TIMEOUT: Duration = Duration::from_secs(5);

fn ev(sequence: u64, event: AgentEvent) -> RunStreamItem {
    RunStreamItem::Event(Box::new(RunEvent { sequence, event }))
}

fn done_event() -> AgentEvent {
    AgentEvent::Done {
        thread_id: ThreadId::from_string("t".to_owned()),
        total_turns: 1,
        total_usage: TokenUsage::default(),
        duration: Duration::from_millis(1),
        estimated_cost_usd: None,
        emitter_task_id: None,
    }
}

fn cancelled_event() -> AgentEvent {
    AgentEvent::Cancelled {
        turn: 1,
        usage: TokenUsage::default(),
        reason: None,
        emitter_task_id: None,
    }
}

/// Scripted backend: serves a fixed item list, optionally followed by a live
/// channel fed by `cancel()`. Records every `open_events` cursor and cancel.
struct MockBackend {
    first_event_sequence: u64,
    scripted: Mutex<Vec<RunStreamItem>>,
    /// When set, the stream stays open after the scripted items and
    /// `cancel()` pushes a terminal `Cancelled` event through it.
    live_on_cancel: bool,
    live_tx: Mutex<Option<mpsc::UnboundedSender<RunStreamItem>>>,
    opened_after: Mutex<Vec<Option<u64>>>,
    cancels: AtomicUsize,
}

impl MockBackend {
    const fn new(
        first_event_sequence: u64,
        scripted: Vec<RunStreamItem>,
        live_on_cancel: bool,
    ) -> Self {
        Self {
            first_event_sequence,
            scripted: Mutex::new(scripted),
            live_on_cancel,
            live_tx: Mutex::new(None),
            opened_after: Mutex::new(Vec::new()),
            cancels: AtomicUsize::new(0),
        }
    }
}

#[async_trait::async_trait]
impl AcpBackend for MockBackend {
    async fn submit_prompt(
        &self,
        _session_id: &str,
        _session: &NewSessionParams,
        _blocks: &[String],
    ) -> Result<AcpRunHandle, BackendError> {
        Ok(AcpRunHandle {
            thread_id: "buzz:test-channel".to_owned(),
            task_id: "task-1".to_owned(),
            first_event_sequence: self.first_event_sequence,
        })
    }

    async fn open_events(
        &self,
        _thread_id: &str,
        after_sequence: Option<u64>,
    ) -> Result<EventStream, BackendError> {
        self.opened_after.lock().expect("lock").push(after_sequence);
        let scripted = std::mem::take(&mut *self.scripted.lock().expect("lock"));
        let head = futures::stream::iter(scripted);
        if self.live_on_cancel {
            let (tx, rx) = mpsc::unbounded_channel();
            *self.live_tx.lock().expect("lock") = Some(tx);
            Ok(head.chain(UnboundedReceiverStream::new(rx)).boxed())
        } else {
            Ok(head.boxed())
        }
    }

    async fn cancel(&self, _thread_id: &str, _task_id: &str) -> Result<(), BackendError> {
        self.cancels.fetch_add(1, Ordering::SeqCst);
        if let Some(tx) = self.live_tx.lock().expect("lock").as_ref() {
            let _ = tx.send(ev(99, cancelled_event()));
        }
        Ok(())
    }
}

struct Client {
    write: WriteHalf<DuplexStream>,
    read: BufReader<ReadHalf<DuplexStream>>,
}

impl Client {
    fn start(backend: Arc<MockBackend>) -> Self {
        let (client_side, server_side) = tokio::io::duplex(1 << 20);
        let (server_read, server_write) = tokio::io::split(server_side);
        tokio::spawn(async move {
            AcpServer::new(BackendPromptHandler::new(backend as Arc<dyn AcpBackend>))
                .serve(server_read, server_write)
                .await
                .expect("server exits cleanly");
        });
        let (client_read, client_write) = tokio::io::split(client_side);
        Self {
            write: client_write,
            read: BufReader::new(client_read),
        }
    }

    async fn send(&mut self, msg: Value) {
        let mut line = msg.to_string();
        line.push('\n');
        self.write.write_all(line.as_bytes()).await.expect("write");
        self.write.flush().await.expect("flush");
    }

    async fn next_message(&mut self) -> Value {
        let mut line = String::new();
        let read = tokio::time::timeout(TIMEOUT, self.read.read_line(&mut line))
            .await
            .expect("message within timeout")
            .expect("read succeeds");
        assert!(read > 0, "unexpected EOF from server");
        serde_json::from_str(&line).expect("valid JSON from server")
    }

    async fn until_response(&mut self, id: u64) -> (Vec<Value>, Value) {
        let mut notifications = Vec::new();
        loop {
            let msg = self.next_message().await;
            if msg.get("id") == Some(&json!(id)) {
                return (notifications, msg);
            }
            notifications.push(msg);
        }
    }

    async fn handshake_and_prompt(&mut self) -> (Vec<Value>, Value) {
        self.send(
            json!({"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":2}}),
        )
        .await;
        let (_, _init) = self.until_response(0).await;
        self.send(json!({"jsonrpc":"2.0","id":1,"method":"session/new","params":{"cwd":"/tmp","mcpServers":[]}}))
            .await;
        let (_, new) = self.until_response(1).await;
        let session_id = new["result"]["sessionId"].as_str().expect("sessionId");
        self.send(
            json!({"jsonrpc":"2.0","id":2,"method":"session/prompt","params":{
                "sessionId": session_id,
                "prompt": [{"type":"text","text":"hello"}],
            }}),
        )
        .await;
        self.until_response(2).await
    }
}

fn chunk_texts(notifications: &[Value]) -> Vec<String> {
    notifications
        .iter()
        .filter(|n| n["params"]["update"]["sessionUpdate"] == json!("agent_message_chunk"))
        .filter_map(|n| n["params"]["update"]["content"]["text"].as_str())
        .map(str::to_owned)
        .collect()
}

#[tokio::test]
async fn cursor_zero_opens_from_beginning_and_delivers_sequence_zero() {
    let backend = Arc::new(MockBackend::new(
        0,
        vec![
            ev(0, AgentEvent::text_delta("m1", "first")),
            ev(1, done_event()),
        ],
        false,
    ));
    let mut client = Client::start(Arc::clone(&backend));
    let (notifications, response) = client.handshake_and_prompt().await;

    // C-c: first_event_sequence 0 → open from the very beginning…
    assert_eq!(*backend.opened_after.lock().expect("lock"), vec![None]);
    // …and the sequence-0 event IS delivered.
    assert_eq!(chunk_texts(&notifications), vec!["first"]);
    assert_eq!(response["result"]["stopReason"], json!("end_turn"));
}

#[tokio::test]
async fn cursor_mid_thread_converts_to_exclusive_lower_bound() {
    let backend = Arc::new(MockBackend::new(
        5,
        vec![
            ev(5, AgentEvent::text_delta("m1", "turn-start")),
            ev(6, done_event()),
        ],
        false,
    ));
    let mut client = Client::start(Arc::clone(&backend));
    let (notifications, response) = client.handshake_and_prompt().await;

    // C-c: first_event_sequence 5 → after_sequence Some(4), never Some(5).
    assert_eq!(*backend.opened_after.lock().expect("lock"), vec![Some(4)]);
    assert_eq!(chunk_texts(&notifications), vec!["turn-start"]);
    assert_eq!(response["result"]["stopReason"], json!("end_turn"));
}

#[tokio::test]
async fn duplicate_sequences_are_suppressed() {
    let backend = Arc::new(MockBackend::new(
        0,
        vec![
            ev(0, AgentEvent::text_delta("m1", "a")),
            ev(0, AgentEvent::text_delta("m1", "a-again")),
            ev(1, AgentEvent::text_delta("m1", "b")),
            ev(2, done_event()),
        ],
        false,
    ));
    let mut client = Client::start(Arc::clone(&backend));
    let (notifications, response) = client.handshake_and_prompt().await;

    assert_eq!(chunk_texts(&notifications), vec!["a", "b"]);
    assert_eq!(response["result"]["stopReason"], json!("end_turn"));
}

#[tokio::test]
async fn cancel_forwards_to_backend_and_drains_to_cancelled() {
    let backend = Arc::new(MockBackend::new(
        0,
        vec![ev(0, AgentEvent::text_delta("m1", "working"))],
        true,
    ));
    let mut client = Client::start(Arc::clone(&backend));

    client
        .send(json!({"jsonrpc":"2.0","id":0,"method":"initialize","params":{}}))
        .await;
    let (_, _) = client.until_response(0).await;
    client
        .send(json!({"jsonrpc":"2.0","id":1,"method":"session/new","params":{}}))
        .await;
    let (_, new) = client.until_response(1).await;
    let session_id = new["result"]["sessionId"]
        .as_str()
        .expect("sessionId")
        .to_owned();

    client
        .send(
            json!({"jsonrpc":"2.0","id":2,"method":"session/prompt","params":{
                "sessionId": session_id, "prompt": [{"type":"text","text":"go"}],
            }}),
        )
        .await;
    // Turn is provably in flight once the first chunk lands.
    let first = client.next_message().await;
    assert_eq!(first["method"], json!("session/update"));

    client
        .send(json!({"jsonrpc":"2.0","method":"session/cancel","params":{"sessionId": session_id}}))
        .await;
    let (_, response) = client.until_response(2).await;

    assert_eq!(response["result"]["stopReason"], json!("cancelled"));
    assert_eq!(backend.cancels.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn error_event_resolves_prompt_as_internal_error() {
    let backend = Arc::new(MockBackend::new(
        0,
        vec![ev(0, AgentEvent::error("provider exploded", false))],
        false,
    ));
    let mut client = Client::start(backend);
    let (_, response) = client.handshake_and_prompt().await;
    assert_eq!(response["error"]["code"], json!(-32603));
    assert_eq!(response["error"]["message"], json!("provider exploded"));
}

#[tokio::test]
async fn stream_end_without_terminal_and_lagged_fail_loudly() {
    let backend = Arc::new(MockBackend::new(
        0,
        vec![ev(0, AgentEvent::text_delta("m1", "a"))],
        false,
    ));
    let mut client = Client::start(backend);
    let (_, response) = client.handshake_and_prompt().await;
    assert_eq!(response["error"]["code"], json!(-32603));

    let backend = Arc::new(MockBackend::new(0, vec![RunStreamItem::Lagged], false));
    let mut client = Client::start(backend);
    let (_, response) = client.handshake_and_prompt().await;
    assert_eq!(response["error"]["code"], json!(-32603));
}
