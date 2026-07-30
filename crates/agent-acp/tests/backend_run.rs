//! End-to-end tests for the backend seam: the wire server + the SDK run
//! loop driving a scripted [`AcpBackend`], asserting the card's contracts —
//! typed cursor + lag-reopen (C-c), task-scoped completion + stall
//! reconciliation (C-d), duplicate suppression, terminal mapping, and the
//! cancel-forwarding drain.
//!
//! Scripts are C-d-conformant where the durable host would be: OUR task
//! (`task-1`) commits an attributed `Start` before its content, and its
//! terminals carry `emitter_task_id`. Stale-predecessor traffic is played
//! as `task-0`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_acp::{
    AcpBackend, AcpRunHandle, AcpServer, BackendError, BackendPromptHandler, BackendTaskStatus,
    EventStream, NewSessionParams, RunEvent, RunStreamItem,
};
use agent_sdk_foundation::{AgentEvent, ThreadId, TokenUsage};
use futures::StreamExt;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream, ReadHalf, WriteHalf};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

const TIMEOUT: Duration = Duration::from_secs(10);

/// OUR task for every scripted turn.
const TASK: &str = "task-1";
/// A stale predecessor on the same thread.
const PREDECESSOR: &str = "task-0";

fn ev(sequence: u64, event: AgentEvent) -> RunStreamItem {
    RunStreamItem::Event(Box::new(RunEvent { sequence, event }))
}

fn start_event(task: &str) -> AgentEvent {
    AgentEvent::start(ThreadId::from_string("t".to_owned()), 1).with_emitter_task_id(task)
}

fn done_event(task: Option<&str>) -> AgentEvent {
    let done = AgentEvent::Done {
        thread_id: ThreadId::from_string("t".to_owned()),
        total_turns: 1,
        total_usage: TokenUsage::default(),
        duration: Duration::from_millis(1),
        estimated_cost_usd: None,
        emitter_task_id: None,
    };
    match task {
        Some(task) => done.with_emitter_task_id(task),
        None => done,
    }
}

fn cancelled_event(task: Option<&str>) -> AgentEvent {
    let cancelled = AgentEvent::Cancelled {
        turn: 1,
        usage: TokenUsage::default(),
        reason: None,
        emitter_task_id: None,
    };
    match task {
        Some(task) => cancelled.with_emitter_task_id(task),
        None => cancelled,
    }
}

/// Scripted backend. `streams` is a queue of event scripts — each
/// `open_events` call pops the next one (so lag-reopen gets the second
/// script). The LAST script may stay open as a live channel (fed by
/// `cancel()`, or simply pending forever for stall-poll tests).
struct MockBackend {
    first_event_sequence: u64,
    streams: Mutex<Vec<Vec<RunStreamItem>>>,
    /// Keep the final stream open after its scripted items.
    hold_last_open: bool,
    /// Durable status served to every `task_status` probe.
    status: Mutex<BackendTaskStatus>,
    status_polls: AtomicUsize,
    live_tx: Mutex<Option<mpsc::UnboundedSender<RunStreamItem>>>,
    opened_after: Mutex<Vec<Option<u64>>>,
    cancels: AtomicUsize,
}

impl MockBackend {
    const fn new(first_event_sequence: u64, streams: Vec<Vec<RunStreamItem>>) -> Self {
        Self {
            first_event_sequence,
            streams: Mutex::new(streams),
            hold_last_open: false,
            status: Mutex::new(BackendTaskStatus::Running),
            status_polls: AtomicUsize::new(0),
            live_tx: Mutex::new(None),
            opened_after: Mutex::new(Vec::new()),
            cancels: AtomicUsize::new(0),
        }
    }

    const fn hold_open(mut self) -> Self {
        self.hold_last_open = true;
        self
    }

    fn with_status(self, status: BackendTaskStatus) -> Self {
        *self.status.lock().expect("lock") = status;
        self
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
            task_id: TASK.to_owned(),
            first_event_sequence: self.first_event_sequence,
        })
    }

    async fn open_events(
        &self,
        _thread_id: &str,
        after_sequence: Option<u64>,
    ) -> Result<EventStream, BackendError> {
        self.opened_after.lock().expect("lock").push(after_sequence);
        let mut streams = self.streams.lock().expect("lock");
        let scripted = if streams.is_empty() {
            Vec::new()
        } else {
            streams.remove(0)
        };
        let is_last = streams.is_empty();
        drop(streams);
        let head = futures::stream::iter(scripted);
        if is_last && self.hold_last_open {
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
            let _ = tx.send(ev(99, cancelled_event(Some(TASK))));
        }
        Ok(())
    }

    async fn task_status(
        &self,
        _thread_id: &str,
        _task_id: &str,
    ) -> Result<BackendTaskStatus, BackendError> {
        self.status_polls.fetch_add(1, Ordering::SeqCst);
        Ok(self.status.lock().expect("lock").clone())
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

    async fn handshake(&mut self) -> String {
        self.send(
            json!({"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":2}}),
        )
        .await;
        let (_, _init) = self.until_response(0).await;
        self.send(json!({"jsonrpc":"2.0","id":1,"method":"session/new","params":{"cwd":"/tmp","mcpServers":[]}}))
            .await;
        let (_, new) = self.until_response(1).await;
        new["result"]["sessionId"]
            .as_str()
            .expect("sessionId")
            .to_owned()
    }

    async fn handshake_and_prompt(&mut self) -> (Vec<Value>, Value) {
        let session_id = self.handshake().await;
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
        vec![vec![
            ev(0, start_event(TASK)),
            ev(1, AgentEvent::text_delta("m1", "first")),
            ev(2, done_event(Some(TASK))),
        ]],
    ));
    let mut client = Client::start(Arc::clone(&backend));
    let (notifications, response) = client.handshake_and_prompt().await;

    // C-c: first_event_sequence 0 → open from the very beginning…
    assert_eq!(*backend.opened_after.lock().expect("lock"), vec![None]);
    // …and the sequence-0 event IS delivered (Start gates the stream on).
    assert_eq!(chunk_texts(&notifications), vec!["first"]);
    assert_eq!(response["result"]["stopReason"], json!("end_turn"));
}

#[tokio::test]
async fn cursor_mid_thread_converts_to_exclusive_lower_bound() {
    let backend = Arc::new(MockBackend::new(
        5,
        vec![vec![
            ev(5, start_event(TASK)),
            ev(6, AgentEvent::text_delta("m1", "turn-start")),
            ev(7, done_event(Some(TASK))),
        ]],
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
        vec![vec![
            ev(0, start_event(TASK)),
            ev(1, AgentEvent::text_delta("m1", "a")),
            ev(1, AgentEvent::text_delta("m1", "a-again")),
            ev(2, AgentEvent::text_delta("m1", "b")),
            ev(3, done_event(Some(TASK))),
        ]],
    ));
    let mut client = Client::start(Arc::clone(&backend));
    let (notifications, response) = client.handshake_and_prompt().await;

    assert_eq!(chunk_texts(&notifications), vec!["a", "b"]);
    assert_eq!(response["result"]["stopReason"], json!("end_turn"));
}

/// C-d acceptance: a stale predecessor's terminal — before OR after our
/// `Start` — never closes the prompt; our attributed terminal does. The
/// predecessor's late content never streams either.
#[tokio::test]
async fn stale_predecessor_terminals_and_text_never_touch_the_prompt() {
    let backend = Arc::new(MockBackend::new(
        10,
        vec![vec![
            // Late predecessor traffic ahead of our turn:
            ev(10, AgentEvent::text_delta("m0", "stale-text")),
            ev(11, done_event(Some(PREDECESSOR))),
            // Our turn begins.
            ev(12, start_event(TASK)),
            ev(13, AgentEvent::text_delta("m1", "ours")),
            // A predecessor salvage terminal AFTER our Start (V4: a
            // cancelled root's late commit names the cancelled root).
            ev(14, cancelled_event(Some(PREDECESSOR))),
            ev(15, AgentEvent::text_delta("m1", " continues")),
            ev(16, done_event(Some(TASK))),
        ]],
    ));
    let mut client = Client::start(Arc::clone(&backend));
    let (notifications, response) = client.handshake_and_prompt().await;

    assert_eq!(
        chunk_texts(&notifications),
        vec!["ours", " continues"],
        "predecessor text must not stream; our text must survive its stale terminal"
    );
    assert_eq!(response["result"]["stopReason"], json!("end_turn"));
}

/// ENG-9422 acceptance: the exact salvage-flush interleaving — a
/// cancelled predecessor's late deltas commit AFTER our `Start`, and
/// attribution (not phase) is what keeps them out of our answer.
/// Unattributed content (pre-attribution journals) must keep streaming.
#[tokio::test]
async fn stale_attributed_deltas_after_our_start_never_stream() {
    let backend = Arc::new(MockBackend::new(
        0,
        vec![vec![
            ev(0, start_event(TASK)),
            // Start(B) → late TextDelta(A): the race from the card.
            ev(
                1,
                AgentEvent::text_delta("m0", "stale-salvage").with_emitter_task_id(PREDECESSOR),
            ),
            ev(
                2,
                AgentEvent::text_delta("m1", "ours").with_emitter_task_id(TASK),
            ),
            // Old-journal compatibility: unattributed content streams.
            ev(3, AgentEvent::text_delta("m1", " and compat")),
            ev(4, done_event(Some(TASK))),
        ]],
    ));
    let mut client = Client::start(Arc::clone(&backend));
    let (notifications, response) = client.handshake_and_prompt().await;

    assert_eq!(
        chunk_texts(&notifications),
        vec!["ours", " and compat"],
        "a predecessor-attributed delta must never render; unattributed content must"
    );
    assert_eq!(response["result"]["stopReason"], json!("end_turn"));
}

/// C-d acceptance: an UNATTRIBUTED terminal while streaming is reconciled
/// against the task's durable status — still running means it was not
/// ours, and the turn continues to its real terminal.
#[tokio::test]
async fn unattributed_terminal_with_running_status_is_ignored() {
    let backend = Arc::new(MockBackend::new(
        0,
        vec![vec![
            ev(0, start_event(TASK)),
            ev(1, AgentEvent::text_delta("m1", "a")),
            ev(2, done_event(None)), // no emitter: pre-attribution journal
            ev(3, AgentEvent::text_delta("m1", "b")),
            ev(4, done_event(Some(TASK))),
        ]],
    ));
    let mut client = Client::start(Arc::clone(&backend));
    let (notifications, response) = client.handshake_and_prompt().await;

    assert_eq!(chunk_texts(&notifications), vec!["a", "b"]);
    assert_eq!(response["result"]["stopReason"], json!("end_turn"));
    assert!(
        backend.status_polls.load(Ordering::SeqCst) >= 1,
        "the unattributed terminal must be reconciled via task_status"
    );
}

/// C-d: an unattributed terminal while streaming, when OUR task really is
/// complete, resolves from STATUS.
#[tokio::test]
async fn unattributed_terminal_with_completed_status_resolves() {
    let backend = Arc::new(
        MockBackend::new(
            0,
            vec![vec![
                ev(0, start_event(TASK)),
                ev(1, AgentEvent::text_delta("m1", "a")),
                ev(2, done_event(None)),
            ]],
        )
        .hold_open()
        .with_status(BackendTaskStatus::Completed),
    );
    let mut client = Client::start(Arc::clone(&backend));
    let (notifications, response) = client.handshake_and_prompt().await;

    assert_eq!(chunk_texts(&notifications), vec!["a"]);
    assert_eq!(response["result"]["stopReason"], json!("end_turn"));
}

/// C-d acceptance: a task that reaches `Failed` WITHOUT ever committing a
/// journal `Error` event still resolves the prompt — with the task's
/// recorded error — via the bounded stall poll.
#[tokio::test]
async fn task_failed_without_journal_error_resolves_with_task_error() {
    let backend = Arc::new(
        MockBackend::new(0, vec![vec![ev(0, start_event(TASK))]])
            .hold_open()
            .with_status(BackendTaskStatus::Failed {
                error: Some("provider melted".to_owned()),
            }),
    );
    let mut client = Client::start(Arc::clone(&backend));
    let (_, response) = client.handshake_and_prompt().await;

    assert_eq!(response["error"]["code"], json!(-32603));
    assert!(
        response["error"]["message"]
            .as_str()
            .expect("message")
            .contains("provider melted"),
        "the task's recorded error must reach the client"
    );
    assert!(
        backend.status_polls.load(Ordering::SeqCst) >= 2,
        "resolution requires two consecutive terminal status readings"
    );
}

/// C-c acceptance: a `Lagged` stream reopens from the last yielded
/// sequence — gapless, duplicate-free — and the turn completes normally.
#[tokio::test]
async fn lagged_stream_reopens_gapless_and_duplicate_free() {
    let backend = Arc::new(MockBackend::new(
        5,
        vec![
            vec![
                ev(5, start_event(TASK)),
                ev(6, AgentEvent::text_delta("m1", "a")),
                RunStreamItem::Lagged,
            ],
            vec![
                // The reopened stream overlaps: the duplicate is dropped.
                ev(6, AgentEvent::text_delta("m1", "a-dup")),
                ev(7, AgentEvent::text_delta("m1", "b")),
                ev(8, done_event(Some(TASK))),
            ],
        ],
    ));
    let mut client = Client::start(Arc::clone(&backend));
    let (notifications, response) = client.handshake_and_prompt().await;

    assert_eq!(
        *backend.opened_after.lock().expect("lock"),
        vec![Some(4), Some(6)],
        "reopen must resume strictly after the last yielded sequence"
    );
    assert_eq!(chunk_texts(&notifications), vec!["a", "b"]);
    assert_eq!(response["result"]["stopReason"], json!("end_turn"));
}

/// C-c: retention loss is NOT recoverable — the prompt fails loudly.
#[tokio::test]
async fn retention_gap_mid_turn_fails_the_prompt() {
    let backend = Arc::new(MockBackend::new(
        0,
        vec![vec![ev(0, start_event(TASK)), RunStreamItem::RetentionGap]],
    ));
    let mut client = Client::start(backend);
    let (_, response) = client.handshake_and_prompt().await;
    assert_eq!(response["error"]["code"], json!(-32603));
    assert!(
        response["error"]["message"]
            .as_str()
            .expect("message")
            .contains("retention"),
    );
}

#[tokio::test]
async fn cancel_forwards_to_backend_and_drains_to_cancelled() {
    let backend = Arc::new(
        MockBackend::new(
            0,
            vec![vec![
                ev(0, start_event(TASK)),
                ev(1, AgentEvent::text_delta("m1", "working")),
            ]],
        )
        .hold_open(),
    );
    let mut client = Client::start(Arc::clone(&backend));

    let session_id = client.handshake().await;
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

/// The documented C-d asymmetry: a queued turn cancelled BEFORE its
/// `Start` resolves via its OWN attributed `Cancelled` — an event naming
/// our task can never be a predecessor's, whatever the phase.
#[tokio::test]
async fn queued_turn_cancelled_before_start_still_resolves() {
    let backend = Arc::new(MockBackend::new(0, vec![vec![]]).hold_open());
    let mut client = Client::start(Arc::clone(&backend));

    let session_id = client.handshake().await;
    client
        .send(
            json!({"jsonrpc":"2.0","id":2,"method":"session/prompt","params":{
                "sessionId": session_id, "prompt": [{"type":"text","text":"go"}],
            }}),
        )
        .await;
    // Give the loop a beat to open the stream, then cancel the queued turn.
    tokio::time::sleep(Duration::from_millis(100)).await;
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
        vec![vec![
            ev(0, start_event(TASK)),
            ev(
                1,
                AgentEvent::error("provider exploded", false).with_emitter_task_id(TASK),
            ),
        ]],
    ));
    let mut client = Client::start(backend);
    let (_, response) = client.handshake_and_prompt().await;
    assert_eq!(response["error"]["code"], json!(-32603));
    assert_eq!(response["error"]["message"], json!("provider exploded"));
}

#[tokio::test]
async fn stream_end_without_terminal_fails_loudly() {
    let backend = Arc::new(MockBackend::new(
        0,
        vec![vec![
            ev(0, start_event(TASK)),
            ev(1, AgentEvent::text_delta("m1", "a")),
        ]],
    ));
    let mut client = Client::start(backend);
    let (_, response) = client.handshake_and_prompt().await;
    assert_eq!(response["error"]["code"], json!(-32603));
}
