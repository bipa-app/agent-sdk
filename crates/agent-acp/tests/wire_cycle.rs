//! Fixture-driven wire tests: the recorded buzz-acp exchanges from
//! `tests/fixtures/` (block/buzz rev `7e34bee`) replayed against the server
//! over an in-memory duplex transport.

use std::time::Duration;

use agent_acp::{AcpServer, PromptError, PromptHandler, PromptRequest, StopReason, UpdateSink};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream, ReadHalf, WriteHalf};
use tokio_util::sync::CancellationToken;

const TIMEOUT: Duration = Duration::from_secs(5);

fn fixture(name: &str) -> Value {
    let raw = match name {
        "initialize" => include_str!("fixtures/initialize.json"),
        "session_new" => include_str!("fixtures/session_new.json"),
        "session_prompt" => include_str!("fixtures/session_prompt.json"),
        "session_cancel" => include_str!("fixtures/session_cancel.json"),
        other => panic!("unknown fixture {other}"),
    };
    serde_json::from_str(raw).expect("fixture parses")
}

fn with_session_id(mut params: Value, session_id: &str) -> Value {
    params["sessionId"] = json!(session_id);
    params
}

/// Echoes the joined prompt text as one message chunk, then ends the turn.
struct EchoHandler;

#[async_trait::async_trait]
impl PromptHandler for EchoHandler {
    async fn prompt(
        &self,
        request: PromptRequest,
        updates: UpdateSink,
        _cancel: CancellationToken,
    ) -> Result<StopReason, PromptError> {
        let text = format!("echo: {}", request.blocks.join(" | "));
        updates
            .agent_message_chunk(&text)
            .await
            .map_err(|e| PromptError::new(e.to_string()))?;
        Ok(StopReason::EndTurn)
    }
}

/// Streams one chunk, then waits for cancellation — the "long turn" shape.
struct HangUntilCancelled;

#[async_trait::async_trait]
impl PromptHandler for HangUntilCancelled {
    async fn prompt(
        &self,
        _request: PromptRequest,
        updates: UpdateSink,
        cancel: CancellationToken,
    ) -> Result<StopReason, PromptError> {
        updates
            .agent_message_chunk("working…")
            .await
            .map_err(|e| PromptError::new(e.to_string()))?;
        cancel.cancelled().await;
        Ok(StopReason::Cancelled)
    }
}

struct Client {
    write: WriteHalf<DuplexStream>,
    read: BufReader<ReadHalf<DuplexStream>>,
}

impl Client {
    fn start<H: PromptHandler>(handler: H) -> Self {
        let (client_side, server_side) = tokio::io::duplex(1 << 20);
        let (server_read, server_write) = tokio::io::split(server_side);
        tokio::spawn(async move {
            AcpServer::new(handler)
                .with_agent_info("agent-acp-test", "0.0.0")
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

    async fn send_raw(&mut self, line: &str) {
        self.write.write_all(line.as_bytes()).await.expect("write");
        self.write.write_all(b"\n").await.expect("write newline");
        self.write.flush().await.expect("flush");
    }

    async fn request(&mut self, id: u64, method: &str, params: Value) {
        let msg = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        self.send_raw(&msg.to_string()).await;
    }

    async fn notify(&mut self, method: &str, params: Value) {
        let msg = json!({ "jsonrpc": "2.0", "method": method, "params": params });
        self.send_raw(&msg.to_string()).await;
    }

    async fn next_message(&mut self) -> Value {
        let mut line = String::new();
        let read = tokio::time::timeout(TIMEOUT, self.read.read_line(&mut line))
            .await
            .expect("message within timeout")
            .expect("read succeeds");
        assert!(read > 0, "unexpected EOF from server");
        serde_json::from_str(&line).expect("server emits valid JSON")
    }

    /// Read messages until the response with `id` arrives; return the
    /// notifications seen on the way plus the response.
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

    /// Run the recorded initialize → session/new preamble; returns sessionId.
    async fn handshake(&mut self) -> String {
        self.request(0, "initialize", fixture("initialize")).await;
        let (_, init) = self.until_response(0).await;
        assert_eq!(init["result"]["protocolVersion"], json!(2));
        assert_eq!(init["result"]["agentInfo"]["name"], json!("agent-acp-test"));

        self.request(1, "session/new", fixture("session_new")).await;
        let (_, new) = self.until_response(1).await;
        new["result"]["sessionId"]
            .as_str()
            .expect("session/new returns a sessionId")
            .to_owned()
    }
}

#[tokio::test]
async fn recorded_cycle_initialize_new_prompt() {
    let mut client = Client::start(EchoHandler);
    let session_id = client.handshake().await;

    client
        .request(
            2,
            "session/prompt",
            with_session_id(fixture("session_prompt"), &session_id),
        )
        .await;
    let (notifications, response) = client.until_response(2).await;

    // The streamed chunk precedes the response and echoes both text blocks.
    let chunk = notifications
        .iter()
        .find(|n| {
            n["method"] == json!("session/update")
                && n["params"]["update"]["sessionUpdate"] == json!("agent_message_chunk")
        })
        .expect("an agent_message_chunk notification before the response");
    assert_eq!(chunk["params"]["sessionId"], json!(session_id));
    let text = chunk["params"]["update"]["content"]["text"]
        .as_str()
        .expect("chunk carries text");
    assert!(text.contains("/status now"), "first block echoed: {text}");
    assert!(text.contains("[Context]"), "second block echoed: {text}");

    assert_eq!(response["result"]["stopReason"], json!("end_turn"));
}

#[tokio::test]
async fn cancel_notification_resolves_prompt_as_cancelled() {
    let mut client = Client::start(HangUntilCancelled);
    let session_id = client.handshake().await;

    client
        .request(
            2,
            "session/prompt",
            with_session_id(fixture("session_prompt"), &session_id),
        )
        .await;
    // Wait for the first streamed chunk so the turn is provably in flight.
    let first = client.next_message().await;
    assert_eq!(first["method"], json!("session/update"));

    client
        .notify(
            "session/cancel",
            with_session_id(fixture("session_cancel"), &session_id),
        )
        .await;
    let (_, response) = client.until_response(2).await;
    assert_eq!(response["result"]["stopReason"], json!("cancelled"));
}

#[tokio::test]
async fn malformed_and_unknown_traffic_never_kills_the_server() {
    let mut client = Client::start(EchoHandler);

    // Garbage line, unknown notification, and a bare response: all ignored.
    client.send_raw("this is not json {{{").await;
    client
        .notify("_goose/unstable/session/steer", json!({}))
        .await;
    client
        .send_raw(&json!({"jsonrpc": "2.0", "id": 99, "result": {}}).to_string())
        .await;

    // Unknown method WITH an id must get -32601 (silence would hang buzz-acp).
    client.request(5, "session/set_model", json!({})).await;
    let (_, err) = client.until_response(5).await;
    assert_eq!(err["error"]["code"], json!(-32601));

    // And the server still completes a full recorded cycle afterwards.
    let session_id = client.handshake().await;
    client
        .request(
            6,
            "session/prompt",
            with_session_id(fixture("session_prompt"), &session_id),
        )
        .await;
    let (_, response) = client.until_response(6).await;
    assert_eq!(response["result"]["stopReason"], json!("end_turn"));
}

#[tokio::test]
async fn prompt_errors_are_wire_errors_not_crashes() {
    let mut client = Client::start(EchoHandler);
    let _session_id = client.handshake().await;

    // Unknown session.
    client
        .request(
            2,
            "session/prompt",
            json!({ "sessionId": "nope", "prompt": [] }),
        )
        .await;
    let (_, err) = client.until_response(2).await;
    assert_eq!(err["error"]["code"], json!(-32602));

    // Missing sessionId.
    client
        .request(3, "session/prompt", json!({ "prompt": [] }))
        .await;
    let (_, err) = client.until_response(3).await;
    assert_eq!(err["error"]["code"], json!(-32602));

    // Overlap: second prompt on the same session while one is in flight.
    let mut hang_client = Client::start(HangUntilCancelled);
    let hang_session = hang_client.handshake().await;
    hang_client
        .request(
            4,
            "session/prompt",
            with_session_id(fixture("session_prompt"), &hang_session),
        )
        .await;
    let first = hang_client.next_message().await;
    assert_eq!(first["method"], json!("session/update"));
    hang_client
        .request(
            5,
            "session/prompt",
            with_session_id(fixture("session_prompt"), &hang_session),
        )
        .await;
    let (_, overlap) = hang_client.until_response(5).await;
    assert_eq!(overlap["error"]["code"], json!(-32602));

    // The original prompt still resolves after cancel.
    hang_client
        .notify(
            "session/cancel",
            with_session_id(fixture("session_cancel"), &hang_session),
        )
        .await;
    let (_, response) = hang_client.until_response(4).await;
    assert_eq!(response["result"]["stopReason"], json!("cancelled"));
}

/// Panics mid-turn — the server must resolve the prompt and free the session.
struct PanicHandler;

#[async_trait::async_trait]
impl PromptHandler for PanicHandler {
    async fn prompt(
        &self,
        _request: PromptRequest,
        _updates: UpdateSink,
        _cancel: CancellationToken,
    ) -> Result<StopReason, PromptError> {
        panic!("handler exploded");
    }
}

#[tokio::test]
async fn handler_panic_resolves_prompt_and_frees_the_session() {
    let mut client = Client::start(PanicHandler);
    let session_id = client.handshake().await;

    client
        .request(
            2,
            "session/prompt",
            with_session_id(fixture("session_prompt"), &session_id),
        )
        .await;
    let (_, err) = client.until_response(2).await;
    assert_eq!(err["error"]["code"], json!(-32603));

    // The session is NOT stranded in-flight: a new prompt is accepted (and
    // resolves the same way, proving begin/end bookkeeping survived).
    client
        .request(
            3,
            "session/prompt",
            with_session_id(fixture("session_prompt"), &session_id),
        )
        .await;
    let (_, err) = client.until_response(3).await;
    assert_eq!(err["error"]["code"], json!(-32603));
}
