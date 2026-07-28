//! The ACP stdio server: read loop, dispatch, and the prompt-handler seam.
//!
//! Transport shape: one JSON-RPC message per line on stdin/stdout. All
//! outgoing traffic (responses AND `session/update` notifications) funnels
//! through one writer channel, so a prompt's streamed updates are always
//! written before its final response — ordering is structural, not timed.
//!
//! Liveness matters to the harness: buzz-acp enforces an idle timeout keyed
//! to stdout activity. Handlers are expected to stream updates through the
//! [`UpdateSink`] while they work; a silent handler will get its turn killed
//! by the client, not by this server.

use std::sync::Arc;

use futures::{FutureExt, StreamExt};
use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio_util::codec::{FramedRead, LinesCodec, LinesCodecError};
use tokio_util::sync::CancellationToken;

use crate::session::{BeginPromptError, NewSessionParams, SessionStore};
use crate::wire::{self, Incoming, PROTOCOL_VERSION, StopReason, error_codes};

/// Maximum accepted line length. Matches the harness's own read-side cap so
/// neither side can OOM the other with a runaway unterminated line.
const MAX_LINE_BYTES: usize = 10_000_000;

/// How long shutdown waits for in-flight prompt tasks after cancelling them.
const SHUTDOWN_DRAIN: std::time::Duration = std::time::Duration::from_secs(5);

/// Identity reported in the `initialize` response.
#[derive(Debug, Clone)]
pub struct AgentInfo {
    /// Agent name shown by clients (e.g. in harness logs).
    pub name: String,
    /// Agent version string.
    pub version: String,
}

impl Default for AgentInfo {
    fn default() -> Self {
        Self {
            name: "agent-acp".to_owned(),
            version: env!("CARGO_PKG_VERSION").to_owned(),
        }
    }
}

/// Error returned when emitting an update after the transport writer closed.
#[derive(Debug, thiserror::Error)]
#[error("update sink closed: the transport writer has shut down")]
pub struct UpdateSinkClosed;

/// Streaming outlet for `session/update` notifications during a prompt.
#[derive(Clone)]
pub struct UpdateSink {
    session_id: String,
    out: mpsc::Sender<Value>,
}

impl UpdateSink {
    /// Emit an `agent_message_chunk` update (assistant text delta).
    ///
    /// # Errors
    ///
    /// Returns [`UpdateSinkClosed`] when the transport writer has shut down
    /// (client disconnected); handlers should treat that as cancellation.
    pub async fn agent_message_chunk(&self, text: &str) -> Result<(), UpdateSinkClosed> {
        self.session_update(json!({
            "sessionUpdate": "agent_message_chunk",
            "content": { "type": "text", "text": text },
        }))
        .await
    }

    /// Emit a raw `session/update` with an arbitrary `update` payload.
    ///
    /// The `sessionId` is injected by the sink; callers supply only the
    /// `update` object. This is the escape hatch later milestones build the
    /// full event mapping on.
    ///
    /// # Errors
    ///
    /// Returns [`UpdateSinkClosed`] when the transport writer has shut down.
    pub async fn session_update(&self, update: Value) -> Result<(), UpdateSinkClosed> {
        let msg = wire::notification(
            "session/update",
            json!({ "sessionId": self.session_id, "update": update }),
        );
        self.out.send(msg).await.map_err(|_| UpdateSinkClosed)
    }
}

/// Failure surfaced by a prompt handler, mapped to a JSON-RPC internal error
/// on the wire.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct PromptError {
    /// Human-readable failure description sent to the client.
    pub message: String,
}

impl PromptError {
    /// Build a prompt error from any displayable message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// One prompt (turn) request, as handed to a [`PromptHandler`].
#[derive(Debug)]
pub struct PromptRequest {
    /// The session the prompt belongs to.
    pub session_id: String,
    /// Parameters captured at `session/new`.
    pub session: Arc<NewSessionParams>,
    /// Text content blocks, in wire order. Non-text blocks are dropped.
    pub blocks: Vec<String>,
}

/// The behavior behind the wire: what happens when a prompt arrives.
///
/// This is the milestone-M0.1 seam; the durable-host backend contract
/// (`AcpBackend` / `AcpRunHandle`) layers on top of it in M0.2. Handlers
/// must honor `cancel` promptly and return [`StopReason::Cancelled`] when
/// interrupted — `session/cancel` only signals the token, it cannot force
/// a return.
#[async_trait::async_trait]
pub trait PromptHandler: Send + Sync + 'static {
    /// Run one turn: stream updates through `updates`, then return the
    /// turn's stop reason.
    ///
    /// # Errors
    ///
    /// A [`PromptError`] resolves the pending `session/prompt` request as a
    /// JSON-RPC internal error (code −32603) carrying the message.
    async fn prompt(
        &self,
        request: PromptRequest,
        updates: UpdateSink,
        cancel: CancellationToken,
    ) -> Result<StopReason, PromptError>;
}

/// ACP server over a generic line-oriented transport.
pub struct AcpServer<H> {
    handler: Arc<H>,
    agent_info: AgentInfo,
}

impl<H: PromptHandler> AcpServer<H> {
    /// Create a server around a prompt handler.
    pub fn new(handler: H) -> Self {
        Self {
            handler: Arc::new(handler),
            agent_info: AgentInfo::default(),
        }
    }

    /// Override the identity reported by `initialize`.
    #[must_use]
    pub fn with_agent_info(mut self, name: impl Into<String>, version: impl Into<String>) -> Self {
        self.agent_info = AgentInfo {
            name: name.into(),
            version: version.into(),
        };
        self
    }

    /// Serve ACP on this process's stdin/stdout until EOF.
    ///
    /// # Errors
    ///
    /// Returns an [`std::io::Error`] only for transport-level read failures;
    /// protocol-level problems (malformed lines, unknown methods) are
    /// handled on the wire and never abort the server.
    pub async fn serve_stdio(self) -> std::io::Result<()> {
        self.serve(tokio::io::stdin(), tokio::io::stdout()).await
    }

    /// Serve ACP over arbitrary reader/writer halves (used by tests).
    ///
    /// # Errors
    ///
    /// Returns an [`std::io::Error`] for transport-level read failures.
    pub async fn serve<R, W>(self, reader: R, writer: W) -> std::io::Result<()>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin + Send + 'static,
    {
        let (out_tx, out_rx) = mpsc::channel::<Value>(256);
        let writer_task = tokio::spawn(write_loop(writer, out_rx));

        let store = Arc::new(SessionStore::default());
        let mut prompts: JoinSet<()> = JoinSet::new();
        let mut lines = FramedRead::new(reader, LinesCodec::new_with_max_length(MAX_LINE_BYTES));

        while let Some(line) = lines.next().await {
            let line = match line {
                Ok(line) => line,
                Err(LinesCodecError::MaxLineLengthExceeded) => {
                    log::warn!("acp: dropping over-long line (> {MAX_LINE_BYTES} bytes)");
                    continue;
                }
                Err(LinesCodecError::Io(e)) => return Err(e),
            };
            let Ok(value) = serde_json::from_str::<Value>(&line) else {
                log::warn!("acp: skipping non-JSON line");
                continue;
            };
            match Incoming::classify(value) {
                Incoming::Request { id, method, params } => {
                    self.dispatch_request(&out_tx, &store, &mut prompts, id, &method, params)
                        .await;
                }
                Incoming::Notification { method, params } => {
                    dispatch_notification(&store, &method, &params);
                }
                Incoming::Response { id } => {
                    log::debug!("acp: ignoring response to unknown request id {id}");
                }
                Incoming::Malformed => {
                    log::warn!("acp: skipping malformed JSON-RPC line (no method, no id)");
                }
            }
        }

        // EOF: the client is gone. Cancel in-flight turns, give them a
        // bounded window to resolve, then let the writer drain and stop.
        store.cancel_all();
        let drain = async { while prompts.join_next().await.is_some() {} };
        if tokio::time::timeout(SHUTDOWN_DRAIN, drain).await.is_err() {
            log::warn!("acp: in-flight prompts did not resolve within shutdown drain window");
            prompts.abort_all();
        }
        drop(out_tx);
        let _ = writer_task.await;
        Ok(())
    }

    async fn dispatch_request(
        &self,
        out: &mpsc::Sender<Value>,
        store: &Arc<SessionStore>,
        prompts: &mut JoinSet<()>,
        id: Value,
        method: &str,
        params: Value,
    ) {
        match method {
            "initialize" => {
                let result = json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "agentInfo": {
                        "name": self.agent_info.name,
                        "version": self.agent_info.version,
                    },
                    "agentCapabilities": { "loadSession": false },
                    "authMethods": [],
                });
                send(out, wire::response(&id, result)).await;
            }
            "session/new" => {
                let session_id = store.create(NewSessionParams::from_params(&params));
                log::info!("acp: session created: {session_id}");
                send(out, wire::response(&id, json!({ "sessionId": session_id }))).await;
            }
            "session/prompt" => {
                self.dispatch_prompt(out, store, prompts, id, &params).await;
            }
            other => {
                log::debug!("acp: method not found: {other}");
                let msg = format!("Method not found: {other}");
                send(
                    out,
                    wire::error_response(&id, error_codes::METHOD_NOT_FOUND, &msg),
                )
                .await;
            }
        }
    }

    async fn dispatch_prompt(
        &self,
        out: &mpsc::Sender<Value>,
        store: &Arc<SessionStore>,
        prompts: &mut JoinSet<()>,
        id: Value,
        params: &Value,
    ) {
        let Some(session_id) = params.get("sessionId").and_then(Value::as_str) else {
            send(
                out,
                wire::error_response(
                    &id,
                    error_codes::INVALID_PARAMS,
                    "session/prompt requires a sessionId",
                ),
            )
            .await;
            return;
        };
        let session_id = session_id.to_owned();
        let blocks = text_blocks(params);

        let (session, cancel) = match store.begin_prompt(&session_id) {
            Ok(pair) => pair,
            Err(BeginPromptError::UnknownSession) => {
                let msg = format!("unknown sessionId: {session_id}");
                send(
                    out,
                    wire::error_response(&id, error_codes::INVALID_PARAMS, &msg),
                )
                .await;
                return;
            }
            Err(BeginPromptError::PromptInFlight) => {
                send(
                    out,
                    wire::error_response(
                        &id,
                        error_codes::INVALID_PARAMS,
                        "a prompt is already in flight on this session",
                    ),
                )
                .await;
                return;
            }
        };

        let handler = Arc::clone(&self.handler);
        let store = Arc::clone(store);
        let out = out.clone();
        prompts.spawn(async move {
            let updates = UpdateSink {
                session_id: session_id.clone(),
                out: out.clone(),
            };
            let request = PromptRequest {
                session_id: session_id.clone(),
                session,
                blocks,
            };
            // A panicking handler must not strand the session in-flight or
            // leave the client's prompt request unanswered forever — resolve
            // it as an internal error and free the session either way.
            let outcome = std::panic::AssertUnwindSafe(handler.prompt(request, updates, cancel))
                .catch_unwind()
                .await;
            store.end_prompt(&session_id);
            let reply = match outcome {
                Ok(Ok(reason)) => wire::response(&id, json!({ "stopReason": reason.as_str() })),
                Ok(Err(e)) => {
                    log::warn!("acp: prompt handler failed: {e}");
                    wire::error_response(&id, error_codes::INTERNAL_ERROR, &e.message)
                }
                Err(_panic) => {
                    log::error!("acp: prompt handler panicked; resolving prompt as internal error");
                    wire::error_response(
                        &id,
                        error_codes::INTERNAL_ERROR,
                        "prompt handler panicked",
                    )
                }
            };
            send(&out, reply).await;
        });
    }
}

fn dispatch_notification(store: &SessionStore, method: &str, params: &Value) {
    match method {
        "session/cancel" => {
            let Some(session_id) = params.get("sessionId").and_then(Value::as_str) else {
                log::warn!("acp: session/cancel without sessionId");
                return;
            };
            if store.cancel(session_id) {
                log::info!("acp: cancel signalled for session {session_id}");
            } else {
                log::debug!("acp: cancel with no in-flight prompt for {session_id}");
            }
        }
        other => {
            log::debug!("acp: ignoring unknown notification: {other}");
        }
    }
}

/// Extract the text content blocks of a `session/prompt` request, in order.
fn text_blocks(params: &Value) -> Vec<String> {
    params
        .get("prompt")
        .and_then(Value::as_array)
        .map(|blocks| {
            blocks
                .iter()
                .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

async fn send(out: &mpsc::Sender<Value>, msg: Value) {
    if out.send(msg).await.is_err() {
        log::debug!("acp: writer closed; dropping outgoing message");
    }
}

async fn write_loop<W: AsyncWrite + Unpin>(mut writer: W, mut rx: mpsc::Receiver<Value>) {
    while let Some(msg) = rx.recv().await {
        let mut line = msg.to_string();
        line.push('\n');
        if let Err(e) = writer.write_all(line.as_bytes()).await {
            log::warn!("acp: write failed, stopping writer: {e}");
            return;
        }
        if let Err(e) = writer.flush().await {
            log::warn!("acp: flush failed, stopping writer: {e}");
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_blocks_extracts_only_text_in_order() {
        let params = json!({
            "prompt": [
                { "type": "text", "text": "first" },
                { "type": "image", "data": "..." },
                { "type": "text", "text": "second" },
            ]
        });
        assert_eq!(text_blocks(&params), vec!["first", "second"]);
        assert!(text_blocks(&json!({})).is_empty());
    }
}
