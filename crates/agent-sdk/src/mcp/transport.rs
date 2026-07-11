//! MCP transport implementations.

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdout, Command};
use tokio::sync::{Mutex, oneshot};

use super::protocol::{JsonRpcRequest, JsonRpcResponse, RequestId};

/// Serialize a JSON-RPC *notification*: a request message with its `id` field
/// removed.
///
/// JSON-RPC 2.0 (and the MCP spec) require notifications to omit `id`; a
/// notification that carries an id is malformed and strict servers reject it or
/// emit an orphan response. Our [`JsonRpcRequest`] type always carries an id for
/// serialization, so we strip it here before sending.
pub(crate) fn notification_body(request: &JsonRpcRequest) -> Result<String> {
    let mut value =
        serde_json::to_value(request).context("failed to serialize MCP notification")?;
    if let Some(object) = value.as_object_mut() {
        object.remove("id");
    }
    serde_json::to_string(&value).context("failed to serialize MCP notification")
}

/// Trait for MCP transports.
///
/// Transports handle the communication protocol with MCP servers.
///
/// # Example
///
/// ```ignore
/// use agent_sdk::mcp::{McpTransport, StdioTransport};
///
/// let transport = StdioTransport::spawn("npx", &["-y", "mcp-server"]).await?;
/// let response = transport.send(request).await?;
/// ```
#[async_trait]
pub trait McpTransport: Send + Sync {
    /// Send a request and wait for a response.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails to send or the response fails to parse.
    async fn send(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse>;

    /// Send a notification (fire-and-forget, no response expected).
    ///
    /// # Errors
    ///
    /// Returns an error if the message fails to serialize or write.
    async fn send_notification(&self, request: JsonRpcRequest) -> Result<()>;

    /// Record the protocol revision negotiated during `initialize`.
    ///
    /// Transports that carry the revision out-of-band (e.g. the streamable-HTTP
    /// transport, which must send a `MCP-Protocol-Version` header on every
    /// request after initialization) override this. The default is a no-op for
    /// transports like stdio where the revision lives only in the JSON-RPC
    /// payload.
    async fn set_protocol_version(&self, _version: &str) {}

    /// Close the transport connection.
    ///
    /// # Errors
    ///
    /// Returns an error if the transport fails to close cleanly.
    async fn close(&self) -> Result<()>;
}

/// Map of in-flight requests to the channel that delivers their response.
///
/// A `std::sync::Mutex` (not a `tokio::sync::Mutex`) is used deliberately: the
/// critical sections are a single `HashMap` insert or remove and are never held
/// across an `.await`. Keeping the lock synchronous lets [`PendingGuard`]'s
/// `Drop` remove its entry without blocking inside an async context.
type PendingMap = std::sync::Mutex<HashMap<RequestId, oneshot::Sender<JsonRpcResponse>>>;

/// RAII guard that removes a pending-request entry on drop.
///
/// The `send` path inserts a `oneshot::Sender` into the `pending` map before
/// writing the request, then awaits the matching response with a timeout. If
/// the future returns early for *any* reason — timeout, the response channel
/// closing, or the task being cancelled/dropped mid-await — the entry would
/// otherwise leak. For a server that times out frequently this grows the map
/// without bound and eventually OOMs a long-lived host (ENG-8736).
///
/// Holding this guard for the duration of the await guarantees the entry is
/// removed on every exit path. On the success path the reader task has already
/// removed the entry by the time the guard drops, so the removal is a harmless
/// no-op.
struct PendingGuard<'a> {
    pending: &'a PendingMap,
    request_id: RequestId,
}

impl Drop for PendingGuard<'_> {
    fn drop(&mut self) {
        // A poisoned lock means a prior holder panicked while mutating the map;
        // recover the guard and still remove our entry so it cannot leak.
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        pending.remove(&self.request_id);
    }
}

/// Stdio transport for MCP servers.
///
/// Spawns a subprocess and communicates via stdin/stdout using JSON-RPC.
pub struct StdioTransport {
    /// Request ID counter.
    next_id: AtomicU64,
    /// Pending requests awaiting responses.
    ///
    /// Wrapped in `Arc` so the reader task can hold the map *without* keeping
    /// the rest of the transport (the child handle, the stdin writer) alive.
    /// Holding only the map means dropping every user handle drops the writer
    /// (closing the child's stdin → the child exits → stdout EOF → the reader
    /// task ends), instead of the reader's strong reference pinning the process
    /// forever.
    pending: Arc<PendingMap>,
    /// Writer to send requests.
    writer: Mutex<tokio::io::BufWriter<tokio::process::ChildStdin>>,
    /// Child process handle (`kill_on_drop`); also killed explicitly by `close`.
    child: Arc<Mutex<Child>>,
    /// Timeout for awaiting responses.
    response_timeout: Duration,
}

impl StdioTransport {
    /// Default per-request response timeout (60 seconds).
    ///
    /// Applied by [`StdioTransport::spawn`] and [`StdioTransport::spawn_with_env`].
    /// Override it per connection with [`StdioTransport::spawn_with_timeout`] or
    /// [`StdioTransport::spawn_with_env_and_timeout`]; reference it to base a
    /// custom timeout off the default (e.g. `DEFAULT_RESPONSE_TIMEOUT * 5`).
    pub const DEFAULT_RESPONSE_TIMEOUT: Duration = Duration::from_mins(1);

    /// Spawn a new MCP server process.
    ///
    /// # Arguments
    ///
    /// * `command` - The command to execute
    /// * `args` - Arguments to pass to the command
    ///
    /// # Errors
    ///
    /// Returns an error if the process fails to spawn.
    pub fn spawn(command: &str, args: &[&str]) -> Result<Arc<Self>> {
        Self::spawn_with_env(command, args, &[])
    }

    /// Spawn a new MCP server process with environment variables.
    ///
    /// # Arguments
    ///
    /// * `command` - The command to execute
    /// * `args` - Arguments to pass to the command
    /// * `env` - Environment variables to set
    ///
    /// # Errors
    ///
    /// Returns an error if the process fails to spawn.
    pub fn spawn_with_env(command: &str, args: &[&str], env: &[(&str, &str)]) -> Result<Arc<Self>> {
        Self::spawn_inner(command, args, env, Self::DEFAULT_RESPONSE_TIMEOUT)
    }

    /// Spawn a transport with a custom response timeout.
    ///
    /// MCP tool calls routinely exceed the 60s default (builds, codegen);
    /// this lets callers raise (or lower) the per-request response deadline.
    ///
    /// # Errors
    ///
    /// Returns an error if the process fails to spawn.
    pub fn spawn_with_timeout(
        command: &str,
        args: &[&str],
        response_timeout: Duration,
    ) -> Result<Arc<Self>> {
        Self::spawn_inner(command, args, &[], response_timeout)
    }

    /// Spawn a transport with both environment variables and a custom response
    /// timeout.
    ///
    /// # Errors
    ///
    /// Returns an error if the process fails to spawn.
    pub fn spawn_with_env_and_timeout(
        command: &str,
        args: &[&str],
        env: &[(&str, &str)],
        response_timeout: Duration,
    ) -> Result<Arc<Self>> {
        Self::spawn_inner(command, args, env, response_timeout)
    }

    /// Shared constructor: spawn the child, wire up the reader task, and return
    /// the transport. All public constructors funnel through here so the
    /// reader-loop and process setup live in exactly one place.
    fn spawn_inner(
        command: &str,
        args: &[&str],
        env: &[(&str, &str)],
        response_timeout: Duration,
    ) -> Result<Arc<Self>> {
        let mut cmd = Command::new(command);
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);

        for (key, value) in env {
            cmd.env(key, value);
        }

        let mut child = cmd
            .spawn()
            .with_context(|| format!("Failed to spawn MCP server: {command}"))?;

        let stdin = child.stdin.take().context("Failed to get stdin")?;
        let stdout = child.stdout.take().context("Failed to get stdout")?;

        let transport = Arc::new(Self {
            next_id: AtomicU64::new(1),
            pending: Arc::new(std::sync::Mutex::new(HashMap::new())),
            writer: Mutex::new(tokio::io::BufWriter::new(stdin)),
            child: Arc::new(Mutex::new(child)),
            response_timeout,
        });

        // The reader task holds only the pending map (not the transport), so it
        // never keeps the child/writer alive. See [`StdioTransport::pending`].
        Self::spawn_reader(stdout, Arc::clone(&transport.pending));

        Ok(transport)
    }

    /// Spawn the background task that reads JSON-RPC responses off the child's
    /// stdout and dispatches them to waiting senders.
    ///
    /// On EOF or read error (the child died or closed stdout) the task drains
    /// the pending map, dropping every sender. Each waiting `send` then fails
    /// immediately via the "Response channel closed" path instead of burning
    /// the full response timeout.
    fn spawn_reader(stdout: ChildStdout, pending: Arc<PendingMap>) {
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();

            loop {
                line.clear();
                match reader.read_line(&mut line).await {
                    Ok(0) | Err(_) => break, // EOF or error
                    Ok(_) => {
                        const MAX_LINE_LEN: usize = 10 * 1024 * 1024; // 10 MiB
                        if line.len() > MAX_LINE_LEN {
                            log::warn!(
                                "MCP stdout line exceeds {} bytes (got {}), skipping",
                                MAX_LINE_LEN,
                                line.len()
                            );
                            continue;
                        }
                        if let Ok(response) = serde_json::from_str::<JsonRpcResponse>(&line) {
                            let sender = {
                                let mut map = pending
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                                map.remove(&response.id)
                            };
                            if let Some(sender) = sender {
                                let _ = sender.send(response);
                            }
                        }
                    }
                }
            }

            // Reader exited: the child closed stdout (it died or finished).
            // Fail all in-flight requests now so callers don't wait out the
            // full response timeout.
            let mut map = pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            map.clear();
        });
    }

    /// Get the next request ID.
    fn next_request_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Number of in-flight requests still registered in the pending map.
    ///
    /// Test-only accessor used to assert the map does not grow unbounded when
    /// requests time out (ENG-8736).
    #[cfg(test)]
    fn pending_len(&self) -> usize {
        self.pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Per-request response timeout configured on this transport.
    ///
    /// Test-only accessor used to assert constructors store the timeout the
    /// caller supplied (or the documented default when none is given).
    #[cfg(test)]
    const fn response_timeout(&self) -> Duration {
        self.response_timeout
    }
}

#[async_trait]
impl McpTransport for StdioTransport {
    async fn send(&self, mut request: JsonRpcRequest) -> Result<JsonRpcResponse> {
        // Assign request ID
        let id = self.next_request_id();
        request.id = RequestId::Number(id);

        // Create response channel
        let (tx, rx) = oneshot::channel();
        {
            let mut pending = self
                .pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            pending.insert(request.id.clone(), tx);
        }

        // From here on, this guard removes the pending entry on every exit
        // path — timeout, channel-closed, JSON-RPC error, or this future being
        // cancelled/dropped mid-await. Without it the entry leaks and the map
        // grows unbounded for a server that times out frequently (ENG-8736).
        let _pending_guard = PendingGuard {
            pending: self.pending.as_ref(),
            request_id: request.id.clone(),
        };

        // Send request
        let json = serde_json::to_string(&request)?;
        let mut writer = self.writer.lock().await;
        writer.write_all(json.as_bytes()).await?;
        writer.write_all(b"\n").await?;
        writer.flush().await?;
        drop(writer);

        // Wait for response with timeout
        let response = tokio::time::timeout(self.response_timeout, rx)
            .await
            .context("MCP response timed out")?
            .context("Response channel closed")?;

        // Check for JSON-RPC error
        if let Some(ref error) = response.error {
            bail!("JSON-RPC error {}: {}", error.code, error.message);
        }

        Ok(response)
    }

    async fn send_notification(&self, mut request: JsonRpcRequest) -> Result<()> {
        // Advance the shared id counter so request ids stay monotonic across the
        // connection, but strip the id on the wire: JSON-RPC 2.0 / MCP
        // notifications must not carry one.
        let id = self.next_request_id();
        request.id = RequestId::Number(id);
        let json = notification_body(&request)?;
        let mut writer = self.writer.lock().await;
        writer.write_all(json.as_bytes()).await?;
        writer.write_all(b"\n").await?;
        writer.flush().await?;
        drop(writer);
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        // Close stdin so the child sees EOF and can exit cleanly.
        {
            let mut writer = self.writer.lock().await;
            let _ = writer.flush().await;
            let _ = writer.shutdown().await;
            drop(writer);
        }

        // Kill the child process explicitly rather than relying on a drop that
        // may never happen while user handles are alive.
        {
            let mut child = self.child.lock().await;
            let _ = child.start_kill();
        }

        // Fail any in-flight requests immediately so awaiting `send` calls do
        // not wait out the response timeout.
        {
            let mut map = self
                .pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            map.clear();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::ensure;

    #[test]
    fn test_request_id_generation() {
        let next_id = AtomicU64::new(1);
        assert_eq!(next_id.fetch_add(1, Ordering::SeqCst), 1);
        assert_eq!(next_id.fetch_add(1, Ordering::SeqCst), 2);
        assert_eq!(next_id.fetch_add(1, Ordering::SeqCst), 3);
    }

    /// The documented default response timeout must stay at 60 seconds so
    /// existing callers see no behavioural change.
    #[test]
    fn default_response_timeout_is_sixty_seconds() {
        assert_eq!(
            StdioTransport::DEFAULT_RESPONSE_TIMEOUT,
            Duration::from_mins(1)
        );
    }

    /// Constructors with no timeout argument must store the documented default,
    /// while the `*_with_timeout` constructors must store the caller's value.
    #[tokio::test]
    async fn constructors_store_configured_timeout() -> Result<()> {
        let default = StdioTransport::spawn_with_env("sh", &["-c", "cat > /dev/null"], &[])?;
        ensure!(
            default.response_timeout() == StdioTransport::DEFAULT_RESPONSE_TIMEOUT,
            "spawn_with_env must store the default timeout"
        );

        let custom = StdioTransport::spawn_with_timeout(
            "sh",
            &["-c", "cat > /dev/null"],
            Duration::from_millis(250),
        )?;
        ensure!(
            custom.response_timeout() == Duration::from_millis(250),
            "spawn_with_timeout must store the caller's timeout"
        );
        Ok(())
    }

    /// Regression test for ENG-8736: a server that never replies must not leak
    /// pending-request entries on timeout. We spawn a child that drains stdin
    /// to `/dev/null` and never writes to stdout, so every `send` hits the
    /// timeout branch. After N timed-out requests the pending map must be empty
    /// — without the [`PendingGuard`] it would hold N stale senders and grow
    /// without bound, eventually exhausting the memory of a long-lived host.
    #[tokio::test]
    async fn timed_out_requests_do_not_leak_pending_entries() -> Result<()> {
        const N: usize = 8;

        // `cat > /dev/null` accepts everything written to stdin and emits
        // nothing on stdout, guaranteeing a timeout on every request.
        let transport = StdioTransport::spawn_with_timeout(
            "sh",
            &["-c", "cat > /dev/null"],
            std::time::Duration::from_millis(50),
        )?;

        ensure!(
            transport.pending_len() == 0,
            "pending map should start empty"
        );

        for _ in 0..N {
            let request = JsonRpcRequest::new("tools/list", None, 0);
            let result = transport.send(request).await;
            ensure!(
                result.is_err(),
                "request should time out when the server never replies"
            );
        }

        // The leak would manifest here: each timed-out request would leave its
        // sender behind. With the guard, every exit path removes the entry.
        ensure!(
            transport.pending_len() == 0,
            "pending map must be empty after {N} timeouts, found {} stale entries",
            transport.pending_len(),
        );

        Ok(())
    }

    /// JSON-RPC notifications must not carry an `id` (finding 13).
    #[test]
    fn notification_body_omits_id() -> Result<()> {
        let request = JsonRpcRequest::new("notifications/initialized", None, 7);
        let body = notification_body(&request)?;
        let value: serde_json::Value = serde_json::from_str(&body)?;
        ensure!(
            value.get("id").is_none(),
            "notification must not carry an id, got: {body}"
        );
        ensure!(
            value.get("method").and_then(serde_json::Value::as_str)
                == Some("notifications/initialized"),
            "notification method must be preserved"
        );
        ensure!(
            value.get("jsonrpc").and_then(serde_json::Value::as_str) == Some("2.0"),
            "jsonrpc version must be preserved"
        );
        Ok(())
    }

    /// Regression test for findings 3 & 12: `close()` must terminate the child
    /// and fail in-flight requests immediately instead of leaving them to wait
    /// out the full response timeout. The child drains stdin and never replies,
    /// so without the fix `send` would block for the entire (30s) timeout.
    #[tokio::test]
    async fn close_fails_in_flight_requests_promptly() -> Result<()> {
        let transport = StdioTransport::spawn_with_timeout(
            "sh",
            &["-c", "cat > /dev/null"],
            Duration::from_secs(30),
        )?;

        let sender = Arc::clone(&transport);
        let handle = tokio::spawn(async move {
            sender
                .send(JsonRpcRequest::new("tools/list", None, 0))
                .await
        });

        // Wait until the request is registered in the pending map.
        for _ in 0..200 {
            if transport.pending_len() == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        ensure!(
            transport.pending_len() == 1,
            "request should be registered before close()"
        );

        transport.close().await?;

        // The in-flight request must fail well under the 30s response timeout.
        let send_result = tokio::time::timeout(Duration::from_secs(5), handle)
            .await
            .context("in-flight request did not fail promptly after close()")?
            .context("send task panicked")?;
        ensure!(
            send_result.is_err(),
            "in-flight request must fail when the transport is closed"
        );

        Ok(())
    }
}
