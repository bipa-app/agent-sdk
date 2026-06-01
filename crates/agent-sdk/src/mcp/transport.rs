//! MCP transport implementations.

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, oneshot};

use super::protocol::{JsonRpcRequest, JsonRpcResponse, RequestId};

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

/// Default response timeout for MCP requests (60 seconds).
const DEFAULT_RESPONSE_TIMEOUT: std::time::Duration = std::time::Duration::from_mins(1);

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
    pending: PendingMap,
    /// Writer to send requests.
    writer: Mutex<tokio::io::BufWriter<tokio::process::ChildStdin>>,
    /// Child process handle.
    _child: Arc<Mutex<Child>>,
    /// Timeout for awaiting responses.
    response_timeout: std::time::Duration,
}

impl StdioTransport {
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
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("Failed to spawn MCP server: {command}"))?;

        let stdin = child.stdin.take().context("Failed to get stdin")?;
        let stdout = child.stdout.take().context("Failed to get stdout")?;

        let transport = Arc::new(Self {
            next_id: AtomicU64::new(1),
            pending: std::sync::Mutex::new(HashMap::new()),
            writer: Mutex::new(tokio::io::BufWriter::new(stdin)),
            _child: Arc::new(Mutex::new(child)),
            response_timeout: DEFAULT_RESPONSE_TIMEOUT,
        });

        // Spawn reader task
        let transport_clone = Arc::clone(&transport);
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
                                let mut pending = match transport_clone.pending.lock() {
                                    Ok(pending) => pending,
                                    Err(poisoned) => poisoned.into_inner(),
                                };
                                pending.remove(&response.id)
                            };
                            if let Some(sender) = sender {
                                let _ = sender.send(response);
                            }
                        }
                    }
                }
            }
        });

        Ok(transport)
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
            pending: std::sync::Mutex::new(HashMap::new()),
            writer: Mutex::new(tokio::io::BufWriter::new(stdin)),
            _child: Arc::new(Mutex::new(child)),
            response_timeout: DEFAULT_RESPONSE_TIMEOUT,
        });

        // Spawn reader task
        let transport_clone = Arc::clone(&transport);
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
                                let mut pending = match transport_clone.pending.lock() {
                                    Ok(pending) => pending,
                                    Err(poisoned) => poisoned.into_inner(),
                                };
                                pending.remove(&response.id)
                            };
                            if let Some(sender) = sender {
                                let _ = sender.send(response);
                            }
                        }
                    }
                }
            }
        });

        Ok(transport)
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

    /// Spawn a transport with a custom response timeout.
    ///
    /// Test-only helper so timeout behaviour can be exercised deterministically
    /// without waiting for the production 60s default.
    #[cfg(test)]
    fn spawn_with_timeout(
        command: &str,
        args: &[&str],
        response_timeout: std::time::Duration,
    ) -> Result<Arc<Self>> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("Failed to spawn MCP server: {command}"))?;

        let stdin = child.stdin.take().context("Failed to get stdin")?;
        let stdout = child.stdout.take().context("Failed to get stdout")?;

        let transport = Arc::new(Self {
            next_id: AtomicU64::new(1),
            pending: std::sync::Mutex::new(HashMap::new()),
            writer: Mutex::new(tokio::io::BufWriter::new(stdin)),
            _child: Arc::new(Mutex::new(child)),
            response_timeout,
        });

        let transport_clone = Arc::clone(&transport);
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();

            loop {
                line.clear();
                match reader.read_line(&mut line).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        if let Ok(response) = serde_json::from_str::<JsonRpcResponse>(&line) {
                            let sender = {
                                let mut pending = match transport_clone.pending.lock() {
                                    Ok(pending) => pending,
                                    Err(poisoned) => poisoned.into_inner(),
                                };
                                pending.remove(&response.id)
                            };
                            if let Some(sender) = sender {
                                let _ = sender.send(response);
                            }
                        }
                    }
                }
            }
        });

        Ok(transport)
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
            pending: &self.pending,
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
        // Assign an ID for serialization but don't register a pending response
        let id = self.next_request_id();
        request.id = RequestId::Number(id);

        let json = serde_json::to_string(&request)?;
        let mut writer = self.writer.lock().await;
        writer.write_all(json.as_bytes()).await?;
        writer.write_all(b"\n").await?;
        writer.flush().await?;
        drop(writer);
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        // Closing is handled by dropping the transport (kill_on_drop)
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

    /// Regression test for ENG-8736: a server that never replies must not leak
    /// pending-request entries on timeout. We spawn a child that drains stdin
    /// to `/dev/null` and never writes to stdout, so every `send` hits the
    /// timeout branch. After N timed-out requests the pending map must be empty
    /// — without the [`PendingGuard`] it would hold N stale senders and grow
    /// without bound, eventually OOMing a long-lived host.
    #[tokio::test]
    async fn timed_out_requests_do_not_leak_pending_entries() -> Result<()> {
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

        const N: usize = 8;
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
}
