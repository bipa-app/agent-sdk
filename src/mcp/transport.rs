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

    /// Close the transport connection.
    ///
    /// # Errors
    ///
    /// Returns an error if the transport fails to close cleanly.
    async fn close(&self) -> Result<()>;
}

/// Stdio transport for MCP servers.
///
/// Spawns a subprocess and communicates via stdin/stdout using JSON-RPC.
pub struct StdioTransport {
    /// Request ID counter.
    next_id: AtomicU64,
    /// Pending requests awaiting responses.
    pending: Mutex<HashMap<RequestId, oneshot::Sender<JsonRpcResponse>>>,
    /// Writer to send requests.
    writer: Mutex<tokio::io::BufWriter<tokio::process::ChildStdin>>,
    /// Child process handle.
    _child: Arc<Mutex<Child>>,
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
            pending: Mutex::new(HashMap::new()),
            writer: Mutex::new(tokio::io::BufWriter::new(stdin)),
            _child: Arc::new(Mutex::new(child)),
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
                        if let Ok(response) = serde_json::from_str::<JsonRpcResponse>(&line) {
                            let mut pending = transport_clone.pending.lock().await;
                            if let Some(sender) = pending.remove(&response.id) {
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
            pending: Mutex::new(HashMap::new()),
            writer: Mutex::new(tokio::io::BufWriter::new(stdin)),
            _child: Arc::new(Mutex::new(child)),
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
                        if let Ok(response) = serde_json::from_str::<JsonRpcResponse>(&line) {
                            let mut pending = transport_clone.pending.lock().await;
                            if let Some(sender) = pending.remove(&response.id) {
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
            let mut pending = self.pending.lock().await;
            pending.insert(request.id.clone(), tx);
        }

        // Send request
        let json = serde_json::to_string(&request)?;
        let mut writer = self.writer.lock().await;
        writer.write_all(json.as_bytes()).await?;
        writer.write_all(b"\n").await?;
        writer.flush().await?;
        drop(writer);

        // Wait for response
        let response = rx.await.context("Response channel closed")?;

        // Check for JSON-RPC error
        if let Some(ref error) = response.error {
            bail!("JSON-RPC error {}: {}", error.code, error.message);
        }

        Ok(response)
    }

    async fn close(&self) -> Result<()> {
        // Closing is handled by dropping the transport (kill_on_drop)
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_id_generation() {
        let next_id = AtomicU64::new(1);
        assert_eq!(next_id.fetch_add(1, Ordering::SeqCst), 1);
        assert_eq!(next_id.fetch_add(1, Ordering::SeqCst), 2);
        assert_eq!(next_id.fetch_add(1, Ordering::SeqCst), 3);
    }
}
