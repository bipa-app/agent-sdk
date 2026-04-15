//! Minimal HTTP health endpoint for Kubernetes-style probes.
//!
//! Serves three routes on the configured `http_addr`:
//!
//! | Route | Purpose | Healthy | Unhealthy |
//! |-------|---------|---------|-----------|
//! | `GET /healthz` | Liveness | 200 | 503 |
//! | `GET /readyz` | Readiness | 200 | 503 |
//! | `GET /health` | Full snapshot | 200 (always) | 200 (always) |
//!
//! `/healthz` and `/readyz` return a JSON body with the full
//! [`ServiceHealth`] snapshot so operators can inspect the reason for a
//! non-200 response.  `/health` always returns 200 with the snapshot —
//! it is meant for dashboards, not load-balancer probes.
//!
//! The server is intentionally minimal: raw TCP + hand-crafted HTTP
//! responses so no additional dependencies are needed beyond `tokio`.
//!
//! [`ServiceHealth`]: crate::health::ServiceHealth

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::task::{JoinHandle, JoinSet};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::health::HealthSurface;

/// Handle for a running HTTP health server.
///
/// Cancelling the token or calling [`HttpHealthHandle::shutdown`] stops
/// the listener and awaits all in-flight connection tasks before
/// returning.
pub struct HttpHealthHandle {
    cancel: CancellationToken,
    join: JoinHandle<Result<()>>,
}

impl HttpHealthHandle {
    /// Cancel the server and await its drain.
    ///
    /// # Errors
    /// Returns an error if the server task panicked.
    pub async fn shutdown(self) -> Result<()> {
        self.cancel.cancel();
        self.join
            .await
            .context("HTTP health server task panicked")?
    }
}

/// Spawn the HTTP health server on `addr`.
///
/// The server observes `cancel` for graceful shutdown.
///
/// # Errors
/// Returns an error if binding the TCP listener fails.
pub async fn spawn(
    addr: SocketAddr,
    health: Arc<HealthSurface>,
    cancel: CancellationToken,
) -> Result<HttpHealthHandle> {
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding HTTP health listener on {addr}"))?;
    let local_addr = listener
        .local_addr()
        .context("resolving HTTP health local address")?;
    info!(%local_addr, "HTTP health endpoint listening");

    let token = cancel.clone();
    let join = tokio::spawn(async move { serve_loop(listener, health, cancel).await });
    Ok(HttpHealthHandle {
        cancel: token,
        join,
    })
}

async fn serve_loop(
    listener: TcpListener,
    health: Arc<HealthSurface>,
    cancel: CancellationToken,
) -> Result<()> {
    let mut connections = JoinSet::new();

    loop {
        tokio::select! {
            biased;
            () = cancel.cancelled() => {
                info!("HTTP health server shutting down");
                break;
            }
            accept = listener.accept() => {
                match accept {
                    Ok((stream, peer)) => {
                        let health = Arc::clone(&health);
                        connections.spawn(async move {
                            if let Err(err) = handle_connection(stream, &health).await {
                                debug!(%peer, error = %err, "HTTP health connection error");
                            }
                        });
                        while connections.try_join_next().is_some() {}
                    }
                    Err(err) => {
                        warn!(error = %err, "HTTP health accept failed");
                    }
                }
            }
        }
    }

    while connections.join_next().await.is_some() {}
    Ok(())
}

async fn handle_connection(
    mut stream: tokio::net::TcpStream,
    health: &HealthSurface,
) -> Result<()> {
    let mut buf = [0u8; 1024];
    let n = tokio::time::timeout(std::time::Duration::from_secs(5), stream.read(&mut buf))
        .await
        .context("HTTP health read timeout")?
        .context("reading HTTP request")?;
    if n == 0 {
        return Ok(());
    }

    let request = std::str::from_utf8(&buf[..n]).unwrap_or("");
    let path = parse_request_path(request);
    let snapshot = health.snapshot();

    let body = snapshot
        .to_json_bytes()
        .context("serializing health snapshot")?;

    let (status_code, status_text) = match path {
        "/healthz" if snapshot.is_live() => (200, "OK"),
        "/readyz" if snapshot.is_ready() => (200, "OK"),
        "/healthz" | "/readyz" => (503, "Service Unavailable"),
        "/health" => (200, "OK"),
        _ => {
            let not_found = b"{\"error\":\"not found\"}";
            let response = format!(
                "HTTP/1.1 404 Not Found\r\n\
                 Content-Type: application/json\r\n\
                 Content-Length: {}\r\n\
                 Connection: close\r\n\
                 \r\n",
                not_found.len(),
            );
            stream
                .write_all(response.as_bytes())
                .await
                .context("writing 404 response headers")?;
            stream
                .write_all(not_found)
                .await
                .context("writing 404 response body")?;
            stream.flush().await.context("flushing 404 response")?;
            return Ok(());
        }
    };

    let response = format!(
        "HTTP/1.1 {status_code} {status_text}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n",
        body.len(),
    );
    stream
        .write_all(response.as_bytes())
        .await
        .context("writing HTTP response headers")?;
    stream
        .write_all(&body)
        .await
        .context("writing HTTP response body")?;
    stream.flush().await.context("flushing HTTP response")?;
    Ok(())
}

fn parse_request_path(request: &str) -> &str {
    // HTTP request line: "GET /path?query HTTP/1.1\r\n..."
    let target = request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/");
    target.split('?').next().unwrap_or("/")
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::health::{CoreHealth, HealthSurface, LatencyLayerHealth};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpStream;

    async fn http_get(addr: SocketAddr, path: &str) -> Result<(u16, String)> {
        let mut stream = TcpStream::connect(addr)
            .await
            .context("connecting to health server")?;
        let request = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n");
        stream.write_all(request.as_bytes()).await?;
        stream.flush().await?;

        let mut buf = Vec::new();
        stream.read_to_end(&mut buf).await?;
        let response = String::from_utf8_lossy(&buf).to_string();

        let status_code = response
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|code| code.parse::<u16>().ok())
            .unwrap_or(0);

        let body = response.split("\r\n\r\n").nth(1).unwrap_or("").to_string();

        Ok((status_code, body))
    }

    #[tokio::test]
    async fn health_endpoints_respond_correctly() -> Result<()> {
        let health = HealthSurface::shared();
        health.set_sweep_alive(true);
        health.set_workers_alive(true);
        health.set_core(CoreHealth::Healthy);

        let cancel = CancellationToken::new();
        // Bind to port 0 to let the OS assign a free port.
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let token = cancel.clone();
        let h = Arc::clone(&health);
        let join = tokio::spawn(async move { serve_loop(listener, h, token).await });

        // Give the server a moment to start accepting.
        tokio::task::yield_now().await;

        // /healthz should be 200 when healthy.
        let (code, body) = http_get(addr, "/healthz").await?;
        assert_eq!(code, 200);
        let json: serde_json::Value = serde_json::from_str(&body)?;
        assert_eq!(json["status"], "healthy");

        // /readyz should be 200 when core is healthy.
        let (code, _) = http_get(addr, "/readyz").await?;
        assert_eq!(code, 200);

        // /health always returns 200.
        let (code, body) = http_get(addr, "/health").await?;
        assert_eq!(code, 200);
        let json: serde_json::Value = serde_json::from_str(&body)?;
        assert_eq!(json["core"], "healthy");

        cancel.cancel();
        join.await??;
        Ok(())
    }

    #[tokio::test]
    async fn readyz_returns_503_when_core_unhealthy() -> Result<()> {
        let health = HealthSurface::shared();
        // Don't set sweep/workers alive — core stays unhealthy.

        let cancel = CancellationToken::new();
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let token = cancel.clone();
        let h = Arc::clone(&health);
        let join = tokio::spawn(async move { serve_loop(listener, h, token).await });
        tokio::task::yield_now().await;

        let (code, body) = http_get(addr, "/readyz").await?;
        assert_eq!(code, 503);
        let json: serde_json::Value = serde_json::from_str(&body)?;
        assert_eq!(json["core"], "unhealthy");

        cancel.cancel();
        join.await??;
        Ok(())
    }

    #[tokio::test]
    async fn degraded_relay_keeps_readyz_200() -> Result<()> {
        let health = HealthSurface::shared();
        health.set_sweep_alive(true);
        health.set_workers_alive(true);
        health.set_core(CoreHealth::Healthy);
        health.set_latency_layer(LatencyLayerHealth::Degraded);

        let cancel = CancellationToken::new();
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let token = cancel.clone();
        let h = Arc::clone(&health);
        let join = tokio::spawn(async move { serve_loop(listener, h, token).await });
        tokio::task::yield_now().await;

        // readyz: still 200 because core is healthy.
        let (code, _) = http_get(addr, "/readyz").await?;
        assert_eq!(code, 200);

        // healthz: still 200 because Degraded is live.
        let (code, body) = http_get(addr, "/healthz").await?;
        assert_eq!(code, 200);
        let json: serde_json::Value = serde_json::from_str(&body)?;
        assert_eq!(json["status"], "degraded");
        assert_eq!(json["latency_layer"], "degraded");

        cancel.cancel();
        join.await??;
        Ok(())
    }

    #[tokio::test]
    async fn unknown_path_returns_404() -> Result<()> {
        let health = HealthSurface::shared();
        let cancel = CancellationToken::new();
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let token = cancel.clone();
        let h = Arc::clone(&health);
        let join = tokio::spawn(async move { serve_loop(listener, h, token).await });
        tokio::task::yield_now().await;

        let (code, _) = http_get(addr, "/unknown").await?;
        assert_eq!(code, 404);

        cancel.cancel();
        join.await??;
        Ok(())
    }

    #[test]
    fn parse_request_path_extracts_path() {
        assert_eq!(
            parse_request_path("GET /healthz HTTP/1.1\r\nHost: localhost\r\n"),
            "/healthz"
        );
        assert_eq!(parse_request_path("GET /readyz HTTP/1.1\r\n"), "/readyz");
        assert_eq!(
            parse_request_path("GET /healthz?verbose=true HTTP/1.1\r\n"),
            "/healthz"
        );
        assert_eq!(
            parse_request_path("GET /readyz?full=1 HTTP/1.1\r\n"),
            "/readyz"
        );
        assert_eq!(parse_request_path(""), "/");
        assert_eq!(parse_request_path("MALFORMED"), "/");
    }
}
