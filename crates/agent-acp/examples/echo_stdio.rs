//! Minimal ACP agent over real stdin/stdout: echoes each prompt back as one
//! `agent_message_chunk`, then ends the turn.
//!
//! Drive it by piping a scripted session:
//!
//! ```sh
//! printf '%s\n%s\n%s\n' \
//!   '{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":2}}' \
//!   '{"jsonrpc":"2.0","id":1,"method":"session/new","params":{"cwd":"/tmp","mcpServers":[]}}' \
//!   '{"jsonrpc":"2.0","id":2,"method":"session/prompt","params":{"sessionId":"<paste>","prompt":[{"type":"text","text":"hi"}]}}' \
//!   | cargo run -p agent-acp --example echo_stdio
//! ```
//!
//! (For a non-interactive smoke run, the sessionId line can be scripted by
//! reading the `session/new` response first — see the ENG-9395 PR for the
//! recorded evidence transcript.)

use agent_acp::{AcpServer, PromptError, PromptHandler, PromptRequest, StopReason, UpdateSink};
use tokio_util::sync::CancellationToken;

struct Echo;

#[async_trait::async_trait]
impl PromptHandler for Echo {
    async fn prompt(
        &self,
        request: PromptRequest,
        updates: UpdateSink,
        cancel: CancellationToken,
    ) -> Result<StopReason, PromptError> {
        if cancel.is_cancelled() {
            return Ok(StopReason::Cancelled);
        }
        let text = format!("echo: {}", request.blocks.join(" | "));
        updates
            .agent_message_chunk(&text)
            .await
            .map_err(|e| PromptError::new(e.to_string()))?;
        Ok(StopReason::EndTurn)
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> std::io::Result<()> {
    env_logger::init();
    AcpServer::new(Echo)
        .with_agent_info("echo-stdio-example", env!("CARGO_PKG_VERSION"))
        .serve_stdio()
        .await
}
