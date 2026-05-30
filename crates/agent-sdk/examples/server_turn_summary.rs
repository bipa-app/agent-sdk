//! Server-facing `TurnSummary` contract example.
//!
//! This example demonstrates how a server-side orchestrator reads the
//! authoritative [`TurnSummary`] that `run_turn` attaches to every
//! terminal `TurnOutcome` variant. It pairs with Phase 1 of the
//! sdk/v2 rewrite and shows the fields later server
//! phases (`journal`, `workers`, `transport`, `storage`) depend on.
//!
//! The example uses [`ToolRuntime::External`] and `strict_durability:
//! true` — the server profile — so the caller owns tool-task dispatch
//! and the SDK yields a `PendingToolCalls` outcome with a durable
//! continuation and summary. For a `Done` outcome, swap the input or
//! use a text-only prompt.
//!
//! # Running
//!
//! ```bash
//! ANTHROPIC_API_KEY=your_key cargo run --example server_turn_summary
//! ```

use agent_sdk::advanced::TurnOutcome;
use agent_sdk::{
    AgentInput, CancellationToken, InMemoryEventStore, ThreadId, ToolContext, ToolRuntime,
    TurnOptions, builder, providers::AnthropicProvider,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY environment variable must be set");

    // Build the agent with an event store the server would own.
    let event_store = Arc::new(InMemoryEventStore::new());
    let agent = builder::<()>()
        .provider(AnthropicProvider::sonnet(api_key))
        .event_store(event_store.clone())
        .build();

    let thread_id = ThreadId::new();
    let tool_ctx = ToolContext::new(());

    // Server profile: external tool runtime + strict durability.
    let server_options = TurnOptions {
        tool_runtime: ToolRuntime::External,
        strict_durability: true,
    };

    // Run exactly one turn directly in this task — no spawn, no extra
    // indirection. This is the authoritative `run_turn` boundary.
    let outcome = Box::pin(agent.run_turn(
        thread_id.clone(),
        AgentInput::Text("What is the capital of France?".into()),
        tool_ctx,
        CancellationToken::new(),
        server_options,
    ))
    .await;

    // Every variant except `Error` carries a structured `TurnSummary`.
    // Server code reads from the summary, not the legacy per-variant
    // fields, so durable turn rows stay consistent with the event store.
    if let Some(summary) = outcome.summary() {
        println!("── TurnSummary ──");
        println!("thread_id        : {}", summary.thread_id);
        println!("turn             : {}", summary.turn);
        println!("total_turns      : {}", summary.total_turns);
        println!(
            "provider / model : {} / {}",
            summary.provenance.provider, summary.provenance.model
        );
        println!(
            "response_id      : {}",
            summary.response_id.as_deref().unwrap_or("<none>")
        );
        println!("stop_reason      : {:?}", summary.stop_reason);
        println!("tool_call_count  : {}", summary.tool_call_count);
        println!(
            "turn_usage       : {} in / {} out",
            summary.turn_usage.input_tokens, summary.turn_usage.output_tokens
        );
        println!(
            "total_usage      : {} in / {} out",
            summary.total_usage.input_tokens, summary.total_usage.output_tokens
        );
        println!("duration_ms      : {}", summary.duration_ms);
        println!("tool_runtime     : {:?}", summary.tool_runtime);
        println!("strict_durability: {}", summary.strict_durability);

        // Summaries are serde-stable so a server can persist them
        // directly as part of a durable turn row.
        let json = serde_json::to_string_pretty(summary)?;
        println!("\n── Serialised (durable server row) ──\n{json}");
    }

    match outcome {
        TurnOutcome::Done { .. } => println!("\nAgent finished."),
        TurnOutcome::NeedsMoreTurns { .. } => {
            println!("\nAgent needs another turn (tool results already appended).");
        }
        TurnOutcome::PendingToolCalls {
            tool_calls,
            continuation,
            ..
        } => {
            println!(
                "\n{} tool call(s) need external execution:",
                tool_calls.len()
            );
            for call in &tool_calls {
                println!("  • {} ({}) — {}", call.display_name, call.name, call.id);
            }
            // The server would persist `continuation` and dispatch tool
            // tasks, then resume with `AgentInput::SubmitToolResults`.
            let _ = continuation;
        }
        TurnOutcome::AwaitingConfirmation { description, .. } => {
            println!("\nTool needs confirmation: {description}");
        }
        TurnOutcome::Refusal { .. } => println!("\nModel refused the request."),
        TurnOutcome::Cancelled { .. } => println!("\nTurn was cancelled."),
        TurnOutcome::Error(err) => eprintln!("\nError: {err}"),
    }

    Ok(())
}
