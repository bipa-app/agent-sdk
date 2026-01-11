//! Custom hooks implementation example.
//!
//! This example shows how to implement custom hooks to control agent behavior.
//!
//! # Running
//!
//! ```bash
//! ANTHROPIC_API_KEY=your_key cargo run --example custom_hooks
//! ```

use agent_sdk::{
    AgentEvent, AgentHooks, InMemoryStore, ThreadId, Tool, ToolContext, ToolDecision, ToolRegistry,
    ToolResult, ToolTier, builder, providers::AnthropicProvider,
};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A simple tool that simulates sending an email.
struct SendEmailTool;

#[async_trait]
impl Tool<()> for SendEmailTool {
    fn name(&self) -> &str {
        "send_email"
    }

    fn description(&self) -> &str {
        "Send an email to a recipient. Use this to send messages to people."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Email address of the recipient"
                },
                "subject": {
                    "type": "string",
                    "description": "Subject line of the email"
                },
                "body": {
                    "type": "string",
                    "description": "Body content of the email"
                }
            },
            "required": ["to", "subject", "body"]
        })
    }

    fn tier(&self) -> ToolTier {
        // This is a sensitive operation that would normally require confirmation
        ToolTier::Confirm
    }

    async fn execute(&self, _ctx: &ToolContext<()>, input: Value) -> Result<ToolResult> {
        let to = input
            .get("to")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let subject = input
            .get("subject")
            .and_then(|v| v.as_str())
            .unwrap_or("(no subject)");

        // In a real implementation, this would actually send the email
        Ok(ToolResult::success(format!(
            "Email sent successfully to {to} with subject '{subject}'"
        )))
    }
}

/// Custom hooks that log all events and implement rate limiting.
struct CustomHooks {
    tool_call_count: AtomicUsize,
    max_tool_calls: usize,
}

impl CustomHooks {
    fn new(max_tool_calls: usize) -> Self {
        Self {
            tool_call_count: AtomicUsize::new(0),
            max_tool_calls,
        }
    }
}

#[async_trait]
impl AgentHooks for CustomHooks {
    async fn pre_tool_use(&self, tool_name: &str, input: &Value, tier: ToolTier) -> ToolDecision {
        let count = self.tool_call_count.fetch_add(1, Ordering::SeqCst);

        println!(
            "[Hooks] pre_tool_use: {tool_name} (call #{}, tier: {tier:?})",
            count + 1
        );

        // Rate limiting: block if too many tool calls
        if count >= self.max_tool_calls {
            println!(
                "[Hooks] BLOCKED: Rate limit exceeded ({} calls)",
                self.max_tool_calls
            );
            return ToolDecision::Block(format!(
                "Rate limit exceeded: maximum {} tool calls allowed",
                self.max_tool_calls
            ));
        }

        // For Confirm tier tools, we could prompt the user
        // For this example, we'll auto-approve with logging
        match tier {
            ToolTier::Observe => {
                println!("[Hooks] ALLOWED: Observe tier tool");
                ToolDecision::Allow
            }
            ToolTier::Confirm => {
                // In a real app, you might prompt the user here
                println!("[Hooks] AUTO-APPROVED: Confirm tier tool (input: {input})");
                ToolDecision::Allow
            }
            ToolTier::RequiresPin => {
                println!("[Hooks] BLOCKED: PIN required for this tool");
                ToolDecision::RequiresPin("This action requires PIN verification".to_string())
            }
        }
    }

    async fn post_tool_use(&self, tool_name: &str, result: &ToolResult) {
        println!(
            "[Hooks] post_tool_use: {tool_name} - success={}, duration={:?}ms",
            result.success, result.duration_ms
        );
    }

    async fn on_event(&self, event: &AgentEvent) {
        match event {
            AgentEvent::Start { turn, .. } => {
                println!("[Hooks] Event: Turn {turn} starting");
            }
            AgentEvent::TurnComplete { turn, usage } => {
                println!(
                    "[Hooks] Event: Turn {turn} complete (tokens: {} in, {} out)",
                    usage.input_tokens, usage.output_tokens
                );
            }
            _ => {}
        }
    }

    async fn on_error(&self, error: &anyhow::Error) -> bool {
        println!("[Hooks] Error occurred: {error}");
        // Return true to attempt recovery, false to abort
        false
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY environment variable must be set");

    // Create tools
    let mut tools = ToolRegistry::new();
    tools.register(SendEmailTool);

    // Create custom hooks with a rate limit of 3 tool calls
    let hooks = CustomHooks::new(3);

    println!("Starting agent with custom hooks (max 3 tool calls)\n");

    // Build the agent with custom hooks
    let agent = builder::<()>()
        .provider(AnthropicProvider::sonnet(api_key))
        .tools(tools)
        .hooks(hooks)
        .message_store(InMemoryStore::new())
        .state_store(InMemoryStore::new())
        .build_with_stores();

    let thread_id = ThreadId::new();
    let tool_ctx = ToolContext::new(());

    // Ask the agent to send an email
    let mut events = agent.run(
        thread_id,
        "Please send an email to test@example.com with subject 'Hello' and body 'This is a test message.'".to_string(),
        tool_ctx,
    );

    println!("\n--- Agent Output ---\n");

    while let Some(event) = events.recv().await {
        match event {
            AgentEvent::Text { text } => {
                println!("Agent: {text}");
            }
            AgentEvent::Done { total_turns, .. } => {
                println!("\n(Completed in {total_turns} turns)");
            }
            AgentEvent::Error { message, .. } => {
                eprintln!("Error: {message}");
            }
            _ => {}
        }
    }

    Ok(())
}
