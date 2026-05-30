//! Runtime coverage for the Phase 12 · A ergonomics surface:
//! `AgentLoop::ask` / `AgentLoop::send`, `run()` returning an
//! `impl Future`, and the builder defaulting to an in-memory event store.

#[path = "support/stub_provider.rs"]
mod stub_provider;

use agent_sdk::{AgentInput, CancellationToken, ThreadId, ToolContext, builder};
use anyhow::Result;
use stub_provider::StubProvider;

#[tokio::test]
async fn ask_returns_assembled_text_without_ceremony() -> Result<()> {
    // No Arc, no event_store wiring, no oneshot — the builder defaults to
    // an in-memory event store and `ask` reassembles the reply text.
    let agent = builder::<()>()
        .provider(StubProvider::new(vec![StubProvider::text_response(
            "Paris is the capital of France.",
        )]))
        .build();

    let answer = agent
        .ask(ThreadId::new(), "What is the capital of France?")
        .await?;

    assert_eq!(answer, "Paris is the capital of France.");
    Ok(())
}

#[tokio::test]
async fn send_accepts_agent_input_and_returns_text() -> Result<()> {
    let agent = builder::<()>()
        .provider(StubProvider::new(vec![StubProvider::text_response("pong")]))
        .build();

    let answer = agent
        .send(ThreadId::new(), AgentInput::Text("ping".into()))
        .await?;

    assert_eq!(answer, "pong");
    Ok(())
}

#[tokio::test]
async fn ask_returns_only_this_calls_text_per_thread() -> Result<()> {
    // Each fresh-thread `ask` returns only its own reply, never an empty or
    // concatenated-across-threads string.
    let agent = builder::<()>()
        .provider(StubProvider::new(vec![
            StubProvider::text_response("first"),
            StubProvider::text_response("second"),
        ]))
        .build();

    let first = agent.ask(ThreadId::new(), "one").await?;
    let second = agent.ask(ThreadId::new(), "two").await?;

    assert_eq!(first, "first");
    assert_eq!(second, "second");
    Ok(())
}

#[tokio::test]
async fn run_is_an_awaitable_future() -> Result<()> {
    // `run` returns `impl Future<Output = Result<AgentRunState>>` — a single
    // `.await?` drives it, no `oneshot::Receiver` double-await.
    let agent = builder::<()>()
        .provider(StubProvider::new(vec![StubProvider::text_response("ok")]))
        .build();

    let state = agent
        .run(
            ThreadId::new(),
            AgentInput::Text("hi".into()),
            ToolContext::new(()),
            CancellationToken::new(),
        )
        .await?;

    assert!(
        matches!(state, agent_sdk::AgentRunState::Done { .. }),
        "expected Done; got {state:?}",
    );
    Ok(())
}
