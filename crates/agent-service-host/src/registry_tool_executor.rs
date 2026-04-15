//! Concrete [`ToolCallExecutor`] that dispatches tool calls through a
//! [`ToolRegistry<Ctx>`].
//!
//! This is the bridge between the durable v2 tool runtime
//! ([`agent_server::worker::tool_task::execute_tool_task`]) and the SDK
//! tool traits defined in [`agent_sdk_tools`].  Each invocation:
//!
//! 1. Looks up the tool in the registry by the `PendingToolCallInfo`'s
//!    name.
//! 2. Builds a [`ToolContext<Ctx>`] via the configured
//!    [`ExecutionContextFactory`] using a [`ToolContextSeed`] derived
//!    from the parent task's continuation and a [`HostDependencies`]
//!    whose event sink is a [`CollectorEventStore`] wrapping the
//!    supplied [`ToolEventCollector`].
//! 3. Calls [`Tool::execute`] with the pending call's
//!    `effective_input` and returns the resulting [`ToolResult`].
//!
//! Progress events emitted via [`ToolContext::emit_event`] are
//! forwarded into the worker's collector and committed by
//! [`execute_tool_task`] after the CAS-guarded state transition —
//! never written straight to the durable event store.
//!
//! [`ExecutionContextFactory`]: agent_sdk_tools::seed::ExecutionContextFactory
//! [`HostDependencies`]: agent_sdk_tools::seed::HostDependencies
//! [`ToolContext<Ctx>`]: agent_sdk_tools::ToolContext
//! [`ToolContext::emit_event`]: agent_sdk_tools::ToolContext::emit_event
//! [`ToolContextSeed`]: agent_sdk_tools::seed::ToolContextSeed
//! [`Tool::execute`]: agent_sdk_tools::Tool::execute
//! [`execute_tool_task`]: agent_server::worker::tool_task::execute_tool_task

use std::sync::Arc;

use agent_sdk_core::ToolResult;
use agent_sdk_tools::seed::{
    DefaultContextFactory, ExecutionContextFactory, HostDependencies, ToolContextSeed,
};
use agent_sdk_tools::tools::ToolRegistry;
use agent_server::worker::{ToolEventCollector, ToolTaskBootstrap};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::collector_event_store::CollectorEventStore;
use crate::runtime::ToolCallExecutor;

/// [`ToolCallExecutor`] implementation that dispatches via a shared
/// [`ToolRegistry<Ctx>`].
///
/// Hosts build one of these per application context and pass it to
/// [`ExecutionRuntime::new`](crate::runtime::ExecutionRuntime::new) in
/// place of [`NoopToolExecutor`](crate::runtime::NoopToolExecutor).
pub struct RegistryToolExecutor<Ctx>
where
    Ctx: Clone + Send + Sync + 'static,
{
    registry: Arc<ToolRegistry<Ctx>>,
    factory: Arc<dyn ExecutionContextFactory<Ctx>>,
    app: Ctx,
}

impl<Ctx> RegistryToolExecutor<Ctx>
where
    Ctx: Clone + Send + Sync + 'static,
{
    /// Build an executor with an explicit context factory.
    ///
    /// The factory is invoked per tool call to materialise the
    /// [`ToolContext<Ctx>`] passed to [`Tool::execute`].  Supply a
    /// custom factory when the context needs ambient state beyond
    /// `app` (for example, a database connection cloned from a pool
    /// per call); otherwise prefer [`Self::with_default_factory`].
    ///
    /// [`ToolContext<Ctx>`]: agent_sdk_tools::ToolContext
    /// [`Tool::execute`]: agent_sdk_tools::Tool::execute
    pub fn new(
        registry: Arc<ToolRegistry<Ctx>>,
        factory: Arc<dyn ExecutionContextFactory<Ctx>>,
        app: Ctx,
    ) -> Self {
        Self {
            registry,
            factory,
            app,
        }
    }

    /// Build an executor using the SDK-default [`DefaultContextFactory`].
    ///
    /// Equivalent to calling [`Self::new`] with
    /// `Arc::new(DefaultContextFactory)`.
    pub fn with_default_factory(registry: Arc<ToolRegistry<Ctx>>, app: Ctx) -> Self {
        Self::new(registry, Arc::new(DefaultContextFactory), app)
    }
}

impl<Ctx> RegistryToolExecutor<Ctx>
where
    Ctx: Clone + Send + Sync + 'static,
{
    /// Core dispatch — tool lookup, context reconstruction, and
    /// execution — extracted for testing without a full
    /// [`ToolTaskBootstrap`] fixture.
    ///
    /// `execute_tool_call` is a thin wrapper that extracts these
    /// inputs from the bootstrap.
    async fn dispatch(
        &self,
        tool_name: &str,
        input: serde_json::Value,
        seed: ToolContextSeed,
        collector: ToolEventCollector,
        cancel: CancellationToken,
    ) -> Result<ToolResult> {
        let tool = self
            .registry
            .get(tool_name)
            .ok_or_else(|| anyhow!("tool '{tool_name}' is not registered on this host runtime"))?;

        let deps = HostDependencies {
            event_store: Arc::new(CollectorEventStore::new(collector)),
            cancel_token: cancel,
            subagent_semaphore: None,
        };

        let ctx = self.factory.build(&seed, self.app.clone(), deps);

        tool.execute(&ctx, input).await
    }
}

#[async_trait]
impl<Ctx> ToolCallExecutor for RegistryToolExecutor<Ctx>
where
    Ctx: Clone + Send + Sync + 'static,
{
    async fn execute_tool_call(
        &self,
        bootstrap: &ToolTaskBootstrap,
        collector: ToolEventCollector,
        cancel: CancellationToken,
    ) -> Result<ToolResult> {
        let seed = tool_context_seed(bootstrap)?;

        self.dispatch(
            &bootstrap.tool_call.name,
            bootstrap.tool_call.effective_input.clone(),
            seed,
            collector,
            cancel,
        )
        .await
    }
}

/// Reconstruct the durable [`ToolContextSeed`] for a tool-runtime child.
fn tool_context_seed(bootstrap: &ToolTaskBootstrap) -> Result<ToolContextSeed> {
    let continuation = bootstrap.parent_task.state.continuation().ok_or_else(|| {
        anyhow!(
            "parent task {} is not paused on a continuation carrying tool context",
            bootstrap.parent_task.id,
        )
    })?;

    Ok(ToolContextSeed {
        thread_id: bootstrap.thread_id.clone(),
        turn: continuation.payload.turn,
        // The current tool-runtime bootstrap does not carry the durable
        // event-sequence offset. `CollectorEventStore` drops the wrapped
        // envelope before forwarding the inner event to the worker-owned
        // collector, so the placeholder authority created from this offset
        // is not persisted.
        sequence_offset: 0,
        metadata: continuation.payload.state.metadata.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_sdk_core::ThreadId;
    use agent_sdk_core::events::AgentEvent;
    use agent_sdk_core::types::ToolTier;
    use agent_sdk_tools::tools::{DynamicToolName, Tool, ToolContext};
    use serde_json::{Value, json};

    /// Minimal in-test tool that echoes its `text` input back as the
    /// output and optionally emits a progress event through the
    /// context.  Lets us verify both the dispatch path and the event
    /// forwarding path without pulling in `agent-sdk`.
    struct EchoTool {
        emit: bool,
    }

    impl<Ctx: Send + Sync + 'static> Tool<Ctx> for EchoTool {
        type Name = DynamicToolName;

        fn name(&self) -> DynamicToolName {
            DynamicToolName::new("echo")
        }
        fn display_name(&self) -> &'static str {
            "Echo"
        }
        fn description(&self) -> &'static str {
            "Echo input back"
        }
        fn input_schema(&self) -> Value {
            json!({"type": "object"})
        }
        fn tier(&self) -> ToolTier {
            ToolTier::Observe
        }

        async fn execute(&self, ctx: &ToolContext<Ctx>, input: Value) -> Result<ToolResult> {
            if self.emit {
                ctx.emit_event(AgentEvent::text("echo", "progress")).await?;
            }
            let text = input
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            Ok(ToolResult::success(text))
        }
    }

    fn build_executor(emit: bool) -> RegistryToolExecutor<()> {
        let mut registry = ToolRegistry::<()>::new();
        registry.register(EchoTool { emit });
        RegistryToolExecutor::with_default_factory(Arc::new(registry), ())
    }

    #[tokio::test]
    async fn dispatch_runs_registered_tool() -> Result<()> {
        let executor = build_executor(false);
        let collector = ToolEventCollector::new();

        let result = executor
            .dispatch(
                "echo",
                json!({"text": "hello"}),
                ToolContextSeed {
                    thread_id: ThreadId::from_string("t-dispatch"),
                    turn: 3,
                    sequence_offset: 0,
                    metadata: std::collections::HashMap::new(),
                },
                collector.clone(),
                CancellationToken::new(),
            )
            .await?;

        assert!(result.success, "echo tool should succeed");
        assert_eq!(result.output, "hello");
        assert!(collector.drain().is_empty(), "no events emitted");
        Ok(())
    }

    #[tokio::test]
    async fn dispatch_forwards_progress_events_to_collector() -> Result<()> {
        let executor = build_executor(true);
        let collector = ToolEventCollector::new();

        executor
            .dispatch(
                "echo",
                json!({"text": "x"}),
                ToolContextSeed {
                    thread_id: ThreadId::from_string("t-progress"),
                    turn: 1,
                    sequence_offset: 0,
                    metadata: std::collections::HashMap::new(),
                },
                collector.clone(),
                CancellationToken::new(),
            )
            .await?;

        let events = collector.drain();
        assert_eq!(events.len(), 1, "expected one forwarded progress event");
        match &events[0] {
            AgentEvent::Text { text, .. } => assert_eq!(text, "progress"),
            other => panic!("unexpected event: {other:?}"),
        }
        Ok(())
    }

    #[tokio::test]
    async fn dispatch_rejects_unknown_tool() {
        let executor = build_executor(false);
        let err = executor
            .dispatch(
                "not_registered",
                json!({}),
                ToolContextSeed {
                    thread_id: ThreadId::from_string("t-unknown"),
                    turn: 1,
                    sequence_offset: 0,
                    metadata: std::collections::HashMap::new(),
                },
                ToolEventCollector::new(),
                CancellationToken::new(),
            )
            .await
            .expect_err("unknown tool should fail dispatch");
        assert!(err.to_string().contains("not_registered"));
    }
}
