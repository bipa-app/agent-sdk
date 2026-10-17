//! The signals of the subagent leg of the stall budget. A child root failed
//! for silence is counted by leg and recorded on its own span, as a stalled
//! tool call is (`tool_stall.rs`). Both legs run under no span of their own:
//! the running leg in the child root's heartbeat, the parked leg in the sweep.

use agent_server::journal::task::{AgentTask, AgentTaskId};
use agent_server::observability::{ServerMetrics, SubagentStallSignal};

use crate::stores::StoreRegistry;

/// Where a stalled child root's signal attaches.
///
/// A child root stores no `otel_traceparent` of its own today, so the span
/// falls back to its invocation's, which the invocation inherits from the
/// parent turn's task; the stall then shows in the trace of the parent that
/// waits on it.
#[derive(Debug)]
pub(super) struct SubagentStallTrace {
    invocation_id: Option<AgentTaskId>,
    traceparent: Option<String>,
}

impl SubagentStallTrace {
    pub(super) fn of(child_root: &AgentTask, invocation: &AgentTask) -> Self {
        Self {
            invocation_id: Some(invocation.id.clone()),
            traceparent: child_root
                .otel_traceparent
                .clone()
                .or_else(|| invocation.otel_traceparent.clone()),
        }
    }

    /// The running leg holds neither row, so it reads them, before the fail
    /// wakes the invocation. A read that fails leaves its part empty: the
    /// stall is still counted.
    pub(super) async fn read(stores: &StoreRegistry, child_root: &AgentTaskId) -> Self {
        let invocation = stores
            .task_store
            .find_subagent_invocation_for_child_root(child_root)
            .await
            .ok()
            .flatten();
        let root = stores.task_store.get(child_root).await.ok().flatten();
        Self {
            invocation_id: invocation.as_ref().map(|invocation| invocation.id.clone()),
            traceparent: root
                .and_then(|root| root.otel_traceparent)
                .or_else(|| invocation.and_then(|invocation| invocation.otel_traceparent)),
        }
    }

    pub(super) fn record(&self, leg: &'static str, child_root: &AgentTaskId, budget_ms: u64) {
        ServerMetrics::global().record_subagent_stall(
            leg,
            &SubagentStallSignal {
                task_id: child_root.as_str(),
                invocation_id: self.invocation_id.as_ref().map(AgentTaskId::as_str),
                traceparent: self.traceparent.as_deref(),
                budget_ms,
            },
        );
    }
}
