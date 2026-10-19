//! The store-side phases of a tool call: the prior-effect lookup, the
//! effect claim and the settlement commit.
//!
//! With `otel`, each phase is an `agent_server.tool.phase` span and one
//! sample on `agent_server.tool.phase.duration{phase, outcome}`. The span
//! nests under the caller's current span, or else under the tool child's
//! stored `traceparent`, so a stalled call can be attributed to one phase
//! from its trace. The phase values are bip's `bip.tool.phase` values
//! (`lookup`, `claim`, `settlement`); the executor's `execution` phase
//! stays in the host. Every phase span carries `agent_server.task.id` (the
//! tool child) and `gen_ai.tool.call.id`, the keys that join one call's
//! phases with bip's `execution` phase.
//!
//! [`in_call_context`] runs the effect (`run_effect`) in that same parent
//! context, so its `effect.run` span is a sibling of the phases rather than
//! a trace of its own.
//!
//! The stored-`traceparent` fallback decodes through the **global**
//! text-map propagator ([`context_from_traceparent`]). With none installed
//! the decode yields no valid span and the phases (and `effect.run`) start
//! unparented. `agent_sdk_otel::install_global_provider` installs
//! `TraceContextPropagator`; a host that sets its tracer provider by hand
//! must also call `opentelemetry::global::set_text_map_propagator`.
//!
//! This only observes the work: it changes no result, retry or deadline.
//!
//! [`context_from_traceparent`]: agent_sdk::observability::loop_instrument::context_from_traceparent

use std::future::Future;

use async_trait::async_trait;
use time::OffsetDateTime;

use crate::journal::effect::{EffectKey, EffectRecord, EffectWrite};
use crate::journal::task::AgentTaskId;
use crate::journal::transition::{Committed, DurableStore, Rejected, Transition, Write};
use crate::worker::tool_task::ToolTaskBootstrap;

/// A measured store boundary of one tool call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ToolPhase {
    /// The read of the call's prior effect record (`Unsafe` calls only).
    Lookup,
    /// The commit that starts the call's effect record (`Unsafe` calls only).
    Claim,
    /// The commit that settles the tool child and closes its call.
    Settlement,
}

#[cfg(feature = "otel")]
impl ToolPhase {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Lookup => "lookup",
            Self::Claim => "claim",
            Self::Settlement => "settlement",
        }
    }
}

/// Run `work` as `phase` of `bootstrap`'s call. `Ok` is the outcome
/// `success`, `Err` is `error`, and a future dropped first records
/// `interrupted`. The outcome describes the boundary: a committed settle of
/// a failed tool is a successful settlement.
pub(super) async fn observe<T, E>(
    phase: ToolPhase,
    bootstrap: &ToolTaskBootstrap,
    work: impl Future<Output = Result<T, E>>,
) -> Result<T, E> {
    #[cfg(feature = "otel")]
    {
        use opentelemetry::context::FutureExt as _;

        let mut observation = otel::Observation::start(phase, bootstrap);
        let result = work.with_context(observation.context()).await;
        observation.finish(result.is_ok());
        result
    }
    #[cfg(not(feature = "otel"))]
    {
        let _ = (phase, bootstrap);
        work.await
    }
}

/// Run `work` in the context `bootstrap`'s phases nest in (the caller's
/// current span, else the tool child's stored `traceparent`), so spans it
/// starts from the current context, such as `run_effect`'s `effect.run`,
/// share the phases' parent. Without `otel` this is `work.await`.
pub(super) async fn in_call_context<F: Future>(
    bootstrap: &ToolTaskBootstrap,
    work: F,
) -> F::Output {
    #[cfg(feature = "otel")]
    {
        use opentelemetry::context::FutureExt as _;

        work.with_context(otel::parent_context(bootstrap)).await
    }
    #[cfg(not(feature = "otel"))]
    {
        let _ = bootstrap;
        work.await
    }
}

/// The store a tool call's effect runs against: the commit that starts its
/// record is its claim, any other commit its settlement. Reads pass through.
pub(super) struct PhaseStore<'a> {
    pub(super) inner: &'a dyn DurableStore,
    pub(super) bootstrap: &'a ToolTaskBootstrap,
}

#[async_trait]
impl DurableStore for PhaseStore<'_> {
    async fn commit(
        &self,
        transition: Transition,
        now: OffsetDateTime,
    ) -> Result<Committed, Rejected> {
        let phase = if starts_effect(&transition) {
            ToolPhase::Claim
        } else {
            ToolPhase::Settlement
        };
        observe(phase, self.bootstrap, self.inner.commit(transition, now)).await
    }

    async fn effect(&self, key: &EffectKey) -> anyhow::Result<Option<EffectRecord>> {
        self.inner.effect(key).await
    }

    async fn effects_for_task(&self, owner: &AgentTaskId) -> anyhow::Result<Vec<EffectRecord>> {
        self.inner.effects_for_task(owner).await
    }
}

fn starts_effect(transition: &Transition) -> bool {
    transition
        .writes
        .iter()
        .any(|write| matches!(write, Write::Effect(EffectWrite::Start(_))))
}

#[cfg(feature = "otel")]
mod otel {
    use std::time::Instant;

    use agent_sdk::observability::attrs as sdk_attrs;
    use agent_sdk::observability::loop_instrument::context_from_traceparent;
    use opentelemetry::trace::{TraceContextExt as _, Tracer as _};
    use opentelemetry::{Context, KeyValue, global};

    use super::ToolPhase;
    use crate::observability::{ServerMetrics, attrs};
    use crate::worker::tool_task::ToolTaskBootstrap;

    /// One phase in flight: its span (inside `context`) and its clock. Drop
    /// ends the span and records the duration.
    pub(super) struct Observation {
        phase: ToolPhase,
        context: Context,
        started: Instant,
        outcome: &'static str,
    }

    impl Observation {
        pub(super) fn start(phase: ToolPhase, bootstrap: &ToolTaskBootstrap) -> Self {
            let parent = parent_context(bootstrap);
            let tracer = global::tracer("agent-server");
            let span = tracer
                .span_builder("agent_server.tool.phase")
                .with_attributes([
                    KeyValue::new(attrs::SPAN_TOOL_PHASE, phase.as_str()),
                    KeyValue::new(attrs::SPAN_TASK_ID, bootstrap.task_id.as_str().to_owned()),
                    KeyValue::new(
                        sdk_attrs::GEN_AI_TOOL_CALL_ID,
                        bootstrap.tool_call.id.clone(),
                    ),
                    KeyValue::new(
                        sdk_attrs::GEN_AI_TOOL_NAME,
                        bootstrap.tool_call.name.clone(),
                    ),
                ])
                .start_with_context(&tracer, &parent);
            Self {
                phase,
                context: parent.with_span(span),
                started: Instant::now(),
                outcome: attrs::OUTCOME_INTERRUPTED,
            }
        }

        /// The context the phase's work runs in, so its own spans nest here.
        pub(super) fn context(&self) -> Context {
            self.context.clone()
        }

        pub(super) const fn finish(&mut self, ok: bool) {
            self.outcome = if ok {
                attrs::OUTCOME_SUCCESS
            } else {
                attrs::OUTCOME_ERROR
            };
        }
    }

    impl Drop for Observation {
        fn drop(&mut self) {
            let span = self.context.span();
            span.set_attribute(KeyValue::new(attrs::OUTCOME, self.outcome));
            span.end();
            ServerMetrics::global().record_tool_phase(
                self.phase.as_str(),
                self.outcome,
                self.started.elapsed().as_secs_f64(),
            );
        }
    }

    /// The caller's current span when it has one, else the tool child's
    /// stored `traceparent` (the host runs a tool task under no span).
    pub(super) fn parent_context(bootstrap: &ToolTaskBootstrap) -> Context {
        let current = Context::current();
        if current.span().span_context().is_valid() {
            return current;
        }
        bootstrap
            .child_task
            .otel_traceparent
            .as_deref()
            .and_then(context_from_traceparent)
            .unwrap_or(current)
    }
}
