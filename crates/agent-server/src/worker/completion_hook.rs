//! The subagent completion hook (primitive 10): host work that runs when a
//! subagent invocation completes, recorded as a [`EffectKind::GitStep`]
//! effect the invocation owns.
//!
//! The hook runs in the invocation's `Running` phase: the child root is
//! terminal, the parent still waits on its children, and the invocation
//! holds a lease the host heartbeat renews. With no hook installed the
//! invocation settles exactly as before.
//!
//! The SDK does not use [`run_effect`](crate::journal::effect::run_effect)
//! here, because a later release lets the hook park the invocation with its
//! effect still `Started`, and a requeued invocation continues the record
//! of its earlier attempt. It commits the parts itself:
//!
//! 1. [`SubagentCompletionHook::plan`] names the git step, or none.
//! 2. With a step, the SDK reads the effect `{invocation}:git:{step}`. If
//!    there is no record, it commits the `Started` record under the
//!    invocation's lease before the hook does any work.
//! 3. [`SubagentCompletionHook::run`] runs, with the earlier attempt's
//!    `Started` record as `prior` if one exists, and with
//!    [`RunMode::Inspect`] on the invocation's last attempt when it is a
//!    retry (a first attempt always executes).
//! 4. One transition then finishes the effect, settles the invocation with
//!    the hook's result, appends the parent's completed `SubagentProgress`
//!    and applies the hook's extra writes. The parent resumes in it.
//!
//! A crash or an expired lease anywhere before step 4 commits leaves the
//! record `Started`; the requeued invocation (up to
//! [`AgentTask::SUBAGENT_INVOCATION_MAX_ATTEMPTS`](crate::journal::task::AgentTask::SUBAGENT_INVOCATION_MAX_ATTEMPTS))
//! continues it at step 3. A cancel, a purge or a fail-closed sweep makes
//! the invocation terminal and abandons the record in the same transaction.

use std::sync::Arc;

use agent_sdk_foundation::{ThreadId, ToolReplay, ToolResult};
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use time::OffsetDateTime;

use super::subagent::{SubagentResult, SubagentTaskBootstrap, SubagentTaskOutcome};
use crate::journal::effect::{
    EffectIntent, EffectKey, EffectKind, EffectRecord, EffectState, EffectSubject, EffectWrite,
    kind_wire, record_effect,
};
use crate::journal::store::AgentTaskStore;
use crate::journal::task::{AgentTaskId, TaskStatus};
use crate::journal::transition::{
    Committed, DurableStore, Guard, Rejected, Settle, Transition, Write,
};

/// How [`SubagentCompletionHook::run`] may act.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunMode {
    /// Do the work, continuing from `prior` if it is set.
    Execute,
    /// The invocation's last attempt, when it is a retry (attempt above 1):
    /// read state and report it, change nothing. A first attempt always
    /// executes, also under a budget of 1. A typical result is [`EffectState::Salvaged`] with the state
    /// found.
    Inspect,
}

impl RunMode {
    /// The mode as the `agent_server.completion_hook.mode` span attribute
    /// labels it.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Execute => "execute",
            Self::Inspect => "inspect",
        }
    }
}

/// What the hook knows about the completion.
pub struct CompletionContext {
    pub invocation: AgentTaskId,
    /// This acquisition of the invocation, from 1.
    pub attempt: u32,
    pub max_attempts: u32,
    pub parent_thread: ThreadId,
    pub child_thread: ThreadId,
    /// The tool result the SDK materialized from the terminal child. It
    /// can be a failure.
    pub child_result: ToolResult,
    /// The task store, read only: its `commit` is refused. A hook reads
    /// effect records with it (`effect`, `effects_for_task`) and returns its
    /// writes in [`CompletionOutput::extra_writes`].
    pub store: Arc<dyn DurableStore>,
}

/// The git step a completion runs: the subject of its effect record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionPlan {
    /// For example `merge_back`. The effect key is
    /// `{invocation}:git:{step}`.
    pub step: String,
    pub repo: String,
    pub unit: String,
}

/// A request to park the invocation on a human decision. Placeholder: the
/// SDK refuses it ([`CompletionHookError::Unsupported`]) until the durable
/// prompt row (primitive 6) can own a parked invocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptRequest {
    pub payload: serde_json::Value,
}

/// What [`SubagentCompletionHook::run`] returns.
#[derive(Clone, Debug)]
pub struct CompletionOutput {
    /// The tool result the parent gets, in place of the child's.
    pub result: ToolResult,
    /// The state the effect finishes in. Not `Started`. Ignored when
    /// [`SubagentCompletionHook::plan`] returned `None`.
    pub state: EffectState,
    /// More writes for the settle transition: [`Write::Document`] and
    /// [`Write::AppendEvents`] only.
    pub extra_writes: Vec<Write>,
    /// Must be `None` in this release.
    pub park: Option<PromptRequest>,
}

/// Host work for each completed subagent invocation.
///
/// The SDK calls it for every invocation that settles from its terminal
/// child, also when the child failed. It does not see an invocation that fails before the settle
/// (a materialization error, or the lease sweep failing it closed).
///
/// An `Err` from [`Self::run`] fails the invocation closed, which abandons
/// its effect. Return a failure the parent should see as
/// `Ok(CompletionOutput { state: Failed { .. }, .. })` instead.
#[async_trait]
pub trait SubagentCompletionHook: Send + Sync {
    /// Pure. `None` when this completion has no git step (for example a
    /// read-only or a failed child). [`Self::run`] is still called, with no
    /// effect record, so the host can rewrite the result and add writes.
    fn plan(&self, ctx: &CompletionContext) -> Option<CompletionPlan>;

    /// Do the step. `prior` is the `Started` record an earlier attempt
    /// left, if any; continue from the host's own state. Each step must be
    /// safe to run again, because a crash before the settle runs it again.
    ///
    /// # Errors
    /// Fails the invocation closed.
    async fn run(
        &self,
        ctx: &CompletionContext,
        prior: Option<&EffectRecord>,
        mode: RunMode,
    ) -> Result<CompletionOutput>;
}

/// An installed hook and the store whose effect records its context reads
/// (the task store the invocation settles through).
#[derive(Clone)]
pub struct InstalledCompletionHook {
    pub hook: Arc<dyn SubagentCompletionHook>,
    pub store: Arc<dyn DurableStore>,
}

/// Why the SDK refused a hook's output.
#[derive(Debug, thiserror::Error)]
pub enum CompletionHookError {
    /// The hook asked for something this release cannot do.
    #[error("unsupported: {0}")]
    Unsupported(&'static str),
    /// An extra write of a kind the settle cannot carry.
    #[error(
        "a completion hook may add only Document and AppendEvents writes; extra write {index} is {write}"
    )]
    RefusedExtraWrite { index: usize, write: &'static str },
    /// The hook finished its effect as `Started`.
    #[error("a completion hook must finish its effect in an end state, not Started")]
    StartedFinish,
    /// The effect's record already ended, or is not the invocation's, while
    /// the invocation still runs.
    #[error("effect {key} is {state} under {owner}; invocation {invocation} cannot continue it")]
    EffectNotContinuable {
        key: EffectKey,
        state: &'static str,
        owner: AgentTaskId,
        invocation: AgentTaskId,
    },
}

/// The context's store: reads pass through, `commit` is refused.
struct ReadOnlyStore(Arc<dyn DurableStore>);

#[async_trait]
impl DurableStore for ReadOnlyStore {
    async fn commit(
        &self,
        _transition: Transition,
        _now: OffsetDateTime,
    ) -> Result<Committed, Rejected> {
        Err(Rejected::Store(anyhow::anyhow!(
            "a completion hook's store is read only; return its writes in CompletionOutput::extra_writes"
        )))
    }

    async fn effect(&self, key: &EffectKey) -> Result<Option<EffectRecord>> {
        self.0.effect(key).await
    }

    async fn effects_for_task(&self, owner: &AgentTaskId) -> Result<Vec<EffectRecord>> {
        self.0.effects_for_task(owner).await
    }
}

const fn write_name(write: &Write) -> &'static str {
    match write {
        Write::SettleTask { .. } => "SettleTask",
        Write::RejectConfirmation { .. } => "RejectConfirmation",
        Write::PauseOnConfirmation { .. } => "PauseOnConfirmation",
        Write::ReparkAfterSteering { .. } => "ReparkAfterSteering",
        Write::AppendCompaction { .. } => "AppendCompaction",
        Write::AppendEvents { .. } => "AppendEvents",
        Write::SpawnChildren { .. } => "SpawnChildren",
        Write::Document(_) => "Document",
        Write::Effect(_) => "Effect",
        Write::EnqueueSteeringResume { .. } => "EnqueueSteeringResume",
    }
}

/// Check the hook's output before the SDK builds the settle from it.
fn check_output(output: &CompletionOutput) -> Result<(), CompletionHookError> {
    if output.park.is_some() {
        return Err(CompletionHookError::Unsupported(
            "park: an invocation cannot wait on a prompt before primitive 6",
        ));
    }
    if output.state.is_started() {
        return Err(CompletionHookError::StartedFinish);
    }
    for (index, write) in output.extra_writes.iter().enumerate() {
        if !matches!(write, Write::Document(_) | Write::AppendEvents { .. }) {
            return Err(CompletionHookError::RefusedExtraWrite {
                index,
                write: write_name(write),
            });
        }
    }
    Ok(())
}

/// The span `agent_server.subagent.completion_hook` and the hook's metrics.
struct HookSignals {
    #[cfg(feature = "otel")]
    span: opentelemetry::global::BoxedSpan,
    #[cfg(feature = "otel")]
    started: Option<std::time::Instant>,
}

impl HookSignals {
    #[cfg(feature = "otel")]
    fn begin(ctx: &CompletionContext, key: Option<&EffectKey>, mode: RunMode) -> Self {
        use crate::observability::attrs;
        use opentelemetry::KeyValue;
        use opentelemetry::trace::{Span, Tracer};

        let mut span = opentelemetry::global::tracer("agent-server")
            .start("agent_server.subagent.completion_hook");
        if let Some(key) = key {
            span.set_attribute(KeyValue::new(
                attrs::SPAN_EFFECT_ID,
                key.as_str().to_owned(),
            ));
        }
        span.set_attribute(KeyValue::new(
            attrs::SPAN_TASK_ID,
            ctx.invocation.as_str().to_owned(),
        ));
        span.set_attribute(KeyValue::new(
            attrs::SPAN_TASK_ATTEMPT,
            i64::from(ctx.attempt),
        ));
        span.set_attribute(KeyValue::new(
            attrs::SPAN_COMPLETION_HOOK_MODE,
            mode.label(),
        ));
        Self {
            span,
            started: None,
        }
    }

    #[cfg(not(feature = "otel"))]
    const fn begin(_ctx: &CompletionContext, _key: Option<&EffectKey>, _mode: RunMode) -> Self {
        Self {}
    }

    /// The effect's record committed (or an earlier attempt's was found).
    fn claimed(&mut self, transition: &'static str) {
        record_effect(EffectKind::GitStep, transition);
        #[cfg(feature = "otel")]
        {
            self.started = Some(std::time::Instant::now());
        }
    }

    fn refused(&self, reason: &'static str) {
        #[cfg(feature = "otel")]
        crate::observability::ServerMetrics::global().record_completion_hook_refused(reason);
        let _ = (self, reason);
    }

    /// End the span with `outcome`: the effect's end state, `no_effect`,
    /// `refused` or `error`.
    #[cfg(feature = "otel")]
    fn end(mut self, outcome: &'static str) {
        use opentelemetry::trace::Span;

        self.span.set_attribute(opentelemetry::KeyValue::new(
            crate::observability::attrs::SPAN_COMPLETION_HOOK_OUTCOME,
            outcome,
        ));
        self.span.end();
    }

    #[cfg(not(feature = "otel"))]
    const fn end(self, _outcome: &'static str) {}

    fn finished(&self, outcome: &'static str) {
        record_effect(EffectKind::GitStep, "finished");
        #[cfg(feature = "otel")]
        if let Some(started) = self.started {
            crate::observability::ServerMetrics::global().record_effect_duration(
                EffectKind::GitStep,
                outcome,
                started.elapsed().as_secs_f64(),
            );
        }
        let _ = (self, outcome);
    }
}

/// The settle of [`execute_subagent_task`](super::execute_subagent_task)
/// when a hook is installed (see the module documentation).
pub(super) async fn settle_with_hook(
    installed: &InstalledCompletionHook,
    bootstrap: &SubagentTaskBootstrap,
    task_store: &dyn AgentTaskStore,
    subagent_result: SubagentResult,
    child_result: ToolResult,
    now: OffsetDateTime,
) -> Result<SubagentTaskOutcome> {
    let ctx = CompletionContext {
        invocation: bootstrap.task_id.clone(),
        attempt: bootstrap.invocation_task.attempt,
        max_attempts: bootstrap.invocation_task.max_attempts,
        parent_thread: bootstrap.thread_id.clone(),
        child_thread: bootstrap.child_thread_id.clone(),
        child_result,
        store: Arc::new(ReadOnlyStore(Arc::clone(&installed.store))),
    };
    // A first attempt always executes: an invocation stored with a budget
    // of 1 (spawned before SUBAGENT_INVOCATION_MAX_ATTEMPTS, or by another
    // path) must still land its work. Only a retry that is the last attempt
    // inspects.
    let mode = if ctx.attempt > 1 && ctx.attempt >= ctx.max_attempts {
        RunMode::Inspect
    } else {
        RunMode::Execute
    };
    let plan = installed.hook.plan(&ctx);
    let key = plan
        .as_ref()
        .map(|plan| EffectKey::git(&ctx.invocation, &plan.step));
    let mut signals = HookSignals::begin(&ctx, key.as_ref(), mode);
    let run = HookRun {
        installed,
        bootstrap,
        task_store,
        ctx: &ctx,
        mode,
        now,
    };
    let settled = run
        .settle(subagent_result, plan.zip(key), &mut signals)
        .await;
    match settled {
        Ok((outcome, label)) => {
            signals.end(label);
            Ok(outcome)
        }
        Err(error) => {
            let label = if error.downcast_ref::<CompletionHookError>().is_some() {
                "refused"
            } else {
                "error"
            };
            log::warn!(
                task_id = ctx.invocation.as_str(),
                attempt = ctx.attempt,
                mode = mode.label(),
                effect_kind = kind_wire(EffectKind::GitStep);
                "subagent completion hook did not settle the invocation: {error:#}"
            );
            signals.end(label);
            Err(error)
        }
    }
}

/// One hook settle, split out of [`settle_with_hook`] so its span ends on
/// every return.
struct HookRun<'a> {
    installed: &'a InstalledCompletionHook,
    bootstrap: &'a SubagentTaskBootstrap,
    task_store: &'a dyn AgentTaskStore,
    ctx: &'a CompletionContext,
    mode: RunMode,
    now: OffsetDateTime,
}

impl HookRun<'_> {
    async fn settle(
        &self,
        subagent_result: SubagentResult,
        planned: Option<(CompletionPlan, EffectKey)>,
        signals: &mut HookSignals,
    ) -> Result<(SubagentTaskOutcome, &'static str)> {
        let Self {
            installed,
            bootstrap,
            task_store,
            ctx,
            mode,
            now,
        } = *self;
        let leased = Guard::TaskIs {
            task: ctx.invocation.clone(),
            status: TaskStatus::Running,
            lease: Some((bootstrap.worker_id.clone(), bootstrap.lease_id.clone())),
        };
        let prior = match &planned {
            Some((plan, key)) => self.claim(plan, key, &leased, signals).await?,
            None => None,
        };

        let output = installed
            .hook
            .run(ctx, prior.as_ref(), mode)
            .await
            .context("run the subagent completion hook")?;
        if let Err(refused) = check_output(&output) {
            signals.refused(match refused {
                CompletionHookError::RefusedExtraWrite { .. } => "extra_write",
                CompletionHookError::Unsupported(_) => "park",
                CompletionHookError::StartedFinish
                | CompletionHookError::EffectNotContinuable { .. } => "state",
            });
            return Err(refused.into());
        }

        let result = serde_json::to_value(&output.result)
            .context("serialize the completion hook's tool result")?;
        let mut end = Transition::settle(
            ctx.invocation.clone(),
            bootstrap.worker_id.clone(),
            bootstrap.lease_id.clone(),
            Settle::Complete { result },
            ctx.parent_thread.clone(),
            vec![super::subagent::completed_progress_event(
                bootstrap,
                &subagent_result.summary,
            )],
        );
        debug_assert!(matches!(end.guards.as_slice(), [guard] if *guard == leased));
        let label = match planned {
            Some((_, key)) => {
                let label = output.state.outcome_label();
                end.writes.insert(
                    0,
                    Write::Effect(EffectWrite::Finish {
                        key,
                        owner: ctx.invocation.clone(),
                        state: output.state,
                    }),
                );
                Some(label)
            }
            None => None,
        };
        end.writes.extend(output.extra_writes);
        crate::fail_point!("subagent.completion_hook.before_settle");
        let committed = task_store
            .commit(end, now)
            .await
            .map_err(|rejected| {
                crate::journal::transition::wrapper_error("complete_task_with_result", rejected)
            })
            .context("settle subagent invocation after its completion hook")?;
        if let Some(label) = label {
            signals.finished(label);
        }
        let (invocation_task, parent_task) = committed
            .settled
            .into_iter()
            .next()
            .context("a settle transition returns its settled task")?;
        Ok((
            SubagentTaskOutcome {
                invocation_task,
                parent_task,
                subagent_result,
                tool_result: output.result,
                committed_events: committed.events,
            },
            label.unwrap_or("no_effect"),
        ))
    }

    /// Read the effect's record; commit its `Started` record if there is none.
    /// Returns the record an earlier attempt left.
    async fn claim(
        &self,
        plan: &CompletionPlan,
        key: &EffectKey,
        leased: &Guard,
        signals: &mut HookSignals,
    ) -> Result<Option<EffectRecord>> {
        let Self {
            task_store,
            ctx,
            now,
            ..
        } = *self;
        match task_store
            .effect(key)
            .await
            .context("read the completion effect")?
        {
            None => {
                let start = Transition {
                    guards: vec![leased.clone(), Guard::EffectFree { key: key.clone() }],
                    writes: vec![Write::Effect(EffectWrite::Start(EffectIntent {
                        key: key.clone(),
                        replay: ToolReplay::Safe,
                        owner: ctx.invocation.clone(),
                        thread: ctx.parent_thread.clone(),
                        subject: EffectSubject::Git {
                            step: plan.step.clone(),
                            repo: plan.repo.clone(),
                            unit: plan.unit.clone(),
                        },
                        attempt: ctx.attempt,
                    }))],
                };
                if let Err(rejected) = task_store.commit(start, now).await {
                    record_effect(EffectKind::GitStep, "refused");
                    return Err(anyhow::Error::from(rejected)
                        .context("start the completion effect; the hook did not run"));
                }
                signals.claimed("started");
                Ok(None)
            }
            Some(record) if record.state.is_started() && record.owner == ctx.invocation => {
                signals.claimed("continued");
                log::info!(
                    effect_key = key.as_str(),
                    task_id = ctx.invocation.as_str(),
                    attempt = ctx.attempt,
                    started_attempt = record.attempt;
                    "a requeued subagent invocation continues its completion effect"
                );
                Ok(Some(record))
            }
            Some(record) => bail!(CompletionHookError::EffectNotContinuable {
                key: key.clone(),
                state: record.state.outcome_label(),
                owner: record.owner,
                invocation: ctx.invocation.clone(),
            }),
        }
    }
}
