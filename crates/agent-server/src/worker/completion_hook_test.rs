//! The subagent completion hook on the in-memory store: its effect record,
//! the retry of a requeued invocation, inspect mode and the refusals.
use std::sync::{Arc, Mutex};

use agent_sdk_foundation::{ToolResult, events::AgentEvent};
use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use futures::FutureExt;
use serde_json::json;

use super::completion_hook::{
    CompletionContext, CompletionHookError, CompletionOutput, CompletionPlan,
    InstalledCompletionHook, PromptRequest, RunMode, SubagentCompletionHook,
};
use super::subagent::{
    SpawnedSubagentInvocation, SubagentInvocationDeps, SubagentResultDeps, SubagentTaskOutcome,
    execute_subagent_task, resolve_subagent_bootstrap, spawn_subagent_invocation,
};
use super::subagent_execution_test::{TestStores, child_spawn, create_running_parent_root, t_plus};
use crate::journal::document::{DocExpect, DocKind, DocScope, DocumentKey, DocumentWrite};
use crate::journal::effect::{EffectKey, EffectKind, EffectRecord, EffectState, EffectSubject};
use crate::journal::event_repository::EventRepository;
use crate::journal::recovery::RecoveryAction;
use crate::journal::store::AgentTaskStore;
use crate::journal::task::{AgentTask, LeaseId, TaskStatus, WorkerId};
use crate::journal::transition::{DurableStore, Settle, Transition, Write};
use agent_sdk_foundation::ThreadId;

/// What one call of [`FakeHook::run`] saw.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Seen {
    attempt: u32,
    mode: RunMode,
    prior: Option<EffectState>,
}

/// What [`FakeHook::run`] does on an attempt.
#[derive(Clone)]
enum Act {
    /// Panic: a crash after the start record committed.
    Crash,
    Return(Box<CompletionOutput>),
    /// Try to commit through the context's store.
    CommitThroughContext,
}

struct FakeHook {
    plan: Option<CompletionPlan>,
    acts: Mutex<Vec<Act>>,
    seen: Mutex<Vec<Seen>>,
}

impl FakeHook {
    fn new(plan: Option<CompletionPlan>, acts: Vec<Act>) -> Arc<Self> {
        Arc::new(Self {
            plan,
            acts: Mutex::new(acts),
            seen: Mutex::new(Vec::new()),
        })
    }

    fn seen(&self) -> Result<Vec<Seen>> {
        Ok(self.seen.lock().ok().context("seen lock")?.clone())
    }
}

#[async_trait]
impl SubagentCompletionHook for FakeHook {
    fn plan(&self, _ctx: &CompletionContext) -> Option<CompletionPlan> {
        self.plan.clone()
    }

    async fn run(
        &self,
        ctx: &CompletionContext,
        prior: Option<&EffectRecord>,
        mode: RunMode,
    ) -> Result<CompletionOutput> {
        self.seen.lock().ok().context("seen lock")?.push(Seen {
            attempt: ctx.attempt,
            mode,
            prior: prior.map(|record| record.state.clone()),
        });
        let act = {
            let mut acts = self.acts.lock().ok().context("acts lock")?;
            ensure!(!acts.is_empty(), "the hook ran more often than scripted");
            acts.remove(0)
        };
        match act {
            Act::Crash => panic!("simulated crash while the completion hook runs"),
            Act::Return(output) => Ok(*output),
            Act::CommitThroughContext => {
                let refused = ctx.store.commit(Transition::default(), t_plus(20)).await;
                ensure!(refused.is_err(), "the context's store must refuse a commit");
                Ok(output(EffectState::Completed, Vec::new()))
            }
        }
    }
}

fn merge_back() -> CompletionPlan {
    CompletionPlan {
        step: "merge_back".into(),
        repo: "repo_1".into(),
        unit: "unit_1".into(),
    }
}

fn output(state: EffectState, extra_writes: Vec<Write>) -> CompletionOutput {
    CompletionOutput {
        result: ToolResult::success("merged unit_1 at abc123"),
        state,
        extra_writes,
        park: None,
    }
}

/// A parent waiting on one invocation whose child root was cancelled, so
/// the invocation is runnable.
async fn runnable_invocation(stores: &TestStores, name: &str) -> Result<SpawnedSubagentInvocation> {
    let parent_thread = ThreadId::from_string(name);
    let (parent, worker, lease) = create_running_parent_root(&stores.tasks, &parent_thread).await?;
    let spawned = spawn_subagent_invocation(
        &parent.id,
        &worker,
        &lease,
        child_spawn(&parent_thread),
        Vec::new(),
        &SubagentInvocationDeps {
            task_store: &stores.tasks,
            thread_store: &stores.threads,
            event_repo: &stores.events,
        },
        t_plus(2),
    )
    .await?;
    stores
        .tasks
        .cancel_tree(&spawned.child_root_task.id, t_plus(3))
        .await?;
    Ok(spawned)
}

fn deps<'a>(
    stores: &'a TestStores,
    hook: &Arc<FakeHook>,
) -> (SubagentResultDeps<'a>, InstalledCompletionHook) {
    let installed = InstalledCompletionHook {
        hook: Arc::clone(hook) as Arc<dyn SubagentCompletionHook>,
        store: Arc::new(stores.tasks.clone()) as Arc<dyn DurableStore>,
    };
    let mut deps = stores.subagent_deps();
    deps.completion_hook = Some(installed.clone());
    (deps, installed)
}

/// Acquire the invocation at `secs` with a lease to `secs + 5`, and run it
/// with `hook`.
async fn run_attempt(
    stores: &TestStores,
    invocation: &AgentTask,
    hook: &Arc<FakeHook>,
    secs: i64,
) -> Result<SubagentTaskOutcome> {
    let running = stores
        .tasks
        .try_acquire_task(
            &invocation.id,
            WorkerId::new(),
            LeaseId::new(),
            t_plus(secs + 5),
            t_plus(secs),
        )
        .await?
        .context("the invocation must acquire")?;
    let bootstrap = resolve_subagent_bootstrap(running, &stores.tasks).await?;
    let (deps, _installed) = deps(stores, hook);
    execute_subagent_task(bootstrap, &deps, t_plus(secs + 1)).await
}

async fn effect(stores: &TestStores, invocation: &AgentTask) -> Result<Option<EffectRecord>> {
    stores
        .tasks
        .effect(&EffectKey::git(&invocation.id, "merge_back"))
        .await
}

async fn completed_progress(stores: &TestStores, thread: &ThreadId) -> Result<usize> {
    Ok(stores
        .events
        .get_events(thread)
        .await?
        .iter()
        .filter(|event| {
            matches!(
                event.event,
                AgentEvent::SubagentProgress {
                    completed: true,
                    ..
                }
            )
        })
        .count())
}

/// The hook runs once, the record commits `Started` before it and finishes
/// in the settle, and the parent resumes in that settle.
#[tokio::test]
async fn the_hook_finishes_its_effect_in_the_settle_that_resumes_the_parent() -> Result<()> {
    let stores = TestStores::new();
    let spawned = runnable_invocation(&stores, "t-hook-settle").await?;
    let hook = FakeHook::new(
        Some(merge_back()),
        vec![Act::Return(Box::new(output(
            EffectState::Completed,
            Vec::new(),
        )))],
    );

    let outcome = run_attempt(&stores, &spawned.invocation_task, &hook, 10).await?;

    assert_eq!(
        hook.seen()?,
        vec![Seen {
            attempt: 1,
            mode: RunMode::Execute,
            prior: None
        }]
    );
    assert_eq!(outcome.invocation_task.status, TaskStatus::Completed);
    assert_eq!(outcome.tool_result.output, "merged unit_1 at abc123");
    let parent = outcome
        .parent_task
        .context("the settle returns the parent")?;
    assert_eq!(parent.status, TaskStatus::Pending, "the parent resumes");
    let record = effect(&stores, &spawned.invocation_task)
        .await?
        .context("the git record")?;
    assert_eq!(record.state, EffectState::Completed);
    assert_eq!(record.kind(), EffectKind::GitStep);
    assert_eq!(
        record.subject,
        EffectSubject::Git {
            step: "merge_back".into(),
            repo: "repo_1".into(),
            unit: "unit_1".into(),
        }
    );
    assert_eq!(record.attempt, 1);
    assert_eq!(
        record.thread.as_ref(),
        Some(&spawned.invocation_task.thread_id)
    );
    Ok(())
}

/// Attempts 1 and 2 crash while the hook runs, after the start record
/// committed. Each requeue continues the same `Started` record; attempt 3,
/// the last, runs in inspect mode and settles.
#[tokio::test]
async fn a_requeued_invocation_continues_its_effect_and_the_last_attempt_inspects() -> Result<()> {
    let stores = TestStores::new();
    let spawned = runnable_invocation(&stores, "t-hook-retry").await?;
    let invocation = &spawned.invocation_task;
    let salvaged = EffectState::Salvaged {
        captured_output: Some("branch bip/thread/c1 at abc123, receipt staged".into()),
    };
    let hook = FakeHook::new(
        Some(merge_back()),
        vec![
            Act::Crash,
            Act::Crash,
            Act::Return(Box::new(output(salvaged.clone(), Vec::new()))),
        ],
    );

    for (attempt, secs) in [(1_u32, 10_i64), (2, 20)] {
        let crashed = std::panic::AssertUnwindSafe(run_attempt(&stores, invocation, &hook, secs))
            .catch_unwind()
            .await;
        assert!(crashed.is_err(), "attempt {attempt} must crash in the hook");
        let record = effect(&stores, invocation)
            .await?
            .context("the start committed before the hook ran")?;
        assert_eq!(record.state, EffectState::Started, "attempt {attempt}");
        assert_eq!(record.attempt, 1, "the first attempt's record is kept");
        let row = stores.tasks.get(&invocation.id).await?.context("row")?;
        assert_eq!(row.status, TaskStatus::Running, "attempt {attempt}");
        assert_eq!(completed_progress(&stores, &invocation.thread_id).await?, 0);
        let records = stores
            .tasks
            .release_expired_leases(t_plus(secs + 6))
            .await?;
        assert!(
            records.iter().any(
                |record| record.id == invocation.id && record.action == RecoveryAction::Requeue
            ),
            "attempt {attempt} of 3 must requeue: {records:?}"
        );
    }

    let outcome = run_attempt(&stores, invocation, &hook, 30).await?;
    assert_eq!(
        hook.seen()?,
        vec![
            Seen {
                attempt: 1,
                mode: RunMode::Execute,
                prior: None
            },
            Seen {
                attempt: 2,
                mode: RunMode::Execute,
                prior: Some(EffectState::Started)
            },
            Seen {
                attempt: 3,
                mode: RunMode::Inspect,
                prior: Some(EffectState::Started)
            },
        ]
    );
    assert_eq!(outcome.invocation_task.status, TaskStatus::Completed);
    assert_eq!(
        outcome.parent_task.map(|parent| parent.status),
        Some(TaskStatus::Pending)
    );
    let record = effect(&stores, invocation).await?.context("record")?;
    assert_eq!(record.state, salvaged);
    assert_eq!(completed_progress(&stores, &invocation.thread_id).await?, 1);
    Ok(())
}

/// With no plan there is no record: the hook still runs, rewrites the
/// result and adds its events and documents to the settle.
#[tokio::test]
async fn without_a_plan_the_hook_rewrites_the_result_and_writes_no_effect() -> Result<()> {
    let stores = TestStores::new();
    let spawned = runnable_invocation(&stores, "t-hook-no-plan").await?;
    let invocation = &spawned.invocation_task;
    let document = DocumentWrite::put(
        DocumentKey::new(
            DocScope::Thread(invocation.thread_id.clone()),
            DocKind::new("bip.linear_outbox")?,
            "unit_1",
        ),
        1,
        DocExpect::Absent,
        json!({"event": "unit_failed"}),
    );
    let extra_event = Write::AppendEvents {
        thread: invocation.thread_id.clone(),
        events: vec![AgentEvent::text("msg_hook", "unit_1 failed; no merge")],
    };
    let hook = FakeHook::new(
        None,
        vec![Act::Return(Box::new(output(
            EffectState::Completed,
            vec![Write::Document(document), extra_event],
        )))],
    );

    let outcome = run_attempt(&stores, invocation, &hook, 10).await?;

    assert_eq!(outcome.tool_result.output, "merged unit_1 at abc123");
    assert!(
        stores
            .tasks
            .effects_for_task(&invocation.id)
            .await?
            .is_empty(),
        "no plan writes no effect record"
    );
    assert_eq!(
        outcome.committed_events.len(),
        2,
        "the progress event and the hook's event: {:?}",
        outcome.committed_events
    );
    let written = crate::journal::DocumentStore::get_document(
        &stores.tasks,
        &DocumentKey::new(
            DocScope::Thread(invocation.thread_id.clone()),
            DocKind::new("bip.linear_outbox")?,
            "unit_1",
        ),
    )
    .await?;
    assert!(
        written.is_some(),
        "the hook's document commits with the settle"
    );
    Ok(())
}

/// A hook output the settle cannot carry is refused before anything is
/// written: the invocation keeps running and its record stays `Started`.
#[tokio::test]
async fn a_park_or_a_foreign_extra_write_is_refused_and_settles_nothing() -> Result<()> {
    let refusals = [
        CompletionOutput {
            park: Some(PromptRequest {
                payload: json!({"merge": "needs approval"}),
            }),
            ..output(EffectState::Completed, Vec::new())
        },
        output(
            EffectState::Completed,
            vec![Write::SettleTask {
                task: crate::journal::task::AgentTaskId::new(),
                outcome: Settle::Fail {
                    error: "not the hook's to settle".into(),
                },
            }],
        ),
        output(EffectState::Started, Vec::new()),
    ];
    for (index, refused) in refusals.into_iter().enumerate() {
        let stores = TestStores::new();
        let spawned = runnable_invocation(&stores, &format!("t-hook-refused-{index}")).await?;
        let invocation = &spawned.invocation_task;
        let hook = FakeHook::new(Some(merge_back()), vec![Act::Return(Box::new(refused))]);

        let error = run_attempt(&stores, invocation, &hook, 10)
            .await
            .err()
            .with_context(|| format!("output {index} must be refused"))?;
        let reason = error
            .downcast_ref::<CompletionHookError>()
            .with_context(|| format!("output {index}: {error:#}"))?;
        match (index, reason) {
            (0, CompletionHookError::Unsupported(_))
            | (1, CompletionHookError::RefusedExtraWrite { index: 0, .. })
            | (2, CompletionHookError::StartedFinish) => {}
            _ => anyhow::bail!("output {index} refused for the wrong reason: {reason}"),
        }
        let row = stores.tasks.get(&invocation.id).await?.context("row")?;
        assert_eq!(row.status, TaskStatus::Running, "output {index}");
        let record = effect(&stores, invocation).await?.context("record")?;
        assert_eq!(record.state, EffectState::Started, "output {index}");
        assert_eq!(completed_progress(&stores, &invocation.thread_id).await?, 0);
    }
    Ok(())
}

/// The context's store reads; it does not commit.
#[tokio::test]
async fn the_hook_context_store_is_read_only() -> Result<()> {
    let stores = TestStores::new();
    let spawned = runnable_invocation(&stores, "t-hook-read-only").await?;
    let hook = FakeHook::new(Some(merge_back()), vec![Act::CommitThroughContext]);
    let outcome = run_attempt(&stores, &spawned.invocation_task, &hook, 10).await?;
    assert_eq!(outcome.invocation_task.status, TaskStatus::Completed);
    Ok(())
}
