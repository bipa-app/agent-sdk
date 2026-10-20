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
    InstalledCompletionHook, RunMode, SubagentCompletionHook,
};
use super::root_turn::RootTurnOutcome;
use super::subagent::{
    SpawnedSubagentInvocation, SubagentInvocationDeps, SubagentResultDeps, SubagentTaskOutcome,
    execute_subagent_task, resolve_subagent_bootstrap, spawn_subagent_invocation,
};
use super::subagent_execution_test::{TestStores, child_spawn, create_running_parent_root, t_plus};
use crate::journal::document::{
    DocExpect, DocKind, DocScope, Document, DocumentKey, DocumentWrite,
};
use crate::journal::effect::{EffectKey, EffectKind, EffectRecord, EffectState, EffectSubject};
use crate::journal::event_repository::EventRepository;
use crate::journal::extension::{FoldedSteering, TransitionExtension, TransitionView};
use crate::journal::recovery::RecoveryAction;
use crate::journal::store::AgentTaskStore;
use crate::journal::task::{AgentTask, AgentTaskId, LeaseId, TaskStatus, WorkerId};
use crate::journal::transition::{DurableStore, Settle, SteeringInput, Transition, Write};
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

/// [`run_attempt`] with any hook.
async fn run_attempt_with<H: SubagentCompletionHook + 'static>(
    stores: &TestStores,
    invocation: &AgentTask,
    hook: &Arc<H>,
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
    let mut deps = stores.subagent_deps();
    deps.completion_hook = Some(InstalledCompletionHook {
        hook: Arc::clone(hook) as Arc<dyn SubagentCompletionHook>,
        store: Arc::new(stores.tasks.clone()) as Arc<dyn DurableStore>,
    });
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
            .release_expired_leases(
                t_plus(secs + 6),
                crate::journal::LeaseRecoveryScope::Expired,
            )
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

/// An invocation stored with a budget of 1 (spawned before the invocation
/// budget, or by another path) runs its only attempt in `Execute`, not in
/// `Inspect`, so its work lands.
#[tokio::test]
async fn a_first_attempt_executes_even_under_a_budget_of_one() -> Result<()> {
    let stores = TestStores::new();
    let spawned = runnable_invocation(&stores, "t-hook-budget-one").await?;
    let hook = FakeHook::new(
        Some(merge_back()),
        vec![Act::Return(Box::new(output(
            EffectState::Completed,
            Vec::new(),
        )))],
    );
    let running = stores
        .tasks
        .try_acquire_task(
            &spawned.invocation_task.id,
            WorkerId::new(),
            LeaseId::new(),
            t_plus(15),
            t_plus(10),
        )
        .await?
        .context("the invocation must acquire")?;
    let mut bootstrap = resolve_subagent_bootstrap(running, &stores.tasks).await?;
    bootstrap.invocation_task.max_attempts = 1;
    let (deps, _installed) = deps(&stores, &hook);
    let outcome = execute_subagent_task(bootstrap, &deps, t_plus(11)).await?;
    assert_eq!(
        hook.seen()?,
        vec![Seen {
            attempt: 1,
            mode: RunMode::Execute,
            prior: None
        }]
    );
    assert_eq!(outcome.invocation_task.status, TaskStatus::Completed);
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
async fn a_foreign_extra_write_or_a_started_finish_is_refused_and_settles_nothing() -> Result<()> {
    let refusals = [
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
            (0, CompletionHookError::RefusedExtraWrite { index: 0, .. })
            | (1, CompletionHookError::StartedFinish) => {}
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

/// A host prompt for `invocation`'s merge decision, on its parent thread.
fn merge_prompt(invocation: &AgentTask, id: &str) -> crate::journal::prompt::NewPrompt {
    use crate::journal::prompt::{NewPrompt, PromptCloseFrame, PromptFrame, PromptOwner};
    NewPrompt {
        id: id.to_owned(),
        surface_thread: invocation.thread_id.clone(),
        owner: PromptOwner::Host {
            name: "test.merge_gate".into(),
        },
        owner_task: None,
        frame: PromptFrame {
            name: "apply_worker_merge".into(),
            display_name: "Merge worker".into(),
            input: json!({"unit": "unit_1"}),
            description: "Merge unit_1?".into(),
        },
        close: PromptCloseFrame {
            name: "apply_worker_merge".into(),
            display_name: "Merge worker".into(),
            input: json!({"unit": "unit_1"}),
        },
        must_reach_human: true,
        host_state: json!({}),
    }
}

fn parking(prompt: crate::journal::prompt::NewPrompt) -> CompletionOutput {
    CompletionOutput {
        park: Some(prompt),
        ..output(EffectState::Started, Vec::new())
    }
}

/// Close `prompt` and wake `invocation` in one transition, as a host's
/// prompt handler does (`PromptResolution::Close { extra }`).
async fn close_and_wake(
    stores: &TestStores,
    invocation: &AgentTask,
    prompt: &str,
    secs: i64,
) -> std::result::Result<crate::journal::transition::Committed, crate::journal::transition::Rejected>
{
    let answered = crate::journal::prompt::get_prompt(&stores.tasks, &invocation.thread_id, prompt)
        .await
        .map_err(crate::journal::transition::Rejected::Store)?
        .ok_or_else(|| {
            crate::journal::transition::Rejected::Store(anyhow::anyhow!("no prompt {prompt}"))
        })?;
    let mut writes = vec![Write::WakeInvocation {
        invocation: invocation.id.clone(),
        prompt: prompt.to_owned(),
    }];
    writes.extend(
        answered
            .close_writes(ToolResult::success("approved"), false, t_plus(secs))
            .map_err(crate::journal::transition::Rejected::Store)?,
    );
    stores
        .tasks
        .commit(
            Transition {
                guards: vec![crate::journal::transition::Guard::TaskIs {
                    task: invocation.id.clone(),
                    status: TaskStatus::WaitingOnChildren,
                    lease: None,
                }],
                writes,
            },
            t_plus(secs),
        )
        .await
}

/// A hook that asks for a decision parks the invocation: the prompt row
/// opens (owned by the host, with the invocation as its task), the
/// invocation waits on it with no lease, the effect stays `Started`, and the
/// parent keeps waiting with no completed progress.
#[tokio::test]
async fn a_park_opens_the_prompt_and_keeps_the_effect_started() -> Result<()> {
    let stores = TestStores::new();
    let spawned = runnable_invocation(&stores, "t-hook-park").await?;
    let invocation = &spawned.invocation_task;
    let hook = FakeHook::new(
        Some(merge_back()),
        vec![Act::Return(Box::new(parking(merge_prompt(
            invocation, "merge-1",
        ))))],
    );

    let outcome = run_attempt(&stores, invocation, &hook, 10).await?;

    assert_eq!(outcome.parked_on.as_deref(), Some("merge-1"));
    assert!(outcome.parent_task.is_none());
    let row = stores.tasks.get(&invocation.id).await?.context("row")?;
    assert_eq!(row.status, TaskStatus::WaitingOnChildren);
    assert_eq!(row.pending_child_count, 1);
    assert!(row.lease_id.is_none());
    assert_eq!(row.state.awaited_prompt(), Some("merge-1"));
    let record = effect(&stores, invocation).await?.context("record")?;
    assert_eq!(record.state, EffectState::Started);
    let prompt =
        crate::journal::prompt::get_prompt(&stores.tasks, &invocation.thread_id, "merge-1")
            .await?
            .context("the prompt row opened")?;
    assert_eq!(prompt.record.owner_task.as_ref(), Some(&invocation.id));
    let parent = stores
        .tasks
        .get(invocation.parent_id.as_ref().context("parent id")?)
        .await?
        .context("parent")?;
    assert_eq!(parent.status, TaskStatus::WaitingOnChildren);
    assert_eq!(completed_progress(&stores, &invocation.thread_id).await?, 0);
    Ok(())
}

/// The handler's close wakes the invocation in the same transition; its
/// next acquisition charges no attempt, and the hook runs again with the
/// `Started` record as `prior` and the prompt as `resumed_prompt`, then
/// settles: the parent resumes with one completed result.
#[tokio::test]
async fn a_woken_invocation_resumes_without_an_attempt_and_settles() -> Result<()> {
    let stores = TestStores::new();
    let spawned = runnable_invocation(&stores, "t-hook-wake").await?;
    let invocation = &spawned.invocation_task;
    let resumed = Arc::new(Mutex::new(Vec::new()));
    let hook = Arc::new(ResumeHook {
        inner: FakeHook::new(
            Some(merge_back()),
            vec![
                Act::Return(Box::new(parking(merge_prompt(invocation, "merge-2")))),
                Act::Return(Box::new(output(EffectState::Completed, Vec::new()))),
            ],
        ),
        resumed: Arc::clone(&resumed),
    });
    run_attempt_with(&stores, invocation, &hook, 10).await?;
    close_and_wake(&stores, invocation, "merge-2", 20).await?;
    let woken = stores.tasks.get(&invocation.id).await?.context("row")?;
    assert_eq!(woken.status, TaskStatus::Pending);
    assert_eq!(woken.pending_child_count, 0);
    assert_eq!(woken.attempt, 1);

    let outcome = run_attempt_with(&stores, invocation, &hook, 30).await?;

    assert_eq!(
        outcome.invocation_task.attempt, 1,
        "the wake charged no attempt"
    );
    assert_eq!(outcome.invocation_task.status, TaskStatus::Completed);
    assert_eq!(
        outcome.parent_task.map(|parent| parent.status),
        Some(TaskStatus::Pending)
    );
    assert_eq!(
        hook.inner.seen()?,
        vec![
            Seen {
                attempt: 1,
                mode: RunMode::Execute,
                prior: None
            },
            Seen {
                attempt: 1,
                mode: RunMode::Execute,
                prior: Some(EffectState::Started)
            },
        ]
    );
    assert_eq!(
        resumed.lock().ok().context("resumed lock")?.clone(),
        vec![None, Some("merge-2".to_owned())]
    );
    let record = effect(&stores, invocation).await?.context("record")?;
    assert_eq!(record.state, EffectState::Completed);
    assert_eq!(completed_progress(&stores, &invocation.thread_id).await?, 1);
    Ok(())
}

/// A wake that names another prompt is refused by a guard and writes
/// nothing: the invocation stays parked and the prompt stays open.
#[tokio::test]
async fn a_wake_for_another_prompt_is_refused() -> Result<()> {
    let stores = TestStores::new();
    let spawned = runnable_invocation(&stores, "t-hook-wrong-wake").await?;
    let invocation = &spawned.invocation_task;
    let hook = FakeHook::new(
        Some(merge_back()),
        vec![Act::Return(Box::new(parking(merge_prompt(
            invocation, "merge-3",
        ))))],
    );
    run_attempt(&stores, invocation, &hook, 10).await?;
    let mut wrong = Transition {
        guards: vec![crate::journal::transition::Guard::TaskIs {
            task: invocation.id.clone(),
            status: TaskStatus::WaitingOnChildren,
            lease: None,
        }],
        writes: vec![Write::WakeInvocation {
            invocation: invocation.id.clone(),
            prompt: "some-other-prompt".into(),
        }],
    };
    let refused = stores.tasks.commit(wrong.clone(), t_plus(20)).await;
    assert!(
        matches!(
            refused,
            Err(crate::journal::transition::Rejected::Guard { index: 1, .. })
        ),
        "{refused:?}"
    );
    wrong.guards.clear();
    assert!(
        matches!(
            stores.tasks.commit(wrong, t_plus(21)).await,
            Err(crate::journal::transition::Rejected::Store(_))
        ),
        "a wake needs its waiting guard"
    );
    let row = stores.tasks.get(&invocation.id).await?.context("row")?;
    assert_eq!(row.state.awaited_prompt(), Some("merge-3"));
    Ok(())
}

/// A park and its wake count `agent_server.subagent.invocation.prompts`
/// once each, `step=parked` then `step=woken`, labelled only by backend and
/// step.
#[cfg(feature = "otel")]
#[tokio::test]
async fn a_park_and_its_wake_are_counted_by_backend_and_step() -> Result<()> {
    use crate::observability::attrs;
    use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData, SumDataPoint};
    use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};

    let exporter = InMemoryMetricExporter::default();
    let provider = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(exporter.clone()).build())
        .build();
    opentelemetry::global::set_meter_provider(provider.clone());
    crate::observability::ServerMetrics::reset_for_testing();

    let stores = TestStores::new();
    let spawned = runnable_invocation(&stores, "t-hook-park-metrics").await?;
    let invocation = &spawned.invocation_task;
    let hook = FakeHook::new(
        Some(merge_back()),
        vec![Act::Return(Box::new(parking(merge_prompt(
            invocation, "merge-6",
        ))))],
    );
    run_attempt(&stores, invocation, &hook, 10).await?;
    close_and_wake(&stores, invocation, "merge-6", 20).await?;

    provider.force_flush()?;
    let points: Vec<(Vec<(String, String)>, u64)> = exporter
        .get_finished_metrics()?
        .iter()
        .rev()
        .take(1)
        .flat_map(opentelemetry_sdk::metrics::data::ResourceMetrics::scope_metrics)
        .flat_map(opentelemetry_sdk::metrics::data::ScopeMetrics::metrics)
        .filter(|metric| metric.name() == "agent_server.subagent.invocation.prompts")
        .flat_map(|metric| match metric.data() {
            AggregatedMetrics::U64(MetricData::Sum(sum)) => sum
                .data_points()
                .map(|point| {
                    let labels = point
                        .attributes()
                        .map(|kv| (kv.key.as_str().to_owned(), kv.value.to_string()))
                        .collect();
                    (labels, SumDataPoint::value(point))
                })
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect();
    let counted = |step: &str| {
        points
            .iter()
            .filter(|(labels, _)| {
                labels.contains(&(attrs::BACKEND.to_owned(), attrs::BACKEND_MEMORY.to_owned()))
                    && labels.contains(&(attrs::STEP.to_owned(), step.to_owned()))
            })
            .map(|(_, count)| count)
            .sum::<u64>()
    };
    // A lower bound: other lib tests share the global meter under plain
    // `cargo test` (nextest runs one per process).
    assert!(counted(attrs::STEP_PARKED) >= 1, "{points:?}");
    assert!(counted(attrs::STEP_WOKEN) >= 1, "{points:?}");
    for (labels, _) in &points {
        let mut keys: Vec<&str> = labels.iter().map(|(key, _)| key.as_str()).collect();
        keys.sort_unstable();
        assert_eq!(keys, [attrs::BACKEND, attrs::STEP], "bounded labels only");
    }
    Ok(())
}

/// The last attempt only inspects: a park there is refused and nothing is
/// written.
#[tokio::test]
async fn parking_on_the_last_attempt_is_refused() -> Result<()> {
    let stores = TestStores::new();
    let spawned = runnable_invocation(&stores, "t-hook-park-last").await?;
    let invocation = &spawned.invocation_task;
    let hook = FakeHook::new(
        Some(merge_back()),
        vec![
            Act::Crash,
            Act::Crash,
            Act::Return(Box::new(parking(merge_prompt(invocation, "merge-4")))),
        ],
    );
    for secs in [10_i64, 20] {
        let crashed = std::panic::AssertUnwindSafe(run_attempt(&stores, invocation, &hook, secs))
            .catch_unwind()
            .await;
        assert!(crashed.is_err());
        stores
            .tasks
            .release_expired_leases(
                t_plus(secs + 6),
                crate::journal::LeaseRecoveryScope::Expired,
            )
            .await?;
    }
    let error = run_attempt(&stores, invocation, &hook, 30)
        .await
        .err()
        .context("a park on the last attempt is refused")?;
    assert!(
        matches!(
            error.downcast_ref::<CompletionHookError>(),
            Some(CompletionHookError::ParkOnLastAttempt)
        ),
        "{error:#}"
    );
    let row = stores.tasks.get(&invocation.id).await?.context("row")?;
    assert_eq!(row.status, TaskStatus::Running);
    assert!(
        crate::journal::prompt::get_prompt(&stores.tasks, &invocation.thread_id, "merge-4")
            .await?
            .is_none()
    );
    Ok(())
}

/// A [`FakeHook`] that also records the context's `resumed_prompt`.
struct ResumeHook {
    inner: Arc<FakeHook>,
    resumed: Arc<Mutex<Vec<Option<String>>>>,
}

#[async_trait]
impl SubagentCompletionHook for ResumeHook {
    fn plan(&self, ctx: &CompletionContext) -> Option<CompletionPlan> {
        self.inner.plan(ctx)
    }

    async fn run(
        &self,
        ctx: &CompletionContext,
        prior: Option<&EffectRecord>,
        mode: RunMode,
    ) -> Result<CompletionOutput> {
        self.resumed
            .lock()
            .ok()
            .context("resumed lock")?
            .push(ctx.resumed_prompt.clone());
        self.inner.run(ctx, prior, mode).await
    }
}

// ── steering at the wave's end (S-WS) ───────────────────────────────

/// The `test.steering` queue on `thread`.
fn steering_queue(thread: &ThreadId) -> Result<DocumentKey> {
    Ok(DocumentKey::new(
        DocScope::Thread(thread.clone()),
        DocKind::new("test.steering")?,
        "",
    ))
}

/// The settle's steering write for `parent`'s queue.
fn queued_steering(parent: &AgentTaskId) -> Result<Write> {
    Ok(Write::EnqueueSteeringResume {
        parent: parent.clone(),
        steering: SteeringInput::Queued {
            kind: DocKind::new("test.steering")?,
            key: String::new(),
        },
    })
}

/// Folds the `test.steering` queue (`{"notes": [..]}`) into the fan-in
/// result's output as `[steering] <notes>`, and counts its folds.
#[derive(Default)]
struct NoteFold {
    folds: std::sync::atomic::AtomicUsize,
}

impl NoteFold {
    fn folds(&self) -> usize {
        self.folds.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl TransitionExtension for NoteFold {
    fn extend(&self, _view: TransitionView<'_>) -> Vec<Write> {
        Vec::new()
    }

    fn fold_queued_steering(
        &self,
        _parent: &AgentTask,
        _settled: &AgentTask,
        result: &serde_json::Value,
        queued: &Document,
    ) -> Option<FoldedSteering> {
        self.folds.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let notes: Vec<String> = serde_json::from_value(queued.value.get("notes")?.clone()).ok()?;
        if notes.is_empty() {
            return None;
        }
        let mut folded: ToolResult = serde_json::from_value(result.clone()).ok()?;
        folded.output = format!("{}\n[steering] {}", folded.output, notes.join("; "));
        Some(FoldedSteering {
            result: serde_json::to_value(folded).ok()?,
            schema_version: queued.schema_version,
            value: json!({ "notes": [] }),
        })
    }
}

/// [`TestStores`] whose task store folds with a [`NoteFold`].
fn stores_with_fold() -> (TestStores, Arc<NoteFold>) {
    let fold = Arc::new(NoteFold::default());
    let mut stores = TestStores::new();
    stores.tasks = stores
        .tasks
        .clone()
        .with_transition_extension(Arc::clone(&fold) as Arc<dyn TransitionExtension>);
    (stores, fold)
}

/// Queue `notes` for the parent on `thread`, from nothing.
async fn queue_notes(stores: &TestStores, thread: &ThreadId, notes: &[&str]) -> Result<()> {
    stores
        .tasks
        .commit(
            Transition {
                guards: Vec::new(),
                writes: vec![Write::Document(DocumentWrite::put(
                    steering_queue(thread)?,
                    1,
                    DocExpect::Absent,
                    json!({ "notes": notes }),
                ))],
            },
            t_plus(4),
        )
        .await
        .map_err(anyhow::Error::from)?;
    Ok(())
}

async fn queue(stores: &TestStores, thread: &ThreadId) -> Result<Option<(u64, serde_json::Value)>> {
    Ok(
        crate::journal::DocumentStore::get_document(&stores.tasks, &steering_queue(thread)?)
            .await?
            .map(|document| (document.revision, document.value)),
    )
}

fn parent_of(invocation: &AgentTask) -> Result<AgentTaskId> {
    invocation
        .parent_id
        .clone()
        .context("an invocation has a parent")
}

/// H1: the hook's settle may carry a queued steering write of the
/// invocation's own parent. The settle fans the parent in, so the store
/// folds the parent's notes into the settle's own result and empties the
/// queue in the same transaction; the outcome reports the folded result.
#[tokio::test]
async fn a_hook_settle_folds_the_notes_queued_for_its_own_parent() -> Result<()> {
    let (stores, fold) = stores_with_fold();
    let spawned = runnable_invocation(&stores, "t-hook-fold").await?;
    let invocation = &spawned.invocation_task;
    let parent = parent_of(invocation)?;
    queue_notes(&stores, &invocation.thread_id, &["skip the docs crate"]).await?;
    let hook = FakeHook::new(
        Some(merge_back()),
        vec![Act::Return(Box::new(output(
            EffectState::Completed,
            vec![queued_steering(&parent)?],
        )))],
    );

    let outcome = run_attempt(&stores, invocation, &hook, 10).await?;

    assert_eq!(fold.folds(), 1);
    let folded = "merged unit_1 at abc123\n[steering] skip the docs crate";
    assert_eq!(outcome.tool_result.output, folded);
    let persisted: ToolResult = serde_json::from_value(
        stores
            .tasks
            .get(&invocation.id)
            .await?
            .and_then(|row| row.result_payload)
            .context("the stored result")?,
    )?;
    assert_eq!(persisted.output, folded);
    let resumed = stores.tasks.get(&parent).await?.context("parent")?;
    assert_eq!(resumed.status, TaskStatus::Pending, "fanned in");
    assert!(
        matches!(&resumed.state, crate::journal::TaskState::ReadyToResume { steering, .. } if steering.is_empty()),
        "{:?}",
        resumed.state
    );
    assert_eq!(
        queue(&stores, &invocation.thread_id).await?,
        Some((2, json!({ "notes": [] })))
    );
    Ok(())
}

/// H2–H4: a steering write the settle cannot carry is refused with a typed
/// error before anything is written: on a park, for another task, with
/// inline steering, or a second one. The note stays queued.
#[tokio::test]
async fn a_steering_write_a_hook_may_not_add_is_refused() -> Result<()> {
    for (index, case) in ["park", "foreign", "inline", "twice"]
        .into_iter()
        .enumerate()
    {
        let (stores, fold) = stores_with_fold();
        let spawned = runnable_invocation(&stores, &format!("t-hook-steering-{case}")).await?;
        let invocation = &spawned.invocation_task;
        let parent = parent_of(invocation)?;
        queue_notes(&stores, &invocation.thread_id, &["status?"]).await?;
        let output = match case {
            "park" => CompletionOutput {
                extra_writes: vec![queued_steering(&parent)?],
                ..parking(merge_prompt(invocation, "merge_1"))
            },
            "foreign" => output(
                EffectState::Completed,
                vec![queued_steering(&AgentTaskId::new())?],
            ),
            "inline" => output(
                EffectState::Completed,
                vec![Write::EnqueueSteeringResume {
                    parent: parent.clone(),
                    steering: SteeringInput::Blocks(vec![
                        agent_sdk_foundation::llm::ContentBlock::Text {
                            text: "inline".into(),
                        },
                    ]),
                }],
            ),
            _ => output(
                EffectState::Completed,
                vec![queued_steering(&parent)?, queued_steering(&parent)?],
            ),
        };
        let hook = FakeHook::new(Some(merge_back()), vec![Act::Return(Box::new(output))]);

        let error = run_attempt(&stores, invocation, &hook, 10)
            .await
            .err()
            .with_context(|| format!("{case} must be refused"))?;
        let refused = error
            .downcast_ref::<CompletionHookError>()
            .with_context(|| format!("{case}: {error:#}"))?;
        let expected_index = usize::from(case == "twice");
        assert!(
            matches!(refused, CompletionHookError::RefusedSteeringWrite { index, .. } if *index == expected_index),
            "case {index} ({case}): {refused}"
        );
        let row = stores.tasks.get(&invocation.id).await?.context("row")?;
        assert_eq!(row.status, TaskStatus::Running, "{case} wrote nothing");
        assert_eq!(
            stores.tasks.get(&parent).await?.map(|row| row.status),
            Some(TaskStatus::WaitingOnChildren),
            "{case}"
        );
        assert_eq!(fold.folds(), 0, "{case}");
        assert_eq!(
            queue(&stores, &invocation.thread_id).await?,
            Some((1, json!({ "notes": ["status?"] }))),
            "{case}"
        );
    }
    Ok(())
}

/// A crash while the hook runs, before its settle commits, leaves the note
/// queued; the requeued invocation's settle folds it exactly once.
#[tokio::test]
async fn a_crash_before_the_hook_settle_leaves_the_note_for_the_requeued_settle() -> Result<()> {
    let (stores, fold) = stores_with_fold();
    let spawned = runnable_invocation(&stores, "t-hook-fold-crash").await?;
    let invocation = &spawned.invocation_task;
    let parent = parent_of(invocation)?;
    queue_notes(&stores, &invocation.thread_id, &["skip the docs crate"]).await?;
    let hook = FakeHook::new(
        Some(merge_back()),
        vec![
            Act::Crash,
            Act::Return(Box::new(output(
                EffectState::Completed,
                vec![queued_steering(&parent)?],
            ))),
        ],
    );
    let crashed = std::panic::AssertUnwindSafe(run_attempt(&stores, invocation, &hook, 10))
        .catch_unwind()
        .await;
    assert!(crashed.is_err(), "the first attempt crashes in the hook");
    assert_eq!(fold.folds(), 0);
    assert_eq!(
        queue(&stores, &invocation.thread_id).await?,
        Some((1, json!({ "notes": ["skip the docs crate"] })))
    );
    stores
        .tasks
        .release_expired_leases(t_plus(16), crate::journal::LeaseRecoveryScope::Expired)
        .await?;

    let outcome = run_attempt(&stores, invocation, &hook, 20).await?;
    assert_eq!(
        outcome.tool_result.output,
        "merged unit_1 at abc123\n[steering] skip the docs crate"
    );
    assert_eq!(fold.folds(), 1);
    assert_eq!(
        queue(&stores, &invocation.thread_id).await?,
        Some((2, json!({ "notes": [] })))
    );
    Ok(())
}

/// Records each request's messages as text, and fails the first `fail`
/// calls with a server error.
struct RecordingProvider {
    fail: usize,
    requests: Mutex<Vec<String>>,
}

impl RecordingProvider {
    fn new(fail: usize) -> Self {
        Self {
            fail,
            requests: Mutex::new(Vec::new()),
        }
    }

    fn requests(&self) -> Result<Vec<String>> {
        Ok(self.requests.lock().ok().context("requests lock")?.clone())
    }
}

#[async_trait]
impl agent_sdk_providers::LlmProvider for RecordingProvider {
    async fn chat(
        &self,
        request: agent_sdk_foundation::llm::ChatRequest,
    ) -> Result<agent_sdk_foundation::llm::ChatOutcome> {
        use agent_sdk_foundation::llm::{
            ChatOutcome, ChatResponse, ContentBlock, StopReason, Usage,
        };

        let call = {
            let mut requests = self.requests.lock().ok().context("requests lock")?;
            requests.push(format!("{:?}", request.messages));
            requests.len()
        };
        if call <= self.fail {
            return Ok(ChatOutcome::ServerError("simulated outage".into()));
        }
        Ok(ChatOutcome::Success(ChatResponse {
            id: format!("msg_{call}"),
            content: vec![ContentBlock::Text {
                text: "noted".into(),
            }],
            model: "mock-model".into(),
            stop_reason: Some(StopReason::EndTurn),
            usage: Usage {
                served_speed: None,
                input_tokens: 10,
                output_tokens: 5,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
                request_input_tokens: None,
            },
        }))
    }

    fn model(&self) -> &'static str {
        "mock-model"
    }

    fn provider(&self) -> &'static str {
        "mock"
    }
}

/// H5, H6: the resumed parent's first LLM request carries the folded note
/// in the wave's last tool result, and so does the request after that call
/// failed: the note sits in the durable result, not in a transient
/// steering slot a failed call drops.
#[tokio::test]
async fn the_resumed_parent_reads_the_folded_note_also_after_a_failed_call() -> Result<()> {
    for fail in [0, 1] {
        let (stores, _fold) = stores_with_fold();
        let spawned = runnable_invocation(&stores, &format!("t-hook-fold-resume-{fail}")).await?;
        let invocation = &spawned.invocation_task;
        let parent = parent_of(invocation)?;
        queue_notes(&stores, &invocation.thread_id, &["skip the docs crate"]).await?;
        let hook = FakeHook::new(
            None,
            vec![Act::Return(Box::new(output(
                EffectState::Completed,
                vec![queued_steering(&parent)?],
            )))],
        );
        run_attempt(&stores, invocation, &hook, 10).await?;

        let provider = RecordingProvider::new(fail);
        let resumed = Box::pin(stores.resume_parent(&parent, &provider, 20)).await;
        let requests = provider.requests()?;
        assert!(
            matches!(resumed, Ok(RootTurnOutcome::Completed { .. })),
            "fail {fail}: {resumed:?}"
        );
        assert_eq!(
            requests.len(),
            fail + 1,
            "fail {fail}: one retry per failure"
        );
        for (call, request) in requests.iter().enumerate() {
            assert!(
                request.contains("[steering] skip the docs crate"),
                "fail {fail}: request {call} carries the note: {request}"
            );
        }
    }
    Ok(())
}
