#![cfg(feature = "sqlite")]
//! `kill -9` while a subagent invocation's completion hook runs (primitive
//! 10). The hook's `GitStep` record committed before the hook ran, so after
//! the restart the requeued invocation continues that record, the hook runs
//! once more with it as `prior`, and the parent gets one result.

use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_sdk_foundation::audit::AuditProvenance;
use agent_sdk_foundation::{
    AgentContinuation, AgentEvent, AgentState, ContinuationEnvelope, PendingToolCallInfo, ThreadId,
    TokenUsage, ToolConcurrency, ToolReplay, ToolResult, ToolTier,
};
use agent_server::journal::effect::{EffectKey, EffectRecord, EffectState};
use agent_server::journal::event_repository::EventRepository;
use agent_server::journal::recovery::RecoveryAction;
use agent_server::journal::store::{AgentTaskStore, SubagentInvocationSpawn};
use agent_server::journal::task::{
    AgentTask, AgentTaskId, LeaseId, SubmittedInputItem, SuspensionPayload, TaskStatus, WorkerId,
};
use agent_server::journal::thread_store::ThreadStore;
use agent_server::journal::{DurableStore, SpawnChildrenExt};
use agent_server::worker::subagent::{
    EffectiveSubagentCapabilities, EffectiveSubagentMcpPolicy, EffectiveSubagentSpec,
    InheritedSubagentPolicy, SubagentCapabilityProfile, SubagentSandboxPolicy,
};
use agent_server::worker::{
    CompletionContext, CompletionOutput, CompletionPlan, InstalledCompletionHook, RunMode,
    SubagentCompletionHook, SubagentResultDeps, SubagentTaskOutcome, execute_subagent_task,
    resolve_subagent_bootstrap,
};
use agent_service_host::sqlite::SqliteDurableStore;
use anyhow::{Context, Result, bail, ensure};
use async_trait::async_trait;
use serde_json::json;
use time::OffsetDateTime;

const CHILD_ENV: &str = "AGENT_SDK_COMPLETION_HOOK_CRASH_CHILD";
const DB_ENV: &str = "AGENT_SDK_COMPLETION_HOOK_CRASH_DB";
const READY_ENV: &str = "AGENT_SDK_COMPLETION_HOOK_CRASH_READY";

const PARENT_THREAD: &str = "kill-nine-hook-parent";

fn at(secs: i64) -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + time::Duration::seconds(secs)
}

fn set(values: &[&str]) -> std::collections::BTreeSet<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

fn spec() -> EffectiveSubagentSpec {
    EffectiveSubagentSpec {
        task: "Change one file".into(),
        prompt: "Work in your worktree.".into(),
        model: "mock-model".into(),
        max_turns: 4,
        timeout_ms: 30_000,
        depth: 1,
        max_parallel_subagents: 1,
        nickname: Some("Worker".into()),
        sandbox: SubagentSandboxPolicy::read_only(),
        mcp: EffectiveSubagentMcpPolicy {
            allowed_servers: set(&[]),
        },
        audit_provenance: Some(AuditProvenance::new("mock", "mock-model")),
        inherited_policy: InheritedSubagentPolicy {
            default_model: "mock-model".into(),
            allowed_models: set(&["mock-model"]),
            default_max_turns: 4,
            max_turns: 4,
            default_timeout_ms: 30_000,
            max_timeout_ms: 30_000,
            capability_profiles: std::collections::BTreeMap::from([(
                "worker".into(),
                SubagentCapabilityProfile {
                    capabilities: set(&["read_file"]),
                    sandbox: SubagentSandboxPolicy::read_only(),
                    allowed_mcp_servers: set(&[]),
                },
            )]),
            allowed_capabilities: set(&["read_file"]),
            max_depth: 3,
            max_parallel_subagents: 1,
            sandbox: SubagentSandboxPolicy::read_only(),
            allowed_mcp_servers: set(&[]),
            audit_provider: "mock".into(),
        },
        capabilities: EffectiveSubagentCapabilities {
            profile: "worker".into(),
            allowed: set(&["read_file"]),
        },
    }
}

fn parent_payload(thread: &ThreadId) -> SuspensionPayload {
    let input = json!({"task": "Change one file"});
    SuspensionPayload {
        continuation: ContinuationEnvelope::wrap(AgentContinuation {
            thread_id: thread.clone(),
            turn: 1,
            total_usage: TokenUsage::default(),
            turn_usage: TokenUsage::default(),
            pending_tool_calls: vec![PendingToolCallInfo {
                id: "call_worker".into(),
                name: "subagent_worker".into(),
                display_name: "Subagent: Worker".into(),
                tier: ToolTier::Confirm,
                concurrency: ToolConcurrency::Exclusive,
                replay: ToolReplay::Unsafe,
                input: input.clone(),
                effective_input: input,
                listen_context: None,
            }],
            awaiting_index: 0,
            completed_results: Vec::new(),
            state: AgentState::new(thread.clone()),
            response_id: None,
            stop_reason: None,
            response_content: Vec::new(),
        }),
        suspended_messages: Vec::new(),
        child_join_policy: agent_server::ChildJoinPolicy::default(),
    }
}

/// A parent waiting on one invocation whose child root was cancelled, so
/// the invocation is runnable. Returns `(parent, invocation)`.
async fn runnable_invocation(store: &SqliteDurableStore) -> Result<(AgentTask, AgentTask)> {
    let parent_thread = ThreadId::from_string(PARENT_THREAD);
    ThreadStore::get_or_create(store, &parent_thread, at(0)).await?;
    let parent = AgentTask::new_root_turn(parent_thread.clone(), at(0), 3);
    let parent_id = parent.id.clone();
    store.submit_root_turn(parent).await?;
    let worker = WorkerId::from_string("seed-worker");
    let lease = LeaseId::from_string("seed-lease");
    store
        .try_acquire_task(&parent_id, worker.clone(), lease.clone(), at(60), at(1))
        .await?
        .context("the parent must acquire")?;
    let child_thread = ThreadId::from_string("kill-nine-hook-child");
    ThreadStore::get_or_create(store, &child_thread, at(1)).await?;
    let (parent, invocation, child_root) = store
        .spawn_subagent_invocation(
            &parent_id,
            &worker,
            &lease,
            SubagentInvocationSpawn {
                child_thread_id: child_thread,
                spec: spec(),
                child_root_input: vec![SubmittedInputItem::Text {
                    text: "Change one file".into(),
                }],
                spawn_index: 0,
                child_caller_metadata: None,
                payload: parent_payload(&parent_thread),
            },
            Vec::new(),
            at(2),
        )
        .await?;
    store.cancel_tree(&child_root.id, at(3)).await?;
    Ok((parent, invocation))
}

fn merge_back() -> CompletionPlan {
    CompletionPlan {
        step: "merge_back".into(),
        repo: "repo_1".into(),
        unit: "unit_1".into(),
    }
}

/// The crashing process's hook: publish the invocation id, then wait to be
/// killed.
struct HangingHook {
    ready: std::path::PathBuf,
}

#[async_trait]
impl SubagentCompletionHook for HangingHook {
    fn plan(&self, _ctx: &CompletionContext) -> Option<CompletionPlan> {
        Some(merge_back())
    }

    async fn run(
        &self,
        ctx: &CompletionContext,
        _prior: Option<&EffectRecord>,
        _mode: RunMode,
    ) -> Result<CompletionOutput> {
        std::fs::write(&self.ready, ctx.invocation.as_str())?;
        tokio::time::sleep(Duration::from_mins(1)).await;
        bail!("the child was not killed while the hook ran")
    }
}

/// The restarted process's hook: record what it saw and finish the step.
#[derive(Default)]
struct RecordingHook {
    seen: Mutex<Vec<(u32, RunMode, Option<EffectState>)>>,
}

#[async_trait]
impl SubagentCompletionHook for RecordingHook {
    fn plan(&self, _ctx: &CompletionContext) -> Option<CompletionPlan> {
        Some(merge_back())
    }

    async fn run(
        &self,
        ctx: &CompletionContext,
        prior: Option<&EffectRecord>,
        mode: RunMode,
    ) -> Result<CompletionOutput> {
        self.seen.lock().ok().context("seen lock")?.push((
            ctx.attempt,
            mode,
            prior.map(|record| record.state.clone()),
        ));
        Ok(CompletionOutput {
            result: ToolResult::success("merged unit_1 at abc123"),
            state: EffectState::Completed,
            extra_writes: Vec::new(),
            park: None,
        })
    }
}

async fn run_invocation(
    store: &Arc<SqliteDurableStore>,
    invocation: &AgentTaskId,
    hook: Arc<dyn SubagentCompletionHook>,
    lease: (&str, i64, i64),
) -> Result<SubagentTaskOutcome> {
    let (name, acquired_at, expires_at) = lease;
    let running = store
        .try_acquire_task(
            invocation,
            WorkerId::from_string(format!("{name}-worker")),
            LeaseId::from_string(format!("{name}-lease")),
            at(expires_at),
            at(acquired_at),
        )
        .await?
        .context("the invocation must acquire")?;
    let bootstrap = resolve_subagent_bootstrap(running, store.as_ref()).await?;
    let deps = SubagentResultDeps {
        task_store: store.as_ref(),
        thread_store: store.as_ref(),
        message_store: store.as_ref(),
        event_repo: store.as_ref(),
        artifact_store: None,
        completion_hook: Some(InstalledCompletionHook {
            hook,
            store: Arc::clone(store) as Arc<dyn DurableStore>,
        }),
    };
    execute_subagent_task(bootstrap, &deps, at(acquired_at + 1)).await
}

/// Child-process half: settle the invocation through a hook that hangs,
/// and wait to be killed while it runs.
#[tokio::test]
async fn completion_hook_crash_child() -> Result<()> {
    if std::env::var_os(CHILD_ENV).is_none() {
        return Ok(());
    }
    let url = std::env::var(DB_ENV).context("missing crash database URL")?;
    let ready = std::env::var(READY_ENV).context("missing crash ready path")?;
    let store = Arc::new(SqliteDurableStore::connect(&url).await?);
    let (_parent, invocation) = runnable_invocation(&store).await?;
    run_invocation(
        &store,
        &invocation.id,
        Arc::new(HangingHook {
            ready: ready.into(),
        }),
        ("crashed", 10, 600),
    )
    .await?;
    bail!("the child was not killed while the hook ran")
}

/// Spawn the child, wait until its hook runs, then `kill -9` it.
async fn crash_while_the_hook_runs(url: &str, ready: &std::path::Path) -> Result<AgentTaskId> {
    let mut child = Command::new(std::env::current_exe()?)
        .arg("completion_hook_crash_child")
        .arg("--exact")
        .arg("--nocapture")
        .env(CHILD_ENV, "1")
        .env(DB_ENV, url)
        .env(READY_ENV, ready)
        .spawn()
        .context("spawn the hook child process")?;
    for _ in 0..200 {
        if let Ok(task_id) = std::fs::read_to_string(ready)
            && !task_id.is_empty()
        {
            let status = Command::new("kill")
                .arg("-9")
                .arg(child.id().to_string())
                .status()
                .context("send kill -9 to the hook child process")?;
            ensure!(status.success(), "kill -9 failed with {status}");
            let _ = child.wait()?;
            return Ok(AgentTaskId::from_string(task_id.trim()));
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let _ = child.kill();
    let output = child.wait_with_output()?;
    bail!(
        "the hook child process did not reach its hook: {}",
        String::from_utf8_lossy(&output.stderr)
    )
}

#[tokio::test]
async fn kill_nine_during_the_completion_hook_continues_its_effect_after_restart() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let url = format!(
        "sqlite://{}?mode=rwc",
        temp.path().join("completion-hook-crash.db").display()
    );
    let invocation = crash_while_the_hook_runs(&url, &temp.path().join("hook-running")).await?;

    let store = Arc::new(SqliteDurableStore::connect(&url).await?);
    let key = EffectKey::git(&invocation, "merge_back");
    let crashed = store
        .effect(&key)
        .await?
        .context("the start record committed in the killed process before its hook ran")?;
    ensure!(
        crashed.state == EffectState::Started && crashed.attempt == 1,
        "the crashed record: {crashed:?}"
    );
    let row = AgentTaskStore::get(store.as_ref(), &invocation)
        .await?
        .context("the invocation row")?;
    ensure!(row.status == TaskStatus::Running, "{:?}", row.status);

    let records = store.release_expired_leases(at(601)).await?;
    ensure!(
        records
            .iter()
            .any(|record| record.id == invocation && record.action == RecoveryAction::Requeue),
        "the lease sweep did not requeue the crashed invocation: {records:?}"
    );
    let hook = Arc::new(RecordingHook::default());
    let outcome = run_invocation(
        &store,
        &invocation,
        Arc::clone(&hook) as Arc<dyn SubagentCompletionHook>,
        ("boot", 602, 1200),
    )
    .await?;

    let seen = hook.seen.lock().ok().context("seen lock")?.clone();
    ensure!(
        seen == vec![(2, RunMode::Execute, Some(EffectState::Started))],
        "the restarted hook saw {seen:?}"
    );
    ensure!(outcome.invocation_task.status == TaskStatus::Completed);
    let parent = outcome
        .parent_task
        .context("the settle returns the parent")?;
    ensure!(
        parent.status == TaskStatus::Pending,
        "the parent resumes: {:?}",
        parent.status
    );
    let finished = store.effect(&key).await?.context("the record")?;
    ensure!(
        finished.state == EffectState::Completed,
        "{:?}",
        finished.state
    );
    let results: Vec<_> =
        EventRepository::get_events(store.as_ref(), &ThreadId::from_string(PARENT_THREAD))
            .await?
            .into_iter()
            .filter(|event| {
                matches!(
                    event.event,
                    AgentEvent::SubagentProgress {
                        completed: true,
                        ..
                    }
                )
            })
            .collect();
    ensure!(
        results.len() == 1,
        "the parent gets one completion, got {}",
        results.len()
    );
    let settled = AgentTaskStore::get(store.as_ref(), &invocation)
        .await?
        .context("the invocation row")?;
    ensure!(
        settled
            .result_payload
            .as_ref()
            .and_then(|result| result.get("output"))
            == Some(&json!("merged unit_1 at abc123")),
        "the invocation carries the hook's result: {:?}",
        settled.result_payload
    );
    Ok(())
}
