//! `ExecutionContextFactory` inputs and checkpoint-seeded context builder.
//!
//! The root worker needs a deterministic execution context built from
//! durable task, thread, and checkpoint state plus host-provided runtime
//! dependencies. This module captures those inputs and provides the
//! builder function that wires them together.
//!
//! # Design properties
//!
//! 1. **Deterministic reconstruction** — given the same durable inputs
//!    the builder always produces the same staged context, so crash
//!    recovery picks up exactly where the last committed checkpoint
//!    left off.
//! 2. **No durable mid-turn writes** — the builder returns staged
//!    adapters ([`super::staged::StagedStores`]) that buffer all
//!    mutations in memory until the commit path runs.
//! 3. **Separation of durable vs runtime inputs** — durable state
//!    (`AgentTask`, `ThreadRecoveryView`, `AgentDefinition`) is
//!    recoverable from storage; runtime dependencies (`HostDependencies`)
//!    are created fresh by the orchestration layer each time.
//!
//! # Relationship to Phase 4.1
//!
//! Phase 4.1 defines [`crate::worker::AgentDefinition`] and the
//! [`crate::worker::WorkerBootstrapContext`] that validates task
//! preconditions and resolves the definition. This module picks up
//! *after* bootstrap: it takes the validated bootstrap context, recovers
//! thread state, and seeds the staged stores.
//!
//! # What this module does **not** own
//!
//! - Text-only turn execution (Phase 4.3).
//! - Tool-batch suspension and child-task creation (Phase 4.4+).
//! - Commit orchestration (Phase 3.4).
//! - Turn-attempt audit lifecycle (Phase 3.3).

use agent_sdk_core::ThreadId;
use anyhow::{Context, Result};
use time::OffsetDateTime;

use super::checkpoint_store::CheckpointStore;
use super::staged::StagedStores;
use super::thread_recover::{ThreadRecoveryView, recover_thread};
use super::thread_store::ThreadStore;
use crate::worker::bootstrap::WorkerBootstrapContext;
use crate::worker::definition::AgentDefinition;

// ─────────────────────────────────────────────────────────────────────
// RootWorkerInputs
// ─────────────────────────────────────────────────────────────────────

/// Everything the root worker needs to begin (or resume) a turn.
///
/// This is the "factory input" struct the orchestration layer constructs
/// from durable storage before handing control to the worker. It builds
/// on the Phase 4.1 [`WorkerBootstrapContext`] by adding the recovered
/// thread state and staged stores.
///
/// # Fields
///
/// | Group | Fields |
/// |-------|--------|
/// | Bootstrap | `bootstrap` |
/// | Thread state | `recovery_view` |
/// | Staged adapters | `staged_stores` |
///
/// The worker uses `staged_stores` for all message/state reads and
/// writes during the turn. At commit time the staged data is drained
/// and fed into [`super::commit::commit_completed_turn`].
pub struct RootWorkerInputs {
    /// The validated bootstrap context from Phase 4.1 (task, definition,
    /// lease identifiers).
    pub bootstrap: WorkerBootstrapContext,

    /// Thread recovery view with committed messages, agent-state
    /// snapshot, and next turn number.
    pub recovery_view: ThreadRecoveryView,

    /// Staged message and state stores seeded from the recovery view.
    ///
    /// The worker reads from and writes to these during the turn; the
    /// commit path drains them.
    pub staged_stores: StagedStores,
}

impl RootWorkerInputs {
    /// Convenience accessor for the resolved agent definition.
    #[must_use]
    pub const fn definition(&self) -> &AgentDefinition {
        &self.bootstrap.definition
    }
}

// ─────────────────────────────────────────────────────────────────────
// Builder
// ─────────────────────────────────────────────────────────────────────

/// Reconstruct trusted execution context for a root worker from a
/// validated bootstrap context and durable thread state.
///
/// This is the primary entry point for Phase 4.2 context reconstruction.
/// It:
///
/// 1. Recovers the thread state via Phase 3.5's [`recover_thread`].
/// 2. Validates the task is bound to the same thread.
/// 3. Seeds staged stores from the recovery view.
/// 4. Returns [`RootWorkerInputs`] ready for turn execution.
///
/// # Errors
///
/// - Thread recovery fails (completed thread, missing checkpoint,
///   consistency mismatch).
/// - Task `thread_id` does not match the recovery view.
/// - Agent-state snapshot deserialization fails.
pub async fn build_root_worker_inputs(
    bootstrap: WorkerBootstrapContext,
    thread_store: &dyn ThreadStore,
    checkpoint_store: &dyn CheckpointStore,
    now: OffsetDateTime,
) -> Result<RootWorkerInputs> {
    // 1. Recover thread state from the latest checkpoint.
    let recovery_view = recover_thread(&bootstrap.thread_id, thread_store, checkpoint_store, now)
        .await
        .context("build_root_worker_inputs: recover thread")?;

    // 2. Validate task is bound to the bootstrap thread.
    ensure_thread_match(&bootstrap.thread_id, &bootstrap.task.thread_id)?;

    // 3. Seed staged stores from the recovery view.
    let staged_stores = StagedStores::from_recovery_view(&recovery_view)
        .context("build_root_worker_inputs: seed staged stores")?;

    Ok(RootWorkerInputs {
        bootstrap,
        recovery_view,
        staged_stores,
    })
}

fn ensure_thread_match(task_thread: &ThreadId, recovered_thread: &ThreadId) -> Result<()> {
    anyhow::ensure!(
        task_thread == recovered_thread,
        "task thread_id ({task_thread}) does not match recovered thread ({recovered_thread})",
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::checkpoint_store::InMemoryCheckpointStore;
    use super::super::commit::{CompletedTurnCommit, commit_completed_turn};
    use super::super::message_store::{InMemoryMessageProjectionStore, MessageProjectionStore};
    use super::super::task::{AgentTask, AgentTaskId, LeaseId, WorkerId};
    use super::super::thread_store::InMemoryThreadStore;
    use super::super::turn_attempt::{OpenAttemptParams, TurnAttemptOutcome};
    use super::super::turn_attempt_store::{InMemoryTurnAttemptStore, TurnAttemptStore};
    use super::*;
    use crate::worker::definition::{RuntimePolicy, ThinkingPolicy};
    use agent_sdk_core::audit::AuditProvenance;
    use agent_sdk_core::{TokenUsage, llm};
    use agent_sdk_tools::stores::{MessageStore, StateStore};
    use time::Duration;

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn thread_a() -> ThreadId {
        ThreadId::from_string("t-exec-ctx-a")
    }

    fn usage(input: u32, output: u32) -> TokenUsage {
        TokenUsage {
            input_tokens: input,
            output_tokens: output,
        }
    }

    fn sample_definition() -> AgentDefinition {
        AgentDefinition {
            provider: "anthropic".into(),
            model: "claude-sonnet-4-5-20250929".into(),
            system_prompt: "You are a helpful assistant.".into(),
            max_tokens: 4096,
            tools: Vec::new(),
            thinking: ThinkingPolicy::default(),
            policy: RuntimePolicy::server_default(),
        }
    }

    /// Build a `WorkerBootstrapContext` for testing.
    ///
    /// In production this comes from `resolve_bootstrap_context` which
    /// requires a Running task. For unit tests of the staged-store
    /// seeding path we construct it directly.
    fn sample_bootstrap(task: AgentTask) -> WorkerBootstrapContext {
        let thread_id = task.thread_id.clone();
        let task_id = task.id.clone();
        WorkerBootstrapContext {
            task,
            definition: sample_definition(),
            thread_id,
            task_id,
            worker_id: WorkerId::from_string("worker_test"),
            lease_id: LeaseId::from_string("lease_test"),
        }
    }

    fn sample_close_params() -> super::super::turn_attempt::CloseAttemptParams {
        super::super::turn_attempt::CloseAttemptParams {
            response_blob: serde_json::json!({"id": "msg_01"}),
            response_id: Some("msg_01".into()),
            response_model: Some("claude-sonnet-4-5-20250929".into()),
            stop_reason: Some(llm::StopReason::EndTurn),
            outcome: TurnAttemptOutcome::Success,
            input_tokens: 100,
            output_tokens: 50,
            cached_input_tokens: 10,
        }
    }

    fn root_task(thread_id: &ThreadId) -> AgentTask {
        AgentTask::new_root_turn(thread_id.clone(), t0(), 3)
    }

    struct Stores {
        threads: InMemoryThreadStore,
        messages: InMemoryMessageProjectionStore,
        attempts: InMemoryTurnAttemptStore,
        checkpoints: InMemoryCheckpointStore,
    }

    impl Stores {
        fn new() -> Self {
            Self {
                threads: InMemoryThreadStore::new(),
                messages: InMemoryMessageProjectionStore::new(),
                attempts: InMemoryTurnAttemptStore::new(),
                checkpoints: InMemoryCheckpointStore::new(),
            }
        }

        async fn commit_turn(
            &self,
            thread_id: &ThreadId,
            task_id: &AgentTaskId,
            messages: Vec<llm::Message>,
            state_snapshot: serde_json::Value,
            at: OffsetDateTime,
        ) -> Result<super::super::commit::CommitOutcome> {
            let attempt = self
                .attempts
                .open_attempt(OpenAttemptParams {
                    task_id: task_id.clone(),
                    attempt_number: 1,
                    provenance: AuditProvenance::new("anthropic", "claude-sonnet-4-5-20250929"),
                    request_blob: serde_json::json!({"messages": []}),
                    now: t0(),
                })
                .await
                .context("open attempt")?;
            commit_completed_turn(
                CompletedTurnCommit {
                    thread_id: thread_id.clone(),
                    task_id: task_id.clone(),
                    turn_attempt_id: attempt.id,
                    close_attempt_params: sample_close_params(),
                    messages,
                    turn_usage: usage(100, 50),
                    agent_state_snapshot: state_snapshot,
                    now: at,
                },
                &self.threads,
                &self.messages,
                &self.attempts,
                &self.checkpoints,
            )
            .await
        }
    }

    // ── Fresh thread ─────────────────────────────────────────────

    #[tokio::test]
    async fn build_inputs_for_fresh_thread() -> Result<()> {
        let s = Stores::new();
        let task = root_task(&thread_a());
        let bootstrap = sample_bootstrap(task);

        let inputs = build_root_worker_inputs(bootstrap, &s.threads, &s.checkpoints, t0())
            .await
            .context("build")?;

        // Recovery view is for a fresh thread.
        assert_eq!(inputs.recovery_view.next_turn_number, 1);
        assert!(inputs.recovery_view.messages.is_empty());

        // Staged messages start empty.
        let msgs = inputs
            .staged_stores
            .messages
            .get_history(&thread_a())
            .await?;
        assert!(msgs.is_empty());

        // Staged state is a fresh AgentState.
        let state = inputs
            .staged_stores
            .state
            .load(&thread_a())
            .await?
            .context("state")?;
        assert_eq!(state.thread_id, thread_a());
        assert_eq!(state.turn_count, 0);

        // Definition forwarded.
        assert_eq!(inputs.definition().model, "claude-sonnet-4-5-20250929");

        Ok(())
    }

    // ── Existing thread with checkpoint ──────────────────────────

    #[tokio::test]
    async fn build_inputs_seeded_from_checkpoint() -> Result<()> {
        let s = Stores::new();
        let prior_task = AgentTaskId::from_string("task_prior");

        // Commit two turns so the thread has a checkpoint.
        let state_snapshot = serde_json::to_value(&agent_sdk_core::AgentState {
            thread_id: thread_a(),
            turn_count: 2,
            total_usage: usage(200, 100),
            metadata: std::collections::HashMap::default(),
            created_at: t0(),
        })?;

        s.commit_turn(
            &thread_a(),
            &prior_task,
            vec![llm::Message::user("turn 1")],
            serde_json::json!({"turn": 1}),
            t_plus(1),
        )
        .await
        .context("turn 1")?;

        s.commit_turn(
            &thread_a(),
            &prior_task,
            vec![llm::Message::assistant("turn 2")],
            state_snapshot,
            t_plus(2),
        )
        .await
        .context("turn 2")?;

        // Build inputs for a new task on the same thread.
        let task = root_task(&thread_a());
        let bootstrap = sample_bootstrap(task);
        let inputs = build_root_worker_inputs(bootstrap, &s.threads, &s.checkpoints, t_plus(3))
            .await
            .context("build")?;

        // Recovery view has the committed state.
        assert_eq!(inputs.recovery_view.next_turn_number, 3);
        assert_eq!(inputs.recovery_view.messages.len(), 2);

        // Staged messages seeded from checkpoint.
        let msgs = inputs
            .staged_stores
            .messages
            .get_history(&thread_a())
            .await?;
        assert_eq!(msgs.len(), 2);

        // Staged state deserialized from snapshot.
        let state = inputs
            .staged_stores
            .state
            .load(&thread_a())
            .await?
            .context("state")?;
        assert_eq!(state.turn_count, 2);
        assert_eq!(state.total_usage.input_tokens, 200);

        Ok(())
    }

    // ── Staged mutations stay buffered ───────────────────────────

    #[tokio::test]
    async fn staged_mutations_do_not_affect_durable_stores() -> Result<()> {
        let s = Stores::new();
        let prior_task = AgentTaskId::from_string("task_prior");

        let snapshot = serde_json::to_value(&agent_sdk_core::AgentState {
            thread_id: thread_a(),
            turn_count: 1,
            total_usage: usage(100, 50),
            metadata: std::collections::HashMap::default(),
            created_at: t0(),
        })?;

        s.commit_turn(
            &thread_a(),
            &prior_task,
            vec![llm::Message::user("committed")],
            snapshot,
            t_plus(1),
        )
        .await
        .context("commit")?;

        let task = root_task(&thread_a());
        let bootstrap = sample_bootstrap(task);
        let inputs = build_root_worker_inputs(bootstrap, &s.threads, &s.checkpoints, t_plus(2))
            .await
            .context("build")?;

        // Mutate staged stores.
        inputs
            .staged_stores
            .messages
            .append(&thread_a(), llm::Message::assistant("buffered"))
            .await?;
        let mut new_state = agent_sdk_core::AgentState::new(thread_a());
        new_state.turn_count = 99;
        inputs.staged_stores.state.save(&new_state).await?;

        // Staged stores reflect mutations.
        let staged_msgs = inputs
            .staged_stores
            .messages
            .get_history(&thread_a())
            .await?;
        assert_eq!(staged_msgs.len(), 2); // 1 committed + 1 buffered

        // Durable message projection unchanged — still only the
        // committed message.
        let durable_msgs = s.messages.get_history(&thread_a()).await?;
        assert_eq!(durable_msgs.len(), 1);

        // Durable thread aggregate unchanged.
        let thread = s.threads.get(&thread_a()).await?.context("thread")?;
        assert_eq!(thread.committed_turns, 1);

        Ok(())
    }

    // ── Drain semantics ─────────────────────────────────────────

    #[tokio::test]
    async fn drain_yields_staged_data_for_commit() -> Result<()> {
        let s = Stores::new();
        let task = root_task(&thread_a());
        let bootstrap = sample_bootstrap(task);
        let inputs = build_root_worker_inputs(bootstrap, &s.threads, &s.checkpoints, t0())
            .await
            .context("build")?;

        // Simulate turn work.
        inputs
            .staged_stores
            .messages
            .append(&thread_a(), llm::Message::user("hello"))
            .await?;
        inputs
            .staged_stores
            .messages
            .append(&thread_a(), llm::Message::assistant("hi"))
            .await?;

        let mut state = agent_sdk_core::AgentState::new(thread_a());
        state.turn_count = 1;
        inputs.staged_stores.state.save(&state).await?;

        // Drain for commit.
        let msgs = inputs.staged_stores.messages.drain_messages()?;
        assert_eq!(msgs.len(), 2);

        let drained_state = inputs.staged_stores.state.drain_state()?.context("state")?;
        assert_eq!(drained_state.turn_count, 1);

        // After drain, stores are empty.
        let remaining = inputs
            .staged_stores
            .messages
            .get_history(&thread_a())
            .await?;
        assert!(remaining.is_empty());
        let remaining_state = inputs.staged_stores.state.load(&thread_a()).await?;
        assert!(remaining_state.is_none());

        Ok(())
    }

    // ── Thread mismatch guard ────────────────────────────────────

    #[tokio::test]
    async fn build_rejects_thread_mismatch() -> Result<()> {
        let s = Stores::new();

        // Construct a bootstrap where thread_id != task.thread_id.
        let task = AgentTask::new_root_turn(ThreadId::from_string("t-task-thread"), t0(), 3);
        let bootstrap = WorkerBootstrapContext {
            thread_id: ThreadId::from_string("t-bootstrap-thread"),
            task_id: task.id.clone(),
            worker_id: WorkerId::from_string("worker_test"),
            lease_id: LeaseId::from_string("lease_test"),
            definition: sample_definition(),
            task,
        };

        let err = build_root_worker_inputs(bootstrap, &s.threads, &s.checkpoints, t0())
            .await
            .err()
            .expect("should reject mismatched thread_id");
        assert!(
            err.to_string().contains("does not match"),
            "expected mismatch error, got: {err}",
        );

        Ok(())
    }

    // ── Definition is preserved ──────────────────────────────────

    #[tokio::test]
    async fn definition_fields_are_accessible() -> Result<()> {
        let s = Stores::new();
        let task = root_task(&thread_a());

        let def = AgentDefinition {
            provider: "openai".into(),
            model: "gpt-5".into(),
            system_prompt: "test prompt".into(),
            max_tokens: 8192,
            tools: Vec::new(),
            thinking: ThinkingPolicy::Enabled {
                budget_tokens: 1000,
            },
            policy: RuntimePolicy {
                max_attempts: 5,
                strict_durability: false,
                ..RuntimePolicy::server_default()
            },
        };

        let bootstrap = WorkerBootstrapContext {
            thread_id: task.thread_id.clone(),
            task_id: task.id.clone(),
            worker_id: WorkerId::from_string("worker_test"),
            lease_id: LeaseId::from_string("lease_test"),
            definition: def,
            task,
        };

        let inputs = build_root_worker_inputs(bootstrap, &s.threads, &s.checkpoints, t0())
            .await
            .context("build")?;

        assert_eq!(inputs.definition().model, "gpt-5");
        assert_eq!(inputs.definition().system_prompt, "test prompt");
        assert_eq!(inputs.definition().policy.max_attempts, 5);
        assert!(!inputs.definition().policy.strict_durability);

        Ok(())
    }
}
