//! Cross-backend conformance test suite.
//!
//! This module defines backend-agnostic test functions that verify the
//! critical correctness invariants of the journal and checkpoint stores.
//! Each test function accepts trait-object references so they can be
//! invoked against the in-memory, `SQLite`, and `PostgreSQL` backends from
//! the same code path.
//!
//! # Covered invariants
//!
//! 1. **Root admission** — at most one blocking root per thread.
//! 2. **FIFO queueing** — second root on same thread gets `Queued`;
//!    order preserved.
//! 3. **Lease acquisition** — `try_acquire_task` /
//!    `acquire_next_runnable` move `Pending` → `Running` with correct
//!    lease fields.
//! 4. **Heartbeat** — heartbeats extend lease expiry.
//! 5. **Lease expiry sweep** — expired leases are requeued.
//! 6. **Completed-turn commit** — atomic thread advance, message
//!    projection, attempt close, and checkpoint creation.
//! 7. **Child spawn and resume** — parent pauses on children;
//!    completing all children makes parent runnable again.
//! 8. **Cancel tree** — cancellation propagates to all descendants.
//! 9. **Queued root promotion** — completing an active root promotes
//!    the FIFO head.
//! 10. **Retry exhaustion** — budget-exhausted row is failed closed.
//! 11. **Fail-closed child wakes parent** — recovery-path fail-close
//!     propagates to `WaitingOnChildren` parent so it does not stay
//!     blocked forever.
//! 12. **Clear with parent-child** — `clear()` wipes parent/child
//!     chains despite `ON DELETE RESTRICT` self-referential FKs.
//!
//! # Backend-specific differences (documented, not hidden)
//!
//! | Concern | In-memory | SQLite | PostgreSQL |
//! |---------|-----------|--------|------------|
//! | Locking | `RwLock` on `Inner` | database-level (`BEGIN IMMEDIATE`) | row-level (`FOR UPDATE` / `SKIP LOCKED`) |
//! | Concurrency | single-writer, multi-reader | single-writer, multi-reader (WAL) | multi-writer, row-locked |
//! | Persistence | none — lost on restart | file-backed, survives restart | server-backed, survives restart |
//! | `acquire_next_runnable` skip | in-memory index scan | full table scan (OK for local) | `SKIP LOCKED` avoids contention |

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use time::{Duration, OffsetDateTime};

    use agent_sdk_foundation::audit::AuditProvenance;
    use agent_sdk_foundation::events::AgentEvent;
    use agent_sdk_foundation::{ThreadId, TokenUsage};
    use agent_server::journal::checkpoint::CheckpointKind;
    use agent_server::journal::checkpoint_store::CheckpointStore;
    use agent_server::journal::commit::{CompletedTurnCommit, commit_completed_turn};
    use agent_server::journal::event_repository::EventRepository;
    use agent_server::journal::execution_intent::{
        ExecutionIntent, ExecutionIntentStore, IntentStatus, OperationId, ToolEffectClass,
    };
    use agent_server::journal::idempotency::{
        IdempotencyClaim, IdempotencyKind, IdempotencyRecord,
    };
    use agent_server::journal::message_store::MessageProjectionStore;
    use agent_server::journal::outbox::OutboxStore;
    use agent_server::journal::outbox_message::OutboxMessageKind;
    use agent_server::journal::recovery::RecoveryAction;
    use agent_server::journal::store::{AgentTaskStore, SubagentInvocationSpawn};
    use agent_server::journal::task::{
        AgentTask, AgentTaskId, ChildSpawnSpec, LeaseId, SubmittedInputItem, SuspensionPayload,
        TaskStatus, WorkerId,
    };
    use agent_server::journal::task_state::TaskState;
    use agent_server::journal::thread_store::ThreadStore;
    use agent_server::journal::turn_attempt::{
        CloseAttemptParams, OpenAttemptParams, TurnAttemptOutcome,
    };
    use agent_server::journal::turn_attempt_store::TurnAttemptStore;
    use agent_server::worker::subagent::{
        EffectiveSubagentCapabilities, EffectiveSubagentMcpPolicy, EffectiveSubagentSpec,
        InheritedSubagentPolicy, SubagentCapabilityProfile, SubagentSandboxPolicy,
    };

    fn t0() -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
    }

    fn t_plus(secs: i64) -> OffsetDateTime {
        t0() + Duration::seconds(secs)
    }

    fn thread_id(name: &str) -> ThreadId {
        ThreadId::from_string(name)
    }

    fn fresh_root(name: &str, secs: i64) -> AgentTask {
        AgentTask::new_root_turn(thread_id(name), t_plus(secs), 3)
    }

    /// A minimal suspension payload bound to `name`'s thread. Shared by the
    /// child-spawn / cancel-cascade conformance cases so they do not each
    /// hand-roll the full continuation envelope.
    fn suspension_payload(name: &str) -> SuspensionPayload {
        SuspensionPayload {
            continuation: agent_sdk_foundation::ContinuationEnvelope::wrap(
                agent_sdk_foundation::AgentContinuation {
                    thread_id: thread_id(name),
                    turn: 1,
                    total_usage: TokenUsage::default(),
                    turn_usage: TokenUsage::default(),
                    pending_tool_calls: vec![],
                    awaiting_index: 0,
                    completed_results: vec![],
                    state: agent_sdk_foundation::AgentState::new(thread_id(name)),
                    response_id: None,
                    stop_reason: None,
                    response_content: vec![],
                },
            ),
            suspended_messages: vec![],
        }
    }

    // ── Subagent fixtures (cross-thread cancellation / fail-closed) ──

    /// The same effective subagent spec the in-memory store tests use,
    /// replicated here so the durable backends can be driven through the
    /// cross-thread cancellation BFS and the fail-closed invocation-wake
    /// path that previously had **no** SQL-backend coverage.
    fn sample_subagent_spec() -> EffectiveSubagentSpec {
        EffectiveSubagentSpec {
            task: "Inspect durable linkage".into(),
            prompt: "Stay in read-only mode.".into(),
            model: "claude-sonnet-4-5-20250929".into(),
            max_turns: 5,
            timeout_ms: 15_000,
            depth: 1,
            max_parallel_subagents: 1,
            nickname: Some("Scout".into()),
            sandbox: SubagentSandboxPolicy::read_only(),
            mcp: EffectiveSubagentMcpPolicy {
                allowed_servers: std::collections::BTreeSet::from(["docs".to_owned()]),
            },
            audit_provenance: Some(AuditProvenance::new(
                "anthropic",
                "claude-sonnet-4-5-20250929",
            )),
            inherited_policy: InheritedSubagentPolicy {
                default_model: "claude-sonnet-4-5-20250929".into(),
                allowed_models: std::collections::BTreeSet::from([String::from(
                    "claude-sonnet-4-5-20250929",
                )]),
                default_max_turns: 5,
                max_turns: 5,
                default_timeout_ms: 15_000,
                max_timeout_ms: 15_000,
                capability_profiles: std::collections::BTreeMap::from([(
                    "research".into(),
                    SubagentCapabilityProfile {
                        capabilities: ["read_file", "rg"].into_iter().map(str::to_owned).collect(),
                        sandbox: SubagentSandboxPolicy::read_only(),
                        allowed_mcp_servers: std::collections::BTreeSet::from(["docs".to_owned()]),
                    },
                )]),
                allowed_capabilities: ["read_file", "rg"].into_iter().map(str::to_owned).collect(),
                max_depth: 3,
                max_parallel_subagents: 1,
                sandbox: SubagentSandboxPolicy::read_only(),
                allowed_mcp_servers: std::collections::BTreeSet::from(["docs".to_owned()]),
                audit_provider: "anthropic".into(),
            },
            capabilities: EffectiveSubagentCapabilities {
                profile: "research".into(),
                allowed: ["read_file", "rg"].into_iter().map(str::to_owned).collect(),
            },
        }
    }

    fn sample_subagent_input() -> Vec<SubmittedInputItem> {
        vec![SubmittedInputItem::Text {
            text: "Stay in read-only mode.\n\nInspect durable linkage".into(),
        }]
    }

    /// Build a parent root → subagent invocation → child-thread root tree.
    ///
    /// Returns `(parent, invocation, child_root)`. The child thread row is
    /// materialised first so the durable backends satisfy the task→thread
    /// FK when inserting the child root. Takes the thread store separately
    /// because the in-memory backend keeps tasks and threads in distinct
    /// stores (the SQL backends pass the same store for both).
    async fn spawn_subagent_fixture(
        task_store: &dyn AgentTaskStore,
        thread_store: &dyn ThreadStore,
        parent_thread: &str,
    ) -> Result<(AgentTask, AgentTask, AgentTask)> {
        let parent = AgentTask::new_root_turn(thread_id(parent_thread), t0(), 3);
        let parent_id = parent.id.clone();
        task_store.submit_root_turn(parent).await?;
        let worker = WorkerId::new();
        let lease = LeaseId::new();
        task_store
            .try_acquire_task(
                &parent_id,
                worker.clone(),
                lease.clone(),
                t_plus(600),
                t_plus(1),
            )
            .await?
            .context("parent root must acquire before spawning a subagent")?;

        // Durable backends require the child thread row to exist before the
        // child root is inserted (task → thread FK).
        let child_thread = thread_id(&format!("{parent_thread}-child"));
        thread_store.get_or_create(&child_thread, t0()).await?;

        let triple = task_store
            .spawn_subagent_invocation(
                &parent_id,
                &worker,
                &lease,
                SubagentInvocationSpawn {
                    child_thread_id: child_thread,
                    spec: sample_subagent_spec(),
                    child_root_input: sample_subagent_input(),
                    spawn_index: 0,
                    child_caller_metadata: None,
                    payload: suspension_payload(parent_thread),
                },
                t_plus(2),
            )
            .await
            .context("spawn subagent invocation")?;
        Ok(triple)
    }

    /// Acquire `task_id` and let its lease expire `max_attempts` times so
    /// the final sweep fails it closed (`LeaseExpiredBudgetExhausted`).
    /// Returns the records from the final sweep.
    async fn exhaust_retry_budget(
        task_store: &dyn AgentTaskStore,
        task_id: &AgentTaskId,
        max_attempts: u32,
        start_secs: i64,
    ) -> Result<Vec<agent_server::journal::recovery::RecoveryRecord>> {
        let mut last = Vec::new();
        for attempt in 0..max_attempts {
            let offset = i64::from(attempt) * 10;
            task_store
                .try_acquire_task(
                    task_id,
                    WorkerId::new(),
                    LeaseId::new(),
                    t_plus(start_secs + offset + 5),
                    t_plus(start_secs + offset + 1),
                )
                .await?
                .context("claim for retry-budget exhaustion")?;
            last = task_store
                .release_expired_leases(t_plus(start_secs + offset + 6))
                .await?;
        }
        Ok(last)
    }

    // ── Core conformance tests ───────────────────────────────────────

    /// Root admission: submit a root, verify it's Pending.
    async fn test_root_admission(task_store: &dyn AgentTaskStore) -> Result<()> {
        let root = fresh_root("conformance-root-admit", 1);
        let admitted = task_store.submit_root_turn(root.clone()).await?;
        assert_eq!(admitted.status, TaskStatus::Pending);
        assert_eq!(admitted.id, root.id);

        let fetched = task_store
            .get(&root.id)
            .await?
            .context("root should exist")?;
        assert_eq!(fetched.status, TaskStatus::Pending);
        Ok(())
    }

    /// Root FIFO queueing: second root on same thread gets Queued.
    async fn test_root_fifo_queueing(task_store: &dyn AgentTaskStore) -> Result<()> {
        let tid = "conformance-fifo";
        let root1 = fresh_root(tid, 10);
        let root2 = fresh_root(tid, 11);

        let admitted1 = task_store.submit_root_turn(root1).await?;
        assert_eq!(admitted1.status, TaskStatus::Pending);

        let admitted2 = task_store.submit_root_turn(root2).await?;
        assert_eq!(admitted2.status, TaskStatus::Queued);

        let active = task_store
            .active_root_for_thread(&thread_id(tid))
            .await?
            .context("active root missing")?;
        assert_eq!(active.id, admitted1.id);

        let queued = task_store.list_queued_roots(&thread_id(tid)).await?;
        assert_eq!(queued.len(), 1);
        assert_eq!(queued[0].id, admitted2.id);
        Ok(())
    }

    /// Lease acquisition via `try_acquire_task`.
    async fn test_lease_acquisition(task_store: &dyn AgentTaskStore) -> Result<()> {
        let root = fresh_root("conformance-lease", 20);
        let _ = task_store.submit_root_turn(root.clone()).await?;

        let worker = WorkerId::new();
        let lease = LeaseId::new();
        let expires = t_plus(50);

        let acquired = task_store
            .try_acquire_task(&root.id, worker.clone(), lease.clone(), expires, t_plus(21))
            .await?;
        let acquired = acquired.context("should acquire")?;
        assert_eq!(acquired.status, TaskStatus::Running);
        assert_eq!(acquired.worker_id, Some(worker.clone()));
        assert_eq!(acquired.lease_id, Some(lease.clone()));
        assert_eq!(acquired.attempt, 1);

        // Second acquire on the same task should return None.
        let second = task_store
            .try_acquire_task(
                &root.id,
                WorkerId::new(),
                LeaseId::new(),
                expires,
                t_plus(22),
            )
            .await?;
        assert!(second.is_none());
        Ok(())
    }

    /// Heartbeat extends a lease.
    async fn test_heartbeat(task_store: &dyn AgentTaskStore) -> Result<()> {
        let root = fresh_root("conformance-heartbeat", 30);
        let _ = task_store.submit_root_turn(root.clone()).await?;
        let worker = WorkerId::new();
        let lease = LeaseId::new();
        let _ = task_store
            .try_acquire_task(
                &root.id,
                worker.clone(),
                lease.clone(),
                t_plus(40),
                t_plus(31),
            )
            .await?;

        let refreshed = task_store
            .heartbeat_task(&root.id, &worker, &lease, t_plus(60), None, t_plus(35))
            .await?;
        assert_eq!(refreshed.lease_expires_at, Some(t_plus(60)));
        Ok(())
    }

    /// Expired leases are swept and requeued.
    async fn test_lease_expiry_sweep(task_store: &dyn AgentTaskStore) -> Result<()> {
        let root = fresh_root("conformance-expiry", 40);
        let _ = task_store.submit_root_turn(root.clone()).await?;
        let _ = task_store
            .try_acquire_task(
                &root.id,
                WorkerId::new(),
                LeaseId::new(),
                t_plus(42),
                t_plus(41),
            )
            .await?;

        // Sweep after lease expires.
        let records = task_store.release_expired_leases(t_plus(43)).await?;
        assert_eq!(records.len(), 1);
        assert!(matches!(records[0].action, RecoveryAction::Requeue));

        // Task should be back to Pending.
        let task = task_store.get(&root.id).await?.context("task exists")?;
        assert_eq!(task.status, TaskStatus::Pending);
        Ok(())
    }

    /// Worker-initiated requeue of an owned Running row (issue #354):
    /// the collision path's twin of the expiry sweep. Owned rows
    /// return to `Pending` with the lease cleared; a stale lease or a
    /// cancelled row reports `NotOwned` without touching the row; a
    /// budget-exhausted row reports `BudgetExhausted` leaving the row
    /// owned so the caller can run its terminal envelope.
    async fn test_requeue_owned_task(task_store: &dyn AgentTaskStore) -> Result<()> {
        use agent_server::journal::store::RequeueOutcome;

        let root = fresh_root("conformance-requeue", 20);
        let _ = task_store.submit_root_turn(root.clone()).await?;
        let worker = WorkerId::new();
        let lease = LeaseId::new();
        let _ = task_store
            .try_acquire_task(
                &root.id,
                worker.clone(),
                lease.clone(),
                t_plus(50),
                t_plus(21),
            )
            .await?;

        // Stale lease: NotOwned, row untouched.
        let stale = task_store
            .requeue_owned_task(&root.id, &worker, &LeaseId::new(), None, t_plus(22))
            .await?;
        assert!(matches!(stale, RequeueOutcome::NotOwned));
        let row = task_store.get(&root.id).await?.context("row")?;
        assert_eq!(row.status, TaskStatus::Running);

        // Owned: requeued to Pending, lease cleared, re-acquirable.
        let outcome = task_store
            .requeue_owned_task(&root.id, &worker, &lease, None, t_plus(23))
            .await?;
        let RequeueOutcome::Requeued(row) = outcome else {
            anyhow::bail!("expected Requeued, got {outcome:?}");
        };
        assert_eq!(row.status, TaskStatus::Pending);
        assert!(row.worker_id.is_none() && row.lease_id.is_none());
        let reacquired = task_store
            .try_acquire_task(
                &root.id,
                WorkerId::new(),
                LeaseId::new(),
                t_plus(80),
                t_plus(24),
            )
            .await?;
        assert!(reacquired.is_some(), "requeued row must be re-acquirable");

        // Budget-exhausted (max_attempts = 1, attempt consumed at
        // acquire): reported without transitioning.
        let capped =
            AgentTask::new_root_turn(thread_id("conformance-requeue-capped"), t_plus(30), 1);
        let _ = task_store.submit_root_turn(capped.clone()).await?;
        let cw = WorkerId::new();
        let cl = LeaseId::new();
        let _ = task_store
            .try_acquire_task(&capped.id, cw.clone(), cl.clone(), t_plus(60), t_plus(31))
            .await?;
        let outcome = task_store
            .requeue_owned_task(&capped.id, &cw, &cl, None, t_plus(32))
            .await?;
        assert!(matches!(outcome, RequeueOutcome::BudgetExhausted));
        let row = task_store.get(&capped.id).await?.context("capped row")?;
        assert_eq!(row.status, TaskStatus::Running);
        assert_eq!(row.worker_id, Some(cw));

        // Cancelled row: NotOwned even with the original lease — a
        // requeue must never resurrect a cancelled task.
        let doomed = fresh_root("conformance-requeue-cancelled", 40);
        let _ = task_store.submit_root_turn(doomed.clone()).await?;
        let dw = WorkerId::new();
        let dl = LeaseId::new();
        let _ = task_store
            .try_acquire_task(&doomed.id, dw.clone(), dl.clone(), t_plus(90), t_plus(41))
            .await?;
        let _ = task_store.cancel_tree(&doomed.id, t_plus(42)).await?;
        let outcome = task_store
            .requeue_owned_task(&doomed.id, &dw, &dl, None, t_plus(43))
            .await?;
        assert!(matches!(outcome, RequeueOutcome::NotOwned));
        let row = task_store.get(&doomed.id).await?.context("doomed row")?;
        assert_eq!(row.status, TaskStatus::Cancelled);
        Ok(())
    }

    fn evidence_close_params() -> CloseAttemptParams {
        CloseAttemptParams {
            response_blob: serde_json::json!({"id": "msg_conf_1"}),
            response_id: Some("msg_conf_1".into()),
            response_model: Some("claude-sonnet-4-5-20250929".into()),
            stop_reason: Some(agent_sdk_foundation::llm::StopReason::EndTurn),
            outcome: TurnAttemptOutcome::Success,
            input_tokens: 100,
            output_tokens: 50,
            cached_input_tokens: 30,
            cache_creation_input_tokens: 40,
            route_provider: Some("anthropic".into()),
            thinking_mode: Some(
                agent_server::journal::turn_attempt::ThinkingModeEvidence::Adaptive,
            ),
            thinking_budget_tokens: None,
            thinking_effort: Some(agent_sdk_foundation::llm::Effort::XHigh),
        }
    }

    /// Completed-turn commit creates a checkpoint.
    async fn test_completed_turn_commit(
        task_store: &dyn AgentTaskStore,
        thread_store: &dyn ThreadStore,
        message_store: &dyn MessageProjectionStore,
        attempt_store: &dyn TurnAttemptStore,
        checkpoint_store: &dyn CheckpointStore,
        event_repo: &dyn EventRepository,
    ) -> Result<()> {
        let tid = thread_id("conformance-commit");
        let root = AgentTask::new_root_turn(tid.clone(), t_plus(50), 3);
        let _ = task_store.submit_root_turn(root.clone()).await?;
        let worker = WorkerId::new();
        let lease = LeaseId::new();
        let _ = task_store
            .try_acquire_task(&root.id, worker, lease, t_plus(80), t_plus(51))
            .await?;

        let attempt = attempt_store
            .open_attempt(OpenAttemptParams {
                task_id: root.id.clone(),
                attempt_number: 1,
                provenance: AuditProvenance::new("anthropic", "claude-sonnet-4-5-20250929"),
                request_blob: serde_json::json!({"messages": []}),
                now: t_plus(52),
                otel_trace_id: None,
                otel_span_id: None,
            })
            .await?;

        let turn_usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        };
        let outcome = commit_completed_turn(
            CompletedTurnCommit {
                checkpoint_kind: CheckpointKind::FullTurn,
                thread_id: tid.clone(),
                task_id: root.id.clone(),
                expected_turn: 1,
                turn_attempt_id: attempt.id.clone(),
                close_attempt_params: evidence_close_params(),
                messages: vec![agent_sdk_foundation::llm::Message::assistant(
                    "test response",
                )],
                turn_usage,
                agent_state_snapshot: serde_json::json!({}),
                events: vec![],
                outbox_max_attempts: 3,
                owner_guard: None,
                now: t_plus(55),
            },
            thread_store,
            message_store,
            attempt_store,
            checkpoint_store,
            event_repo,
        )
        .await?;

        assert_eq!(outcome.thread.committed_turns, 1);
        assert_eq!(outcome.checkpoint.turn_number, 1);
        assert_eq!(outcome.closed_attempt.cache_creation_input_tokens, Some(40));
        assert_eq!(
            outcome.closed_attempt.route_provider.as_deref(),
            Some("anthropic"),
        );
        assert_eq!(
            outcome.closed_attempt.thinking_mode,
            Some(agent_server::journal::turn_attempt::ThinkingModeEvidence::Adaptive),
            "the thinking mode must survive the close",
        );
        assert_eq!(
            outcome.closed_attempt.thinking_effort,
            Some(agent_sdk_foundation::llm::Effort::XHigh),
        );
        let stored_attempt = attempt_store
            .get(&attempt.id)
            .await?
            .context("committed attempt missing")?;
        assert_eq!(stored_attempt.cache_creation_input_tokens, Some(40));
        assert_eq!(stored_attempt.route_provider.as_deref(), Some("anthropic"));
        assert_eq!(
            stored_attempt.thinking_mode,
            Some(agent_server::journal::turn_attempt::ThinkingModeEvidence::Adaptive),
            "the thinking mode must survive a store round-trip",
        );
        assert_eq!(
            stored_attempt.thinking_effort,
            Some(agent_sdk_foundation::llm::Effort::XHigh),
        );

        let latest = checkpoint_store
            .get_latest_by_thread(&tid)
            .await?
            .context("latest checkpoint missing")?;
        assert_eq!(latest.turn_number, 1);
        Ok(())
    }

    /// Shared plumbing for the owner-guard / typed-stale-commit
    /// conformance cases: submit + acquire a root under a fresh
    /// `(worker, lease)`, open its first attempt, and return a
    /// ready-to-commit [`CompletedTurnCommit`] for turn 1.
    async fn guarded_commit_fixture(
        task_store: &dyn AgentTaskStore,
        attempt_store: &dyn TurnAttemptStore,
        tid: &str,
        base_secs: i64,
    ) -> Result<(AgentTask, WorkerId, LeaseId, CompletedTurnCommit)> {
        let root = AgentTask::new_root_turn(thread_id(tid), t_plus(base_secs), 3);
        let _ = task_store.submit_root_turn(root.clone()).await?;
        let worker = WorkerId::new();
        let lease = LeaseId::new();
        let _ = task_store
            .try_acquire_task(
                &root.id,
                worker.clone(),
                lease.clone(),
                t_plus(base_secs + 60),
                t_plus(base_secs + 1),
            )
            .await?
            .context("acquire root for guarded commit")?;
        let attempt = attempt_store
            .open_attempt(OpenAttemptParams {
                task_id: root.id.clone(),
                attempt_number: 1,
                provenance: AuditProvenance::new("anthropic", "claude-sonnet-4-5-20250929"),
                request_blob: serde_json::json!({"messages": []}),
                now: t_plus(base_secs + 2),
                otel_trace_id: None,
                otel_span_id: None,
            })
            .await?;
        let params = CompletedTurnCommit {
            checkpoint_kind: CheckpointKind::FullTurn,
            thread_id: thread_id(tid),
            task_id: root.id.clone(),
            expected_turn: 1,
            turn_attempt_id: attempt.id,
            close_attempt_params: CloseAttemptParams {
                response_blob: serde_json::json!({"id": "msg_guard"}),
                response_id: Some("msg_guard".into()),
                response_model: Some("claude-sonnet-4-5-20250929".into()),
                stop_reason: Some(agent_sdk_foundation::llm::StopReason::EndTurn),
                outcome: TurnAttemptOutcome::Success,
                input_tokens: 10,
                output_tokens: 5,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
                route_provider: None,
                thinking_mode: None,
                thinking_budget_tokens: None,
                thinking_effort: None,
            },
            messages: vec![agent_sdk_foundation::llm::Message::assistant("guarded")],
            turn_usage: TokenUsage::default(),
            agent_state_snapshot: serde_json::json!({}),
            events: vec![],
            outbox_max_attempts: 3,
            owner_guard: None,
            now: t_plus(base_secs + 3),
        };
        Ok((root, worker, lease, params))
    }

    /// An owner-guarded commit (the slot-shift retry) must be
    /// rejected INSIDE the commit transaction when the
    /// task row is no longer a `Running` row owned by the presenting
    /// `(worker, lease)` — here, a cancellation that landed between
    /// the caller's shift-eligibility re-read and the retried commit.
    /// Nothing may be written (no thread advance, no checkpoint), and
    /// the rejection carries the typed [`LostCommitOwnership`] root
    /// cause so the shift loop treats it as terminal instead of
    /// shifting again. Durable backends only: the non-atomic
    /// in-memory backend has no cross-store transaction to enforce
    /// the guard in (documented on `CompletedTurnCommit::owner_guard`).
    async fn test_owner_guarded_commit_rejects_lost_ownership(
        task_store: &dyn AgentTaskStore,
        thread_store: &dyn ThreadStore,
        message_store: &dyn MessageProjectionStore,
        attempt_store: &dyn TurnAttemptStore,
        checkpoint_store: &dyn CheckpointStore,
        event_repo: &dyn EventRepository,
    ) -> Result<()> {
        use agent_server::journal::commit::{CommitOwnerGuard, LostCommitOwnership};

        let (root, worker, lease, mut params) =
            guarded_commit_fixture(task_store, attempt_store, "conformance-owner-guard", 500)
                .await?;
        params.owner_guard = Some(CommitOwnerGuard {
            worker_id: worker.clone(),
            lease_id: lease.clone(),
        });

        // The cancel lands between the caller's eligibility check and
        // the retried commit.
        let cancelled = task_store.cancel_tree(&root.id, t_plus(504)).await?;
        assert_eq!(cancelled.transitioned, vec![root.id.clone()]);

        let error = commit_completed_turn(
            params,
            thread_store,
            message_store,
            attempt_store,
            checkpoint_store,
            event_repo,
        )
        .await
        .err()
        .context("owner-guarded commit against a cancelled row must be rejected")?;
        assert!(
            error.downcast_ref::<LostCommitOwnership>().is_some(),
            "expected the typed LostCommitOwnership root cause, got: {error:#}",
        );

        // The rejected transaction wrote nothing.
        let thread = thread_store
            .get(&thread_id("conformance-owner-guard"))
            .await?;
        assert_eq!(
            thread.map_or(0, |thread| thread.committed_turns),
            0,
            "rejected owner-guarded commit must not advance the thread",
        );
        assert!(
            checkpoint_store
                .get_by_turn(&thread_id("conformance-owner-guard"), 1)
                .await?
                .is_none(),
            "rejected owner-guarded commit must not write a checkpoint",
        );
        Ok(())
    }

    /// Owner-guard positive arm: an owner-guarded commit whose worker
    /// still owns the live `Running` row succeeds — the guard only
    /// bites on lost ownership.
    async fn test_owner_guarded_commit_succeeds_while_owned(
        task_store: &dyn AgentTaskStore,
        thread_store: &dyn ThreadStore,
        message_store: &dyn MessageProjectionStore,
        attempt_store: &dyn TurnAttemptStore,
        checkpoint_store: &dyn CheckpointStore,
        event_repo: &dyn EventRepository,
    ) -> Result<()> {
        use agent_server::journal::commit::CommitOwnerGuard;

        let (_root, worker, lease, mut params) =
            guarded_commit_fixture(task_store, attempt_store, "conformance-owner-guard-ok", 520)
                .await?;
        params.owner_guard = Some(CommitOwnerGuard {
            worker_id: worker,
            lease_id: lease,
        });

        let outcome = commit_completed_turn(
            params,
            thread_store,
            message_store,
            attempt_store,
            checkpoint_store,
            event_repo,
        )
        .await
        .context("owner-guarded commit while still owned")?;
        assert_eq!(outcome.thread.committed_turns, 1);
        assert_eq!(outcome.checkpoint.turn_number, 1);
        Ok(())
    }

    /// The completed-turn slot CAS rejection carries the typed
    /// [`StaleTurnCommit`] root cause on every backend, so the
    /// worker's slot-shift path can downcast instead of matching
    /// message strings (the `Display` text stays byte-identical to the
    /// historical message for logs).
    async fn test_stale_turn_commit_rejection_is_typed(
        task_store: &dyn AgentTaskStore,
        thread_store: &dyn ThreadStore,
        message_store: &dyn MessageProjectionStore,
        attempt_store: &dyn TurnAttemptStore,
        checkpoint_store: &dyn CheckpointStore,
        event_repo: &dyn EventRepository,
    ) -> Result<()> {
        use agent_server::journal::commit::StaleTurnCommit;

        let (root, _worker, _lease, params) =
            guarded_commit_fixture(task_store, attempt_store, "conformance-stale-typed", 540)
                .await?;
        commit_completed_turn(
            params,
            thread_store,
            message_store,
            attempt_store,
            checkpoint_store,
            event_repo,
        )
        .await
        .context("first commit of turn 1")?;

        // A second commit of the SAME expected turn (fresh attempt,
        // same slot) must fail the CAS with the typed root cause.
        let attempt = attempt_store
            .open_attempt(OpenAttemptParams {
                task_id: root.id.clone(),
                attempt_number: 2,
                provenance: AuditProvenance::new("anthropic", "claude-sonnet-4-5-20250929"),
                request_blob: serde_json::json!({"messages": []}),
                now: t_plus(546),
                otel_trace_id: None,
                otel_span_id: None,
            })
            .await?;
        let stale = CompletedTurnCommit {
            checkpoint_kind: CheckpointKind::FullTurn,
            thread_id: thread_id("conformance-stale-typed"),
            task_id: root.id.clone(),
            expected_turn: 1,
            turn_attempt_id: attempt.id,
            close_attempt_params: CloseAttemptParams {
                response_blob: serde_json::json!({"id": "msg_stale"}),
                response_id: Some("msg_stale".into()),
                response_model: Some("claude-sonnet-4-5-20250929".into()),
                stop_reason: Some(agent_sdk_foundation::llm::StopReason::EndTurn),
                outcome: TurnAttemptOutcome::Success,
                input_tokens: 1,
                output_tokens: 1,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
                route_provider: None,
                thinking_mode: None,
                thinking_budget_tokens: None,
                thinking_effort: None,
            },
            messages: vec![agent_sdk_foundation::llm::Message::assistant("stale")],
            turn_usage: TokenUsage::default(),
            agent_state_snapshot: serde_json::json!({}),
            events: vec![],
            outbox_max_attempts: 3,
            owner_guard: None,
            now: t_plus(547),
        };
        let error = commit_completed_turn(
            stale,
            thread_store,
            message_store,
            attempt_store,
            checkpoint_store,
            event_repo,
        )
        .await
        .err()
        .context("second commit of the same slot must be rejected")?;
        let typed = error
            .downcast_ref::<StaleTurnCommit>()
            .context("rejection must carry the typed StaleTurnCommit root cause")?;
        assert_eq!(typed.expected_turn, 1);
        assert_eq!(typed.committed_turns, 1);
        assert!(
            format!("{error:#}").contains("stale turn commit"),
            "Display text must stay stable, got: {error:#}",
        );
        Ok(())
    }

    /// Child spawn + complete resumes parent.
    async fn test_child_spawn_and_resume(task_store: &dyn AgentTaskStore) -> Result<()> {
        let root = fresh_root("conformance-children", 60);
        let _ = task_store.submit_root_turn(root.clone()).await?;
        let worker = WorkerId::new();
        let lease = LeaseId::new();
        let _ = task_store
            .try_acquire_task(
                &root.id,
                worker.clone(),
                lease.clone(),
                t_plus(90),
                t_plus(61),
            )
            .await?;

        let spec = ChildSpawnSpec { max_attempts: 3 };
        let payload = SuspensionPayload {
            continuation: agent_sdk_foundation::ContinuationEnvelope::wrap(
                agent_sdk_foundation::AgentContinuation {
                    thread_id: thread_id("conformance-children"),
                    turn: 1,
                    total_usage: TokenUsage {
                        input_tokens: 0,
                        output_tokens: 0,
                        ..Default::default()
                    },
                    turn_usage: TokenUsage {
                        input_tokens: 0,
                        output_tokens: 0,
                        ..Default::default()
                    },
                    pending_tool_calls: vec![],
                    awaiting_index: 0,
                    completed_results: vec![],
                    state: agent_sdk_foundation::AgentState::new(thread_id("conformance-children")),
                    response_id: None,
                    stop_reason: None,
                    response_content: vec![],
                },
            ),
            suspended_messages: vec![],
        };
        let (parent, children) = task_store
            .spawn_tool_children(
                &root.id,
                &worker,
                &lease,
                vec![spec],
                payload,
                None,
                t_plus(62),
            )
            .await?;
        assert_eq!(parent.status, TaskStatus::WaitingOnChildren);
        assert_eq!(children.len(), 1);

        // Acquire and complete the child.
        let child = &children[0];
        let cw = WorkerId::new();
        let cl = LeaseId::new();
        let _ = task_store
            .try_acquire_task(&child.id, cw.clone(), cl.clone(), t_plus(90), t_plus(63))
            .await?;
        let (completed_child, resumed_parent) = task_store
            .complete_task(&child.id, &cw, &cl, t_plus(64))
            .await?;
        assert_eq!(completed_child.status, TaskStatus::Completed);
        let resumed = resumed_parent.context("parent should be returned")?;
        assert_eq!(resumed.status, TaskStatus::Pending);
        Ok(())
    }

    /// R2 steering wake: `enqueue_steering_resume` parks the parent in a
    /// steering `ready_to_resume` state, `repark_after_steering` re-binds
    /// the survivor under a dense `spawn_index` and re-parks in
    /// `waiting_on_children`, and the survivor's eventual completion fans
    /// the parent back in — budget-neutral, no new children. Exercised on
    /// every backend so the SQL child re-index path is covered on the
    /// durable stores.
    async fn test_steering_wake_and_repark(task_store: &dyn AgentTaskStore) -> Result<()> {
        let tid = "conformance-steering";
        let root = fresh_root(tid, 60);
        task_store.submit_root_turn(root.clone()).await?;
        let worker = WorkerId::new();
        let lease = LeaseId::new();
        task_store
            .try_acquire_task(
                &root.id,
                worker.clone(),
                lease.clone(),
                t_plus(90),
                t_plus(61),
            )
            .await?;

        let (parent, children) = task_store
            .spawn_tool_children(
                &root.id,
                &worker,
                &lease,
                vec![
                    ChildSpawnSpec { max_attempts: 3 },
                    ChildSpawnSpec { max_attempts: 3 },
                ],
                suspension_payload(tid),
                None,
                t_plus(62),
            )
            .await?;
        assert_eq!(parent.status, TaskStatus::WaitingOnChildren);
        assert_eq!(children.len(), 2);

        // Child 0 finishes during the wave; child 1 keeps running.
        let cw = WorkerId::new();
        let cl = LeaseId::new();
        task_store
            .try_acquire_task(
                &children[0].id,
                cw.clone(),
                cl.clone(),
                t_plus(90),
                t_plus(63),
            )
            .await?;
        task_store
            .complete_task(&children[0].id, &cw, &cl, t_plus(64))
            .await?;

        // Wake the parked parent.
        let note = vec![agent_sdk_foundation::llm::ContentBlock::Text {
            text: "status?".into(),
        }];
        let woken = task_store
            .enqueue_steering_resume(&root.id, note, t_plus(65))
            .await?
            .context("wake must succeed on a parked parent")?;
        assert_eq!(woken.status, TaskStatus::Pending);
        assert!(woken.state.is_steering_resume());

        // Acquire and re-park on the survivor under a fresh binding.
        let sw = WorkerId::new();
        let sl = LeaseId::new();
        task_store
            .try_acquire_task(&root.id, sw.clone(), sl.clone(), t_plus(200), t_plus(67))
            .await?;
        let reparked = task_store
            .repark_after_steering(
                &root.id,
                &sw,
                &sl,
                suspension_payload(tid),
                vec![children[1].id.clone()],
                t_plus(68),
            )
            .await?;
        assert_eq!(reparked.status, TaskStatus::WaitingOnChildren);
        assert_eq!(reparked.pending_child_count, 1);
        match &reparked.state {
            TaskState::WaitingOnChildren { child_ids, .. } => {
                assert_eq!(child_ids, &vec![children[1].id.clone()]);
            }
            other => panic!("expected WaitingOnChildren re-park, got {other:?}"),
        }
        // Survivor re-indexed to dense position 0; still non-terminal.
        let survivor = task_store.get(&children[1].id).await?.context("survivor")?;
        assert_eq!(survivor.spawn_index, Some(0));
        assert!(!survivor.status.is_terminal());

        // Survivor finishes → the parent fans in to ReadyToResume.
        let c2w = WorkerId::new();
        let c2l = LeaseId::new();
        task_store
            .try_acquire_task(
                &children[1].id,
                c2w.clone(),
                c2l.clone(),
                t_plus(200),
                t_plus(70),
            )
            .await?;
        let (_done, parent_after) = task_store
            .complete_task(&children[1].id, &c2w, &c2l, t_plus(71))
            .await?;
        let ready = parent_after.context("parent returned on final fan-in")?;
        assert_eq!(ready.status, TaskStatus::Pending);
        assert!(matches!(ready.state, TaskState::ReadyToResume { .. }));
        Ok(())
    }

    /// Cancel tree cancels the root **and all its descendants** (invariant
    /// #8). The root spawns two tool children (so the parent parks on
    /// `WaitingOnChildren`) before the cancel, and the cascade must drive
    /// the root and both children to `Cancelled` — exercising the
    /// descendant BFS that previously ran only in production on the SQL
    /// backends (finding #5: the old test cancelled a childless root).
    async fn test_cancel_tree(task_store: &dyn AgentTaskStore) -> Result<()> {
        let tid = "conformance-cancel";
        let root = fresh_root(tid, 70);
        let _ = task_store.submit_root_turn(root.clone()).await?;
        let worker = WorkerId::new();
        let lease = LeaseId::new();
        let _ = task_store
            .try_acquire_task(
                &root.id,
                worker.clone(),
                lease.clone(),
                t_plus(100),
                t_plus(71),
            )
            .await?;

        // Spawn two tool children so the cancel has a real subtree to walk.
        let (parent, children) = task_store
            .spawn_tool_children(
                &root.id,
                &worker,
                &lease,
                vec![
                    ChildSpawnSpec { max_attempts: 3 },
                    ChildSpawnSpec { max_attempts: 3 },
                ],
                suspension_payload(tid),
                None,
                t_plus(72),
            )
            .await?;
        assert_eq!(parent.status, TaskStatus::WaitingOnChildren);
        assert_eq!(children.len(), 2);

        let cancelled = task_store
            .cancel_tree(&root.id, t_plus(73))
            .await?
            .transitioned;
        // Root + both children must all be transitioned.
        assert_eq!(
            cancelled.len(),
            3,
            "cancel must cascade to the root and both tool children, got {cancelled:?}",
        );
        assert!(cancelled.contains(&root.id));
        for child in &children {
            assert!(
                cancelled.contains(&child.id),
                "child {} must be in the cancelled set",
                child.id,
            );
        }

        // Every node is durably Cancelled.
        let root_after = task_store.get(&root.id).await?.context("root exists")?;
        assert_eq!(root_after.status, TaskStatus::Cancelled);
        for child in &children {
            let child_after = task_store
                .get(&child.id)
                .await?
                .with_context(|| format!("child {} exists", child.id))?;
            assert_eq!(
                child_after.status,
                TaskStatus::Cancelled,
                "descendant {} must land Cancelled",
                child.id,
            );
        }
        Ok(())
    }

    /// Cross-thread cancel cascade through a subagent invocation
    /// (finding #5). Topology: parent root → subagent invocation →
    /// child-thread root. `cancel_tree(parent)` must follow the durable
    /// linkage across the thread boundary and cancel all three. This runs
    /// the cross-thread BFS on the in-memory, `SQLite`, and `PostgreSQL`
    /// backends (previously SQL-untested).
    async fn test_cancel_tree_cascades_through_subagent(
        task_store: &dyn AgentTaskStore,
        thread_store: &dyn ThreadStore,
    ) -> Result<()> {
        let (parent, invocation, child_root) =
            spawn_subagent_fixture(task_store, thread_store, "conformance-subagent-cascade")
                .await?;

        let cancelled = task_store
            .cancel_tree(&parent.id, t_plus(10))
            .await?
            .transitioned;
        assert_eq!(
            cancelled.len(),
            3,
            "cancel must cascade parent → invocation → child root, got {cancelled:?}",
        );
        for id in [&parent.id, &invocation.id, &child_root.id] {
            assert!(cancelled.contains(id), "{id} must be in the cancelled set");
            let task = task_store
                .get(id)
                .await?
                .with_context(|| format!("{id} exists"))?;
            assert_eq!(task.status, TaskStatus::Cancelled, "{id} must be Cancelled");
        }
        Ok(())
    }

    /// Count the `AgentEvent::Cancelled` markers committed on `thread`.
    async fn cancelled_marker_count(
        event_repo: &dyn EventRepository,
        thread: &ThreadId,
    ) -> Result<usize> {
        Ok(event_repo
            .get_events(thread)
            .await?
            .iter()
            .filter(|committed| matches!(committed.event, AgentEvent::Cancelled { .. }))
            .count())
    }

    /// Terminal-marker contract (issue #354), part 1: cancelling the
    /// thread's blocking root commits exactly ONE durable `Cancelled`
    /// marker plus its `thread_events_available` outbox advisory,
    /// atomically with the cancellation — and a crash-retry (the
    /// caller "dies" right after `cancel_tree` commits, then retries)
    /// finds the marker already durable and emits nothing new.
    async fn test_cancel_marker_exactly_once_and_crash_retry_idempotent(
        task_store: &dyn AgentTaskStore,
        event_repo: &dyn EventRepository,
        outbox_store: &dyn OutboxStore,
    ) -> Result<()> {
        let tid = "conformance-cancel-marker";
        let root = fresh_root(tid, 300);
        let _ = task_store.submit_root_turn(root.clone()).await?;

        let outcome = task_store.cancel_tree(&root.id, t_plus(301)).await?;
        assert_eq!(outcome.transitioned, vec![root.id.clone()]);
        assert_eq!(
            outcome.markers.len(),
            1,
            "one marker for the cancelled blocking root",
        );
        let marker = &outcome.markers[0];
        assert_eq!(marker.thread_id, thread_id(tid));
        assert!(
            matches!(marker.event, AgentEvent::Cancelled { .. }),
            "marker must be a Cancelled event, got {:?}",
            marker.event,
        );

        // Durable in the committed-event journal (a follower replaying
        // the thread sees the closing frame even if every process died
        // the instant `cancel_tree` returned).
        assert_eq!(
            cancelled_marker_count(event_repo, &thread_id(tid)).await?,
            1
        );

        // The advisory outbox row committed with the marker (cross-host
        // followers are woken through the relay, not a process-local
        // notifier).
        let claimed = outbox_store
            .claim_pending("conformance-marker-relay", 32, t_plus(302))
            .await?;
        let advisories: Vec<_> = claimed
            .iter()
            .filter(|row| {
                row.kind == OutboxMessageKind::ThreadEventsAvailable
                    && row.thread_id == thread_id(tid)
            })
            .collect();
        assert_eq!(
            advisories.len(),
            1,
            "exactly one thread_events_available advisory for the marker",
        );
        assert_eq!(advisories[0].sequence, Some(marker.sequence));

        // Crash-retry: the retry sees a terminal tree — nothing
        // transitions, no duplicate marker, no second advisory.
        let retry = task_store.cancel_tree(&root.id, t_plus(303)).await?;
        assert!(retry.transitioned.is_empty(), "retry must be a no-op");
        assert!(retry.markers.is_empty(), "retry must not re-emit markers");
        assert_eq!(
            cancelled_marker_count(event_repo, &thread_id(tid)).await?,
            1
        );
        let re_claimed = outbox_store
            .claim_pending("conformance-marker-relay", 32, t_plus(304))
            .await?;
        assert!(
            re_claimed
                .iter()
                .all(|row| row.kind != OutboxMessageKind::ThreadEventsAvailable),
            "retry must not write a second advisory, got {re_claimed:?}",
        );
        Ok(())
    }

    /// Terminal-marker contract (issue #354), part 2: a cancel that
    /// cascades across `SubagentInvocation` linkage commits one marker
    /// per affected thread — the parent root's thread AND the
    /// child-thread root's own thread (whose followers previously hung
    /// forever). The invocation task itself (kind `subagent`, parent
    /// thread) adds no extra marker.
    async fn test_cancel_marker_covers_child_threads(
        task_store: &dyn AgentTaskStore,
        thread_store: &dyn ThreadStore,
        event_repo: &dyn EventRepository,
    ) -> Result<()> {
        let (parent, _invocation, child_root) =
            spawn_subagent_fixture(task_store, thread_store, "conformance-marker-cascade").await?;

        let outcome = task_store.cancel_tree(&parent.id, t_plus(10)).await?;
        assert_eq!(outcome.transitioned.len(), 3);
        assert_eq!(
            outcome.markers.len(),
            2,
            "parent root and child-thread root each get a marker, got {:?}",
            outcome.markers,
        );
        for thread in [&parent.thread_id, &child_root.thread_id] {
            assert_eq!(
                cancelled_marker_count(event_repo, thread).await?,
                1,
                "exactly one marker on thread {thread}",
            );
        }
        Ok(())
    }

    /// Terminal-marker contract (issue #354), race arm: two RACING
    /// `cancel_tree` calls on the same blocking root must produce
    /// exactly ONE durable marker, ONE outbox advisory, and report the
    /// transition exactly once across both outcomes. On Postgres the
    /// transition must be gated on a locked re-read (not the
    /// non-locking snapshot), otherwise both racers emit markers and
    /// both report `transitioned = 1`. The two calls run on separate
    /// spawned tasks (separate pool connections on the SQL backends)
    /// released by a barrier, repeated across rounds to give the
    /// interleave a real chance to bite.
    async fn test_concurrent_cancels_emit_single_marker(
        task_store: std::sync::Arc<dyn AgentTaskStore>,
        event_repo: &dyn EventRepository,
        outbox_store: &dyn OutboxStore,
    ) -> Result<()> {
        for round in 0..12i64 {
            let tid = format!("conformance-cancel-race-{round}");
            let root = AgentTask::new_root_turn(thread_id(&tid), t_plus(400 + round * 4), 3);
            let _ = task_store.submit_root_turn(root.clone()).await?;

            let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(2));
            let spawn_cancel = |store: std::sync::Arc<dyn AgentTaskStore>,
                                id: AgentTaskId,
                                gate: std::sync::Arc<tokio::sync::Barrier>,
                                at: OffsetDateTime| {
                tokio::spawn(async move {
                    gate.wait().await;
                    store.cancel_tree(&id, at).await
                })
            };
            let first = spawn_cancel(
                std::sync::Arc::clone(&task_store),
                root.id.clone(),
                std::sync::Arc::clone(&barrier),
                t_plus(401 + round * 4),
            );
            let second = spawn_cancel(
                std::sync::Arc::clone(&task_store),
                root.id.clone(),
                barrier,
                t_plus(401 + round * 4),
            );
            let first = first.await.context("join first racing cancel")??;
            let second = second.await.context("join second racing cancel")??;

            assert_eq!(
                first.transitioned.len() + second.transitioned.len(),
                1,
                "round {round}: exactly one racer may report the transition, got {:?} / {:?}",
                first.transitioned,
                second.transitioned,
            );
            assert_eq!(
                first.markers.len() + second.markers.len(),
                1,
                "round {round}: exactly one marker across both racers",
            );
            assert_eq!(
                cancelled_marker_count(event_repo, &thread_id(&tid)).await?,
                1,
                "round {round}: exactly one durable Cancelled marker",
            );
            let claimed = outbox_store
                .claim_pending("conformance-race-relay", 64, t_plus(403 + round * 4))
                .await?;
            let advisories = claimed
                .iter()
                .filter(|row| {
                    row.kind == OutboxMessageKind::ThreadEventsAvailable
                        && row.thread_id == thread_id(&tid)
                })
                .count();
            assert_eq!(
                advisories, 1,
                "round {round}: exactly one thread_events_available advisory",
            );
        }
        Ok(())
    }

    /// Terminal-marker contract (issue #354), part 3: cancelling a
    /// QUEUED root parked behind a live active root emits NO marker
    /// and no advisory — a thread-terminal frame would close the
    /// active root's followers mid-stream.
    async fn test_queued_root_cancel_emits_no_marker(
        task_store: &dyn AgentTaskStore,
        event_repo: &dyn EventRepository,
        outbox_store: &dyn OutboxStore,
    ) -> Result<()> {
        let tid = "conformance-queued-no-marker";
        let active = fresh_root(tid, 320);
        let active_admitted = task_store.submit_root_turn(active).await?;
        assert_eq!(active_admitted.status, TaskStatus::Pending);
        let queued = fresh_root(tid, 321);
        let queued_admitted = task_store.submit_root_turn(queued).await?;
        assert_eq!(queued_admitted.status, TaskStatus::Queued);

        let outcome = task_store
            .cancel_tree(&queued_admitted.id, t_plus(322))
            .await?;
        assert_eq!(outcome.transitioned, vec![queued_admitted.id.clone()]);
        assert!(
            outcome.markers.is_empty(),
            "queued-behind-active cancel must not emit a thread-terminal marker",
        );
        assert_eq!(
            cancelled_marker_count(event_repo, &thread_id(tid)).await?,
            0
        );
        let claimed = outbox_store
            .claim_pending("conformance-queued-relay", 32, t_plus(323))
            .await?;
        assert!(
            claimed
                .iter()
                .all(|row| row.kind != OutboxMessageKind::ThreadEventsAvailable),
            "no thread_events_available advisory may exist for a queued cancel, got {claimed:?}",
        );

        // The active root is untouched.
        let active_after = task_store
            .get(&active_admitted.id)
            .await?
            .context("active root exists")?;
        assert_eq!(active_after.status, TaskStatus::Pending);
        Ok(())
    }

    /// Cancelling a child-thread root must **wake** its linked parent
    /// subagent invocation (finding #7). Cancelling only the child root
    /// resumes the invocation to `Pending` with no pending children, so the
    /// parent does not stay `WaitingOnChildren` forever — on every backend.
    async fn test_cancel_child_root_wakes_invocation(
        task_store: &dyn AgentTaskStore,
        thread_store: &dyn ThreadStore,
    ) -> Result<()> {
        let (_parent, invocation, child_root) =
            spawn_subagent_fixture(task_store, thread_store, "conformance-subagent-wake").await?;

        let cancelled = task_store
            .cancel_tree(&child_root.id, t_plus(10))
            .await?
            .transitioned;
        assert_eq!(cancelled, vec![child_root.id.clone()]);

        let woken = task_store
            .get(&invocation.id)
            .await?
            .context("invocation exists after child-root cancel")?;
        assert_eq!(
            woken.status,
            TaskStatus::Pending,
            "the linked subagent invocation must wake when its child root is cancelled",
        );
        assert_eq!(woken.pending_child_count, 0);
        Ok(())
    }

    /// A child-thread root that **fails closed** (lease expired, retry
    /// budget exhausted) must wake its linked parent subagent invocation
    /// (finding #7 — the headline durable-liveness gap on the recovery
    /// path). Previously only the in-memory store had this branch; the SQL
    /// backends left the parent invocation `WaitingOnChildren` forever.
    async fn test_fail_closed_child_root_wakes_invocation(
        task_store: &dyn AgentTaskStore,
        thread_store: &dyn ThreadStore,
    ) -> Result<()> {
        let (_parent, invocation, child_root) =
            spawn_subagent_fixture(task_store, thread_store, "conformance-subagent-failclosed")
                .await?;

        // Exhaust the child root's retry budget so its final lease-expiry
        // sweep fails it closed.
        let records =
            exhaust_retry_budget(task_store, &child_root.id, child_root.max_attempts, 5).await?;
        assert!(
            records.iter().any(|r| r.id == child_root.id
                && matches!(r.action, RecoveryAction::FailClosed(_))),
            "the child root must fail closed once its budget is exhausted, got {records:?}",
        );

        let child_after = task_store
            .get(&child_root.id)
            .await?
            .context("child root exists")?;
        assert_eq!(child_after.status, TaskStatus::Failed);

        let woken = task_store
            .get(&invocation.id)
            .await?
            .context("invocation exists after child-root fail-closed")?;
        assert_eq!(
            woken.status,
            TaskStatus::Pending,
            "a fail-closed child-thread root must wake the linked parent invocation",
        );
        assert_eq!(woken.pending_child_count, 0);
        Ok(())
    }

    /// Reverse linkage read (issue #299): while a subagent child-thread
    /// root is live, `find_subagent_invocation_for_child_root` must
    /// return the parked invocation carrying the resolved spec (the
    /// host reads `spec.timeout_ms` off it to derive the child's
    /// wall-clock deadline). Unlinked ids resolve to `None`, and the
    /// window closes once the child root goes terminal and the
    /// invocation has been woken — on every backend.
    async fn test_find_invocation_for_child_root(
        task_store: &dyn AgentTaskStore,
        thread_store: &dyn ThreadStore,
    ) -> Result<()> {
        let (parent, invocation, child_root) =
            spawn_subagent_fixture(task_store, thread_store, "conformance-subagent-linkage")
                .await?;

        let found = task_store
            .find_subagent_invocation_for_child_root(&child_root.id)
            .await?
            .context("linkage lookup must find the parked invocation while the child is live")?;
        assert_eq!(found.id, invocation.id);
        let spec_timeout_ms = found
            .state
            .subagent_invocation()
            .context("found invocation must carry the durable spec linkage")?
            .spec
            .timeout_ms;
        assert_eq!(
            spec_timeout_ms, 15_000,
            "the resolved spec timeout must be readable through the linkage",
        );

        // The parent root is not a linked child root.
        assert!(
            task_store
                .find_subagent_invocation_for_child_root(&parent.id)
                .await?
                .is_none(),
            "unlinked task ids must resolve to None",
        );

        // Terminal child root wakes the invocation and consumes the
        // linkage window.
        task_store.cancel_tree(&child_root.id, t_plus(10)).await?;
        assert!(
            task_store
                .find_subagent_invocation_for_child_root(&child_root.id)
                .await?
                .is_none(),
            "the linkage window must close once the invocation has been woken",
        );
        Ok(())
    }

    /// The deadline sweep's candidate feed (issue #299 round 3):
    /// `list_parked_subagent_invocations` must return exactly the
    /// parked `Subagent` invocations — a plain root parked on tool
    /// children must NOT be materialized — and the window closes once
    /// the linked child root goes terminal and wakes the invocation.
    async fn test_list_parked_subagent_invocations(
        task_store: &dyn AgentTaskStore,
        thread_store: &dyn ThreadStore,
    ) -> Result<()> {
        // A plain root parked on a tool child: same status, wrong kind.
        let tid = "conformance-parked-listing-plain";
        let plain = fresh_root(tid, 1);
        let plain_id = plain.id.clone();
        task_store.submit_root_turn(plain).await?;
        let worker = WorkerId::new();
        let lease = LeaseId::new();
        task_store
            .try_acquire_task(
                &plain_id,
                worker.clone(),
                lease.clone(),
                t_plus(600),
                t_plus(2),
            )
            .await?
            .context("plain parent must acquire before spawning children")?;
        let (parked_plain, _children) = task_store
            .spawn_tool_children(
                &plain_id,
                &worker,
                &lease,
                vec![ChildSpawnSpec { max_attempts: 3 }],
                suspension_payload(tid),
                None,
                t_plus(3),
            )
            .await?;
        assert_eq!(parked_plain.status, TaskStatus::WaitingOnChildren);

        // A parked subagent invocation: the one row the sweep wants.
        let (_parent, invocation, child_root) =
            spawn_subagent_fixture(task_store, thread_store, "conformance-parked-listing").await?;

        let parked = task_store.list_parked_subagent_invocations().await?;
        assert_eq!(
            parked.len(),
            1,
            "only the subagent invocation may be listed (no plain parked parents), got {parked:?}",
        );
        let listed = &parked[0];
        assert_eq!(listed.id, invocation.id);
        assert!(
            listed.state.subagent_invocation().is_some(),
            "the listed row must carry its durable spec linkage",
        );

        // Terminal child root wakes the invocation and closes the window.
        task_store.cancel_tree(&child_root.id, t_plus(10)).await?;
        assert!(
            task_store
                .list_parked_subagent_invocations()
                .await?
                .is_empty(),
            "a woken invocation must no longer be listed",
        );
        Ok(())
    }

    /// Mid-tree cancel must resume a parked parent OUTSIDE the
    /// cancelled subtree (issue #299, parked leg). The host's subagent
    /// deadline sweep cancels a parked root's hung tool children one
    /// subtree at a time; when the last live child cancels, the parked
    /// `WaitingOnChildren` parent must flip back to `Pending` — on
    /// every backend. The in-memory store already propagated this
    /// through its per-row cancel; the SQL backends previously only
    /// handled parents INSIDE the cancelled subtree, leaving an
    /// out-of-subtree parent parked forever.
    async fn test_cancel_tool_child_resumes_parked_parent(
        task_store: &dyn AgentTaskStore,
    ) -> Result<()> {
        let tid = "conformance-midtree-cancel";
        let parent = fresh_root(tid, 1);
        let parent_id = parent.id.clone();
        task_store.submit_root_turn(parent).await?;
        let worker = WorkerId::new();
        let lease = LeaseId::new();
        task_store
            .try_acquire_task(
                &parent_id,
                worker.clone(),
                lease.clone(),
                t_plus(600),
                t_plus(2),
            )
            .await?
            .context("parent must acquire before spawning children")?;
        let (parked, children) = task_store
            .spawn_tool_children(
                &parent_id,
                &worker,
                &lease,
                vec![
                    ChildSpawnSpec { max_attempts: 3 },
                    ChildSpawnSpec { max_attempts: 3 },
                ],
                suspension_payload(tid),
                None,
                t_plus(3),
            )
            .await?;
        assert_eq!(parked.status, TaskStatus::WaitingOnChildren);
        let [first, second] = children.as_slice() else {
            anyhow::bail!("expected exactly two spawned children, got {children:?}");
        };

        // Cancelling one child leaves the parent parked on the other.
        task_store.cancel_tree(&first.id, t_plus(4)).await?;
        let still_parked = task_store
            .get(&parent_id)
            .await?
            .context("parent exists after first child cancel")?;
        assert_eq!(
            still_parked.status,
            TaskStatus::WaitingOnChildren,
            "one live child must keep the parent parked",
        );
        assert_eq!(still_parked.pending_child_count, 1);

        // Cancelling the LAST live child must resume the parked parent.
        task_store.cancel_tree(&second.id, t_plus(5)).await?;
        let resumed = task_store
            .get(&parent_id)
            .await?
            .context("parent exists after last child cancel")?;
        assert_eq!(
            resumed.status,
            TaskStatus::Pending,
            "cancelling the last live child must flip the out-of-subtree parked parent to Pending",
        );
        assert_eq!(resumed.pending_child_count, 0);
        Ok(())
    }

    /// `claim_idempotency` is an atomic reservation (findings #3/#15):
    /// the first claim reserves the key (`Fresh`), a second claim before
    /// the result is recorded observes `Conflict` (reservation in-flight),
    /// and once the winner records the result, retries `Replay`. A
    /// different fingerprint under the same key is also a `Conflict`.
    async fn test_claim_idempotency_reserve_then_replay(
        task_store: &dyn AgentTaskStore,
    ) -> Result<()> {
        let fp: &[u8] = b"fp-seq";
        assert!(matches!(
            task_store
                .claim_idempotency("conf-req-seq", IdempotencyKind::CreateThread, fp)
                .await?,
            IdempotencyClaim::Fresh,
        ));

        // Before the effect records its result, a retry must NOT also be
        // Fresh — the placeholder reservation makes it Conflict.
        assert!(matches!(
            task_store
                .claim_idempotency("conf-req-seq", IdempotencyKind::CreateThread, fp)
                .await?,
            IdempotencyClaim::Conflict,
        ));

        task_store
            .record_idempotency(IdempotencyRecord {
                request_id: "conf-req-seq".into(),
                kind: IdempotencyKind::CreateThread,
                fingerprint: fp.to_vec(),
                result_json: serde_json::json!({ "thread_id": "conf-th" }),
            })
            .await?;

        match task_store
            .claim_idempotency("conf-req-seq", IdempotencyKind::CreateThread, fp)
            .await?
        {
            IdempotencyClaim::Replay(record) => {
                assert_eq!(
                    record.result_json,
                    serde_json::json!({ "thread_id": "conf-th" })
                );
            }
            other => anyhow::bail!("expected Replay after record, got {other:?}"),
        }

        // A different fingerprint under the same key is a conflict, never a
        // silent alias.
        assert!(matches!(
            task_store
                .claim_idempotency(
                    "conf-req-seq",
                    IdempotencyKind::CreateThread,
                    b"fp-different"
                )
                .await?,
            IdempotencyClaim::Conflict,
        ));
        Ok(())
    }

    /// Two concurrent claims on one key yield **exactly one** `Fresh`
    /// (findings #3/#15) — the atomic reservation prevents both retries
    /// from running the effect. The loser observes `Conflict` (in-flight
    /// reservation). Generic over the concrete store so each backend's
    /// real locking (in-memory write lock, `SQLite` `BEGIN IMMEDIATE`,
    /// Postgres `INSERT … ON CONFLICT DO NOTHING`) is exercised.
    async fn test_two_concurrent_claims_one_fresh<S>(store: S) -> Result<()>
    where
        S: AgentTaskStore + Clone + Send + Sync + 'static,
    {
        let s1 = store.clone();
        let s2 = store.clone();
        let (a, b) = tokio::join!(
            async move {
                s1.claim_idempotency("conf-req-conc", IdempotencyKind::CreateThread, b"fp-conc")
                    .await
            },
            async move {
                s2.claim_idempotency("conf-req-conc", IdempotencyKind::CreateThread, b"fp-conc")
                    .await
            },
        );
        let outcomes = [a?, b?];
        let fresh = outcomes
            .iter()
            .filter(|c| matches!(c, IdempotencyClaim::Fresh))
            .count();
        assert_eq!(fresh, 1, "exactly one concurrent claim must win Fresh");
        let conflicts = outcomes
            .iter()
            .filter(|c| matches!(c, IdempotencyClaim::Conflict))
            .count();
        assert_eq!(
            conflicts, 1,
            "the losing concurrent claim must observe Conflict, not a second Fresh",
        );
        Ok(())
    }

    fn make_test_intent(task_name: &str, secs: i64) -> ExecutionIntent {
        let task_id = AgentTaskId::from_string(format!("task_{task_name}"));
        let op_id = OperationId::new(&task_id, "call_1");
        ExecutionIntent {
            operation_id: op_id,
            effect_class: ToolEffectClass::SideEffecting,
            tool_call_id: "call_1".into(),
            child_task_id: task_id,
            tool_name: "test_tool".into(),
            input: serde_json::json!({ "key": "value" }),
            status: IntentStatus::Pending,
            error: None,
            created_at: t_plus(secs),
            updated_at: t_plus(secs),
        }
    }

    /// `persist_intent` is insert-if-absent, not an upsert (findings
    /// #1/#2/#10): the first claim on an `operation_id` wins, and a second
    /// `persist_intent` for the same id returns an **error** (claim
    /// conflict) without clobbering the in-flight record back to a weaker
    /// status. This is the atomic claim the side-effecting-tool guard
    /// relies on to fail closed, and it must hold on every backend.
    async fn test_persist_intent_insert_if_absent(
        intent_store: &dyn ExecutionIntentStore,
    ) -> Result<()> {
        let mut intent = make_test_intent("conformance-claim", 25);
        intent_store.persist_intent(&intent).await?;

        // A racing second claim (e.g. a stale worker) must lose with a
        // conflict and must NOT downgrade the record.
        intent.mark_started(t_plus(26));
        let err = intent_store
            .persist_intent(&intent)
            .await
            .err()
            .context("second persist for the same operation_id must conflict")?;
        assert!(
            err.to_string().contains("conflict"),
            "expected a claim conflict, got: {err}",
        );

        let loaded = intent_store
            .get_intent(&intent.operation_id)
            .await?
            .context("intent should be present after the winning claim")?;
        assert_eq!(
            loaded.status,
            IntentStatus::Pending,
            "the losing claim must not clobber the in-flight Pending record",
        );
        Ok(())
    }

    /// Auto-promotion: completing the active root now promotes the FIFO
    /// head **inside the same store transaction** (findings #4/#8). After
    /// `complete_task`, the queued successor is already `Pending` and holds
    /// the active-root slot — so the host's later explicit
    /// `promote_next_queued_root` is a no-op that returns `None` (the
    /// successor is no longer queued). This contract holds across the
    /// in-memory, `SQLite`, and `PostgreSQL` backends.
    async fn test_promote_queued_root(task_store: &dyn AgentTaskStore) -> Result<()> {
        let tid = "conformance-promote";
        let root1 = fresh_root(tid, 80);
        let root2 = fresh_root(tid, 81);

        let r1 = task_store.submit_root_turn(root1).await?;
        let r2 = task_store.submit_root_turn(root2).await?;
        assert_eq!(r2.status, TaskStatus::Queued);

        // Acquire and complete root1. Completing a root-turn root frees the
        // thread's active-root slot and auto-promotes the queued head in
        // the same transaction.
        let w = WorkerId::new();
        let l = LeaseId::new();
        let _ = task_store
            .try_acquire_task(&r1.id, w.clone(), l.clone(), t_plus(110), t_plus(82))
            .await?;
        let _ = task_store.complete_task(&r1.id, &w, &l, t_plus(83)).await?;

        // The successor is ALREADY Pending — observed via `get`, not via the
        // explicit promote call.
        let promoted = task_store
            .get(&r2.id)
            .await?
            .context("queued successor must survive the terminal transition")?;
        assert_eq!(
            promoted.status,
            TaskStatus::Pending,
            "completing the active root must auto-promote the queued successor",
        );

        // It also holds the thread's active-root slot, and no queued roots
        // remain.
        let active = task_store
            .active_root_for_thread(&thread_id(tid))
            .await?
            .context("active-root slot must be held by the auto-promoted successor")?;
        assert_eq!(active.id, r2.id);
        assert!(
            task_store
                .list_queued_roots(&thread_id(tid))
                .await?
                .is_empty(),
            "no queued roots remain after auto-promotion",
        );

        // The host's explicit promotion now finds nothing to promote — the
        // successor is already Pending, so the call returns None (idempotent
        // no-op), NOT a second promotion.
        let explicit = task_store
            .promote_next_queued_root(&thread_id(tid), t_plus(84))
            .await?;
        assert!(
            explicit.is_none(),
            "an explicit promote after auto-promotion must be a no-op (already Pending)",
        );
        Ok(())
    }

    /// Retry exhaustion fails a task closed.
    async fn test_retry_exhaustion(task_store: &dyn AgentTaskStore) -> Result<()> {
        // Create a root with max_attempts = 1.
        let root = AgentTask::new_root_turn(thread_id("conformance-retry"), t_plus(90), 1);
        let _ = task_store.submit_root_turn(root.clone()).await?;

        // Acquire (attempt 1) and let the lease expire.
        let _ = task_store
            .try_acquire_task(
                &root.id,
                WorkerId::new(),
                LeaseId::new(),
                t_plus(91),
                t_plus(90),
            )
            .await?;
        let records = task_store.release_expired_leases(t_plus(92)).await?;
        assert_eq!(records.len(), 1);
        assert!(matches!(records[0].action, RecoveryAction::FailClosed(_)));

        let task = task_store.get(&root.id).await?.context("task exists")?;
        assert_eq!(task.status, TaskStatus::Failed);
        Ok(())
    }

    /// When a child's lease expires past `max_attempts`, the recovery
    /// path must wake its `WaitingOnChildren` parent — otherwise the
    /// parent stays blocked forever (no future complete/fail event
    /// will fire to decrement `pending_child_count`).
    async fn test_fail_closed_child_wakes_parent(task_store: &dyn AgentTaskStore) -> Result<()> {
        let parent = AgentTask::new_root_turn(thread_id("conformance-orphan"), t_plus(200), 3);
        let _ = task_store.submit_root_turn(parent.clone()).await?;
        let pw = WorkerId::new();
        let pl = LeaseId::new();
        let _ = task_store
            .try_acquire_task(&parent.id, pw.clone(), pl.clone(), t_plus(260), t_plus(201))
            .await?;

        // Spawn a single child with max_attempts=1 so its first lease
        // expiry fail-closes it.
        let payload = SuspensionPayload {
            continuation: agent_sdk_foundation::ContinuationEnvelope::wrap(
                agent_sdk_foundation::AgentContinuation {
                    thread_id: thread_id("conformance-orphan"),
                    turn: 1,
                    total_usage: TokenUsage {
                        input_tokens: 0,
                        output_tokens: 0,
                        ..Default::default()
                    },
                    turn_usage: TokenUsage {
                        input_tokens: 0,
                        output_tokens: 0,
                        ..Default::default()
                    },
                    pending_tool_calls: vec![],
                    awaiting_index: 0,
                    completed_results: vec![],
                    state: agent_sdk_foundation::AgentState::new(thread_id("conformance-orphan")),
                    response_id: None,
                    stop_reason: None,
                    response_content: vec![],
                },
            ),
            suspended_messages: vec![],
        };
        let (_parent, children) = task_store
            .spawn_tool_children(
                &parent.id,
                &pw,
                &pl,
                vec![ChildSpawnSpec { max_attempts: 1 }],
                payload,
                None,
                t_plus(202),
            )
            .await?;
        let child = &children[0];

        // Acquire child, let lease expire, then sweep.  attempt=1 ==
        // max_attempts triggers FailClosed.
        let _ = task_store
            .try_acquire_task(
                &child.id,
                WorkerId::new(),
                LeaseId::new(),
                t_plus(210),
                t_plus(203),
            )
            .await?;
        let records = task_store.release_expired_leases(t_plus(211)).await?;
        assert_eq!(records.len(), 1);
        assert!(matches!(records[0].action, RecoveryAction::FailClosed(_)));

        // Child must be Failed and parent must have woken up to
        // Pending with no live children.
        let child_after = task_store.get(&child.id).await?.context("child exists")?;
        assert_eq!(child_after.status, TaskStatus::Failed);
        let parent_after = task_store.get(&parent.id).await?.context("parent exists")?;
        assert_eq!(parent_after.pending_child_count, 0);
        assert_eq!(parent_after.status, TaskStatus::Pending);
        Ok(())
    }

    /// Clear wipes every task — including parent/child chains guarded
    /// by ON DELETE RESTRICT self-referential FKs on `SQLite`.
    async fn test_clear_with_parent_child(task_store: &dyn AgentTaskStore) -> Result<()> {
        let root = fresh_root("conformance-clear", 100);
        let _ = task_store.submit_root_turn(root.clone()).await?;
        let worker = WorkerId::new();
        let lease = LeaseId::new();
        let _ = task_store
            .try_acquire_task(
                &root.id,
                worker.clone(),
                lease.clone(),
                t_plus(160),
                t_plus(101),
            )
            .await?;

        let spec = ChildSpawnSpec { max_attempts: 3 };
        let payload = SuspensionPayload {
            continuation: agent_sdk_foundation::ContinuationEnvelope::wrap(
                agent_sdk_foundation::AgentContinuation {
                    thread_id: thread_id("conformance-clear"),
                    turn: 1,
                    total_usage: TokenUsage {
                        input_tokens: 0,
                        output_tokens: 0,
                        ..Default::default()
                    },
                    turn_usage: TokenUsage {
                        input_tokens: 0,
                        output_tokens: 0,
                        ..Default::default()
                    },
                    pending_tool_calls: vec![],
                    awaiting_index: 0,
                    completed_results: vec![],
                    state: agent_sdk_foundation::AgentState::new(thread_id("conformance-clear")),
                    response_id: None,
                    stop_reason: None,
                    response_content: vec![],
                },
            ),
            suspended_messages: vec![],
        };
        let (_parent, children) = task_store
            .spawn_tool_children(
                &root.id,
                &worker,
                &lease,
                vec![spec],
                payload,
                None,
                t_plus(102),
            )
            .await?;
        assert_eq!(children.len(), 1);

        // Clear must succeed despite parent→child and child→parent FK edges.
        task_store.clear().await?;
        assert!(task_store.get(&root.id).await?.is_none());
        assert!(task_store.get(&children[0].id).await?.is_none());
        Ok(())
    }

    // ── Phase 10 · E: idempotency + back-pressure ───────────────────

    fn submit_params(
        task: AgentTask,
        request_id: &str,
        fingerprint: &[u8],
        max_queued_depth: Option<u32>,
    ) -> agent_server::journal::store::SubmitRootTurnParams {
        let result_json = serde_json::json!({ "task_id": task.id.to_string() });
        agent_server::journal::store::SubmitRootTurnParams {
            task,
            idempotency: Some(agent_server::journal::store::SubmitRootIdempotency {
                request_id: request_id.to_owned(),
                fingerprint: fingerprint.to_vec(),
                result_json,
            }),
            max_queued_depth,
        }
    }

    /// A retried submission under the same `request_id` admits exactly
    /// one root turn and replays the original on the retry. This is the
    /// at-least-once dedup contract (gap #6).
    async fn test_submit_idempotent_dedup(task_store: &dyn AgentTaskStore) -> Result<()> {
        let tid = thread_id("conformance-idem");
        let first = AgentTask::new_root_turn(tid.clone(), t_plus(10), 3);
        let first_id = first.id.clone();
        let outcome = task_store
            .submit_root_turn_idempotent(submit_params(first, "req-1", b"fp", None))
            .await
            .map_err(|error| anyhow::anyhow!("first submit failed: {error}"))?;
        assert!(!outcome.replayed, "first submit must not be a replay");
        assert_eq!(outcome.task.id, first_id);

        // A retry with a *different* task row but the same request_id +
        // fingerprint replays the original task and admits nothing new.
        let retry = AgentTask::new_root_turn(tid.clone(), t_plus(11), 3);
        let retry_id = retry.id.clone();
        let replay = task_store
            .submit_root_turn_idempotent(submit_params(retry, "req-1", b"fp", None))
            .await
            .map_err(|error| anyhow::anyhow!("retry submit failed: {error}"))?;
        assert!(replay.replayed, "retry must be a replay");
        assert_eq!(replay.task.id, first_id, "replay returns the original task");
        assert_ne!(
            replay.task.id, retry_id,
            "the retry row must not be admitted"
        );

        // Exactly one root task exists on the thread.
        let tasks = task_store.list_by_thread(&tid).await?;
        let roots = tasks.iter().filter(|task| task.is_root()).count();
        assert_eq!(roots, 1, "exactly one root admitted across the retry");
        Ok(())
    }

    /// A retry that re-uses a `request_id` with a *different* payload
    /// fingerprint is a conflict, not a silent alias.
    async fn test_submit_idempotent_conflict(task_store: &dyn AgentTaskStore) -> Result<()> {
        let tid = thread_id("conformance-idem-conflict");
        let first = AgentTask::new_root_turn(tid.clone(), t_plus(10), 3);
        task_store
            .submit_root_turn_idempotent(submit_params(first, "req-c", b"fp-a", None))
            .await
            .map_err(|error| anyhow::anyhow!("first submit failed: {error}"))?;

        let second = AgentTask::new_root_turn(tid.clone(), t_plus(11), 3);
        let result = task_store
            .submit_root_turn_idempotent(submit_params(second, "req-c", b"fp-b", None))
            .await;
        assert!(
            matches!(
                result,
                Err(agent_server::journal::store::SubmitRootTurnError::IdempotencyConflict)
            ),
            "a different fingerprint under the same key must conflict",
        );
        Ok(())
    }

    /// Exceeding the per-thread queued-root depth cap returns
    /// `QueueDepthExceeded` (gap #9). The active root never counts
    /// against the cap; only queued roots do.
    async fn test_submit_queue_depth_cap(task_store: &dyn AgentTaskStore) -> Result<()> {
        let tid = thread_id("conformance-queue-cap");
        // First admission takes the active slot (queued_depth == 0).
        let active = AgentTask::new_root_turn(tid.clone(), t_plus(10), 3);
        let outcome = task_store
            .submit_root_turn_idempotent(submit_params(active, "q-active", b"fp", Some(1)))
            .await
            .map_err(|error| anyhow::anyhow!("active submit failed: {error}"))?;
        assert_eq!(outcome.queued_depth, 0);

        // Second admission queues behind the active slot (queued_depth
        // becomes 1, which equals the cap — still allowed because the
        // check is "would exceed", i.e. current >= cap before insert).
        let queued = AgentTask::new_root_turn(tid.clone(), t_plus(11), 3);
        let outcome = task_store
            .submit_root_turn_idempotent(submit_params(queued, "q-1", b"fp", Some(1)))
            .await
            .map_err(|error| anyhow::anyhow!("queued submit failed: {error}"))?;
        assert_eq!(
            outcome.queued_depth, 1,
            "one queued root after second admit"
        );

        // Third admission would push queued depth to 2 > cap of 1.
        let overflow = AgentTask::new_root_turn(tid.clone(), t_plus(12), 3);
        let result = task_store
            .submit_root_turn_idempotent(submit_params(overflow, "q-2", b"fp", Some(1)))
            .await;
        match result {
            Err(agent_server::journal::store::SubmitRootTurnError::QueueDepthExceeded {
                cap,
                current_depth,
            }) => {
                assert_eq!(cap, 1);
                assert_eq!(current_depth, 1);
            }
            other => anyhow::bail!("expected QueueDepthExceeded, got {other:?}"),
        }

        // The rejected submission left no trace — still exactly two roots.
        let tasks = task_store.list_by_thread(&tid).await?;
        let roots = tasks.iter().filter(|task| task.is_root()).count();
        assert_eq!(roots, 2, "the over-cap submission admitted nothing");
        Ok(())
    }

    /// A root admitted with the production budget that crashes mid-turn
    /// is requeued (not failed-closed). This is the `max_attempts`
    /// reconciliation (gap #10): `DEFAULT_ROOT_MAX_ATTEMPTS == 3`.
    async fn test_root_default_budget_requeues_on_crash(
        task_store: &dyn AgentTaskStore,
    ) -> Result<()> {
        assert_eq!(
            AgentTask::DEFAULT_ROOT_MAX_ATTEMPTS,
            3,
            "root default budget must allow a mid-turn crash to retry",
        );
        let tid = thread_id("conformance-root-budget");
        let root = AgentTask::new_root_turn(
            tid.clone(),
            t_plus(10),
            AgentTask::DEFAULT_ROOT_MAX_ATTEMPTS,
        );
        let _ = task_store.submit_root_turn(root.clone()).await?;

        // Acquire (attempt 1) and let the lease expire — simulating a
        // mid-turn host crash.
        let _ = task_store
            .try_acquire_task(
                &root.id,
                WorkerId::new(),
                LeaseId::new(),
                t_plus(11),
                t_plus(10),
            )
            .await?;
        let records = task_store.release_expired_leases(t_plus(12)).await?;
        assert_eq!(records.len(), 1);
        assert!(
            records[0].action.is_requeue(),
            "budget-3 root must requeue on first crash, got {:?}",
            records[0].action,
        );

        let task = task_store.get(&root.id).await?.context("task exists")?;
        assert_eq!(task.status, TaskStatus::Pending, "requeued back to Pending");
        Ok(())
    }

    // ── Backend runners ──────────────────────────────────────────────

    // ── In-memory backend ────────────────────────────────────────────

    #[tokio::test]
    async fn conformance_in_memory_root_admission() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_root_admission(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_fifo_queueing() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_root_fifo_queueing(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_lease_acquisition() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_lease_acquisition(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_heartbeat() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_heartbeat(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_lease_expiry() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_lease_expiry_sweep(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_requeue_owned_task() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_requeue_owned_task(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_completed_turn() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_completed_turn_commit(
            s.task.as_ref(),
            s.thread.as_ref(),
            s.message.as_ref(),
            s.attempt.as_ref(),
            s.checkpoint.as_ref(),
            s.event.as_ref(),
        )
        .await
    }

    #[tokio::test]
    async fn conformance_in_memory_stale_turn_commit_is_typed() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_stale_turn_commit_rejection_is_typed(
            s.task.as_ref(),
            s.thread.as_ref(),
            s.message.as_ref(),
            s.attempt.as_ref(),
            s.checkpoint.as_ref(),
            s.event.as_ref(),
        )
        .await
    }

    #[tokio::test]
    async fn conformance_in_memory_child_spawn() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_child_spawn_and_resume(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_steering_wake() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_steering_wake_and_repark(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_cancel_tree() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_cancel_tree(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_promote_queued() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_promote_queued_root(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_retry_exhaustion() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_retry_exhaustion(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_clear_parent_child() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_clear_with_parent_child(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_fail_closed_child_wakes_parent() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_fail_closed_child_wakes_parent(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_submit_idempotent_dedup() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_submit_idempotent_dedup(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_submit_idempotent_conflict() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_submit_idempotent_conflict(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_submit_queue_depth_cap() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_submit_queue_depth_cap(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_root_default_budget_requeues() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_root_default_budget_requeues_on_crash(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_cancel_subagent_cascade() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_cancel_tree_cascades_through_subagent(s.task.as_ref(), s.thread.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_cancel_marker_exactly_once_and_crash_retry() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_cancel_marker_exactly_once_and_crash_retry_idempotent(
            s.task.as_ref(),
            s.event.as_ref(),
            s.outbox.as_ref(),
        )
        .await
    }

    #[tokio::test]
    async fn conformance_in_memory_cancel_marker_covers_child_threads() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_cancel_marker_covers_child_threads(
            s.task.as_ref(),
            s.thread.as_ref(),
            s.event.as_ref(),
        )
        .await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_in_memory_concurrent_cancels_single_marker() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_concurrent_cancels_emit_single_marker(
            std::sync::Arc::clone(&s.task),
            s.event.as_ref(),
            s.outbox.as_ref(),
        )
        .await
    }

    #[tokio::test]
    async fn conformance_in_memory_queued_root_cancel_emits_no_marker() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_queued_root_cancel_emits_no_marker(
            s.task.as_ref(),
            s.event.as_ref(),
            s.outbox.as_ref(),
        )
        .await
    }

    #[tokio::test]
    async fn conformance_in_memory_cancel_child_root_wakes_invocation() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_cancel_child_root_wakes_invocation(s.task.as_ref(), s.thread.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_fail_closed_child_root_wakes_invocation() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_fail_closed_child_root_wakes_invocation(s.task.as_ref(), s.thread.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_find_invocation_for_child_root() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_find_invocation_for_child_root(s.task.as_ref(), s.thread.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_cancel_tool_child_resumes_parked_parent() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_cancel_tool_child_resumes_parked_parent(s.task.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_list_parked_subagent_invocations() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_list_parked_subagent_invocations(s.task.as_ref(), s.thread.as_ref()).await
    }

    #[tokio::test]
    async fn conformance_in_memory_claim_idempotency_reserve_then_replay() -> Result<()> {
        let s = fresh_in_memory_stores();
        test_claim_idempotency_reserve_then_replay(s.task.as_ref()).await
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn conformance_in_memory_two_concurrent_claims_one_fresh() -> Result<()> {
        let store = agent_server::journal::store::InMemoryAgentTaskStore::new();
        test_two_concurrent_claims_one_fresh(store).await
    }

    #[tokio::test]
    async fn conformance_in_memory_persist_intent_insert_if_absent() -> Result<()> {
        let store = agent_server::journal::execution_intent::InMemoryExecutionIntentStore::new();
        test_persist_intent_insert_if_absent(&store).await
    }

    struct InMemoryStores {
        task: std::sync::Arc<dyn AgentTaskStore>,
        thread: std::sync::Arc<dyn ThreadStore>,
        message: std::sync::Arc<dyn MessageProjectionStore>,
        attempt: std::sync::Arc<dyn TurnAttemptStore>,
        checkpoint: std::sync::Arc<dyn CheckpointStore>,
        event: std::sync::Arc<dyn EventRepository>,
        outbox: std::sync::Arc<dyn OutboxStore>,
    }

    fn fresh_in_memory_stores() -> InMemoryStores {
        use agent_server::journal::checkpoint_store::InMemoryCheckpointStore;
        use agent_server::journal::event_repository::InMemoryEventRepository;
        use agent_server::journal::message_store::InMemoryMessageProjectionStore;
        use agent_server::journal::outbox::InMemoryOutboxStore;
        use agent_server::journal::store::{CancellationMarkerSink, InMemoryAgentTaskStore};
        use agent_server::journal::thread_store::InMemoryThreadStore;
        use agent_server::journal::turn_attempt_store::InMemoryTurnAttemptStore;

        // Mirror `StoreRegistry::in_memory`: the task store commits its
        // terminal `Cancelled` markers through the shared event /
        // outbox / thread stores (issue #354), so the marker
        // conformance battery runs on the same wiring production uses.
        let thread: std::sync::Arc<InMemoryThreadStore> =
            std::sync::Arc::new(InMemoryThreadStore::new());
        let event: std::sync::Arc<InMemoryEventRepository> =
            std::sync::Arc::new(InMemoryEventRepository::new());
        let outbox: std::sync::Arc<InMemoryOutboxStore> =
            std::sync::Arc::new(InMemoryOutboxStore::new());
        let task =
            InMemoryAgentTaskStore::new().with_cancellation_markers(CancellationMarkerSink {
                event_repo: event.clone(),
                outbox_store: outbox.clone(),
                thread_store: thread.clone(),
            });

        InMemoryStores {
            task: std::sync::Arc::new(task),
            thread,
            message: std::sync::Arc::new(InMemoryMessageProjectionStore::new()),
            attempt: std::sync::Arc::new(InMemoryTurnAttemptStore::new()),
            checkpoint: std::sync::Arc::new(InMemoryCheckpointStore::new()),
            event,
            outbox,
        }
    }

    // ── SQLite backend ───────────────────────────────────────────────

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_root_admission() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_root_admission(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_fifo_queueing() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_root_fifo_queueing(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_lease_acquisition() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_lease_acquisition(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_heartbeat() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_heartbeat(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_lease_expiry() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_lease_expiry_sweep(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_requeue_owned_task() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_requeue_owned_task(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_completed_turn() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_completed_turn_commit(&store, &store, &store, &store, &store, &store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_stale_turn_commit_is_typed() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_stale_turn_commit_rejection_is_typed(&store, &store, &store, &store, &store, &store)
            .await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_owner_guarded_commit_rejects_lost_ownership() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_owner_guarded_commit_rejects_lost_ownership(
            &store, &store, &store, &store, &store, &store,
        )
        .await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_owner_guarded_commit_succeeds_while_owned() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_owner_guarded_commit_succeeds_while_owned(
            &store, &store, &store, &store, &store, &store,
        )
        .await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_child_spawn() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_child_spawn_and_resume(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_steering_wake() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_steering_wake_and_repark(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_cancel_tree() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_cancel_tree(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_promote_queued() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_promote_queued_root(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_retry_exhaustion() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_retry_exhaustion(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_clear_parent_child() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_clear_with_parent_child(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_fail_closed_child_wakes_parent() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_fail_closed_child_wakes_parent(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_submit_idempotent_dedup() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_submit_idempotent_dedup(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_submit_idempotent_conflict() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_submit_idempotent_conflict(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_submit_queue_depth_cap() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_submit_queue_depth_cap(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_root_default_budget_requeues() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_root_default_budget_requeues_on_crash(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_cancel_subagent_cascade() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_cancel_tree_cascades_through_subagent(&store, &store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_cancel_marker_exactly_once_and_crash_retry() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_cancel_marker_exactly_once_and_crash_retry_idempotent(&store, &store, &store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_cancel_marker_covers_child_threads() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_cancel_marker_covers_child_threads(&store, &store, &store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_sqlite_concurrent_cancels_single_marker() -> Result<()> {
        let store = std::sync::Arc::new(
            crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?,
        );
        test_concurrent_cancels_emit_single_marker(
            std::sync::Arc::clone(&store) as std::sync::Arc<dyn AgentTaskStore>,
            store.as_ref(),
            store.as_ref(),
        )
        .await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_queued_root_cancel_emits_no_marker() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_queued_root_cancel_emits_no_marker(&store, &store, &store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_cancel_child_root_wakes_invocation() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_cancel_child_root_wakes_invocation(&store, &store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_fail_closed_child_root_wakes_invocation() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_fail_closed_child_root_wakes_invocation(&store, &store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_find_invocation_for_child_root() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_find_invocation_for_child_root(&store, &store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_cancel_tool_child_resumes_parked_parent() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_cancel_tool_child_resumes_parked_parent(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_list_parked_subagent_invocations() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_list_parked_subagent_invocations(&store, &store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_claim_idempotency_reserve_then_replay() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_claim_idempotency_reserve_then_replay(&store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test(flavor = "multi_thread")]
    async fn conformance_sqlite_two_concurrent_claims_one_fresh() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_two_concurrent_claims_one_fresh(store).await
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_persist_intent_insert_if_absent() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_persist_intent_insert_if_absent(&store).await
    }

    /// Auto-promotion emits the promoted root's durable `task_wakeup`
    /// advisory outbox row inside the same transaction (findings #4/#8/#22)
    /// — verified here because the `SQLite` store has outbox access (the
    /// shared `&dyn AgentTaskStore` battery does not). Two roots are
    /// admitted (one Pending wakeup), then the active root is completed,
    /// auto-promoting the successor and emitting a second wakeup.
    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn conformance_sqlite_auto_promote_emits_wakeup_outbox_row() -> Result<()> {
        use agent_server::journal::outbox::OutboxStore;
        use agent_server::journal::outbox_message::OutboxMessageKind;

        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        let tid = "conformance-promote-wakeup";
        let r1 = store.submit_root_turn(fresh_root(tid, 80)).await?;
        let r2 = store.submit_root_turn(fresh_root(tid, 81)).await?;
        assert_eq!(r2.status, TaskStatus::Queued);

        // One wakeup so far — only the admitted Pending root emitted one;
        // the queued (parked) root did not.
        let wakeups_before = OutboxStore::list_by_thread(&store, &thread_id(tid))
            .await?
            .into_iter()
            .filter(|row| row.kind == OutboxMessageKind::TaskWakeup)
            .count();
        assert_eq!(wakeups_before, 1, "only the runnable root emitted a wakeup");

        let w = WorkerId::new();
        let l = LeaseId::new();
        let _ = store
            .try_acquire_task(&r1.id, w.clone(), l.clone(), t_plus(110), t_plus(82))
            .await?;
        let _ = store.complete_task(&r1.id, &w, &l, t_plus(83)).await?;

        // Completing the active root auto-promoted the successor and must
        // have emitted its wakeup row in the same transaction.
        let promoted = AgentTaskStore::get(&store, &r2.id)
            .await?
            .context("successor exists")?;
        assert_eq!(promoted.status, TaskStatus::Pending);
        let wakeups_after = OutboxStore::list_by_thread(&store, &thread_id(tid))
            .await?
            .into_iter()
            .filter(|row| row.kind == OutboxMessageKind::TaskWakeup)
            .count();
        assert_eq!(
            wakeups_after, 2,
            "auto-promotion must emit a durable task_wakeup row for the promoted root",
        );
        Ok(())
    }

    /// Cross-restart idempotency (gap #6, the headline acceptance
    /// criterion): submit the same `request_id` before and after a
    /// simulated host restart — modeled by dropping the `SQLite` store +
    /// pool and reopening the *same file* — and assert exactly one
    /// `RootTurn` task exists, with the retry replaying the original.
    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn cross_restart_submit_idempotency_admits_exactly_one_root() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let db_path = dir.path().join("restart.db");
        let url = format!("sqlite://{}?mode=rwc", db_path.display());
        let tid = thread_id("restart-idem");

        // ── Before the "restart" ─────────────────────────────────────
        let original_task_id = {
            let store = crate::sqlite::SqliteDurableStore::connect(&url).await?;
            let root = AgentTask::new_root_turn(tid.clone(), t_plus(10), 3);
            let task_id = root.id.clone();
            let outcome = store
                .submit_root_turn_idempotent(submit_params(root, "restart-req", b"fp", None))
                .await
                .map_err(|error| anyhow::anyhow!("pre-restart submit failed: {error}"))?;
            assert!(!outcome.replayed);
            // Drop the store + pool to simulate process death.
            drop(store);
            task_id
        };

        // ── After the "restart": reopen the same file ────────────────
        let store = crate::sqlite::SqliteDurableStore::connect(&url).await?;
        // The in-memory dedup cache is gone, but the durable record
        // survives. A retry under the same request_id must replay the
        // original task, not admit a duplicate.
        let retry = AgentTask::new_root_turn(tid.clone(), t_plus(11), 3);
        let replay = store
            .submit_root_turn_idempotent(submit_params(retry, "restart-req", b"fp", None))
            .await
            .map_err(|error| anyhow::anyhow!("post-restart retry failed: {error}"))?;
        assert!(replay.replayed, "post-restart retry must replay, not admit");
        assert_eq!(
            replay.task.id, original_task_id,
            "replay returns the original task minted before the restart",
        );

        let tasks = AgentTaskStore::list_by_thread(&store, &tid).await?;
        let roots = tasks.iter().filter(|task| task.is_root()).count();
        assert_eq!(
            roots, 1,
            "exactly one RootTurn task survives the request_id retry across restart",
        );
        Ok(())
    }

    // ── Postgres backend (finding #5) ────────────────────────────────
    //
    // The module doc claims this battery runs "against the in-memory,
    // SQLite, and PostgreSQL backends", but runners existed only for the
    // first two. These arms mirror the SQLite runners: each allocates a
    // throwaway schema (so parallel `nextest` runs never collide) and the
    // whole arm is a keyless no-op when no database URL is configured, so a
    // database-less CI lane stays green. With a local Postgres up, point
    // `TEST_DATABASE_URL` at it to exercise the cross-thread cancel BFS,
    // the recovery sweep, fork atomicity, and the idempotency claim path on
    // the flagship durable backend for real.

    /// Allocate a throwaway Postgres schema and a migrated store on it, or
    /// `None` when no database URL is configured. Mirrors the gating used
    /// across `crate::postgres::store`'s own tests and
    /// `journal_conformance.rs`.
    #[cfg(feature = "postgres")]
    async fn pg_test_store() -> Result<
        Option<(
            crate::postgres::store::PostgresDurableStore,
            PgConformanceSchema,
        )>,
    > {
        use sqlx::Connection;
        use sqlx::postgres::{PgConnection, PgPoolOptions};
        use uuid::Uuid;

        let Ok(database_url) =
            std::env::var("TEST_DATABASE_URL").or_else(|_| std::env::var("DATABASE_URL"))
        else {
            return Ok(None);
        };

        let schema = format!("conf_pg_{}", Uuid::new_v4().simple());
        let mut admin = PgConnection::connect(&database_url)
            .await
            .context("connect postgres admin for task-lifecycle conformance")?;
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE SCHEMA {schema}")))
            .execute(&mut admin)
            .await
            .with_context(|| format!("create conformance test schema {schema}"))?;
        drop(admin);

        let search_path = schema.clone();
        let pool = PgPoolOptions::new()
            .max_connections(8)
            .after_connect(move |conn, _meta| {
                let sql = format!("SET search_path TO {search_path}");
                Box::pin(async move {
                    sqlx::query(sqlx::AssertSqlSafe(sql)).execute(conn).await?;
                    Ok(())
                })
            })
            .connect(&database_url)
            .await
            .context("connect postgres conformance test pool")?;
        let store = crate::postgres::store::PostgresDurableStore::from_pool(pool);
        store
            .migrate()
            .await
            .context("migrate postgres conformance test store")?;
        Ok(Some((
            store,
            PgConformanceSchema {
                schema,
                database_url,
            },
        )))
    }

    /// Drops the throwaway schema on `Drop` so a real-database run leaves
    /// nothing behind.
    #[cfg(feature = "postgres")]
    struct PgConformanceSchema {
        schema: String,
        database_url: String,
    }

    #[cfg(feature = "postgres")]
    impl Drop for PgConformanceSchema {
        fn drop(&mut self) {
            let database_url = self.database_url.clone();
            let schema = self.schema.clone();
            let _ = std::thread::spawn(move || {
                use sqlx::Connection;
                use sqlx::postgres::PgConnection;
                let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                else {
                    return;
                };
                runtime.block_on(async move {
                    let Ok(mut conn) = PgConnection::connect(&database_url).await else {
                        return;
                    };
                    let _ = sqlx::query(sqlx::AssertSqlSafe(format!(
                        "DROP SCHEMA IF EXISTS {schema} CASCADE"
                    )))
                    .execute(&mut conn)
                    .await;
                });
            })
            .join();
        }
    }

    /// CI canary (finding #19): every Postgres arm is a keyless no-op when
    /// `TEST_DATABASE_URL` is unset — intentional for database-less lanes,
    /// but it means a wiring regression (renamed secret, dropped service
    /// container) silently turns the *entire* Postgres suite into green
    /// no-ops with zero signal. The dedicated Postgres CI lane sets
    /// `REQUIRE_POSTGRES_TESTS=1`; when that is present this canary
    /// hard-fails unless a database URL is also configured, so the gap is
    /// loud. Local / keyless runs leave the var unset, so it stays a no-op
    /// and never breaks a developer's `cargo nextest run`.
    #[cfg(feature = "postgres")]
    #[tokio::test]
    async fn postgres_required_db_url_canary() -> Result<()> {
        if std::env::var("REQUIRE_POSTGRES_TESTS").is_err() {
            return Ok(());
        }
        let configured =
            std::env::var("TEST_DATABASE_URL").is_ok() || std::env::var("DATABASE_URL").is_ok();
        anyhow::ensure!(
            configured,
            "REQUIRE_POSTGRES_TESTS is set but neither TEST_DATABASE_URL nor DATABASE_URL is \
             configured: the Postgres conformance arms would all silently no-op. Wire the \
             database URL into this CI lane, or unset REQUIRE_POSTGRES_TESTS for keyless runs.",
        );
        Ok(())
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_root_admission() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_root_admission(&store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_fifo_queueing() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_root_fifo_queueing(&store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_lease_acquisition() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_lease_acquisition(&store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_heartbeat() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_heartbeat(&store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_lease_expiry() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_lease_expiry_sweep(&store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_requeue_owned_task() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_requeue_owned_task(&store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_completed_turn() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_completed_turn_commit(&store, &store, &store, &store, &store, &store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_stale_turn_commit_is_typed() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_stale_turn_commit_rejection_is_typed(&store, &store, &store, &store, &store, &store)
            .await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_owner_guarded_commit_rejects_lost_ownership() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_owner_guarded_commit_rejects_lost_ownership(
            &store, &store, &store, &store, &store, &store,
        )
        .await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_owner_guarded_commit_succeeds_while_owned() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_owner_guarded_commit_succeeds_while_owned(
            &store, &store, &store, &store, &store, &store,
        )
        .await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_child_spawn() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_child_spawn_and_resume(&store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_steering_wake() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_steering_wake_and_repark(&store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_cancel_tree() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_cancel_tree(&store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_promote_queued() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_promote_queued_root(&store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_retry_exhaustion() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_retry_exhaustion(&store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_clear_parent_child() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_clear_with_parent_child(&store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_fail_closed_child_wakes_parent() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_fail_closed_child_wakes_parent(&store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_submit_idempotent_dedup() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_submit_idempotent_dedup(&store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_submit_idempotent_conflict() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_submit_idempotent_conflict(&store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_submit_queue_depth_cap() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_submit_queue_depth_cap(&store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_root_default_budget_requeues() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_root_default_budget_requeues_on_crash(&store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_cancel_subagent_cascade() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_cancel_tree_cascades_through_subagent(&store, &store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_cancel_marker_exactly_once_and_crash_retry() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_cancel_marker_exactly_once_and_crash_retry_idempotent(&store, &store, &store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_cancel_marker_covers_child_threads() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_cancel_marker_covers_child_threads(&store, &store, &store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_concurrent_cancels_single_marker() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        let store = std::sync::Arc::new(store);
        test_concurrent_cancels_emit_single_marker(
            std::sync::Arc::clone(&store) as std::sync::Arc<dyn AgentTaskStore>,
            store.as_ref(),
            store.as_ref(),
        )
        .await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_queued_root_cancel_emits_no_marker() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_queued_root_cancel_emits_no_marker(&store, &store, &store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_cancel_child_root_wakes_invocation() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_cancel_child_root_wakes_invocation(&store, &store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_fail_closed_child_root_wakes_invocation() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_fail_closed_child_root_wakes_invocation(&store, &store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_find_invocation_for_child_root() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_find_invocation_for_child_root(&store, &store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_cancel_tool_child_resumes_parked_parent() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_cancel_tool_child_resumes_parked_parent(&store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_list_parked_subagent_invocations() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_list_parked_subagent_invocations(&store, &store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_claim_idempotency_reserve_then_replay() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_claim_idempotency_reserve_then_replay(&store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_two_concurrent_claims_one_fresh() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_two_concurrent_claims_one_fresh(store).await
    }

    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_persist_intent_insert_if_absent() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_persist_intent_insert_if_absent(&store).await
    }

    /// Postgres fork atomicity, happy path (findings #9/#26/#27): the
    /// `AtomicForkCommitter` must commit thread aggregate + projection +
    /// checkpoint + events as one transaction. Mirrors the `SQLite`
    /// `fork_atomic_commits_full_state_in_one_transaction` test.
    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_fork_atomic_commits_full_state() -> Result<()> {
        use agent_sdk_foundation::events::AgentEvent;
        use agent_sdk_foundation::llm::Message;
        use agent_server::journal::checkpoint::{CheckpointKind, NewCheckpointParams};
        use agent_server::journal::fork_transaction::{AtomicForkCommitter, ForkCommitParams};

        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        let now = t0();

        // The source thread + task must exist so the fork's mirrored
        // checkpoint can carry the source task_id without violating the
        // agent_sdk_message_commits.task_id FK.
        let source_thread_id = thread_id("pg-fork-source");
        ThreadStore::get_or_create(&store, &source_thread_id, now).await?;
        let source_task = AgentTask::new_root_turn(source_thread_id.clone(), now, 3);
        let source_task_id = source_task.id.clone();
        AgentTaskStore::submit_root_turn(&store, source_task).await?;

        let new_thread_id = thread_id("pg-forked-thread");
        let messages = vec![
            Message::user("hi from source"),
            Message::assistant("hello from source"),
        ];
        let events = vec![
            AgentEvent::start(source_thread_id.clone(), 1),
            AgentEvent::done(
                source_thread_id.clone(),
                1,
                TokenUsage::default(),
                std::time::Duration::from_secs(1),
            ),
        ];
        let params = ForkCommitParams {
            new_thread_id: new_thread_id.clone(),
            creation: None,
            now,
            committed_turns: 1,
            cumulative_total_usage: TokenUsage::default(),
            messages: messages.clone(),
            checkpoint: Some(NewCheckpointParams {
                kind: CheckpointKind::FullTurn,
                thread_id: new_thread_id.clone(),
                turn_number: 1,
                task_id: source_task_id,
                messages: messages.clone(),
                agent_state_snapshot: serde_json::json!({
                    "thread_id": new_thread_id.0.clone(),
                    "turn_count": 1,
                    "total_usage": TokenUsage::default(),
                    "metadata": {},
                    "created_at": "2023-11-14T00:00:00Z",
                }),
                turn_usage: TokenUsage::default(),
                now,
            }),
            events: events.clone(),
        };

        store.commit_fork_atomic(params).await?;

        let thread = ThreadStore::get(&store, &new_thread_id)
            .await?
            .context("forked thread aggregate must exist after commit")?;
        assert_eq!(thread.committed_turns, 1);
        let projection = MessageProjectionStore::get_history(&store, &new_thread_id).await?;
        assert_eq!(projection.len(), 2);
        let checkpoint = CheckpointStore::get_by_turn(&store, &new_thread_id, 1)
            .await?
            .context("forked checkpoint must exist")?;
        assert_eq!(checkpoint.messages.len(), 2);
        let committed = EventRepository::get_events(&store, &new_thread_id).await?;
        assert_eq!(committed.len(), 2);
        let sequences: Vec<_> = committed.iter().map(|c| c.sequence).collect();
        assert_eq!(
            sequences,
            vec![0, 1],
            "events committed with fresh sequences"
        );
        Ok(())
    }

    /// Postgres fork atomicity, rollback path (findings #9/#26/#27): a
    /// failing fork must leave **zero** observable state on the
    /// destination thread. A pre-existing checkpoint at the same
    /// `(thread_id, turn_number)` collides inside the transaction, so the
    /// whole fork rolls back. Mirrors the `SQLite`
    /// `fork_atomic_rolls_back_on_failure` test.
    #[cfg(feature = "postgres")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conformance_postgres_fork_atomic_rolls_back_on_failure() -> Result<()> {
        use agent_sdk_foundation::events::AgentEvent;
        use agent_sdk_foundation::llm::Message;
        use agent_server::journal::checkpoint::{CheckpointKind, NewCheckpointParams};
        use agent_server::journal::fork_transaction::{AtomicForkCommitter, ForkCommitParams};

        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        let now = t0();
        let new_thread_id = thread_id("pg-forked-rollback");

        // Both checkpoints need real task rows behind them (the
        // message_commits → tasks FK only cares that the task exists).
        let pre_thread_id = thread_id("pg-pre-thread");
        ThreadStore::get_or_create(&store, &pre_thread_id, now).await?;
        let pre_task = AgentTask::new_root_turn(pre_thread_id.clone(), now, 3);
        let pre_task_id = pre_task.id.clone();
        AgentTaskStore::submit_root_turn(&store, pre_task).await?;

        let source_thread_id = thread_id("pg-fork-rollback-source");
        ThreadStore::get_or_create(&store, &source_thread_id, now).await?;
        let source_task = AgentTask::new_root_turn(source_thread_id.clone(), now, 3);
        let source_task_id = source_task.id.clone();
        AgentTaskStore::submit_root_turn(&store, source_task).await?;

        // Pre-insert a checkpoint at turn 1 on the destination thread so the
        // fork's checkpoint insert collides and rolls the transaction back.
        ThreadStore::get_or_create(&store, &new_thread_id, now).await?;
        CheckpointStore::commit_checkpoint(
            &store,
            NewCheckpointParams {
                kind: CheckpointKind::FullTurn,
                thread_id: new_thread_id.clone(),
                turn_number: 1,
                task_id: pre_task_id,
                messages: vec![],
                agent_state_snapshot: serde_json::json!({
                    "thread_id": new_thread_id.0.clone(),
                    "turn_count": 0,
                    "total_usage": TokenUsage::default(),
                    "metadata": {},
                    "created_at": "2023-11-14T00:00:00Z",
                }),
                turn_usage: TokenUsage::default(),
                now,
            },
        )
        .await?;

        let fresh_messages = vec![Message::user("seeded")];
        let params = ForkCommitParams {
            new_thread_id: new_thread_id.clone(),
            creation: None,
            now,
            committed_turns: 1,
            cumulative_total_usage: TokenUsage::default(),
            messages: fresh_messages.clone(),
            checkpoint: Some(NewCheckpointParams {
                kind: CheckpointKind::FullTurn,
                thread_id: new_thread_id.clone(),
                turn_number: 1,
                task_id: source_task_id,
                messages: fresh_messages.clone(),
                agent_state_snapshot: serde_json::json!({
                    "thread_id": new_thread_id.0.clone(),
                    "turn_count": 1,
                    "total_usage": TokenUsage::default(),
                    "metadata": {},
                    "created_at": "2023-11-14T00:00:00Z",
                }),
                turn_usage: TokenUsage::default(),
                now,
            }),
            events: vec![AgentEvent::start(source_thread_id.clone(), 1)],
        };
        let err = store.commit_fork_atomic(params).await;
        anyhow::ensure!(
            err.is_err(),
            "checkpoint collision must fail the fork transaction",
        );

        // The rolled-back fork must not leave the seeded projection / events
        // behind on the destination.
        let projection = MessageProjectionStore::get_history(&store, &new_thread_id).await?;
        assert!(
            projection.is_empty(),
            "rolled-back fork must not leave seeded messages on the destination",
        );
        let committed = EventRepository::get_events(&store, &new_thread_id).await?;
        assert!(
            committed.is_empty(),
            "rolled-back fork must not leave events on the destination",
        );
        Ok(())
    }
}
