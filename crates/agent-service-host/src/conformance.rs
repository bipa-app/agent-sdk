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
    use agent_sdk_foundation::{ThreadId, TokenUsage};
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
            .heartbeat_task(&root.id, &worker, &lease, t_plus(60), t_plus(35))
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
                thread_id: tid.clone(),
                task_id: root.id.clone(),
                expected_turn: 1,
                turn_attempt_id: attempt.id.clone(),
                close_attempt_params: CloseAttemptParams {
                    response_blob: serde_json::json!({"id": "msg_conf_1"}),
                    response_id: Some("msg_conf_1".into()),
                    response_model: Some("claude-sonnet-4-5-20250929".into()),
                    stop_reason: Some(agent_sdk_foundation::llm::StopReason::EndTurn),
                    outcome: TurnAttemptOutcome::Success,
                    input_tokens: 100,
                    output_tokens: 50,
                    cached_input_tokens: 0,
                },
                messages: vec![agent_sdk_foundation::llm::Message::assistant(
                    "test response",
                )],
                turn_usage,
                agent_state_snapshot: serde_json::json!({}),
                events: vec![],
                outbox_max_attempts: 3,
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

        let latest = checkpoint_store
            .get_latest_by_thread(&tid)
            .await?
            .context("latest checkpoint missing")?;
        assert_eq!(latest.turn_number, 1);
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

        let cancelled = task_store.cancel_tree(&root.id, t_plus(73)).await?;
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

        let cancelled = task_store.cancel_tree(&parent.id, t_plus(10)).await?;
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

        let cancelled = task_store.cancel_tree(&child_root.id, t_plus(10)).await?;
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
    }

    fn fresh_in_memory_stores() -> InMemoryStores {
        use agent_server::journal::checkpoint_store::InMemoryCheckpointStore;
        use agent_server::journal::event_repository::InMemoryEventRepository;
        use agent_server::journal::message_store::InMemoryMessageProjectionStore;
        use agent_server::journal::store::InMemoryAgentTaskStore;
        use agent_server::journal::thread_store::InMemoryThreadStore;
        use agent_server::journal::turn_attempt_store::InMemoryTurnAttemptStore;

        InMemoryStores {
            task: std::sync::Arc::new(InMemoryAgentTaskStore::new()),
            thread: std::sync::Arc::new(InMemoryThreadStore::new()),
            message: std::sync::Arc::new(InMemoryMessageProjectionStore::new()),
            attempt: std::sync::Arc::new(InMemoryTurnAttemptStore::new()),
            checkpoint: std::sync::Arc::new(InMemoryCheckpointStore::new()),
            event: std::sync::Arc::new(InMemoryEventRepository::new()),
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
    async fn conformance_sqlite_completed_turn() -> Result<()> {
        let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
        test_completed_turn_commit(&store, &store, &store, &store, &store, &store).await
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
    async fn conformance_postgres_completed_turn() -> Result<()> {
        let Some((store, _guard)) = pg_test_store().await? else {
            return Ok(());
        };
        test_completed_turn_commit(&store, &store, &store, &store, &store, &store).await
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
        use agent_server::journal::checkpoint::NewCheckpointParams;
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
            now,
            committed_turns: 1,
            cumulative_total_usage: TokenUsage::default(),
            messages: messages.clone(),
            checkpoint: Some(NewCheckpointParams {
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
        use agent_server::journal::checkpoint::NewCheckpointParams;
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
            now,
            committed_turns: 1,
            cumulative_total_usage: TokenUsage::default(),
            messages: fresh_messages.clone(),
            checkpoint: Some(NewCheckpointParams {
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
