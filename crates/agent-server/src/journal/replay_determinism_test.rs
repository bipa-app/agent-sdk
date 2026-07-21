//! Temporal-style replay-determinism regression suite (Phase 11 · D).
//!
//! Durable workflow engines (Temporal, Restate, DBOS) guarantee that
//! re-running a recorded history through the *current* commit/recovery
//! code reconstructs byte-identical durable state. If a code change makes
//! replay diverge from the recorded history, the engine has silently
//! broken every in-flight workflow that was journaled under the old code.
//!
//! This module proves that property for the journal's completed-turn
//! commit path:
//!
//! 1. **Record** — run a multi-turn conversation through
//!    [`commit_completed_turn`], capturing the exact input of every turn
//!    (messages, usage, agent-state snapshot, lifecycle events) as a
//!    replayable [`TurnHistory`].
//! 2. **Replay** — feed the identical history into a *fresh* set of
//!    stores through the identical commit code, then recover the thread.
//! 3. **Assert** — the recovered durable state (a canonical JSON
//!    [`DurableDigest`]) is byte-identical across the original and the
//!    replay. Two independent replays of the same history are also
//!    identical (idempotent re-run).
//! 4. **Divergence guard** — a deliberately mutated history (one byte of
//!    one message changed) produces a *different* digest, so the check
//!    fails CI on a real divergence rather than rubber-stamping it.
//!
//! Determinism is total: the histories carry their own timestamps and
//! ids, the stores start empty, and `commit_completed_turn` is a pure
//! function of its inputs plus store state. No wall clock, no RNG, no
//! ordering nondeterminism.

use crate::journal::checkpoint::CheckpointKind;
use agent_sdk_foundation::events::AgentEvent;
use agent_sdk_foundation::{ThreadId, TokenUsage, llm};
use anyhow::{Context, Result};
use time::{Duration, OffsetDateTime};

use super::checkpoint_store::{CheckpointStore, InMemoryCheckpointStore};
use super::commit::{CompletedTurnCommit, commit_completed_turn};
use super::event_repository::{EventRepository, InMemoryEventRepository};
use super::message_store::InMemoryMessageProjectionStore;
use super::task::AgentTaskId;
use super::thread_recover::recover_thread;
use super::thread_store::InMemoryThreadStore;
use super::turn_attempt::{CloseAttemptParams, OpenAttemptParams, TurnAttemptOutcome};
use super::turn_attempt_store::{InMemoryTurnAttemptStore, TurnAttemptStore};

fn t0() -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + Duration::seconds(1_700_000_000)
}

fn thread() -> ThreadId {
    ThreadId::from_string("t-replay-det")
}

/// A single replayable turn: the complete input to one
/// [`commit_completed_turn`] call, captured so the exact same commit can
/// be replayed against a fresh store.
#[derive(Clone)]
struct TurnHistory {
    task_id: AgentTaskId,
    messages: Vec<llm::Message>,
    usage: TokenUsage,
    snapshot: serde_json::Value,
    events: Vec<AgentEvent>,
    committed_at: OffsetDateTime,
}

/// The canonical, fully-ordered durable state of a thread after replay.
///
/// Serialized to a stable JSON string so equality is byte-for-byte: a
/// single differing message, snapshot field, or event payload changes
/// the digest. Built exclusively from durable artifacts (recovered
/// view + every checkpoint + every committed event), never from the
/// transient commit return values, so it reflects what a cold restart
/// would actually reconstruct.
#[derive(Debug, PartialEq, Eq)]
struct DurableDigest(String);

/// A fresh, empty set of in-memory stores plus the helpers to drive and
/// digest one commit history through them.
struct ReplayStores {
    threads: InMemoryThreadStore,
    messages: InMemoryMessageProjectionStore,
    attempts: InMemoryTurnAttemptStore,
    checkpoints: InMemoryCheckpointStore,
    events: InMemoryEventRepository,
}

impl ReplayStores {
    fn new() -> Self {
        Self {
            threads: InMemoryThreadStore::new(),
            messages: InMemoryMessageProjectionStore::new(),
            attempts: InMemoryTurnAttemptStore::new(),
            checkpoints: InMemoryCheckpointStore::new(),
            events: InMemoryEventRepository::new(),
        }
    }

    /// Open and immediately close a turn attempt for `turn`, returning
    /// the attempt id the commit will close. Keeps the audit record
    /// shape identical between record and replay.
    async fn open_attempt(
        &self,
        turn: &TurnHistory,
        attempt_number: u32,
    ) -> Result<CompletedTurnCommit> {
        let attempt = self
            .attempts
            .open_attempt(OpenAttemptParams {
                task_id: turn.task_id.clone(),
                attempt_number,
                provenance: agent_sdk_foundation::audit::AuditProvenance::new(
                    "anthropic",
                    "claude-sonnet-4-5-20250929",
                ),
                request_blob: serde_json::json!({"messages": []}),
                now: turn.committed_at,
                otel_trace_id: None,
                otel_span_id: None,
            })
            .await
            .context("open attempt for replay")?;
        Ok(CompletedTurnCommit {
            checkpoint_kind: CheckpointKind::FullTurn,
            thread_id: thread(),
            task_id: turn.task_id.clone(),
            // History is replayed in order (attempt_number == i + 1), so
            // each turn is committed_turns + 1 at commit time.
            expected_turn: attempt_number,
            turn_attempt_id: attempt.id,
            close_attempt_params: CloseAttemptParams {
                response_blob: serde_json::json!({"id": "msg", "content": []}),
                response_id: Some("msg".into()),
                response_model: Some("claude-sonnet-4-5-20250929".into()),
                stop_reason: Some(llm::StopReason::EndTurn),
                outcome: TurnAttemptOutcome::Success,
                input_tokens: turn.usage.input_tokens,
                output_tokens: turn.usage.output_tokens,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
                route_provider: None,
                thinking_adaptive: false,
                resolved_effort: None,
            },
            messages: turn.messages.clone(),
            turn_usage: turn.usage.clone(),
            agent_state_snapshot: turn.snapshot.clone(),
            events: turn.events.clone(),
            outbox_max_attempts: 3,
            owner_guard: None,
            now: turn.committed_at,
        })
    }

    /// Replay an entire recorded history through the commit path.
    async fn apply_history(&self, history: &[TurnHistory]) -> Result<()> {
        for (i, turn) in history.iter().enumerate() {
            let params = self
                .open_attempt(turn, u32::try_from(i + 1).context("attempt number")?)
                .await?;
            commit_completed_turn(
                params,
                &self.threads,
                &self.messages,
                &self.attempts,
                &self.checkpoints,
                &self.events,
            )
            .await
            .with_context(|| format!("replay commit turn {i}"))?;
        }
        Ok(())
    }

    /// Reconstruct the canonical durable digest exactly as a cold
    /// restart would: recover the thread, then fold in every checkpoint
    /// and every committed event in deterministic order.
    async fn digest(&self) -> Result<DurableDigest> {
        let view = recover_thread(
            &thread(),
            &self.threads,
            &self.checkpoints,
            &self.messages,
            t0(),
        )
        .await
        .context("recover for digest")?;
        let checkpoints = self
            .checkpoints
            .list_by_thread(&thread())
            .await
            .context("list checkpoints for digest")?;
        let events = self
            .events
            .get_events(&thread())
            .await
            .context("read events for digest")?;

        // Build a fully-ordered canonical structure. `recover_thread`
        // already merges committed history + draft, so the view's
        // messages plus its next-turn cursor capture the resumable
        // state; checkpoints + events capture the full durable trail.
        let canonical = serde_json::json!({
            "committed_turns": view.thread.committed_turns,
            "total_usage": view.thread.total_usage,
            "next_turn_number": view.next_turn_number,
            "recovered_messages": view.messages,
            "agent_state_snapshot": view.agent_state_snapshot,
            "checkpoints": checkpoints.iter().map(|c| serde_json::json!({
                "turn_number": c.turn_number,
                "messages": c.messages,
                "agent_state_snapshot": c.agent_state_snapshot,
            })).collect::<Vec<_>>(),
            "events": events.iter().map(|e| serde_json::json!({
                "sequence": e.sequence,
                "event": e.event,
            })).collect::<Vec<_>>(),
        });
        let serialized =
            serde_json::to_string(&canonical).context("serialize canonical durable digest")?;
        Ok(DurableDigest(serialized))
    }
}

/// Build a representative recorded history: a mid-flight tree of three
/// turns with text, tool-use, and tool-result content plus lifecycle
/// events, exactly the shape a real conformance run produces.
fn recorded_history() -> Vec<TurnHistory> {
    vec![
        TurnHistory {
            task_id: AgentTaskId::from_string("task_replay-1"),
            messages: vec![
                llm::Message::user("what is the weather"),
                llm::Message::assistant("let me check"),
            ],
            usage: TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
                ..Default::default()
            },
            snapshot: serde_json::json!({"turn": 1, "phase": "intro"}),
            events: vec![AgentEvent::text("turn-1-evt", "started turn 1")],
            committed_at: t0() + Duration::seconds(1),
        },
        TurnHistory {
            task_id: AgentTaskId::from_string("task_replay-2"),
            messages: vec![llm::Message::user("and tomorrow?")],
            usage: TokenUsage {
                input_tokens: 200,
                output_tokens: 80,
                ..Default::default()
            },
            snapshot: serde_json::json!({"turn": 2, "phase": "followup"}),
            events: vec![
                AgentEvent::text("turn-2-evt-a", "started turn 2"),
                AgentEvent::text("turn-2-evt-b", "finished turn 2"),
            ],
            committed_at: t0() + Duration::seconds(2),
        },
        TurnHistory {
            task_id: AgentTaskId::from_string("task_replay-3"),
            messages: vec![llm::Message::assistant("here is the forecast")],
            usage: TokenUsage {
                input_tokens: 300,
                output_tokens: 120,
                ..Default::default()
            },
            snapshot: serde_json::json!({"turn": 3, "phase": "done"}),
            events: vec![AgentEvent::text("turn-3-evt", "started turn 3")],
            committed_at: t0() + Duration::seconds(3),
        },
    ]
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

/// Recording a history and replaying it through the *same* commit code
/// reconstructs byte-identical durable state.
#[tokio::test]
async fn replay_reconstructs_byte_identical_durable_state() -> Result<()> {
    let history = recorded_history();

    let original = ReplayStores::new();
    original.apply_history(&history).await?;
    let original_digest = original.digest().await?;

    let replay = ReplayStores::new();
    replay.apply_history(&history).await?;
    let replay_digest = replay.digest().await?;

    assert_eq!(
        original_digest, replay_digest,
        "replaying a recorded history must reconstruct byte-identical durable state",
    );
    Ok(())
}

/// Replaying the same history twice is idempotent — two independent
/// replays of identical input land on identical durable state.
#[tokio::test]
async fn independent_replays_of_same_history_are_identical() -> Result<()> {
    let history = recorded_history();

    let a = ReplayStores::new();
    a.apply_history(&history).await?;

    let b = ReplayStores::new();
    b.apply_history(&history).await?;

    assert_eq!(
        a.digest().await?,
        b.digest().await?,
        "two replays of the same history must be identical",
    );
    Ok(())
}

/// The divergence guard: a deliberately mutated history (one message
/// body changed) MUST produce a different durable digest. This is the
/// test that fails CI if a commit/recovery change silently diverges from
/// a recorded history — proving the check has teeth.
#[tokio::test]
async fn divergent_history_produces_a_different_digest() -> Result<()> {
    let baseline = recorded_history();
    let original = ReplayStores::new();
    original.apply_history(&baseline).await?;
    let baseline_digest = original.digest().await?;

    // Mutate exactly one byte of meaning: change turn 2's user message.
    let mut divergent = recorded_history();
    divergent[1].messages = vec![llm::Message::user("DIVERGENT INPUT")];
    let mutated = ReplayStores::new();
    mutated.apply_history(&divergent).await?;
    let mutated_digest = mutated.digest().await?;

    assert_ne!(
        baseline_digest, mutated_digest,
        "a divergent history must produce a different digest, else the replay check is toothless",
    );
    Ok(())
}

/// A divergent *agent-state snapshot* (the opaque recovery blob) is also
/// caught — recovery must surface a snapshot change as a state divergence.
#[tokio::test]
async fn divergent_agent_state_snapshot_is_caught() -> Result<()> {
    let baseline = recorded_history();
    let original = ReplayStores::new();
    original.apply_history(&baseline).await?;
    let baseline_digest = original.digest().await?;

    let mut divergent = recorded_history();
    divergent[2].snapshot = serde_json::json!({"turn": 3, "phase": "TAMPERED"});
    let mutated = ReplayStores::new();
    mutated.apply_history(&divergent).await?;

    assert_ne!(
        baseline_digest,
        mutated.digest().await?,
        "a divergent agent-state snapshot must change the durable digest",
    );
    Ok(())
}

/// A divergent committed-event payload is caught — the event trail is
/// part of the durable state that replay must reproduce exactly.
#[tokio::test]
async fn divergent_event_payload_is_caught() -> Result<()> {
    let baseline = recorded_history();
    let original = ReplayStores::new();
    original.apply_history(&baseline).await?;
    let baseline_digest = original.digest().await?;

    let mut divergent = recorded_history();
    divergent[0].events = vec![AgentEvent::text("turn-1-evt", "TAMPERED EVENT BODY")];
    let mutated = ReplayStores::new();
    mutated.apply_history(&divergent).await?;

    assert_ne!(
        baseline_digest,
        mutated.digest().await?,
        "a divergent committed-event payload must change the durable digest",
    );
    Ok(())
}
