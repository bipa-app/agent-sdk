//! Repository boundaries for the future Postgres backend.
//!
//! The goal here is to keep the eventual SQL implementation aligned to
//! the already-shipped store traits instead of growing a second, subtly
//! different data model inside the backend crate.
//!
//! The eventual implementation is expected to use `sqlx::PgPool` for
//! startup and read paths, and `sqlx::Transaction<'_, sqlx::Postgres>`
//! for the multi-row units of work called out below.

/// A table-backed repository boundary aligned to one existing store trait.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RepositoryBoundary {
    pub name: &'static str,
    pub store_trait: &'static str,
    pub tables: &'static [&'static str],
    pub reads: &'static [&'static str],
    pub writes: &'static [&'static str],
    pub invariants: &'static [&'static str],
    pub transaction_notes: &'static str,
}

/// A multi-repository unit of work that must share one SQL transaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitOfWorkContract {
    pub name: &'static str,
    pub tables: &'static [&'static str],
    pub requirement: &'static str,
}

const AGENT_TASK_READS: &[&str] = &[
    "get",
    "list_by_thread",
    "list_children",
    "list_by_status",
    "active_root_for_thread",
    "list_queued_roots",
];

const AGENT_TASK_WRITES: &[&str] = &[
    "insert",
    "submit_root_turn",
    "update",
    "promote_next_queued_root",
    "try_acquire_task",
    "acquire_next_runnable",
    "heartbeat_task",
    "release_expired_leases",
    "pause_on_children",
    "pause_on_confirmation",
    "spawn_tool_children",
    "complete_task",
    "complete_task_with_result",
    "fail_task",
    "cancel_tree",
    "resume_from_confirmation",
    "reject_confirmation",
    "clear",
];

const THREAD_READS: &[&str] = &["get", "list"];
const THREAD_WRITES: &[&str] = &["get_or_create", "commit_turn", "mark_completed"];
const MESSAGE_READS: &[&str] = &["get_or_create", "get", "get_history"];
const MESSAGE_WRITES: &[&str] = &["commit_messages", "replace_history"];
const TURN_ATTEMPT_READS: &[&str] = &["get", "list_by_task"];
const TURN_ATTEMPT_WRITES: &[&str] = &["open_attempt", "close_attempt"];
const TURN_CHECKPOINT_READS: &[&str] = &[
    "get",
    "get_by_turn",
    "get_latest_by_thread",
    "list_by_thread",
];
const TURN_CHECKPOINT_WRITES: &[&str] = &["commit_checkpoint"];

const REPOSITORIES: &[RepositoryBoundary] = &[
    RepositoryBoundary {
        name: "task_repository",
        store_trait: "agent_server::journal::store::AgentTaskStore",
        tables: &["agent_sdk_tasks"],
        reads: AGENT_TASK_READS,
        writes: AGENT_TASK_WRITES,
        invariants: &[
            "one blocking root per thread",
            "queued roots promote in `(created_at, id)` FIFO order",
            "runnable scans lease only `pending` rows",
            "heartbeat CAS checks both `worker_id` and `lease_id`",
            "lease sweeps walk `lease_expires_at` without scanning live rows",
            "state JSON kind agrees with paused status",
            "parent-child recompute uses durable child rows, not caller-maintained counters",
        ],
        transaction_notes: "Most methods are single-table transactions. `spawn_tool_children`, `complete_task*`, `fail_task`, and `cancel_tree` are multi-row mutations inside `agent_sdk_tasks` and must execute in one SQL transaction or one retry-safe stored procedure boundary.",
    },
    RepositoryBoundary {
        name: "thread_repository",
        store_trait: "agent_server::journal::thread_store::ThreadStore",
        tables: &["agent_sdk_threads"],
        reads: THREAD_READS,
        writes: THREAD_WRITES,
        invariants: &[
            "`commit_turn` is the only path that mutates aggregate counters",
            "completed threads reject later commits",
            "`get_or_create` is idempotent",
        ],
        transaction_notes: "Single-row UPSERT / UPDATE contract. Thread counters must not be exposed to generic updates.",
    },
    RepositoryBoundary {
        name: "message_repository",
        store_trait: "agent_server::journal::message_store::MessageProjectionStore",
        tables: &["agent_sdk_message_heads", "agent_sdk_message_commits"],
        reads: MESSAGE_READS,
        writes: MESSAGE_WRITES,
        invariants: &[
            "one message-head row exists per thread",
            "the current SDK contract persists both tables only from the successful completed-turn path and the compaction path",
            "`commit_messages` appends the raw turn batch to `agent_sdk_message_commits` and updates `agent_sdk_message_heads` in the same transaction as `commit_completed_turn` step 3",
            "`replace_history` rewrites only `agent_sdk_message_heads` during SDK compaction",
            "compaction never deletes or updates rows in `agent_sdk_message_commits`",
            "recovery remains checkpoint-driven; raw committed message batches stay available for audit/rebuild work",
        ],
        transaction_notes: "Under the current SDK contract, `commit_messages` runs only inside the successful `commit_completed_turn` transaction: append immutable raw batch row, then advance the mutable projection head. `replace_history` is a compaction-only, head-only update and must leave `agent_sdk_message_commits` untouched.",
    },
    RepositoryBoundary {
        name: "turn_attempt_repository",
        store_trait: "agent_server::journal::turn_attempt_store::TurnAttemptStore",
        tables: &["agent_sdk_turn_attempts"],
        reads: TURN_ATTEMPT_READS,
        writes: TURN_ATTEMPT_WRITES,
        invariants: &[
            "open rows are append-only placeholders",
            "close_attempt is the only mutation",
            "`(task_id, attempt_number)` defines durable ordering within a task",
        ],
        transaction_notes: "Insert and close are single-row operations, but `close_attempt` participates in the cross-table completed-turn commit transaction.",
    },
    RepositoryBoundary {
        name: "turn_checkpoint_repository",
        store_trait: "agent_server::journal::checkpoint_store::CheckpointStore",
        tables: &["agent_sdk_turn_checkpoints"],
        reads: TURN_CHECKPOINT_READS,
        writes: TURN_CHECKPOINT_WRITES,
        invariants: &[
            "checkpoints are immutable after insert",
            "exactly one row exists per `(thread_id, turn_number)`",
            "latest recovery reads by descending `turn_number`",
        ],
        transaction_notes: "Append-only table. `commit_checkpoint` joins the same transaction as attempt close, thread aggregate update, and message head/batch updates.",
    },
];

const UNITS_OF_WORK: &[UnitOfWorkContract] = &[UnitOfWorkContract {
    name: "commit_completed_turn",
    tables: &[
        "agent_sdk_turn_attempts",
        "agent_sdk_threads",
        "agent_sdk_message_heads",
        "agent_sdk_message_commits",
        "agent_sdk_turn_checkpoints",
    ],
    requirement: "Close the open turn attempt, advance thread aggregates, append the raw message batch, update the current message head, and insert the checkpoint in one SQL transaction so recovery never observes a partial turn commit. This is the only path that persists new rows into `agent_sdk_message_commits`.",
}];

/// Repository boundaries the future Postgres implementation must honor.
#[must_use]
pub const fn repository_boundaries() -> &'static [RepositoryBoundary] {
    REPOSITORIES
}

/// Cross-repository units of work that must share one SQL transaction.
#[must_use]
pub const fn completed_turn_units_of_work() -> &'static [UnitOfWorkContract] {
    UNITS_OF_WORK
}
