//! Table, constraint, and index metadata for the `sqlx` Postgres durable core.
//!
//! The SQL migration text is the review artifact humans read. This file
//! is the machine-readable checklist tests validate against so the
//! contract does not silently drift. These structs are review/test
//! fixtures, not runtime ORM or query-model types.

/// One durable table in the current Postgres contract, captured as
/// review/test metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableContract {
    pub name: &'static str,
    pub purpose: &'static str,
    pub columns: &'static [ColumnContract],
    pub constraints: &'static [ConstraintContract],
    pub indexes: &'static [IndexContract],
}

/// One SQL column in a durable table contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnContract {
    pub name: &'static str,
    pub sql_type: &'static str,
    pub nullable: bool,
    pub notes: &'static str,
}

/// One named constraint in a durable table contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConstraintContract {
    pub name: &'static str,
    pub invariant: &'static str,
}

/// One named secondary index in a durable table contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexContract {
    pub name: &'static str,
    pub key_columns: &'static str,
    pub predicate: Option<&'static str>,
    pub purpose: &'static str,
}

const AGENT_SDK_TASK_COLUMNS: &[ColumnContract] = &[
    ColumnContract {
        name: "id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Primary task identity (`task_<uuid>` wire form).",
    },
    ColumnContract {
        name: "kind",
        sql_type: "TEXT",
        nullable: false,
        notes: "Task kind discriminant (`root_turn`, `tool_runtime`, `subagent`).",
    },
    ColumnContract {
        name: "status",
        sql_type: "TEXT",
        nullable: false,
        notes: "Lifecycle state used for admission, leasing, waiting, and terminal scans.",
    },
    ColumnContract {
        name: "parent_id",
        sql_type: "TEXT",
        nullable: true,
        notes: "Null for roots; points at the direct parent for descendants.",
    },
    ColumnContract {
        name: "root_id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Root task identity for the full task tree.",
    },
    ColumnContract {
        name: "depth",
        sql_type: "BIGINT",
        nullable: false,
        notes: "0 for root, parent depth + 1 for descendants.",
    },
    ColumnContract {
        name: "thread_id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Every task is thread-bound so queueing and recovery stay local, and the row must reference an existing thread.",
    },
    ColumnContract {
        name: "submitted_input_json",
        sql_type: "JSONB",
        nullable: false,
        notes: "Durable root-turn submission payload captured from the external transport.",
    },
    ColumnContract {
        name: "worker_id",
        sql_type: "TEXT",
        nullable: true,
        notes: "Lease owner worker identity.",
    },
    ColumnContract {
        name: "lease_id",
        sql_type: "TEXT",
        nullable: true,
        notes: "Per-acquisition CAS token.",
    },
    ColumnContract {
        name: "lease_expires_at",
        sql_type: "TIMESTAMPTZ",
        nullable: true,
        notes: "Lease expiry cursor for sweep-based requeue/fail-closed recovery.",
    },
    ColumnContract {
        name: "last_heartbeat_at",
        sql_type: "TIMESTAMPTZ",
        nullable: true,
        notes: "Last successful heartbeat under the active lease CAS.",
    },
    ColumnContract {
        name: "state_json",
        sql_type: "JSONB",
        nullable: false,
        notes: "Typed `TaskState` durable payload, including ready-to-resume envelopes.",
    },
    ColumnContract {
        name: "attempt",
        sql_type: "BIGINT",
        nullable: false,
        notes: "Current retry counter.",
    },
    ColumnContract {
        name: "max_attempts",
        sql_type: "BIGINT",
        nullable: false,
        notes: "Retry budget ceiling.",
    },
    ColumnContract {
        name: "last_error",
        sql_type: "TEXT",
        nullable: true,
        notes: "Canonical fail-closed error for `failed` rows.",
    },
    ColumnContract {
        name: "pending_child_count",
        sql_type: "BIGINT",
        nullable: false,
        notes: "Derived live-child count while the parent is waiting on children.",
    },
    ColumnContract {
        name: "spawn_index",
        sql_type: "BIGINT",
        nullable: true,
        notes: "Stable sibling position for tool-runtime child batches.",
    },
    ColumnContract {
        name: "result_payload",
        sql_type: "JSONB",
        nullable: true,
        notes: "Serialized tool result for completed tool-runtime children.",
    },
    ColumnContract {
        name: "created_at",
        sql_type: "TIMESTAMPTZ",
        nullable: false,
        notes: "Admission timestamp and FIFO tiebreak input.",
    },
    ColumnContract {
        name: "updated_at",
        sql_type: "TIMESTAMPTZ",
        nullable: false,
        notes: "Last durable mutation timestamp.",
    },
    ColumnContract {
        name: "completed_at",
        sql_type: "TIMESTAMPTZ",
        nullable: true,
        notes: "Required when the task is terminal.",
    },
];

const AGENT_SDK_TASK_CONSTRAINTS: &[ConstraintContract] = &[
    ConstraintContract {
        name: "agent_sdk_tasks_parent_fk",
        invariant: "Parent linkage is durable and self-referential.",
    },
    ConstraintContract {
        name: "agent_sdk_tasks_root_fk",
        invariant: "Every row points at an existing root task.",
    },
    ConstraintContract {
        name: "agent_sdk_tasks_kind_check",
        invariant: "Only known task kinds can be persisted.",
    },
    ConstraintContract {
        name: "agent_sdk_tasks_status_check",
        invariant: "Only known task statuses can be persisted.",
    },
    ConstraintContract {
        name: "agent_sdk_tasks_root_identity_check",
        invariant: "Depth / parent_id / root_id agree with root-vs-child identity.",
    },
    ConstraintContract {
        name: "agent_sdk_tasks_depth_kind_check",
        invariant: "Depth is zero if and only if the task kind is `root_turn`.",
    },
    ConstraintContract {
        name: "agent_sdk_tasks_depth_non_negative_check",
        invariant: "Depth cannot go negative.",
    },
    ConstraintContract {
        name: "agent_sdk_tasks_thread_fk",
        invariant: "Every task belongs to an existing thread row.",
    },
    ConstraintContract {
        name: "agent_sdk_tasks_attempt_bounds_check",
        invariant: "Retry counters stay within the configured budget.",
    },
    ConstraintContract {
        name: "agent_sdk_tasks_pending_child_count_check",
        invariant: "Pending child count cannot go negative.",
    },
    ConstraintContract {
        name: "agent_sdk_tasks_spawn_index_check",
        invariant: "Spawn index, when present, is non-negative.",
    },
    ConstraintContract {
        name: "agent_sdk_tasks_queue_kind_check",
        invariant: "`queued` rows are root turns only.",
    },
    ConstraintContract {
        name: "agent_sdk_tasks_lease_atomicity_check",
        invariant: "Worker/lease fields are either all absent or all present together.",
    },
    ConstraintContract {
        name: "agent_sdk_tasks_running_lease_check",
        invariant: "Only `running` rows may carry an active lease.",
    },
    ConstraintContract {
        name: "agent_sdk_tasks_terminal_completion_check",
        invariant: "Terminal rows require `completed_at`; non-terminal rows must not set it.",
    },
    ConstraintContract {
        name: "agent_sdk_tasks_failure_payload_check",
        invariant: "Only `failed` rows may carry `last_error`, and every `failed` row must carry one.",
    },
    ConstraintContract {
        name: "agent_sdk_tasks_waiting_state_check",
        invariant: "Paused-state JSON kind must be a non-null known value; `waiting_on_children` rows require `status = waiting_on_children` with a positive pending-child count; `subagent_invocation` rows are either still waiting (`status = waiting_on_children` with a positive pending-child count) or ready to materialize (`status IN ('pending', 'running')` with `pending_child_count = 0`); terminal rows reset state kind to `none`; `ready_to_resume` is valid only for pending/running rows and never for queued rows.",
    },
];

const AGENT_SDK_TASK_INDEXES: &[IndexContract] = &[
    IndexContract {
        name: "agent_sdk_tasks_by_thread_idx",
        key_columns: "(thread_id, created_at, id)",
        predicate: None,
        purpose: "Lists a thread's task history without table scans.",
    },
    IndexContract {
        name: "agent_sdk_tasks_by_parent_idx",
        key_columns: "(parent_id, created_at, id)",
        predicate: Some("parent_id IS NOT NULL"),
        purpose: "Lists direct children and powers child-batch recompute/cancel walks.",
    },
    IndexContract {
        name: "agent_sdk_tasks_by_status_idx",
        key_columns: "(status, created_at, id)",
        predicate: None,
        purpose: "Supports status-bucket inspection and debug scans.",
    },
    IndexContract {
        name: "agent_sdk_tasks_root_admission_slot_idx",
        key_columns: "(thread_id)",
        predicate: Some(
            "kind = 'root_turn' AND status IN ('pending', 'running', 'waiting_on_children', 'awaiting_confirmation')",
        ),
        purpose: "Enforces the one-blocking-root-per-thread admission invariant.",
    },
    IndexContract {
        name: "agent_sdk_tasks_queued_roots_fifo_idx",
        key_columns: "(thread_id, created_at, id)",
        predicate: Some("kind = 'root_turn' AND status = 'queued'"),
        purpose: "Provides deterministic FIFO promotion for queued roots.",
    },
    IndexContract {
        name: "agent_sdk_tasks_runnable_fifo_idx",
        key_columns: "(created_at, id)",
        predicate: Some("status = 'pending'"),
        purpose: "Drives global runnable scans across root and tool-runtime tasks.",
    },
    IndexContract {
        name: "agent_sdk_tasks_running_lease_expiry_idx",
        key_columns: "(lease_expires_at, id)",
        predicate: Some("status = 'running'"),
        purpose: "Drives lease-expiry sweeps proportional to expired rows.",
    },
    IndexContract {
        name: "agent_sdk_tasks_root_tree_idx",
        key_columns: "(root_id, depth, created_at, id)",
        predicate: None,
        purpose: "Keeps task-tree traversals and root-scoped inspection predictable.",
    },
];

const AGENT_SDK_THREAD_COLUMNS: &[ColumnContract] = &[
    ColumnContract {
        name: "thread_id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Primary thread identity.",
    },
    ColumnContract {
        name: "status",
        sql_type: "TEXT",
        nullable: false,
        notes: "`active` or `completed`.",
    },
    ColumnContract {
        name: "committed_turns",
        sql_type: "BIGINT",
        nullable: false,
        notes: "Monotonic count of completed turns on this thread.",
    },
    ColumnContract {
        name: "total_input_tokens",
        sql_type: "BIGINT",
        nullable: false,
        notes: "Aggregate input usage across every committed turn.",
    },
    ColumnContract {
        name: "total_output_tokens",
        sql_type: "BIGINT",
        nullable: false,
        notes: "Aggregate output usage across every committed turn.",
    },
    ColumnContract {
        name: "created_at",
        sql_type: "TIMESTAMPTZ",
        nullable: false,
        notes: "Initial thread creation timestamp.",
    },
    ColumnContract {
        name: "updated_at",
        sql_type: "TIMESTAMPTZ",
        nullable: false,
        notes: "Last aggregate mutation timestamp.",
    },
];

const AGENT_SDK_THREAD_CONSTRAINTS: &[ConstraintContract] = &[
    ConstraintContract {
        name: "agent_sdk_threads_status_check",
        invariant: "Only known thread statuses can be persisted.",
    },
    ConstraintContract {
        name: "agent_sdk_threads_committed_turns_check",
        invariant: "Committed turn count cannot go negative.",
    },
    ConstraintContract {
        name: "agent_sdk_threads_total_usage_check",
        invariant: "Aggregate token usage fields are non-negative.",
    },
    ConstraintContract {
        name: "agent_sdk_threads_completed_turns_check",
        invariant: "Completed threads must have at least one committed turn.",
    },
];

const AGENT_SDK_THREAD_INDEXES: &[IndexContract] = &[IndexContract {
    name: "agent_sdk_threads_active_idx",
    key_columns: "(thread_id)",
    predicate: Some("status = 'active'"),
    purpose: "Keeps active-thread scans cheap for future operational queries.",
}];

const AGENT_SDK_MESSAGE_HEAD_COLUMNS: &[ColumnContract] = &[
    ColumnContract {
        name: "thread_id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Primary key and owning thread identity.",
    },
    ColumnContract {
        name: "history_json",
        sql_type: "JSONB",
        nullable: false,
        notes: "Current SDK-visible history, created/advanced by `commit_messages` and rewritten by `replace_history` during compaction.",
    },
    ColumnContract {
        name: "version",
        sql_type: "BIGINT",
        nullable: false,
        notes: "Monotonic optimistic-concurrency version.",
    },
    ColumnContract {
        name: "created_at",
        sql_type: "TIMESTAMPTZ",
        nullable: false,
        notes: "Projection creation timestamp.",
    },
    ColumnContract {
        name: "updated_at",
        sql_type: "TIMESTAMPTZ",
        nullable: false,
        notes: "Last append or compaction timestamp.",
    },
];

const AGENT_SDK_MESSAGE_HEAD_CONSTRAINTS: &[ConstraintContract] = &[
    ConstraintContract {
        name: "agent_sdk_message_heads_thread_fk",
        invariant: "Every message head belongs to an existing thread row.",
    },
    ConstraintContract {
        name: "agent_sdk_message_heads_history_json_check",
        invariant: "Current history is stored as a JSON array.",
    },
    ConstraintContract {
        name: "agent_sdk_message_heads_version_check",
        invariant: "Projection version is non-negative.",
    },
];

const AGENT_SDK_MESSAGE_HEAD_INDEXES: &[IndexContract] = &[];

const AGENT_SDK_MESSAGE_COMMIT_COLUMNS: &[ColumnContract] = &[
    ColumnContract {
        name: "thread_id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Owning thread identity.",
    },
    ColumnContract {
        name: "turn_number",
        sql_type: "BIGINT",
        nullable: false,
        notes: "1-indexed committed turn number.",
    },
    ColumnContract {
        name: "task_id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Task that produced the committed message batch.",
    },
    ColumnContract {
        name: "head_version_after",
        sql_type: "BIGINT",
        nullable: false,
        notes: "Message-head version immediately after this append batch committed.",
    },
    ColumnContract {
        name: "batch_json",
        sql_type: "JSONB",
        nullable: false,
        notes: "Raw committed message delta for the turn, persisted by `commit_messages` on successful completed-turn commit. Never rewritten by compaction.",
    },
    ColumnContract {
        name: "committed_at",
        sql_type: "TIMESTAMPTZ",
        nullable: false,
        notes: "Commit timestamp for the append batch.",
    },
];

const AGENT_SDK_MESSAGE_COMMIT_CONSTRAINTS: &[ConstraintContract] = &[
    ConstraintContract {
        name: "agent_sdk_message_commits_thread_fk",
        invariant: "Every committed batch belongs to an existing thread row.",
    },
    ConstraintContract {
        name: "agent_sdk_message_commits_task_fk",
        invariant: "Every committed batch points at the task that produced it.",
    },
    ConstraintContract {
        name: "agent_sdk_message_commits_thread_turn_key",
        invariant: "Exactly one raw committed message batch exists per `(thread_id, turn_number)`.",
    },
    ConstraintContract {
        name: "agent_sdk_message_commits_turn_number_check",
        invariant: "Turn numbers are 1-indexed and positive.",
    },
    ConstraintContract {
        name: "agent_sdk_message_commits_head_version_check",
        invariant: "Head versions are positive after a commit append.",
    },
    ConstraintContract {
        name: "agent_sdk_message_commits_batch_json_check",
        invariant: "Committed message batches are stored as JSON arrays and are append-only.",
    },
];

const AGENT_SDK_MESSAGE_COMMIT_INDEXES: &[IndexContract] = &[IndexContract {
    name: "agent_sdk_message_commits_by_task_idx",
    key_columns: "(task_id, turn_number)",
    predicate: None,
    purpose: "Supports task-oriented audit inspection of committed message batches.",
}];

const AGENT_SDK_TURN_ATTEMPT_COLUMNS: &[ColumnContract] = &[
    ColumnContract {
        name: "id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Primary attempt identity (`attempt_<uuid>` wire form).",
    },
    ColumnContract {
        name: "task_id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Owning task identity.",
    },
    ColumnContract {
        name: "attempt_number",
        sql_type: "BIGINT",
        nullable: false,
        notes: "1-indexed attempt counter within the task.",
    },
    ColumnContract {
        name: "provider",
        sql_type: "TEXT",
        nullable: false,
        notes: "Provider provenance.",
    },
    ColumnContract {
        name: "requested_model",
        sql_type: "TEXT",
        nullable: false,
        notes: "Requested model string.",
    },
    ColumnContract {
        name: "request_blob",
        sql_type: "JSONB",
        nullable: false,
        notes: "Opaque request payload.",
    },
    ColumnContract {
        name: "response_blob",
        sql_type: "JSONB",
        nullable: true,
        notes: "Opaque response payload, filled on close.",
    },
    ColumnContract {
        name: "response_id",
        sql_type: "TEXT",
        nullable: true,
        notes: "Provider response identity, when available.",
    },
    ColumnContract {
        name: "response_model",
        sql_type: "TEXT",
        nullable: true,
        notes: "Actual response model string, when available.",
    },
    ColumnContract {
        name: "stop_reason",
        sql_type: "TEXT",
        nullable: true,
        notes: "Provider stop reason discriminant.",
    },
    ColumnContract {
        name: "outcome",
        sql_type: "TEXT",
        nullable: true,
        notes: "Terminal attempt outcome.",
    },
    ColumnContract {
        name: "input_tokens",
        sql_type: "BIGINT",
        nullable: true,
        notes: "Prompt usage captured on close.",
    },
    ColumnContract {
        name: "output_tokens",
        sql_type: "BIGINT",
        nullable: true,
        notes: "Completion usage captured on close.",
    },
    ColumnContract {
        name: "cached_input_tokens",
        sql_type: "BIGINT",
        nullable: true,
        notes: "Provider-reported cached input usage.",
    },
    ColumnContract {
        name: "cache_creation_input_tokens",
        sql_type: "BIGINT",
        nullable: true,
        notes: "Provider-reported cache creation input usage.",
    },
    ColumnContract {
        name: "route_provider",
        sql_type: "TEXT",
        nullable: true,
        notes: "Stable native or gateway route identity captured at dispatch.",
    },
    ColumnContract {
        name: "resolved_effort",
        sql_type: "TEXT",
        nullable: true,
        notes: "Concrete thinking effort dispatched to the provider.",
    },
    ColumnContract {
        name: "opened_at",
        sql_type: "TIMESTAMPTZ",
        nullable: false,
        notes: "Attempt open timestamp.",
    },
    ColumnContract {
        name: "closed_at",
        sql_type: "TIMESTAMPTZ",
        nullable: true,
        notes: "Attempt close timestamp.",
    },
    ColumnContract {
        name: "duration_ms",
        sql_type: "BIGINT",
        nullable: true,
        notes: "Closed attempt wall-clock duration.",
    },
];

const AGENT_SDK_TURN_ATTEMPT_CONSTRAINTS: &[ConstraintContract] = &[
    ConstraintContract {
        name: "agent_sdk_turn_attempts_task_fk",
        invariant: "Every turn attempt belongs to an existing task.",
    },
    ConstraintContract {
        name: "agent_sdk_turn_attempts_attempt_number_check",
        invariant: "Attempt number is 1-indexed and positive.",
    },
    ConstraintContract {
        name: "agent_sdk_turn_attempts_task_attempt_number_key",
        invariant: "Attempt ordering within a task is unique and stable.",
    },
    ConstraintContract {
        name: "agent_sdk_turn_attempts_stop_reason_check",
        invariant: "Only known durable stop reasons can be persisted.",
    },
    ConstraintContract {
        name: "agent_sdk_turn_attempts_outcome_check",
        invariant: "Only known durable attempt outcomes can be persisted.",
    },
    ConstraintContract {
        name: "agent_sdk_turn_attempts_token_bounds_check",
        invariant: "Usage counters are non-negative.",
    },
    ConstraintContract {
        name: "agent_sdk_turn_attempts_open_close_shape_check",
        invariant: "Open rows are empty on the response side; closed rows carry their close payload.",
    },
];

const AGENT_SDK_TURN_ATTEMPT_INDEXES: &[IndexContract] = &[];

const AGENT_SDK_TURN_CHECKPOINT_COLUMNS: &[ColumnContract] = &[
    ColumnContract {
        name: "id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Primary checkpoint identity (`checkpoint_<uuid>` wire form).",
    },
    ColumnContract {
        name: "thread_id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Owning thread identity.",
    },
    ColumnContract {
        name: "turn_number",
        sql_type: "BIGINT",
        nullable: false,
        notes: "1-indexed committed turn number within the thread.",
    },
    ColumnContract {
        name: "task_id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Task that produced the committed turn.",
    },
    ColumnContract {
        name: "messages_json",
        sql_type: "JSONB",
        nullable: false,
        notes: "Full committed message history snapshot at that turn.",
    },
    ColumnContract {
        name: "agent_state_snapshot",
        sql_type: "JSONB",
        nullable: false,
        notes: "Opaque recovery payload.",
    },
    ColumnContract {
        name: "turn_input_tokens",
        sql_type: "BIGINT",
        nullable: false,
        notes: "Per-turn input usage.",
    },
    ColumnContract {
        name: "turn_output_tokens",
        sql_type: "BIGINT",
        nullable: false,
        notes: "Per-turn output usage.",
    },
    ColumnContract {
        name: "kind",
        sql_type: "TEXT",
        nullable: false,
        notes: "Checkpoint provenance (`full_turn` | `cancel_salvage`); drives turn-slot-shift eligibility.",
    },
    ColumnContract {
        name: "created_at",
        sql_type: "TIMESTAMPTZ",
        nullable: false,
        notes: "Checkpoint creation timestamp.",
    },
];

const AGENT_SDK_TURN_CHECKPOINT_CONSTRAINTS: &[ConstraintContract] = &[
    ConstraintContract {
        name: "agent_sdk_turn_checkpoints_thread_fk",
        invariant: "Every checkpoint belongs to an existing thread row.",
    },
    ConstraintContract {
        name: "agent_sdk_turn_checkpoints_task_fk",
        invariant: "Every checkpoint points at the task that produced it.",
    },
    ConstraintContract {
        name: "agent_sdk_turn_checkpoints_turn_number_check",
        invariant: "Turn numbers are 1-indexed and positive.",
    },
    ConstraintContract {
        name: "agent_sdk_turn_checkpoints_thread_turn_key",
        invariant: "There is exactly one checkpoint per `(thread_id, turn_number)`.",
    },
    ConstraintContract {
        name: "agent_sdk_turn_checkpoints_messages_json_check",
        invariant: "Checkpoint message history is stored as a JSON array.",
    },
    ConstraintContract {
        name: "agent_sdk_turn_checkpoints_turn_usage_check",
        invariant: "Per-turn usage fields are non-negative.",
    },
];

const AGENT_SDK_TURN_CHECKPOINT_INDEXES: &[IndexContract] = &[
    IndexContract {
        name: "agent_sdk_turn_checkpoints_latest_by_thread_idx",
        key_columns: "(thread_id, turn_number DESC)",
        predicate: None,
        purpose: "Powers latest-checkpoint recovery with a backward index walk.",
    },
    IndexContract {
        name: "agent_sdk_turn_checkpoints_by_task_idx",
        key_columns: "(task_id, turn_number)",
        predicate: None,
        purpose: "Supports task-oriented audit inspection of committed turns.",
    },
];

const DURABLE_CORE_TABLES: &[TableContract] = &[
    TableContract {
        name: "agent_sdk_tasks",
        purpose: "Root-turn journal, queueing, lease ownership, pause-state, and child orchestration.",
        columns: AGENT_SDK_TASK_COLUMNS,
        constraints: AGENT_SDK_TASK_CONSTRAINTS,
        indexes: AGENT_SDK_TASK_INDEXES,
    },
    TableContract {
        name: "agent_sdk_threads",
        purpose: "Thread-level aggregate counters and completion status.",
        columns: AGENT_SDK_THREAD_COLUMNS,
        constraints: AGENT_SDK_THREAD_CONSTRAINTS,
        indexes: AGENT_SDK_THREAD_INDEXES,
    },
    TableContract {
        name: "agent_sdk_message_heads",
        purpose: "Current SDK-visible message projection head. Successful turn commits create/advance it, and compaction rewrites this row only.",
        columns: AGENT_SDK_MESSAGE_HEAD_COLUMNS,
        constraints: AGENT_SDK_MESSAGE_HEAD_CONSTRAINTS,
        indexes: AGENT_SDK_MESSAGE_HEAD_INDEXES,
    },
    TableContract {
        name: "agent_sdk_message_commits",
        purpose: "Append-only raw committed message batches, written on successful turn commit. Compaction never deletes from this table.",
        columns: AGENT_SDK_MESSAGE_COMMIT_COLUMNS,
        constraints: AGENT_SDK_MESSAGE_COMMIT_CONSTRAINTS,
        indexes: AGENT_SDK_MESSAGE_COMMIT_INDEXES,
    },
    TableContract {
        name: "agent_sdk_turn_attempts",
        purpose: "Append-only LLM request/response audit records, ordered within each task.",
        columns: AGENT_SDK_TURN_ATTEMPT_COLUMNS,
        constraints: AGENT_SDK_TURN_ATTEMPT_CONSTRAINTS,
        indexes: AGENT_SDK_TURN_ATTEMPT_INDEXES,
    },
    TableContract {
        name: "agent_sdk_turn_checkpoints",
        purpose: "Immutable completed-turn recovery snapshots.",
        columns: AGENT_SDK_TURN_CHECKPOINT_COLUMNS,
        constraints: AGENT_SDK_TURN_CHECKPOINT_CONSTRAINTS,
        indexes: AGENT_SDK_TURN_CHECKPOINT_INDEXES,
    },
];

/// All tables in the current durable-core Postgres contract.
#[must_use]
pub const fn durable_core_tables() -> &'static [TableContract] {
    DURABLE_CORE_TABLES
}

// =====================================================================
// Event journal, outbox, and retention tables (migration 0002)
// =====================================================================

const AGENT_SDK_COMMITTED_EVENT_COLUMNS: &[ColumnContract] = &[
    ColumnContract {
        name: "event_id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Primary event identity (UUID v7 string, time-ordered).",
    },
    ColumnContract {
        name: "thread_id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Owning thread identity.",
    },
    ColumnContract {
        name: "sequence",
        sql_type: "BIGINT",
        nullable: false,
        notes: "Monotonically increasing sequence within the thread. 0-indexed.",
    },
    ColumnContract {
        name: "event_json",
        sql_type: "JSONB",
        nullable: false,
        notes: "Serialised `AgentEvent` payload.",
    },
    ColumnContract {
        name: "committed_at",
        sql_type: "TIMESTAMPTZ",
        nullable: false,
        notes: "Server commit timestamp (UTC).",
    },
];

const AGENT_SDK_COMMITTED_EVENT_CONSTRAINTS: &[ConstraintContract] = &[
    ConstraintContract {
        name: "agent_sdk_committed_events_thread_fk",
        invariant: "Every committed event belongs to an existing thread row.",
    },
    ConstraintContract {
        name: "agent_sdk_committed_events_thread_sequence_key",
        invariant: "`(thread_id, sequence)` is the durable unique key for replay and ordering.",
    },
    ConstraintContract {
        name: "agent_sdk_committed_events_sequence_check",
        invariant: "Event sequence is non-negative.",
    },
    ConstraintContract {
        name: "agent_sdk_committed_events_event_json_check",
        invariant: "Event payload is stored as a JSON object.",
    },
];

const AGENT_SDK_COMMITTED_EVENT_INDEXES: &[IndexContract] = &[
    IndexContract {
        name: "agent_sdk_committed_events_replay_idx",
        key_columns: "(thread_id, sequence)",
        predicate: None,
        purpose: "Primary replay query path for reconnecting clients.",
    },
    IndexContract {
        name: "agent_sdk_committed_events_by_time_idx",
        key_columns: "(thread_id, committed_at)",
        predicate: None,
        purpose: "Time-range queries for operational inspection and retention scans.",
    },
];

const AGENT_SDK_OUTBOX_COLUMNS: &[ColumnContract] = &[
    ColumnContract {
        name: "id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Primary outbox row identity (`outbox_<uuid>` wire form).",
    },
    ColumnContract {
        name: "kind",
        sql_type: "TEXT",
        nullable: false,
        notes: "Phase 8.1 logical kind: `task_wakeup` or `thread_events_available`.",
    },
    ColumnContract {
        name: "thread_id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Thread the message refers to.",
    },
    ColumnContract {
        name: "event_id",
        sql_type: "TEXT",
        nullable: true,
        notes: "References `agent_sdk_committed_events.event_id`. Set for `thread_events_available` rows; NULL for `task_wakeup`.",
    },
    ColumnContract {
        name: "sequence",
        sql_type: "BIGINT",
        nullable: true,
        notes: "Highest committed sequence in the triggering batch. Set for `thread_events_available` rows; NULL for `task_wakeup`.",
    },
    ColumnContract {
        name: "status",
        sql_type: "TEXT",
        nullable: false,
        notes: "Relay lifecycle status (`pending`, `claimed`, `delivered`, `expired`).",
    },
    ColumnContract {
        name: "payload_json",
        sql_type: "JSONB",
        nullable: false,
        notes: "Phase 8.1 advisory payload — durable references only, never authoritative state.",
    },
    ColumnContract {
        name: "created_at",
        sql_type: "TIMESTAMPTZ",
        nullable: false,
        notes: "When the outbox row was created (same transaction as event commit).",
    },
    ColumnContract {
        name: "next_attempt_at",
        sql_type: "TIMESTAMPTZ",
        nullable: false,
        notes: "When the relay should next attempt delivery.",
    },
    ColumnContract {
        name: "attempt_count",
        sql_type: "BIGINT",
        nullable: false,
        notes: "Number of relay attempts so far.",
    },
    ColumnContract {
        name: "max_attempts",
        sql_type: "BIGINT",
        nullable: false,
        notes: "Maximum relay attempts before the row expires.",
    },
    ColumnContract {
        name: "last_error",
        sql_type: "TEXT",
        nullable: true,
        notes: "Most recent relay failure message.",
    },
    ColumnContract {
        name: "claimed_by",
        sql_type: "TEXT",
        nullable: true,
        notes: "Relay worker identity that claimed this row.",
    },
    ColumnContract {
        name: "claimed_at",
        sql_type: "TIMESTAMPTZ",
        nullable: true,
        notes: "When the relay worker claimed this row.",
    },
    ColumnContract {
        name: "delivered_at",
        sql_type: "TIMESTAMPTZ",
        nullable: true,
        notes: "When the relay successfully delivered this row.",
    },
];

const AGENT_SDK_OUTBOX_CONSTRAINTS: &[ConstraintContract] = &[
    ConstraintContract {
        name: "agent_sdk_outbox_thread_fk",
        invariant: "Every outbox row belongs to an existing thread.",
    },
    ConstraintContract {
        name: "agent_sdk_outbox_event_fk",
        invariant: "When set, `event_id` references a committed event (NULL allowed for `task_wakeup` rows).",
    },
    ConstraintContract {
        name: "agent_sdk_outbox_kind_check",
        invariant: "Phase 8.1 limits `kind` to `task_wakeup` or `thread_events_available`.",
    },
    ConstraintContract {
        name: "agent_sdk_outbox_kind_payload_check",
        invariant: "`thread_events_available` rows carry both event_id and sequence; `task_wakeup` rows carry neither.",
    },
    ConstraintContract {
        name: "agent_sdk_outbox_status_check",
        invariant: "Only known outbox statuses can be persisted.",
    },
    ConstraintContract {
        name: "agent_sdk_outbox_sequence_check",
        invariant: "When set, outbox sequence (copied from event) is non-negative.",
    },
    ConstraintContract {
        name: "agent_sdk_outbox_attempt_bounds_check",
        invariant: "Retry counters stay within the configured relay budget.",
    },
    ConstraintContract {
        name: "agent_sdk_outbox_payload_json_check",
        invariant: "Relay payload is stored as a JSON object.",
    },
    ConstraintContract {
        name: "agent_sdk_outbox_claim_atomicity_check",
        invariant: "Claim fields are either all absent or all present together.",
    },
    ConstraintContract {
        name: "agent_sdk_outbox_delivered_check",
        invariant: "Delivered rows require `delivered_at`; non-delivered rows must not set it.",
    },
    ConstraintContract {
        name: "agent_sdk_outbox_error_check",
        invariant: "Expired rows require `last_error`; other rows must not carry one.",
    },
];

const AGENT_SDK_OUTBOX_INDEXES: &[IndexContract] = &[
    IndexContract {
        name: "agent_sdk_outbox_relay_scan_idx",
        key_columns: "(next_attempt_at, id)",
        predicate: Some("status = 'pending'"),
        purpose: "Relay worker scan for pending delivery work, ordered by next_attempt_at.",
    },
    IndexContract {
        name: "agent_sdk_outbox_relay_scan_by_kind_idx",
        key_columns: "(kind, next_attempt_at, id)",
        predicate: Some("status = 'pending'"),
        purpose: "Phase 8.1 partial index so the scheduler can drain by kind without scanning every pending row.",
    },
    IndexContract {
        name: "agent_sdk_outbox_by_thread_idx",
        key_columns: "(thread_id, sequence)",
        predicate: None,
        purpose: "Thread-scoped inspection of outbox rows in sequence order.",
    },
    IndexContract {
        name: "agent_sdk_outbox_claimed_sweep_idx",
        key_columns: "(claimed_at, id)",
        predicate: Some("status = 'claimed'"),
        purpose: "Stale-claim sweep for relay workers that failed without releasing.",
    },
];

const AGENT_SDK_RETENTION_CURSOR_COLUMNS: &[ColumnContract] = &[
    ColumnContract {
        name: "thread_id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Primary key and owning thread identity.",
    },
    ColumnContract {
        name: "retention_floor",
        sql_type: "BIGINT",
        nullable: false,
        notes: "Lowest committed-event sequence guaranteed to still exist. Events below this value may have been garbage-collected.",
    },
    ColumnContract {
        name: "updated_at",
        sql_type: "TIMESTAMPTZ",
        nullable: false,
        notes: "When the retention floor was last advanced.",
    },
];

const AGENT_SDK_RETENTION_CURSOR_CONSTRAINTS: &[ConstraintContract] = &[
    ConstraintContract {
        name: "agent_sdk_retention_cursors_thread_fk",
        invariant: "Every retention cursor belongs to an existing thread row.",
    },
    ConstraintContract {
        name: "agent_sdk_retention_cursors_floor_check",
        invariant: "Retention floor is non-negative.",
    },
];

const AGENT_SDK_RETENTION_CURSOR_INDEXES: &[IndexContract] = &[];

const EVENT_JOURNAL_OUTBOX_TABLES: &[TableContract] = &[
    TableContract {
        name: "agent_sdk_committed_events",
        purpose: "Append-only event journal with thread-scoped sequence ordering. Authoritative source of truth for event replay.",
        columns: AGENT_SDK_COMMITTED_EVENT_COLUMNS,
        constraints: AGENT_SDK_COMMITTED_EVENT_CONSTRAINTS,
        indexes: AGENT_SDK_COMMITTED_EVENT_INDEXES,
    },
    TableContract {
        name: "agent_sdk_outbox",
        purpose: "Transactional relay buffer for AMQP/pub-sub delivery. Inserted atomically with committed events. Not the authority for ordering.",
        columns: AGENT_SDK_OUTBOX_COLUMNS,
        constraints: AGENT_SDK_OUTBOX_CONSTRAINTS,
        indexes: AGENT_SDK_OUTBOX_INDEXES,
    },
    TableContract {
        name: "agent_sdk_retention_cursors",
        purpose: "Per-thread retention watermarks for event garbage collection and replay gap detection.",
        columns: AGENT_SDK_RETENTION_CURSOR_COLUMNS,
        constraints: AGENT_SDK_RETENTION_CURSOR_CONSTRAINTS,
        indexes: AGENT_SDK_RETENTION_CURSOR_INDEXES,
    },
];

/// Tables introduced by the event journal and outbox migration (0002).
#[must_use]
pub const fn event_journal_outbox_tables() -> &'static [TableContract] {
    EVENT_JOURNAL_OUTBOX_TABLES
}

// =====================================================================
// Execution intent table (migration 0003)
// =====================================================================

const AGENT_SDK_EXECUTION_INTENT_COLUMNS: &[ColumnContract] = &[
    ColumnContract {
        name: "operation_id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Primary key. Combines child_task_id and tool_call_id.",
    },
    ColumnContract {
        name: "effect_class",
        sql_type: "TEXT",
        nullable: false,
        notes: "Tool effect classification (`replay_safe`, `side_effecting`, `resumable`).",
    },
    ColumnContract {
        name: "tool_call_id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Raw LLM-assigned tool call id.",
    },
    ColumnContract {
        name: "child_task_id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Child task that owns this execution.",
    },
    ColumnContract {
        name: "tool_name",
        sql_type: "TEXT",
        nullable: false,
        notes: "Tool name.",
    },
    ColumnContract {
        name: "input",
        sql_type: "JSONB",
        nullable: false,
        notes: "Tool input snapshot for audit and replay verification.",
    },
    ColumnContract {
        name: "status",
        sql_type: "TEXT",
        nullable: false,
        notes: "Intent lifecycle status (`pending`, `started`, `completed`, `failed`).",
    },
    ColumnContract {
        name: "error",
        sql_type: "TEXT",
        nullable: true,
        notes: "Error message when status is `failed`.",
    },
    ColumnContract {
        name: "created_at",
        sql_type: "TIMESTAMPTZ",
        nullable: false,
        notes: "When the intent was first persisted.",
    },
    ColumnContract {
        name: "updated_at",
        sql_type: "TIMESTAMPTZ",
        nullable: false,
        notes: "When the intent last changed status.",
    },
];

const AGENT_SDK_EXECUTION_INTENT_CONSTRAINTS: &[ConstraintContract] = &[
    ConstraintContract {
        name: "agent_sdk_execution_intents_effect_class_check",
        invariant: "Only known effect classes can be persisted.",
    },
    ConstraintContract {
        name: "agent_sdk_execution_intents_status_check",
        invariant: "Only known intent statuses can be persisted.",
    },
    ConstraintContract {
        name: "agent_sdk_execution_intents_error_check",
        invariant: "Only `failed` rows carry `error`; every `failed` row must carry one.",
    },
];

const AGENT_SDK_EXECUTION_INTENT_INDEXES: &[IndexContract] = &[IndexContract {
    name: "agent_sdk_execution_intents_by_child_task_idx",
    key_columns: "(child_task_id)",
    predicate: None,
    purpose: "Supports restart-recovery lookup by child task id.",
}];

const EXECUTION_INTENT_TABLES: &[TableContract] = &[TableContract {
    name: "agent_sdk_execution_intents",
    purpose: "Durable execution intent records for the Phase 5.2 fail-closed guarded tool execution contract.",
    columns: AGENT_SDK_EXECUTION_INTENT_COLUMNS,
    constraints: AGENT_SDK_EXECUTION_INTENT_CONSTRAINTS,
    indexes: AGENT_SDK_EXECUTION_INTENT_INDEXES,
}];

/// Tables introduced by the execution intents migration (0003).
#[must_use]
pub const fn execution_intent_tables() -> &'static [TableContract] {
    EXECUTION_INTENT_TABLES
}

// =====================================================================
// Tool audit events table (migration 0004)
// =====================================================================

const AGENT_SDK_TOOL_AUDIT_EVENT_COLUMNS: &[ColumnContract] = &[
    ColumnContract {
        name: "id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Primary event identity (`tae_<uuid>` wire form).",
    },
    ColumnContract {
        name: "seq",
        sql_type: "BIGINT",
        nullable: false,
        notes: "Database-assigned monotonic insertion sequence. Stable tie-breaker for `recorded_at` collisions.",
    },
    ColumnContract {
        name: "operation_id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Operation identity (`child_task_id:tool_call_id`).",
    },
    ColumnContract {
        name: "task_id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Child task that owns this tool execution.",
    },
    ColumnContract {
        name: "parent_task_id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Parent root-turn task that spawned the child.",
    },
    ColumnContract {
        name: "thread_id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Thread the task family is bound to.",
    },
    ColumnContract {
        name: "tool_call_id",
        sql_type: "TEXT",
        nullable: false,
        notes: "Raw LLM-assigned tool call id.",
    },
    ColumnContract {
        name: "tool_name",
        sql_type: "TEXT",
        nullable: false,
        notes: "Tool name.",
    },
    ColumnContract {
        name: "effect_class",
        sql_type: "TEXT",
        nullable: false,
        notes: "Tool effect classification (`replay_safe`, `side_effecting`, `resumable`).",
    },
    ColumnContract {
        name: "kind",
        sql_type: "TEXT",
        nullable: false,
        notes: "Lifecycle event discriminant. Constrained to the known `ToolAuditEventKind` variants.",
    },
    ColumnContract {
        name: "kind_payload",
        sql_type: "JSONB",
        nullable: false,
        notes: "Full serialized `ToolAuditEventKind` including any variant-specific payload (reason, error).",
    },
    ColumnContract {
        name: "provider",
        sql_type: "TEXT",
        nullable: false,
        notes: "Provider identifier (e.g. `anthropic`, `openai`).",
    },
    ColumnContract {
        name: "model",
        sql_type: "TEXT",
        nullable: false,
        notes: "Model identifier as reported by the provider for the turn that dispatched the tool.",
    },
    ColumnContract {
        name: "input",
        sql_type: "JSONB",
        nullable: true,
        notes: "Redacted tool input snapshot. NULL when the event kind carries no input.",
    },
    ColumnContract {
        name: "output",
        sql_type: "TEXT",
        nullable: true,
        notes: "Redacted tool output. NULL when the event kind carries no output.",
    },
    ColumnContract {
        name: "error",
        sql_type: "TEXT",
        nullable: true,
        notes: "Redacted error detail. NULL when the event kind carries no error.",
    },
    ColumnContract {
        name: "recorded_at",
        sql_type: "TIMESTAMPTZ",
        nullable: false,
        notes: "When the event was recorded (supplied by the caller, not the database).",
    },
];

const AGENT_SDK_TOOL_AUDIT_EVENT_CONSTRAINTS: &[ConstraintContract] = &[
    ConstraintContract {
        name: "agent_sdk_tool_audit_events_seq_key",
        invariant: "`seq` is the durable monotonic sequence; no two rows share a value.",
    },
    ConstraintContract {
        name: "agent_sdk_tool_audit_events_effect_class_check",
        invariant: "Only known effect classes can be persisted.",
    },
    ConstraintContract {
        name: "agent_sdk_tool_audit_events_kind_check",
        invariant: "Only known lifecycle event kinds can be persisted.",
    },
    ConstraintContract {
        name: "agent_sdk_tool_audit_events_kind_payload_check",
        invariant: "The serialized kind payload is a JSON object carrying the full tagged enum.",
    },
    ConstraintContract {
        name: "agent_sdk_tool_audit_events_kind_payload_discriminant_check",
        invariant: "The payload's embedded `kind` must match the discriminant column.",
    },
];

const AGENT_SDK_TOOL_AUDIT_EVENT_INDEXES: &[IndexContract] = &[
    IndexContract {
        name: "agent_sdk_tool_audit_events_by_operation_idx",
        key_columns: "(operation_id, seq)",
        predicate: None,
        purpose: "Streams the lifecycle for a single operation in insertion order without a sort step.",
    },
    IndexContract {
        name: "agent_sdk_tool_audit_events_by_task_idx",
        key_columns: "(task_id, seq)",
        predicate: None,
        purpose: "Streams every audit event for a child task in insertion order.",
    },
    IndexContract {
        name: "agent_sdk_tool_audit_events_by_thread_idx",
        key_columns: "(thread_id, seq)",
        predicate: None,
        purpose: "Streams every audit event within a thread (covers cross-task audit queries).",
    },
    IndexContract {
        name: "agent_sdk_tool_audit_events_by_recorded_at_idx",
        key_columns: "(recorded_at)",
        predicate: None,
        purpose: "Time-range queries for operational inspection and retention scans.",
    },
];

const TOOL_AUDIT_EVENT_TABLES: &[TableContract] = &[TableContract {
    name: "agent_sdk_tool_audit_events",
    purpose: "Durable append-only tool audit lifecycle events for child-task execution.",
    columns: AGENT_SDK_TOOL_AUDIT_EVENT_COLUMNS,
    constraints: AGENT_SDK_TOOL_AUDIT_EVENT_CONSTRAINTS,
    indexes: AGENT_SDK_TOOL_AUDIT_EVENT_INDEXES,
}];

/// Tables introduced by the tool audit events migration (0004).
#[must_use]
pub const fn tool_audit_event_tables() -> &'static [TableContract] {
    TOOL_AUDIT_EVENT_TABLES
}
