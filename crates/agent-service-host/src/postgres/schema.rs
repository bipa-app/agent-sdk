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
        invariant: "Paused-state JSON kind must be a non-null known value; waiting counters agree with task status; terminal rows reset state kind to `none`; `ready_to_resume` is not valid for queued rows.",
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
