-- ENG-9278: SQLite cannot alter CHECK constraints in place, so rebuild only
-- the task table while preserving every column, row, foreign key, and index.
PRAGMA defer_foreign_keys = ON;

CREATE TABLE agent_sdk_tasks_new (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    parent_id TEXT NULL,
    root_id TEXT NOT NULL,
    depth INTEGER NOT NULL,
    thread_id TEXT NOT NULL,
    submitted_input_json TEXT NOT NULL DEFAULT '[]',
    caller_metadata_json TEXT NULL,
    worker_id TEXT NULL,
    lease_id TEXT NULL,
    lease_expires_at TEXT NULL,
    last_heartbeat_at TEXT NULL,
    state_json TEXT NOT NULL DEFAULT '{"kind":"none"}',
    attempt INTEGER NOT NULL,
    max_attempts INTEGER NOT NULL,
    last_error TEXT NULL,
    terminal_reason_json TEXT NULL,
    pending_child_count INTEGER NOT NULL DEFAULT 0,
    spawn_index INTEGER NULL,
    result_payload TEXT NULL,
    otel_traceparent TEXT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT NULL,
    last_activity_at TEXT NULL,
    CONSTRAINT agent_sdk_tasks_parent_fk
        FOREIGN KEY (parent_id) REFERENCES agent_sdk_tasks_new(id)
        ON DELETE RESTRICT,
    CONSTRAINT agent_sdk_tasks_root_fk
        FOREIGN KEY (root_id) REFERENCES agent_sdk_tasks_new(id)
        ON DELETE RESTRICT,
    -- SQLite does not support ALTER TABLE ADD CONSTRAINT for foreign
    -- keys, but it DOES allow forward-referencing the parent table in
    -- an inline CREATE TABLE FK because the constraint is evaluated at
    -- DML time rather than DDL time. agent_sdk_threads is created below.
    CONSTRAINT agent_sdk_tasks_thread_fk
        FOREIGN KEY (thread_id) REFERENCES agent_sdk_threads(thread_id)
        ON DELETE RESTRICT,
    CONSTRAINT agent_sdk_tasks_kind_check
        CHECK (kind IN ('root_turn', 'tool_runtime', 'subagent', 'input_injection')),
    CONSTRAINT agent_sdk_tasks_status_check
        CHECK (
            status IN (
                'queued',
                'pending',
                'running',
                'waiting_on_children',
                'awaiting_confirmation',
                'completed',
                'failed',
                'cancelled'
            )
        ),
    CONSTRAINT agent_sdk_tasks_root_identity_check
        CHECK (
            (
                depth = 0
                AND parent_id IS NULL
                AND root_id = id
            )
            OR (
                depth > 0
                AND parent_id IS NOT NULL
                AND root_id <> id
            )
        ),
    CONSTRAINT agent_sdk_tasks_depth_kind_check
        CHECK ((depth = 0) = (kind = 'root_turn')),
    CONSTRAINT agent_sdk_tasks_depth_non_negative_check
        CHECK (depth >= 0),
    CONSTRAINT agent_sdk_tasks_attempt_bounds_check
        CHECK (
            attempt >= 0
            AND max_attempts >= 1
            AND attempt <= max_attempts
        ),
    CONSTRAINT agent_sdk_tasks_pending_child_count_check
        CHECK (pending_child_count >= 0),
    CONSTRAINT agent_sdk_tasks_spawn_index_check
        CHECK (spawn_index IS NULL OR spawn_index >= 0),
    CONSTRAINT agent_sdk_tasks_queue_kind_check
        CHECK (status <> 'queued' OR kind IN ('root_turn', 'input_injection')),
    CONSTRAINT agent_sdk_tasks_lease_atomicity_check
        CHECK (
            (
                worker_id IS NULL
                AND lease_id IS NULL
                AND lease_expires_at IS NULL
                AND last_heartbeat_at IS NULL
            )
            OR (
                worker_id IS NOT NULL
                AND lease_id IS NOT NULL
                AND lease_expires_at IS NOT NULL
            )
        ),
    CONSTRAINT agent_sdk_tasks_running_lease_check
        CHECK (
            (
                status = 'running'
                AND worker_id IS NOT NULL
                AND lease_id IS NOT NULL
                AND lease_expires_at IS NOT NULL
            )
            OR (
                status <> 'running'
                AND worker_id IS NULL
                AND lease_id IS NULL
                AND lease_expires_at IS NULL
            )
        ),
    CONSTRAINT agent_sdk_tasks_terminal_completion_check
        CHECK (
            (
                status IN ('completed', 'failed', 'cancelled')
                AND completed_at IS NOT NULL
            )
            OR (
                status NOT IN ('completed', 'failed', 'cancelled')
                AND completed_at IS NULL
            )
        ),
    CONSTRAINT agent_sdk_tasks_failure_payload_check
        CHECK (
            (
                status = 'failed'
                AND last_error IS NOT NULL
            )
            OR (
                status <> 'failed'
                AND last_error IS NULL
            )
        ),
    CONSTRAINT agent_sdk_tasks_waiting_state_check
        CHECK (
            json_type(state_json) = 'object'
            AND json_extract(state_json, '$.kind') IS NOT NULL
            AND json_extract(state_json, '$.kind') IN (
                'none',
                'waiting_on_children',
                'awaiting_confirmation',
                'subagent_invocation',
                'ready_to_resume'
            )
            AND (
                (
                    json_extract(state_json, '$.kind') IN ('waiting_on_children', 'subagent_invocation')
                    AND status = 'waiting_on_children'
                    AND pending_child_count > 0
                )
                OR (
                    json_extract(state_json, '$.kind') = 'subagent_invocation'
                    AND (
                        (
                            status = 'waiting_on_children'
                            AND pending_child_count > 0
                        )
                        OR (
                            status IN ('pending', 'running')
                            AND pending_child_count = 0
                        )
                    )
                )
                OR (
                    json_extract(state_json, '$.kind') = 'awaiting_confirmation'
                    AND status = 'awaiting_confirmation'
                    AND pending_child_count = 0
                )
                OR (
                    json_extract(state_json, '$.kind') = 'none'
                    AND status IN ('queued', 'pending', 'running')
                    AND pending_child_count = 0
                )
                OR (
                    json_extract(state_json, '$.kind') = 'none'
                    AND status IN ('completed', 'failed', 'cancelled')
                    AND pending_child_count = 0
                )
                OR (
                    json_extract(state_json, '$.kind') = 'ready_to_resume'
                    AND status IN ('pending', 'running')
                    AND pending_child_count = 0
                )
            )
        )
);

INSERT INTO agent_sdk_tasks_new (
    id, kind, status, parent_id, root_id, depth, thread_id,
    submitted_input_json, caller_metadata_json, worker_id, lease_id,
    lease_expires_at, last_heartbeat_at, state_json, attempt, max_attempts,
    last_error, terminal_reason_json, pending_child_count, spawn_index,
    result_payload, otel_traceparent, created_at, updated_at, completed_at,
    last_activity_at
)
SELECT
    id, kind, status, parent_id, root_id, depth, thread_id,
    submitted_input_json, caller_metadata_json, worker_id, lease_id,
    lease_expires_at, last_heartbeat_at, state_json, attempt, max_attempts,
    last_error, terminal_reason_json, pending_child_count, spawn_index,
    result_payload, otel_traceparent, created_at, updated_at, completed_at,
    last_activity_at
FROM agent_sdk_tasks;

DROP TABLE agent_sdk_tasks;
ALTER TABLE agent_sdk_tasks_new RENAME TO agent_sdk_tasks;

CREATE INDEX agent_sdk_tasks_by_thread_idx
    ON agent_sdk_tasks (thread_id, created_at, id);

CREATE INDEX agent_sdk_tasks_by_parent_idx
    ON agent_sdk_tasks (parent_id, created_at, id)
    WHERE parent_id IS NOT NULL;

CREATE INDEX agent_sdk_tasks_by_status_idx
    ON agent_sdk_tasks (status, created_at, id);

CREATE UNIQUE INDEX agent_sdk_tasks_root_admission_slot_idx
    ON agent_sdk_tasks (thread_id)
    WHERE kind = 'root_turn'
      AND status IN ('pending', 'running', 'waiting_on_children', 'awaiting_confirmation');

CREATE INDEX agent_sdk_tasks_queued_roots_fifo_idx
    ON agent_sdk_tasks (thread_id, created_at, id)
    WHERE kind = 'root_turn'
      AND status = 'queued';

CREATE INDEX agent_sdk_tasks_runnable_fifo_idx
    ON agent_sdk_tasks (created_at, id)
    WHERE status = 'pending';

CREATE INDEX agent_sdk_tasks_running_lease_expiry_idx
    ON agent_sdk_tasks (lease_expires_at, id)
    WHERE status = 'running';

CREATE INDEX agent_sdk_tasks_root_tree_idx
    ON agent_sdk_tasks (root_id, depth, created_at, id);

CREATE INDEX agent_sdk_tasks_subagent_child_root_waiting_idx
    ON agent_sdk_tasks (json_extract(state_json, '$.invocation.child_root_task_id'))
    WHERE kind = 'subagent' AND status = 'waiting_on_children';


-- Dropping the old table also dropped 0014's terminal-reason triggers;
-- recreate them against the rebuilt table.
CREATE TRIGGER agent_sdk_tasks_terminal_reason_json_insert_check
BEFORE INSERT ON agent_sdk_tasks
WHEN (
    NEW.status IN ('completed', 'failed', 'cancelled')
    AND NEW.terminal_reason_json IS NULL
) OR (
    NEW.status NOT IN ('completed', 'failed', 'cancelled')
    AND NEW.terminal_reason_json IS NOT NULL
)
BEGIN
    SELECT RAISE(ABORT, 'terminal_reason_json must be set exactly on terminal task rows');
END;

CREATE TRIGGER agent_sdk_tasks_terminal_reason_json_update_check
BEFORE UPDATE OF status, terminal_reason_json ON agent_sdk_tasks
WHEN (
    NEW.status IN ('completed', 'failed', 'cancelled')
    AND NEW.terminal_reason_json IS NULL
) OR (
    NEW.status NOT IN ('completed', 'failed', 'cancelled')
    AND NEW.terminal_reason_json IS NOT NULL
)
BEGIN
    SELECT RAISE(ABORT, 'terminal_reason_json must be set exactly on terminal task rows');
END;
