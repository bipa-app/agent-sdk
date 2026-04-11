CREATE TABLE agent_sdk_tasks (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    parent_id TEXT NULL,
    root_id TEXT NOT NULL,
    depth BIGINT NOT NULL,
    thread_id TEXT NOT NULL,
    submitted_input_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    worker_id TEXT NULL,
    lease_id TEXT NULL,
    lease_expires_at TIMESTAMPTZ NULL,
    last_heartbeat_at TIMESTAMPTZ NULL,
    state_json JSONB NOT NULL DEFAULT '{"kind":"none"}'::jsonb,
    attempt BIGINT NOT NULL,
    max_attempts BIGINT NOT NULL,
    last_error TEXT NULL,
    pending_child_count BIGINT NOT NULL DEFAULT 0,
    spawn_index BIGINT NULL,
    result_payload JSONB NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ NULL,
    CONSTRAINT agent_sdk_tasks_parent_fk
        FOREIGN KEY (parent_id) REFERENCES agent_sdk_tasks(id)
        ON DELETE RESTRICT
        DEFERRABLE INITIALLY IMMEDIATE,
    CONSTRAINT agent_sdk_tasks_root_fk
        FOREIGN KEY (root_id) REFERENCES agent_sdk_tasks(id)
        ON DELETE RESTRICT
        DEFERRABLE INITIALLY IMMEDIATE,
    CONSTRAINT agent_sdk_tasks_kind_check
        CHECK (kind IN ('root_turn', 'tool_runtime', 'subagent')),
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
        CHECK (status <> 'queued' OR kind = 'root_turn'),
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
            jsonb_typeof(state_json) = 'object'
            AND state_json ? 'kind'
            AND state_json ->> 'kind' IS NOT NULL
            AND state_json ->> 'kind' IN (
                'none',
                'waiting_on_children',
                'awaiting_confirmation',
                'ready_to_resume'
            )
            AND (
                (
                    state_json ->> 'kind' = 'waiting_on_children'
                    AND status = 'waiting_on_children'
                    AND pending_child_count > 0
                )
                OR (
                    state_json ->> 'kind' = 'awaiting_confirmation'
                    AND status = 'awaiting_confirmation'
                    AND pending_child_count = 0
                )
                OR (
                    state_json ->> 'kind' = 'none'
                    AND status IN ('queued', 'pending', 'running')
                    AND pending_child_count = 0
                )
                OR (
                    state_json ->> 'kind' = 'none'
                    AND status IN ('completed', 'failed', 'cancelled')
                    AND pending_child_count = 0
                )
                OR (
                    state_json ->> 'kind' = 'ready_to_resume'
                    AND status IN ('pending', 'running')
                    AND pending_child_count = 0
                )
            )
        )
);

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

CREATE TABLE agent_sdk_threads (
    thread_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    committed_turns BIGINT NOT NULL,
    total_input_tokens BIGINT NOT NULL,
    total_output_tokens BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT agent_sdk_threads_status_check
        CHECK (status IN ('active', 'completed')),
    CONSTRAINT agent_sdk_threads_committed_turns_check
        CHECK (committed_turns >= 0),
    CONSTRAINT agent_sdk_threads_total_usage_check
        CHECK (
            total_input_tokens >= 0
            AND total_output_tokens >= 0
        ),
    CONSTRAINT agent_sdk_threads_completed_turns_check
        CHECK (status <> 'completed' OR committed_turns > 0)
);

CREATE INDEX agent_sdk_threads_active_idx
    ON agent_sdk_threads (thread_id)
    WHERE status = 'active';

ALTER TABLE agent_sdk_tasks
    ADD CONSTRAINT agent_sdk_tasks_thread_fk
    FOREIGN KEY (thread_id) REFERENCES agent_sdk_threads(thread_id)
    ON DELETE RESTRICT
    DEFERRABLE INITIALLY IMMEDIATE;

-- Mutable per-thread message projection.
-- `commit_completed_turn` / `commit_messages` creates or advances this
-- row when a turn commits successfully, and SDK compaction later
-- rewrites the same row via `replace_history`.
CREATE TABLE agent_sdk_message_heads (
    thread_id TEXT PRIMARY KEY,
    history_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    version BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT agent_sdk_message_heads_thread_fk
        FOREIGN KEY (thread_id) REFERENCES agent_sdk_threads(thread_id)
        ON DELETE CASCADE,
    CONSTRAINT agent_sdk_message_heads_history_json_check
        CHECK (jsonb_typeof(history_json) = 'array'),
    CONSTRAINT agent_sdk_message_heads_version_check
        CHECK (version >= 0)
);

-- Immutable raw turn batches, written only when a completed turn
-- commits successfully. Compaction never updates or deletes these
-- rows; it only rewrites `agent_sdk_message_heads`.
CREATE TABLE agent_sdk_message_commits (
    thread_id TEXT NOT NULL,
    turn_number BIGINT NOT NULL,
    task_id TEXT NOT NULL,
    head_version_after BIGINT NOT NULL,
    batch_json JSONB NOT NULL,
    committed_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT agent_sdk_message_commits_thread_turn_key
        PRIMARY KEY (thread_id, turn_number),
    CONSTRAINT agent_sdk_message_commits_thread_fk
        FOREIGN KEY (thread_id) REFERENCES agent_sdk_threads(thread_id)
        ON DELETE CASCADE,
    CONSTRAINT agent_sdk_message_commits_task_fk
        FOREIGN KEY (task_id) REFERENCES agent_sdk_tasks(id)
        ON DELETE RESTRICT,
    CONSTRAINT agent_sdk_message_commits_turn_number_check
        CHECK (turn_number >= 1),
    CONSTRAINT agent_sdk_message_commits_head_version_check
        CHECK (head_version_after >= 1),
    CONSTRAINT agent_sdk_message_commits_batch_json_check
        CHECK (jsonb_typeof(batch_json) = 'array')
);

CREATE INDEX agent_sdk_message_commits_by_task_idx
    ON agent_sdk_message_commits (task_id, turn_number);

CREATE TABLE agent_sdk_turn_attempts (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    attempt_number BIGINT NOT NULL,
    provider TEXT NOT NULL,
    requested_model TEXT NOT NULL,
    request_blob JSONB NOT NULL,
    response_blob JSONB NULL,
    response_id TEXT NULL,
    response_model TEXT NULL,
    stop_reason TEXT NULL,
    outcome TEXT NULL,
    input_tokens BIGINT NULL,
    output_tokens BIGINT NULL,
    cached_input_tokens BIGINT NULL,
    opened_at TIMESTAMPTZ NOT NULL,
    closed_at TIMESTAMPTZ NULL,
    duration_ms BIGINT NULL,
    CONSTRAINT agent_sdk_turn_attempts_task_fk
        FOREIGN KEY (task_id) REFERENCES agent_sdk_tasks(id)
        ON DELETE CASCADE,
    CONSTRAINT agent_sdk_turn_attempts_attempt_number_check
        CHECK (attempt_number >= 1),
    CONSTRAINT agent_sdk_turn_attempts_task_attempt_number_key
        UNIQUE (task_id, attempt_number),
    CONSTRAINT agent_sdk_turn_attempts_stop_reason_check
        CHECK (
            stop_reason IS NULL
            OR stop_reason IN (
                'end_turn',
                'tool_use',
                'max_tokens',
                'stop_sequence',
                'refusal',
                'model_context_window_exceeded'
            )
        ),
    CONSTRAINT agent_sdk_turn_attempts_outcome_check
        CHECK (
            outcome IS NULL
            OR outcome IN (
                'success',
                'rate_limited',
                'invalid_request',
                'server_error',
                'cancelled'
            )
        ),
    CONSTRAINT agent_sdk_turn_attempts_token_bounds_check
        CHECK (
            (input_tokens IS NULL OR input_tokens >= 0)
            AND (output_tokens IS NULL OR output_tokens >= 0)
            AND (cached_input_tokens IS NULL OR cached_input_tokens >= 0)
            AND (duration_ms IS NULL OR duration_ms >= 0)
        ),
    CONSTRAINT agent_sdk_turn_attempts_open_close_shape_check
        CHECK (
            (
                closed_at IS NULL
                AND outcome IS NULL
                AND response_blob IS NULL
                AND response_id IS NULL
                AND response_model IS NULL
                AND stop_reason IS NULL
                AND input_tokens IS NULL
                AND output_tokens IS NULL
                AND cached_input_tokens IS NULL
                AND duration_ms IS NULL
            )
            OR (
                closed_at IS NOT NULL
                AND outcome IS NOT NULL
                AND response_blob IS NOT NULL
                AND input_tokens IS NOT NULL
                AND output_tokens IS NOT NULL
                AND cached_input_tokens IS NOT NULL
                AND duration_ms IS NOT NULL
            )
        )
);

CREATE TABLE agent_sdk_turn_checkpoints (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    turn_number BIGINT NOT NULL,
    task_id TEXT NOT NULL,
    messages_json JSONB NOT NULL,
    agent_state_snapshot JSONB NOT NULL,
    turn_input_tokens BIGINT NOT NULL,
    turn_output_tokens BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT agent_sdk_turn_checkpoints_thread_fk
        FOREIGN KEY (thread_id) REFERENCES agent_sdk_threads(thread_id)
        ON DELETE CASCADE,
    CONSTRAINT agent_sdk_turn_checkpoints_task_fk
        FOREIGN KEY (task_id) REFERENCES agent_sdk_tasks(id)
        ON DELETE RESTRICT,
    CONSTRAINT agent_sdk_turn_checkpoints_turn_number_check
        CHECK (turn_number >= 1),
    CONSTRAINT agent_sdk_turn_checkpoints_thread_turn_key
        UNIQUE (thread_id, turn_number),
    CONSTRAINT agent_sdk_turn_checkpoints_messages_json_check
        CHECK (jsonb_typeof(messages_json) = 'array'),
    CONSTRAINT agent_sdk_turn_checkpoints_turn_usage_check
        CHECK (
            turn_input_tokens >= 0
            AND turn_output_tokens >= 0
        )
);

CREATE INDEX agent_sdk_turn_checkpoints_latest_by_thread_idx
    ON agent_sdk_turn_checkpoints (thread_id, turn_number DESC);

CREATE INDEX agent_sdk_turn_checkpoints_by_task_idx
    ON agent_sdk_turn_checkpoints (task_id, turn_number);
