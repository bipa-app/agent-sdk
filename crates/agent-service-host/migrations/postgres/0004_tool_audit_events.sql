-- Durable tool audit events for child-task execution lifecycle.
--
-- Each tool call may produce multiple rows as it walks through the
-- lifecycle (dispatched → confirmation → execution → terminal state).
-- Rows are append-only: the application never updates or deletes a row
-- once it is written.
--
-- Ordering: `recorded_at` is caller-supplied metadata and can skew
-- across workers during recovery or failover. A monotonic `seq` column
-- therefore serves as the durable ordering key, and read paths sort by
-- `seq` rather than by wall-clock time.
--
-- Indexes cover the three required query paths: by operation, by task,
-- and by thread, each extended with `seq` so the planner can stream
-- ordered results without a separate sort step.
--
-- Redaction policy: input / output / error columns are written **after**
-- the application applies its `RedactionPolicy`, so this table must
-- never hold raw secrets even if later code paths add new callers.

CREATE TABLE agent_sdk_tool_audit_events (
    id                TEXT        PRIMARY KEY,
    seq               BIGINT      NOT NULL GENERATED ALWAYS AS IDENTITY,
    operation_id      TEXT        NOT NULL,
    task_id           TEXT        NOT NULL,
    parent_task_id    TEXT        NOT NULL,
    thread_id         TEXT        NOT NULL,
    tool_call_id      TEXT        NOT NULL,
    tool_name         TEXT        NOT NULL,
    effect_class      TEXT        NOT NULL,
    kind              TEXT        NOT NULL,
    kind_payload      JSONB       NOT NULL,
    provider          TEXT        NOT NULL,
    model             TEXT        NOT NULL,
    input             JSONB       NULL,
    output            TEXT        NULL,
    error             TEXT        NULL,
    recorded_at       TIMESTAMPTZ NOT NULL,

    CONSTRAINT agent_sdk_tool_audit_events_seq_key
        UNIQUE (seq),

    CONSTRAINT agent_sdk_tool_audit_events_effect_class_check
        CHECK (effect_class IN ('replay_safe', 'side_effecting', 'resumable')),

    CONSTRAINT agent_sdk_tool_audit_events_kind_check
        CHECK (kind IN (
            'dispatched',
            'confirmation_requested',
            'confirmation_approved',
            'confirmation_rejected',
            'confirmation_timed_out',
            'policy_denied',
            'execution_started',
            'completed',
            'failed',
            'cancelled',
            'fail_closed'
        )),

    CONSTRAINT agent_sdk_tool_audit_events_kind_payload_check
        CHECK (jsonb_typeof(kind_payload) = 'object'),

    CONSTRAINT agent_sdk_tool_audit_events_kind_payload_discriminant_check
        CHECK (kind_payload ->> 'kind' = kind)
);

CREATE INDEX agent_sdk_tool_audit_events_by_operation_idx
    ON agent_sdk_tool_audit_events (operation_id, seq);

CREATE INDEX agent_sdk_tool_audit_events_by_task_idx
    ON agent_sdk_tool_audit_events (task_id, seq);

CREATE INDEX agent_sdk_tool_audit_events_by_thread_idx
    ON agent_sdk_tool_audit_events (thread_id, seq);

CREATE INDEX agent_sdk_tool_audit_events_by_recorded_at_idx
    ON agent_sdk_tool_audit_events (recorded_at);
