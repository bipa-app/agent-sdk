-- SQLite variant of migration 0004 (tool audit events).
-- Mirrors the Postgres schema with SQLite type adjustments:
-- BIGINT → INTEGER, JSONB → TEXT, TIMESTAMPTZ → TEXT.
--
-- SQLite only allows AUTOINCREMENT on an INTEGER PRIMARY KEY, so `seq`
-- serves as the physical primary key and `id` carries the logical
-- `tae_<uuid>` identity as a UNIQUE constraint.  Ordering remains
-- `ORDER BY recorded_at, seq`, which matches the Postgres behaviour.
--
-- Redaction policy: input / output / error columns are written **after**
-- the application applies its `RedactionPolicy`, so this table must
-- never hold raw secrets even if later code paths add new callers.

CREATE TABLE agent_sdk_tool_audit_events (
    seq               INTEGER PRIMARY KEY AUTOINCREMENT,
    id                TEXT    NOT NULL UNIQUE,
    operation_id      TEXT    NOT NULL,
    task_id           TEXT    NOT NULL,
    parent_task_id    TEXT    NOT NULL,
    thread_id         TEXT    NOT NULL,
    tool_call_id      TEXT    NOT NULL,
    tool_name         TEXT    NOT NULL,
    effect_class      TEXT    NOT NULL,
    kind              TEXT    NOT NULL,
    kind_payload      TEXT    NOT NULL,
    provider          TEXT    NOT NULL,
    model             TEXT    NOT NULL,
    input             TEXT    NULL,
    output            TEXT    NULL,
    error             TEXT    NULL,
    recorded_at       TEXT    NOT NULL,

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
        CHECK (json_type(kind_payload) = 'object'),

    CONSTRAINT agent_sdk_tool_audit_events_kind_payload_discriminant_check
        CHECK (json_extract(kind_payload, '$.kind') = kind)
);

CREATE INDEX agent_sdk_tool_audit_events_by_operation_idx
    ON agent_sdk_tool_audit_events (operation_id, seq);

CREATE INDEX agent_sdk_tool_audit_events_by_task_idx
    ON agent_sdk_tool_audit_events (task_id, seq);

CREATE INDEX agent_sdk_tool_audit_events_by_thread_idx
    ON agent_sdk_tool_audit_events (thread_id, seq);

CREATE INDEX agent_sdk_tool_audit_events_by_recorded_at_idx
    ON agent_sdk_tool_audit_events (recorded_at);
