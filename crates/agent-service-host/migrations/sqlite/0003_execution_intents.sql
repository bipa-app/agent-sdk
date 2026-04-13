-- SQLite variant of migration 0003 (execution intents).
-- Mirrors the Postgres schema with SQLite type adjustments:
-- JSONB → TEXT, TIMESTAMPTZ → TEXT.

CREATE TABLE agent_sdk_execution_intents (
    operation_id       TEXT    PRIMARY KEY,
    effect_class       TEXT    NOT NULL,
    tool_call_id       TEXT    NOT NULL,
    child_task_id      TEXT    NOT NULL,
    tool_name          TEXT    NOT NULL,
    input              TEXT    NOT NULL,
    status             TEXT    NOT NULL,
    error              TEXT    NULL,
    created_at         TEXT    NOT NULL,
    updated_at         TEXT    NOT NULL,

    CONSTRAINT agent_sdk_execution_intents_effect_class_check
        CHECK (effect_class IN ('replay_safe', 'side_effecting', 'resumable')),

    CONSTRAINT agent_sdk_execution_intents_status_check
        CHECK (status IN ('pending', 'started', 'completed', 'failed')),

    CONSTRAINT agent_sdk_execution_intents_error_check
        CHECK (
            (status = 'failed' AND error IS NOT NULL)
            OR (status <> 'failed' AND error IS NULL)
        )
);

CREATE INDEX agent_sdk_execution_intents_by_child_task_idx
    ON agent_sdk_execution_intents (child_task_id);
