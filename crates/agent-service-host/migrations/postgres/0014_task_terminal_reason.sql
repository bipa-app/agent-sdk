ALTER TABLE agent_sdk_tasks
    ADD COLUMN terminal_reason_json JSONB NULL;

UPDATE agent_sdk_tasks
SET terminal_reason_json = CASE status
    WHEN 'completed' THEN '{"reason":"completed"}'::jsonb
    WHEN 'failed' THEN '{"reason":"internal_error"}'::jsonb
    WHEN 'cancelled' THEN '{"reason":"user_cancel"}'::jsonb
    ELSE NULL
END;

ALTER TABLE agent_sdk_tasks
    ADD CONSTRAINT agent_sdk_tasks_terminal_reason_json_check
    CHECK (
        (status IN ('completed', 'failed', 'cancelled') AND terminal_reason_json IS NOT NULL)
        OR
        (status NOT IN ('completed', 'failed', 'cancelled') AND terminal_reason_json IS NULL)
    );
