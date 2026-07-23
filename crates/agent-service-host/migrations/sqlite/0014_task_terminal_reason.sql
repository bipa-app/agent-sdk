ALTER TABLE agent_sdk_tasks
    ADD COLUMN terminal_reason_json TEXT NULL;

UPDATE agent_sdk_tasks
SET terminal_reason_json = CASE status
    WHEN 'completed' THEN '{"reason":"completed"}'
    WHEN 'failed' THEN '{"reason":"internal_error"}'
    WHEN 'cancelled' THEN '{"reason":"user_cancel"}'
    ELSE NULL
END;

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
