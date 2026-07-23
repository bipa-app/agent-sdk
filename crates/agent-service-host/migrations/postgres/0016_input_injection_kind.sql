-- ENG-9278: pending boundary injections are child rows in the existing task
-- journal, so cancellation reaches them through parent_id without a second
-- queue table.
ALTER TABLE agent_sdk_tasks
    DROP CONSTRAINT agent_sdk_tasks_kind_check,
    DROP CONSTRAINT agent_sdk_tasks_queue_kind_check;

ALTER TABLE agent_sdk_tasks
    ADD CONSTRAINT agent_sdk_tasks_kind_check
        CHECK (kind IN ('root_turn', 'tool_runtime', 'subagent', 'input_injection')),
    ADD CONSTRAINT agent_sdk_tasks_queue_kind_check
        CHECK (status <> 'queued' OR kind IN ('root_turn', 'input_injection'));
