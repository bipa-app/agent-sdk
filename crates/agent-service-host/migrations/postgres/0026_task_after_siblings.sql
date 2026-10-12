-- Earlier children of the same batch that a pending tool child waits for.
-- NULL for every row that may run as soon as it is pending, including
-- every row written before this column existed.

ALTER TABLE agent_sdk_tasks
    ADD COLUMN after_siblings_json JSONB NULL;

-- The acquisition scan reads only pending rows that wait on nothing, so
-- its cost never grows with the rows held behind running siblings.
CREATE INDEX agent_sdk_tasks_ready_fifo_idx
    ON agent_sdk_tasks (created_at, id)
    WHERE status = 'pending' AND after_siblings_json IS NULL;

-- The bounded backstop probe walks held rows in the same order.
CREATE INDEX agent_sdk_tasks_held_fifo_idx
    ON agent_sdk_tasks (created_at, id)
    WHERE status = 'pending' AND after_siblings_json IS NOT NULL;
