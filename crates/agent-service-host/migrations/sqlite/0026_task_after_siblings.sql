-- Earlier children of the same batch that a pending tool child waits for,
-- SQLite dialect. Mirror of postgres/0026_task_after_siblings.sql. Stored as
-- TEXT holding a JSON array, matching this schema's other JSON columns.

ALTER TABLE agent_sdk_tasks
    ADD COLUMN after_siblings_json TEXT NULL;

-- The acquisition scan reads only pending rows that wait on nothing, so
-- its cost never grows with the rows held behind running siblings.
CREATE INDEX agent_sdk_tasks_ready_fifo_idx
    ON agent_sdk_tasks (created_at, id)
    WHERE status = 'pending' AND after_siblings_json IS NULL;

-- The bounded backstop probe walks held rows in the same order.
CREATE INDEX agent_sdk_tasks_held_fifo_idx
    ON agent_sdk_tasks (created_at, id)
    WHERE status = 'pending' AND after_siblings_json IS NOT NULL;
