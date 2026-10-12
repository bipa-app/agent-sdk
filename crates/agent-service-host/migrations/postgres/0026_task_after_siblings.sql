-- Earlier children of the same batch that a pending tool child waits for.
-- NULL for every row that may run as soon as it is pending, including
-- every row written before this column existed.

ALTER TABLE agent_sdk_tasks
    ADD COLUMN after_siblings_json JSONB NULL;

-- Pending rows split by whether they wait on siblings. The queries
-- that use these indexes, and the predicates below, spell `status` as the
-- range `>= 'pending' AND <= 'pending'`: an equality would also match
-- agent_sdk_tasks_by_status_idx (status, created_at, id), which yields the
-- same order but walks every held row, and the planner cannot tell the
-- two apart because it does not know how many pending rows are held.

-- The acquisition scan: pending rows that wait on nothing, oldest first.
CREATE INDEX agent_sdk_tasks_ready_fifo_idx
    ON agent_sdk_tasks (created_at, id) INCLUDE (thread_id)
    WHERE status >= 'pending' AND status <= 'pending' AND after_siblings_json IS NULL;

-- The bounded backstop probe over held rows, in the same order.
CREATE INDEX agent_sdk_tasks_held_fifo_idx
    ON agent_sdk_tasks (created_at, id)
    WHERE status >= 'pending' AND status <= 'pending' AND after_siblings_json IS NOT NULL;

-- The held siblings of one parent, looked up when a sibling settles.
CREATE INDEX agent_sdk_tasks_held_by_parent_idx
    ON agent_sdk_tasks (parent_id, created_at, id)
    WHERE status >= 'pending' AND status <= 'pending' AND after_siblings_json IS NOT NULL;
