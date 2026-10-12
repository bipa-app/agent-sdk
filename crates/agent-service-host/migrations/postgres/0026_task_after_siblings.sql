-- Earlier children of the same batch that a pending tool child waits for.
-- NULL for every row that may run as soon as it is pending, including
-- every row written before this column existed.

ALTER TABLE agent_sdk_tasks
    ADD COLUMN after_siblings_json JSONB NULL;

-- Pending rows split by whether they wait on siblings.
--
-- The ready and held indexes, and the queries that use them, spell
-- `status` as the range `>= 'pending' AND <= 'pending'`. An equality would
-- also match agent_sdk_tasks_by_status_idx (status, created_at, id), which
-- yields the same order but walks every held row, and the planner cannot
-- tell the two apart: it does not know how many pending rows are held.
-- On a status-clustered table with 20k held rows, Postgres chose
-- by_status_idx for the equality form (20k rows filtered, 8.6 ms per scan)
-- and the ready index for the range form (2 pages, 0.4 ms).

-- The acquisition scan: pending rows that wait on nothing, oldest first.
-- It replaces agent_sdk_tasks_runnable_fifo_idx, which nothing reads now.
DROP INDEX agent_sdk_tasks_runnable_fifo_idx;

CREATE INDEX agent_sdk_tasks_ready_fifo_idx
    ON agent_sdk_tasks (created_at, id) INCLUDE (thread_id)
    WHERE status >= 'pending' AND status <= 'pending' AND after_siblings_json IS NULL;

-- The bounded backstop probe over held rows, in the same order. Only a
-- tool child can be held; naming the kind keeps this index out of reach of
-- the per-parent lookup below, whose query does not name it, so that
-- lookup can never walk every held row.
CREATE INDEX agent_sdk_tasks_held_fifo_idx
    ON agent_sdk_tasks (created_at, id)
    WHERE status >= 'pending' AND status <= 'pending' AND after_siblings_json IS NOT NULL
      AND kind = 'tool_runtime';

-- The held siblings of one parent, looked up when a sibling settles.
CREATE INDEX agent_sdk_tasks_held_by_parent_idx
    ON agent_sdk_tasks (parent_id, created_at, id)
    WHERE status >= 'pending' AND status <= 'pending' AND after_siblings_json IS NOT NULL;
