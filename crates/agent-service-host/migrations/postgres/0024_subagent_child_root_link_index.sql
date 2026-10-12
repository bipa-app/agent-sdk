-- Index the child root every subagent invocation links, whatever the
-- invocation's status. A root that ends with input undelivered reads this
-- link to decide whether it is a subagent's child-thread root, under its
-- thread and root locks; the existing index covers only parked
-- invocations, so that read scanned the whole table.
--
-- Built inside the migration's transaction: the migrator's advisory lock
-- already serializes replicas, and a concurrent build would wait on a
-- replica's snapshot while that replica waits for the lock.

CREATE INDEX agent_sdk_tasks_subagent_child_root_idx
    ON agent_sdk_tasks ((state_json -> 'invocation' ->> 'child_root_task_id'))
    WHERE kind = 'subagent';
