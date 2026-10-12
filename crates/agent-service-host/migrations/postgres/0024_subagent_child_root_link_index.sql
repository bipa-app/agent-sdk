-- no-transaction
-- Index the child root every subagent invocation links, whatever the
-- invocation's status. A root that ends with input undelivered reads this
-- link to decide whether it is a subagent's child-thread root, under its
-- thread and root locks; the existing index covers only parked
-- invocations, so that read scanned the whole table.
--
-- Built CONCURRENTLY, outside a transaction, so task writes continue during
-- the build. A failed build leaves an INVALID index under this name, and a
-- retry of this migration then fails on it instead of accepting it: drop
-- the leftover with DROP INDEX CONCURRENTLY before retrying.
CREATE INDEX CONCURRENTLY agent_sdk_tasks_subagent_child_root_idx
    ON agent_sdk_tasks ((state_json -> 'invocation' ->> 'child_root_task_id'))
    WHERE kind = 'subagent';
