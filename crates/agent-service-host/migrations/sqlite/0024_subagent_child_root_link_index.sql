-- Index the child root every subagent invocation links, whatever the
-- invocation's status. A root that ends with input undelivered reads this
-- link to decide whether it is a subagent's child-thread root; the existing
-- index covers only parked invocations, so that read scanned the whole
-- table.

CREATE INDEX agent_sdk_tasks_subagent_child_root_idx
    ON agent_sdk_tasks (json_extract(state_json, '$.invocation.child_root_task_id'))
    WHERE kind = 'subagent';
