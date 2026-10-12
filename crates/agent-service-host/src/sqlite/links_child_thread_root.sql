SELECT EXISTS (
    SELECT 1
    FROM agent_sdk_tasks
    WHERE kind = 'subagent'
      AND json_extract(state_json, '$.invocation.child_root_task_id') = ?1
) AS "linked!: bool"
