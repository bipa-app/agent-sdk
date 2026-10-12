-- Driven from the live rows: `unlikely` tells the planner they are a sliver
-- of the retained history, which per-index averages (ANALYZE without STAT4
-- samples) cannot show, and the unary `+` keeps it from driving from the
-- terminal roots instead. Without both, such statistics make it scan every
-- row.
SELECT DISTINCT descendant.root_id AS "root_id!: String"
FROM agent_sdk_tasks AS descendant
JOIN agent_sdk_tasks AS root ON root.id = descendant.root_id
WHERE unlikely(descendant.status IN (
        'queued', 'pending', 'running', 'waiting_on_children',
        'awaiting_confirmation', 'awaiting_question'
    ))
  AND descendant.parent_id IS NOT NULL
  AND +root.status IN ('completed', 'failed', 'cancelled')
ORDER BY descendant.root_id
