-- no-transaction
-- The parked-invocation index is a subset of the index 0024 builds over
-- every subagent invocation, which serves its lookups too. Dropped
-- CONCURRENTLY so task writes continue meanwhile.
DROP INDEX CONCURRENTLY agent_sdk_tasks_subagent_child_root_waiting_idx;
