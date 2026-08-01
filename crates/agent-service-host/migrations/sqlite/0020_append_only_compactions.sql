ALTER TABLE agent_sdk_message_heads
    ADD COLUMN compactions_json TEXT NULL
        CHECK (compactions_json IS NULL OR json_type(compactions_json) = 'array');

-- Older binaries do not know about compactions_json and therefore
-- omit it from their UPSERT. If one destructively shortens history,
-- invalidate the now-unrelated lineage inside the same SQLite
-- statement transaction. The trigger's own UPDATE does not name
-- history_json, so it remains safe with recursive_triggers enabled.
CREATE TRIGGER agent_sdk_message_heads_invalidate_compactions_on_history_shrink
AFTER UPDATE OF history_json ON agent_sdk_message_heads
FOR EACH ROW
WHEN json_array_length(NEW.history_json) < json_array_length(OLD.history_json)
     AND NEW.compactions_json IS NOT NULL
BEGIN
    UPDATE agent_sdk_message_heads
    SET compactions_json = NULL
    WHERE thread_id = NEW.thread_id;
END;
