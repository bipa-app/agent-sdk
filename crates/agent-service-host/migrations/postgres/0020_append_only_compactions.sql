ALTER TABLE agent_sdk_message_heads
    ADD COLUMN compactions_json JSONB NULL;

ALTER TABLE agent_sdk_message_heads
    ADD CONSTRAINT agent_sdk_message_heads_compactions_json_check
        CHECK (compactions_json IS NULL OR jsonb_typeof(compactions_json) = 'array');

-- A mixed-version deployment can still have a pre-compaction writer
-- updating this row. Those writers do not name compactions_json in
-- their UPSERT, so a destructive history shrink would otherwise leave
-- lineage that points into the replaced transcript. Clear it in the
-- same row update; append-only writes keep their lineage.
CREATE FUNCTION agent_sdk_invalidate_compactions_on_history_shrink()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF jsonb_array_length(NEW.history_json) < jsonb_array_length(OLD.history_json) THEN
        NEW.compactions_json := NULL;
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER agent_sdk_message_heads_invalidate_compactions_on_history_shrink
    BEFORE UPDATE OF history_json ON agent_sdk_message_heads
    FOR EACH ROW
    EXECUTE FUNCTION agent_sdk_invalidate_compactions_on_history_shrink();
