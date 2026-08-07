-- One-shot data-repair ledger (ENG-9651). The startup repair sweep
-- records each thread it repaired so the pass runs once per corruption
-- class per thread: a thread the sweep already fixed is never touched
-- again, and a thread that re-corrupts (it should not — the producers
-- are fixed at the write seams) is repaired exactly once more.

CREATE TABLE agent_sdk_applied_repairs (
    thread_id TEXT NOT NULL,
    repair_key TEXT NOT NULL,
    applied_at TEXT NOT NULL,
    CONSTRAINT agent_sdk_applied_repairs_pk
        PRIMARY KEY (thread_id, repair_key),
    CONSTRAINT agent_sdk_applied_repairs_thread_fk
        FOREIGN KEY (thread_id) REFERENCES agent_sdk_threads(thread_id)
        ON DELETE CASCADE
);
