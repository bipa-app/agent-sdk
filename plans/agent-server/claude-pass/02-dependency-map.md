# Agent Server - Dependency Map and Incremental Build Order

> This document shows what depends on what, so the team can build incrementally
> without taking on too much complexity at once.

---

## Phase Dependency Graph

```
                    Phase 0: Workspace Split
                    (DONE - 6/6 issues closed)
                              |
                              v
                    Phase 1: SDK Contracts
                    (IN PROGRESS - ENG-7908 in review)
                              |
                              v
              +---------------+---------------+
              |                               |
              v                               v
    Phase 2: Task Journal           Phase 3: Persistence*
    (agent_tasks, leases,           (threads, messages,
     retries, recovery)              turn_attempts, checkpoints)
              |                               |
              +---------------+---------------+
                              |
                              v
                    Phase 4: Turn Worker
                    (root turn execution, tool dispatch)
                              |
                              v
                    Phase 5: Tool Runtime
                    (tool workers, confirmation, audit)
                              |
                              v
                    Phase 6: Event Pipeline
                    (durable events, replay, streaming)
                              |
                              v
                    Phase 7: Subagent Hierarchy
                    (child tasks, cascade cancel)
                              |
                              v
                    Phase 8: AMQP + GA Hardening
                    (outbox, retention, health checks)
```

*Phase 3 depends on Phase 2's journal core being in place, but
the thread/message persistence work is somewhat independent of the
deeper task lifecycle features. In practice, build Phase 2 first.

---

## What Each Phase Unlocks

| Phase | What You Can Do After | What You Still Can't Do |
|---|---|---|
| 0 | Build and test SDK crates independently; add agent-server crate to workspace | Run any server code |
| 1 | Call `run_turn` in server mode; seed event sequences; get typed tool handoff | Persist anything; run workers |
| 2 | Create/acquire/lease tasks; retry failed tasks; enforce one-root-per-thread | Store conversation history; run model turns |
| 3 | Persist threads, messages, turn attempts; recover from checkpoints | Actually execute turns (no worker yet) |
| 4 | Execute a full model turn end-to-end; dispatch tool child tasks | Execute tools; replay events |
| 5 | Execute tools durably; handle confirmation; audit tool lifecycle | Replay events after restart |
| 6 | Persist and replay events; live-tail streaming; reconnect-safe replay | Run subagent hierarchies |
| 7 | Spawn subagent trees; cascade cancellation; inherit limits | Production AMQP; retention; health checks |
| 8 | Ship to production with AMQP fanout, retention, metrics, health checks | Nothing - this is GA |

---

## Within-Phase Sub-Task Order

### Phase 1: SDK Contracts (7 sub-tasks)

```
1.1 run_turn boundary + execution modes
    |
    v
1.2 Event envelope authority + event sink
    |
    v
1.3 ToolContextSeed + ExecutionContextFactory inputs
    |
    v
1.4 External ToolInvocation plan + handoff
    |
    v
1.5 ToolInvocation policy + versioned continuations
    |
    v
1.6 ToolAuditSink + full lifecycle audit
    |
    v
1.7 TurnSummary + regression suite + docs
```

### Phase 2: Task Journal (7 sub-tasks)

```
2.1 agent_tasks schema + task kinds + status model
    |
    v
2.2 Root submission queue + FIFO promotion
    |
    v
2.3 Runnable acquisition + lease ownership + heartbeats
    |
    v
2.4 Parent waiting + confirmation pause/resume + typed state
    |
    v
2.5 Retry budget + failure handling + recovery matrix
    |
    v
2.6 Tool runtime child tasks + cancellation tree + parent resume
    |
    v
2.7 Journal CAS APIs + concurrency tests + docs
```

### Phase 3: Persistence (6 sub-tasks)

```
3.1 Threads projection + aggregate ownership
    |
    v
3.2 Message projection + transactional replace_history
    |
    v
3.3 turn_attempts schema + append-only audit repo
    |
    v
3.4 Completed-turn checkpoints + atomic commit path
    |
    v
3.5 Thread-scoped checkpoint recovery + rebuild API
    |
    v
3.6 Crash + compaction + persistence regression suite
```

### Phase 4: Turn Worker (6 sub-tasks)

```
4.1 AgentDefinition resolution + worker bootstrapping
    |
    v
4.2 ExecutionContextFactory + checkpoint-seeding + staged stores
    |
    v
4.3 Text-only root turn execution + completed-turn commit
    |
    v
4.4 Tool-boundary suspension + tool runtime child task dispatch
    |
    v
4.5 Resume root turn from completed tool results
    |
    v
4.6 Root worker cancellation + failure paths + regression coverage
```

### Phase 5: Tool Runtime (NEEDS SUB-TASKS CREATED)

Suggested breakdown based on plan:
```
5.1 Tool runtime worker + task acquisition + basic sync execution
    |
    v
5.2 Durable execution intent + operation identity + fail-closed persistence
    |
    v
5.3 Execution semantics by tool class (sync / async / listen)
    |
    v
5.4 Confirmation flow (prepare -> await -> confirm/reject -> execute)
    |
    v
5.5 Parent resume + batch aggregation + mixed batch failure
    |
    v
5.6 Tool audit log + redaction + full lifecycle outcomes
```

### Phase 6: Event Pipeline (6 sub-tasks)

```
6.1 Durable event committer + thread-scoped sequencing
    |
    v
6.2 Atomic event commit rules for root + tool task transitions
    |
    v
6.3 Replay API + race-free replay-to-live handoff
    |
    v
6.4 Live tail hub + lag detection + bounded-wait disconnect
    |
    v
6.5 Broad replay coverage (deltas, thinking, tool progress, subagent events)
    |
    v
6.6 Regression suite + operational limits + event contract docs
```

### Phase 7: Subagent Hierarchy (7 sub-tasks)

```
7.1 Durable subagent spawn contract + spec resolution
    |
    v
7.2 Subagent invocation task + child thread allocation + durable linkage
    |
    v
7.3 Child thread execution reuse + final result materialization
    |
    v
7.4 Summary-only parent visibility + child reference surfacing
    |
    v
7.5 Inherited limits + capability profiles + policy narrowing
    |
    v
7.6 Cascade cancellation + nested trees + failure mapping
    |
    v
7.7 Restart recovery + nested replayability + regression suite
```

### Phase 8: AMQP + GA Hardening (7 sub-tasks)

```
8.1 Transactional outbox contract + message kinds
    |
    v
8.2 AMQP relay worker + publish ack + startup backfill
    |
    v
8.3 Task wakeup nudges + duplicate-safe consumer rechecks
    |
    v
8.4 Cross-instance thread event fanout + advisory watch
    |
    v
8.5 Degraded mode health readiness + fallback sweeps
    |
    v
8.6 Built-in retention janitor + replay floor semantics
    |
    v
8.7 GA metrics backlog protection + regression suite + ops docs
```

---

## Critical Path

The critical path through the project is:

**Phase 0** -> **Phase 1** -> **Phase 2** -> **Phase 3** -> **Phase 4** -> **Phase 5**

Everything after Phase 5 is important but less architecturally risky. Phases 1-5 establish the core execution model. If those are right, Phases 6-8 are extensions.

**The single riskiest phase is Phase 5** (Tool Runtime). It has:
- The most edge cases (sync/async/listen, confirm/reject, batch semantics)
- External side effects (tools that call APIs)
- The hardest recovery semantics (what if a tool has already acted?)
- No sub-tasks created yet in Linear

---

## Recommended Work Allocation

For a solo developer:
- Phase 1: 2-3 weeks (SDK contract changes require careful API design)
- Phase 2: 2-3 weeks (journal is foundational, must be bulletproof)
- Phase 3: 1-2 weeks (well-understood persistence patterns)
- Phase 4: 2-3 weeks (first integration point, lots of wiring)
- Phase 5: 3-4 weeks (highest complexity, most edge cases)
- Phase 6: 2-3 weeks (event pipeline, replay semantics)
- Phase 7: 2-3 weeks (subagent hierarchy extends existing patterns)
- Phase 8: 1-2 weeks (operational hardening, lower risk)

**Total estimated: 15-23 weeks** for a single developer focused full-time.

For parallel work with 2 developers:
- Dev A focuses on Phases 1-3 (contracts + journal + persistence)
- Dev B focuses on SDK changes and integration tests
- After Phase 3, both converge on Phase 4 (integration point)
- Phase 5+ can be done by either developer
