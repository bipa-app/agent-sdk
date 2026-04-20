# Agent Server - Comprehensive Project Review

> Review date: 2026-04-08
> Linear Initiative: [Agent Server](https://linear.app/bipa/initiative/agent-server-5bffd4d7ee6a)
> Linear Project: [Server SDK](https://linear.app/bipa/project/server-sdk-1f17d8469d74)
> Git branch: `sdk/v2`

---

## 1. Milestone Alignment: BROKEN

### Problem

The Linear milestones **do not match the plan phases**. They appear to be from an older, now-replaced planning effort:

| Linear Milestone | Plan Phase |
|---|---|
| Phase 1: Scaffolding + Crypto + Threads | Phase 0: Workspace Split and Crate Topology |
| Phase 2: Encrypted MessageStore | Phase 1: SDK Contracts and Orchestration |
| Phase 3: Msgs Integration + Turn Dispatch | Phase 2: Journal and Task Lifecycle |
| Phase 4: Turn Processing Engine | Phase 3: Threads, Messages, and Checkpoints |
| Phase 5: Event Pipeline + Streaming | Phase 4: Turn Worker and Context Factory |
| Phase 6: Tool Execution + Confirmation | Phase 5: Tools, Confirmation, and Audit |
| Phase 7: Task Journal + Crash Recovery | Phase 6: Events, Replay, and Streaming |
| Phase 8: AgentServer Builder + Integration | Phase 7: Subagent Hierarchy and Cancellation |
| _(missing)_ | Phase 8: AMQP Outbox and GA Hardening |

### What must change

1. **Delete all 8 existing milestones** - they reference the old plan and will confuse anyone reading the project.
2. **Create 9 new milestones** matching the replacement plan phases (0-8).
3. **Assign all issues to their correct milestones** - currently every `projectMilestone` is null.
4. **Add target dates** - all milestones currently have "No date". Even rough estimates help the team understand sequencing.

---

## 2. Labels/Tags: Uniform and Useless

### Problem

Every backlog issue has the **exact same 4 labels**:
- `agent-server`
- `component:backend`
- `Chore`
- `phase:build`

This defeats the purpose of labels. Specific problems:

1. **`Chore` is wrong** - Most issues are `Feature` work (building new systems), not chores. Only Phase 8 (GA hardening, retention, metrics) might qualify as chore work.

2. **`phase:build` is too generic** - All issues are in build phase. This label adds no information.

3. **`component:backend` is too broad** - Within the agent-server, there are at least 5 distinct architectural domains. Using one label for all of them makes filtering impossible.

4. **`agent-server` is fine** - This correctly scopes to the initiative. Keep it.

### Recommended label strategy

**Keep**: `agent-server` (initiative scope)

**Replace `Chore` with correct type per issue**:
- `Feature` for Phases 0-7 (new systems)
- `Chore` for Phase 8 hardening/ops tasks only
- `Improvement` if any issues refine existing work

**Replace `component:backend` with domain-specific labels**:
- `domain:sdk-contracts` - Phase 1 work
- `domain:task-journal` - Phase 2 work
- `domain:persistence` - Phase 3 work (threads, messages, checkpoints)
- `domain:turn-worker` - Phase 4 work
- `domain:tool-runtime` - Phase 5 work
- `domain:event-pipeline` - Phase 6 work
- `domain:subagent` - Phase 7 work
- `domain:amqp-ops` - Phase 8 work

**Replace `phase:build` with actual lifecycle state** (or remove entirely since Linear status already tracks this):
- Remove `phase:build` from all issues
- Use Linear issue status (Todo, In Progress, Done) instead

---

## 3. Missing Pieces

### 3.1 Phase 5 has NO sub-tasks

Phase 5 (Tools, Confirmation, and Audit) is one of the **most complex phases** in the entire plan - it covers:
- Tool runtime worker implementation
- Durable execution intent and operation identity
- Execution semantics by tool class (sync, async, listen)
- Confirmation flow with prepare/confirm/reject
- Parent resume and batch aggregation with mixed confirm/observe semantics
- Tool audit log with 13+ outcome states

The parent issue `ENG-7897` exists but there are **zero sub-task issues** for it. Every other phase (except Phase 0 which is done) has 6-7 sub-tasks.

**Action needed**: Break Phase 5 into 6-7 sub-tasks matching the plan structure (sections 1-6 in `05-tools-confirmation-and-audit.md`).

### 3.2 No Phase 0 milestone or tracking

Phase 0 is complete (6 issues Done: ENG-7928 to ENG-7933), but:
- No milestone exists for Phase 0
- These issues are not grouped under a parent
- The completion is not visible in project progress

### 3.3 No issue dependencies

The plan has a **strict dependency chain**: Phase 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8.

But no Linear issues have `blocks` / `blocked by` relationships. This means:
- Team members can't see what's ready to work on vs what's blocked
- There's no automatic flow when phases complete
- The critical path is invisible

### 3.4 No project overview document

The project description in Linear is a single sentence. Team members who open this project have no way to understand:
- What is being built and why
- What the architectural decisions are
- What the delivery strategy is
- What the acceptance criteria are at each phase
- How the phases connect

### 3.5 Missing parent-child structure for phases

The plan has clear parent/child structure:
- Phase parents: ENG-7892 (Phase 1), ENG-7893 (Phase 2), etc.
- Sub-tasks: ENG-7908-7914 (Phase 1 sub-tasks), etc.

But Linear doesn't show these as parent/child issues - they appear as flat items. This needs to be verified and fixed if the hierarchy is not set up.

### 3.6 No priority differentiation

Almost every issue has `No Priority` (0) or `High` (1-2). The critical path items (Phases 1-4) should have explicit priority ordering so the team knows what to focus on.

---

## 4. Dependency Map and Incremental Complexity Path

### Phase Dependency Chain

```
Phase 0: Workspace Split (DONE)
    |
    v
Phase 1: SDK Contracts  <-- CURRENT FOCUS (ENG-7908 In Review)
    |
    v
Phase 2: Task Journal
    |
    v
Phase 3: Threads + Checkpoints
    |
    v
Phase 4: Turn Worker
    |
    v
Phase 5: Tool Runtime + Confirmation
    |
    v
Phase 6: Event Pipeline
    |
    v
Phase 7: Subagent Hierarchy
    |
    v
Phase 8: AMQP + GA Hardening
```

### Incremental Complexity Strategy

The plan is designed to be built **bottom-up**. Each phase adds exactly one new capability while standing on proven ground:

| Phase | New Capability Added | Builds On |
|---|---|---|
| 0 | Workspace structure | Nothing (standalone) |
| 1 | SDK/server contract boundary | Workspace layout |
| 2 | Execution journal (tasks, leases, recovery) | Contract types |
| 3 | Conversation persistence + crash recovery | Journal lifecycle |
| 4 | Root turn worker (one model turn end-to-end) | Persistence + journal |
| 5 | Tool execution as child tasks + confirmation | Turn worker + journal |
| 6 | Durable event log + replay + streaming | All execution paths |
| 7 | Subagent hierarchy | All single-agent paths |
| 8 | AMQP transport + production hardening | Everything |

### What this means for the team

1. **Phases 1-3 are foundation** - pure data model, contracts, and persistence. No runtime behavior yet. A developer can understand these by reading SQL schemas and trait definitions.

2. **Phase 4 is the first "it runs" moment** - this is where a root turn worker actually executes a model call. The team should treat Phase 4 completion as the first integration milestone.

3. **Phase 5 is the complexity cliff** - tool execution, confirmation flows, and batch semantics are where most of the edge cases live. This phase should get the most review attention.

4. **Phases 6-7 are horizontal extensions** - they add event replay and subagents on top of a working single-agent system. These can potentially be developed somewhat in parallel.

5. **Phase 8 is polish** - AMQP integration, retention, health checks. This should not drive architectural decisions.

### Sub-task Dependencies Within Phases

Within each phase, sub-tasks should generally be done in numerical order. For example, Phase 4:

```
4.1: AgentDefinition + Bootstrapping
    |
    v
4.2: ExecutionContextFactory + Staged Stores
    |
    v
4.3: Text-Only Turn Execution + Commit
    |
    v
4.4: Tool-Boundary Suspension + Child Dispatch
    |
    v
4.5: Resume from Completed Tool Results
    |
    v
4.6: Cancellation, Failure, Regression
```

---

## 5. Project Overview (for team members)

### What We're Building

A Rust library crate (`agent-server`) that lets a host application run many durable agent threads safely. This is NOT a standalone server binary - it's a library that consumers embed into their own applications.

### Why This Architecture

The previous planning effort had these problems:
1. Execution ownership was split across AMQP, turn rows, and a later task journal
2. Tool execution was modeled as inline blocking loop behavior
3. Event streaming and replay had no durable source of truth
4. The SDK contracts that the server depended on didn't exist yet

The replacement plan fixes all four by establishing:
- **One authoritative journal** (`agent_tasks`) that owns all execution state
- **Task-owned tool execution** instead of inline blocking calls
- **Durable event envelopes** for replay
- **SDK contract lockdown** before any server code is written

### Core Architectural Decisions

1. **Library-first**: Consumers import the crate and provide their own DB pool, cipher, and AMQP connection. No opinions about deployment topology.

2. **Journal-is-truth**: `agent_tasks` is the sole execution authority. AMQP is optional wakeup/fanout, never authoritative. Queue loss affects latency, never correctness.

3. **Single-turn orchestration**: The server executes exactly one externally orchestrated turn at a time via `run_turn`. The old `run()` loop is a convenience wrapper only.

4. **Tool work as child tasks**: In server mode, root turn workers don't execute tools inline. Tool calls become `tool_runtime` child tasks with their own lifecycle, leases, and recovery.

5. **Checkpoint recovery**: After every completed turn, a full-message checkpoint is persisted. Crash recovery always resumes from the latest checkpoint.

6. **Durable events**: Public event replay uses persisted `AgentEventEnvelope` rows with server-owned per-thread sequence numbers. In-process streaming is an optimization, not the source of truth.

### Consumers

| Consumer | Style | Key Concern |
|---|---|---|
| Bipa main agent | In-process library | gRPC loopback tools, encrypted persistence |
| Coding agent | Standalone binary | Filesystem sandbox, subagents, MCP |
| Satoshi agent | Standalone binary | Background durability, event replay |

### Current Status

- **Phase 0**: COMPLETE (workspace split done, 6/6 issues closed)
- **Phase 1**: IN PROGRESS (ENG-7908 in review, 6 more sub-tasks Todo)
- **Phases 2-8**: BACKLOG (not started)
- **50 total issues**, 6 completed, 6 canceled (old plan), 38 in backlog/todo

---

## 6. Summary of Required Actions

### Critical (do before continuing implementation)

1. **Delete old milestones and create new ones** matching plan phases 0-8
2. **Assign all issues to their correct milestones**
3. **Create Phase 5 sub-tasks** (6-7 issues covering tool runtime, execution intent, confirmation, batch aggregation, audit)
4. **Set up issue dependencies** (blocks/blocked-by between phases)

### Important (do this week)

5. **Fix labels**: Replace `Chore` with `Feature`, replace `component:backend` with domain-specific labels, remove `phase:build`
6. **Add milestone target dates** (even rough estimates)
7. **Set up parent/child relationships** for phase parents and their sub-tasks
8. **Write a project overview** in Linear project description (use section 5 above as a starting point)

### Nice to have

9. **Add priority ordering** to critical-path phases (1-4 should be highest)
10. **Create a Phase 0 parent issue** and link the completed sub-tasks under it
