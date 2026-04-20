# Agent Server - Project Overview for Team Members

> Use this document as the starting point for anyone new to the project.

---

## What We're Building

**agent-server** is a Rust library crate that enables a host application to run many durable agent threads safely. It is NOT a standalone server binary -- it's a library that consumers embed and configure.

Think of it as: **"Postgres for agent execution"** -- you import it, hand it your DB pool and crypto keys, and it handles the hard parts of running AI agent conversations durably.

---

## Why We Need This

The previous approach had fundamental problems:

| Problem | Old Approach | New Approach |
|---|---|---|
| Who owns execution state? | Split across AMQP + turn rows + task journal | **One journal** (`agent_tasks`) owns everything |
| How do tools execute? | Inline blocking inside the agent loop | **Child tasks** with their own lifecycle |
| How do we replay events? | No durable source of truth | **Persisted event envelopes** with sequence numbers |
| What are the SDK contracts? | Assumed, not defined | **Locked contracts** before any server code |

---

## Architecture in One Diagram

```
                     +-----------------------+
                     |    Host Application   |
                     |  (Bipa, Coding Agent, |
                     |   Satoshi, etc.)      |
                     +-----------+-----------+
                                 |
                        imports agent-server
                                 |
                     +-----------v-----------+
                     |    Agent Server Lib   |
                     |                       |
                     |  +--- Turn Worker --+ |
                     |  | Acquires tasks   | |
                     |  | Calls run_turn   | |
                     |  | Dispatches tools | |
                     |  +--------+---------+ |
                     |           |           |
                     |  +--- Task Journal -+ |
                     |  | agent_tasks      | |
                     |  | Leases, retries  | |
                     |  | Recovery         | |
                     |  +--------+---------+ |
                     |           |           |
                     |  +--- Persistence --+ |
                     |  | Threads          | |
                     |  | Messages         | |
                     |  | Checkpoints      | |
                     |  | Events           | |
                     |  +------------------+ |
                     |                       |
                     +-----------+-----------+
                                 |
                     +-----------v-----------+
                     |    Postgres + AMQP    |
                     |   (provided by host)  |
                     +-----------------------+
```

---

## Core Concepts (5-minute version)

### 1. Threads = Conversations
A thread is a conversation container. It holds messages, tracks token usage, and has a lifecycle (active -> completed -> archived).

### 2. Tasks = Execution Units
Every piece of work is a **task** in `agent_tasks`:
- **Root turn task**: One model call (user sends message -> model responds)
- **Tool runtime task**: One tool execution (model asked to run a tool)
- **Subagent task**: A child agent with its own conversation

Tasks have leases (to prevent double-execution), retries (for resilience), and a strict state machine.

### 3. Checkpoints = Recovery Points
After every completed turn, we take a full snapshot of the conversation state. If the server crashes, it resumes from the last checkpoint. No work is lost.

### 4. Events = What Happened
Everything that happens is recorded as an event with a per-thread sequence number. Clients can replay events from any point, and live-tail new events in real time.

### 5. The Key Invariant
**One active root task per thread.** This prevents concurrent model calls on the same conversation. Tool tasks run as children of the root task, not as independent root tasks.

---

## How a Turn Works (simplified)

```
User sends message
       |
       v
Server creates root_turn task (queued -> pending)
       |
       v
Worker acquires task (pending -> running)
       |
       v
Worker calls SDK run_turn() with the message
       |
       +--- Model responds with text only -----> Commit checkpoint + advance
       |
       +--- Model requests tool calls ----------+
                                                 |
                                                 v
                                    Create tool_runtime child tasks
                                    Root task -> waiting_on_children
                                                 |
                                                 v
                                    Tool workers execute each tool
                                    Tool tasks -> completed
                                                 |
                                                 v
                                    All tools done -> Resume root task
                                    Root task -> running (with tool results)
                                                 |
                                                 v
                                    Worker calls run_turn() again
                                    (This may loop: more tools, or final text)
```

---

## The 9 Database Tables

| Table | Purpose | Domain |
|---|---|---|
| `threads` | Conversation identity + aggregate status | Conversation |
| `messages` | SDK-compatible message history | Conversation |
| `agent_tasks` | **Authoritative execution journal** | Execution |
| `turn_attempts` | Immutable log of model calls | Execution |
| `turn_checkpoints` | Full snapshots for crash recovery | Execution |
| `tool_executions` | Durable execution intent for tools | Tools |
| `tool_call_log` | Append-only audit trail | Tools |
| `events` | Replayable event log | Events |
| `msgs` | Optional AMQP outbox | Transport |

---

## Build Phases

| Phase | Name | Status | What It Delivers |
|---|---|---|---|
| 0 | Workspace Split | **DONE** | Cargo workspace with SDK + server crates |
| 1 | SDK Contracts | **IN PROGRESS** | `run_turn` boundary, event envelopes, tool handoff |
| 2 | Task Journal | Backlog | `agent_tasks`, leases, retries, recovery |
| 3 | Persistence | Backlog | Threads, messages, checkpoints, turn_attempts |
| 4 | Turn Worker | Backlog | Root turn execution end-to-end |
| 5 | Tool Runtime | Backlog | Tool child tasks, confirmation, audit |
| 6 | Event Pipeline | Backlog | Durable events, replay, streaming |
| 7 | Subagents | Backlog | Multi-agent hierarchy, cascade cancel |
| 8 | GA Hardening | Backlog | AMQP outbox, retention, health checks |

Each phase builds on the previous ones. The critical integration milestone is **Phase 4** -- that's when the first end-to-end turn actually executes.

---

## Key Design Decisions and Why

### "Why not use AMQP as the source of truth?"
Because queue loss or duplication would corrupt execution state. By making `agent_tasks` authoritative and AMQP optional, a lost message only delays work -- it never causes incorrect behavior.

### "Why not execute tools inline like the SDK does?"
Because inline execution blocks the worker and makes recovery impossible. If a tool takes 30 seconds and the worker crashes, all state is lost. With child tasks, the tool work is durable and can be retried.

### "Why checkpoints instead of event sourcing?"
Full-message checkpoints are simpler to reason about for v1. Event sourcing adds complexity around snapshot computation and migration. We persist events for replay, but recovery uses checkpoints.

### "Why separate `turn_attempts` from `agent_tasks`?"
Because a single task may have multiple attempts (retries). The task tracks lifecycle, the attempt tracks what happened during one try. This separation is clean for audit and debugging.

---

## Quick Links

- **Plan documents**: `plans/agent-server/phases/` (one file per phase)
- **Data model**: `plans/agent-server/data-model.md`
- **PRD**: `plans/agent-server/prd.md`
- **Architecture canvas**: `plans/agent-server/claude-pass/architecture-canvas.html`
- **Linear project**: https://linear.app/bipa/project/server-sdk-1f17d8469d74
- **Git branch**: `sdk/v2`
