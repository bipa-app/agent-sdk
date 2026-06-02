# Durable, Crash-Safe Agent Loops in Rust: Inside the Agent SDK

> **Status:** `agent-sdk` is a technical preview (0.9, pre-1.0), MIT-licensed and developed openly at [github.com/bipa-app/agent-sdk](https://github.com/bipa-app/agent-sdk). The core local-agent loop is solid and well-tested; the durable server backend is feature-complete and hardened under fault-injection testing; the production transport layer (gRPC/HTTP) is designed and partially shipped. This article describes what exists in the repository today, with honest notes on what is still hardening.

## The Problem: Agent Loops Are State Machines That Crash

An LLM agent loop is deceptively simple to prototype and genuinely hard to operate. The naive version is a `while` loop: call the model, get back text or tool calls, run the tools, feed results back, repeat until the model stops. It works on your laptop. Then you put it behind a service and the realities arrive:

- A user cancels a turn while a tool is mid-execution. Now the conversation history contains a `tool_use` block with no matching `tool_result`. The next call to the Anthropic API is rejected because the history is unbalanced.
- The process crashes after a tool wrote to a database but before the result was recorded. On retry, the tool runs again — double-charging, double-sending, double-mutating.
- Two workers pick up the same thread and both try to advance it. Now you have interleaved, corrupt history.
- A long conversation blows past the model's context window mid-flight.

These are not edge cases; they are the steady state of running agents in production. `agent-sdk` treats the agent loop as what it actually is — a **durable state machine** — and builds the persistence, ordering, cancellation, and recovery primitives that a state machine needs to survive process death. The framing the project uses internally is apt: a Temporal-grade backend for agent loops.

## Three Layers of Durability

The SDK separates persistence into three independent storage traits, each addressing a distinct failure mode.

**1. Per-thread state and history.** `MessageStore` holds conversation history per `ThreadId`; `StateStore` holds agent state checkpoints. Both are persisted at turn boundaries. After a crash, a thread can be rehydrated to its last committed turn.

**2. Per-tool idempotency via a write-ahead pattern.** This is the layer that prevents double-execution. `ToolExecutionStore` records a tool's *intent* **before** the tool runs, then updates with the *result* **after** completion. The `ToolExecution` record carries an `ExecutionStatus` of `InFlight` or `Completed`. If the process dies between intent and completion, the retry observes the recorded intent, and because the result is cached once written, replay is idempotent rather than re-executing the side effect.

**3. An append-only event journal.** `EventStore` is the durable, replay-safe record of everything that happened on a thread. Every turn writes structured events, and recovery replays without producing duplicate events.

```rust
pub trait EventStore: Send + Sync {
    async fn append(
        &self,
        thread_id: &ThreadId,
        turn: usize,
        envelope: AgentEventEnvelope,
    ) -> Result<()>;
    
    async fn get_events(&self, thread_id: &ThreadId) -> Result<Vec<AgentEventEnvelope>>;
}
```

The default `InMemoryEventStore` is fine for local agents; the serving host swaps in Postgres- or SQLite-backed implementations of the same traits. The agent loop code doesn't change — durability is a deployment decision, not an API decision.

## Cancellation Without Corrupting History

The hardest correctness property in an agent loop is keeping `tool_use` and `tool_result` blocks balanced when a turn is interrupted. The Anthropic API will reject a follow-up request whose history contains a tool call with no corresponding result. `agent-sdk` solves this at the *SDK boundary* rather than relying on tools to be well-behaved.

Every execution path accepts a `CancellationToken` (from `tokio-util`). The integration test `cancel_mid_tool.rs` documents the exact contract, and it is worth quoting because it captures a non-obvious design decision:

```rust
//! Graceful cancellation mid-tool integration test.
//!
//! 1. A run starts. The LLM responds with `tool_use`. The SDK
//!    persists the assistant message and dispatches the tool.
//! 2. The user cancels mid-tool by signalling the [`CancellationToken`].
//! 3. **The tool itself ignores the cancel token** (mirrors bash/cargo).
//! 4. The SDK boundary cancels the in-flight tool future and synthesises
//!    a successful [`ToolResult`] whose content is `"Cancelled by user"`.
//! 5. That synthesised result flows through the normal `append_tool_results`
//!    path — no new branches, no crash-recovery synthesis.
//! 6. The run returns [`AgentRunState::Cancelled`].
//! 7. The next run on the same `thread_id` sees a clean history. The
//!    Anthropic API never receives an unbalanced tool_use/tool_result pair.
```

The insight is that a tool wrapping a subprocess (bash, cargo) cannot always be cooperatively cancelled — so instead of pretending it can, the SDK races the tool future against the cancel token at its own boundary and *synthesises* a balanced `ToolResult`. The cancellation result is not a special crash-recovery branch; it travels the same `append_tool_results` path as any normal result. The same mechanism backs the optional per-tool timeout (`AgentConfig::tool_timeout_ms`): on timeout, the SDK emits a synthetic error result and history stays balanced. Cancellation mid-turn yields `TurnOutcome::Cancelled` with accumulated token usage and a summary, so the caller still gets an accurate accounting.

## Panic Isolation

A panic in a tool, or anywhere in the loop, must not silently drop the agent's event channel and leave callers hanging. The run loop is wrapped via `run_loop_isolated()`, which uses `AssertUnwindSafe(...).catch_unwind()` so a panic becomes an `AgentRunState::Error` rather than a vanished task. Tool-level panics are caught closer to the boundary, again preserving balanced `tool_use`/`tool_result` history.

## Single-Writer Per Thread and Lease-Based Recovery

Durability is meaningless if two workers race on the same thread. The serving host (`agent-service-host`) enforces a **single blocking root per thread**: tasks live in a durable journal (`AgentTaskStore`) with FIFO promotion, lease acquisition, and heartbeat-based expiry. A worker holds a lease while it advances a thread; if it crashes, its heartbeat lapses, the lease is swept, and the task is requeued — no stuck tasks, no concurrent writers.

This is the part of the system that is hardened most aggressively. Phase 11's **Deterministic Simulation Testing (DST)** loop drives the journal with a seeded PRNG, injecting four fault classes and checking invariants after every single step:

```rust
//! Seeded Deterministic Simulation Testing (DST) loop for the journal
//! task store (Phase 11 · D).
//!
//! A single-process scheduler that interleaves `M` simulated workers,
//! injecting four fault classes:
//! - **lease expiry** (worker's heartbeat lapsed; sweep requeues or fails)
//! - **duplicate wakeups** (broker re-delivered advisory)
//! - **cancellation** (caller cancelled thread mid-flight)
//! - **crash** (worker vanished without releasing lease)
//!
//! All nondeterminism drawn from seeded PRNG → fully reproducible.
//! Asserts 5 global invariants after **every** step.
```

Because the simulation is seeded, any failure reproduces deterministically by replaying the seed — the property that makes this style of testing actually useful for chasing concurrency bugs. The concurrency primitives are additionally model-checked with `loom` (in the `agent-server-loom` crate) under `RUSTFLAGS="--cfg loom"` for exhaustive interleaving permutations, with no cost to normal builds.

## Replay Without Gaps

Event replay is what makes crash recovery and live observation possible, and it has a subtle correctness requirement: a consumer reconnecting and draining historical events must not miss events that a producer commits *during* the drain, nor receive them out of order. The `stream_events` API opens a stream from a sequence offset, drains the replay buffer, then transitions to the live tail at the watermark. `CommittedEvent`s carry a monotonic sequence, and commits are atomic (`commit_event` / `commit_event_batch`). The contract is locked in by test:

```rust
#[tokio::test]
async fn producer_during_replay_drain_yields_contiguous_sequence() -> Result<()> {
    // Pre-commit 10 events.
    for i in 0..10i64 {
        repo.commit_event(&thread_a(), AgentEvent::text(format!("m{i}"), "x"), t_plus(i)).await?;
    }

    let mut stream = stream_events(&thread_a(), None, &repo, &InMemoryRetentionStore::new(), &notifier).await?;

    // Drain the first 5 from replay.
    let first_5 = drain_sequences(&mut stream, 5).await;
    assert_eq!(first_5, vec![0, 1, 2, 3, 4]);

    // While draining, the producer commits 5 more.
    for i in 10..15i64 {
        commit_and_notify(&repo, &notifier, &thread_a(), &format!("m{i}"), i).await?;
    }

    // Drain the remaining replay (5..9) and then live (10..14).
    let rest = drain_sequences(&mut stream, 10).await;
    assert_eq!(rest, vec![5, 6, 7, 8, 9, 10, 11, 12, 13, 14]);
    Ok(())
}
```

## The Turn-Level Durable Contract

The bridge between the local loop and the durable server is `run_turn()`, which advances exactly one turn and returns a `TurnOutcome`. Two knobs make it server-grade:

- **`ToolRuntime`** decides who runs tools. `Inline` (the default) means the SDK executes them and feeds results back — ideal for local agents. `External` means the SDK yields `TurnOutcome::PendingToolCalls` with a continuation, and the caller dispatches tools and resumes via `AgentInput::SubmitToolResults`. This externalization is what lets a server schedule tool execution as separate durable tasks.
- **`strict_durability`** (in `TurnOptions`) instructs the SDK to checkpoint at every boundary: before the LLM call, after the LLM response, and after tool execution.

Every `TurnOutcome` variant (except the bare `Error`) carries an authoritative `TurnSummary` — `thread_id`, `turn`, `total_turns`, per-turn and cumulative `Usage`, `provenance` (provider + model), `response_id`, `stop_reason`, `tool_call_count`, `duration_ms`, and the `tool_runtime` / `strict_durability` flags used (so a replay is reproducible). Resumption rides on a **versioned** `ContinuationEnvelope` (`CONTINUATION_VERSION = 1`, validated on resume) wrapping an `AgentContinuation` that holds the pending tool calls, the index being awaited, completed results, an `AgentState` snapshot, and LLM metadata for the audit trail. This is the opaque token that lets a server pause mid-confirmation or mid-tool-execution, persist, and pick up cleanly later.

## The Serving Host

`agent-service-host` is the process-level composition that turns the loop into a durable service. It exposes the durable storage surfaces through a `StoreRegistry` — task journal, thread and message projections, turn attempts, checkpoints, an append-only event repository, a write-ahead execution-intent store, a redaction-aware tool-audit store, a transactional outbox for event ordering, and retention cursors. Storage backs onto Postgres (`storage.backend=postgres`) for full durability across restarts, SQLite for local durability, or in-memory for CLI use. Postgres queries are compile-time-checked via `sqlx` with committed offline metadata. Workers pull from the journal, tool execution is externalized to a tool-executor, and continuation envelopes plus external results are persisted *before* the LLM is resumed.

## Honest Boundaries

This is a pre-1.0 preview, and the maturity is uneven by design. What is solid: the local agent loop (streaming across Anthropic, OpenAI, Gemini, Vertex, Cloudflare, and OpenAI Codex), mid-turn cancellation with balanced history, the event journal and gap-free replay, and lease-based crash recovery verified under DST and `loom`. What is still hardening: the gRPC/HTTP transport (the proto contract exists; production binding is incomplete), multi-instance fanout under chaos (runbooks are written, live chaos tests are not), Postgres schema versioning (migrations work but there's no backward-compatible version negotiation yet), and provider error modes under adversarial network conditions (happy-path streaming is well-covered; rate-limit and partial-drop behavior needs production soak). No `unsafe` code anywhere (`#![forbid(unsafe_code)]`), and every published crate enforces `-D warnings`.

If you control your deployment and you're willing to understand journaling, leasing, and crash recovery rather than treating them as a black box, `agent-sdk` gives you a credible, openly-developed foundation for agent loops that survive the messy realities of production. It is not battle-tested yet — but the durability story is real, the invariants are tested, and the design is honest about where the edges are.
