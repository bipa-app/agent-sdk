# Open-Source Release Plan — Agent SDK

> Status: **draft for review** · Owner: @luiz · Date: 2026-05-29
>
> Decisions locked for this plan: **all 9 crates go public**, distributed via
> **public GitHub repo + crates.io**, licensed **MIT**, and the launch quality
> bar for end-to-end testing is **harness robustness** — proving the runtime
> handles every asynchronous scenario (cancellation mid-turn, message
> start/queueing/ordering, durable journals that persist and replay correctly)
> for any agent type (coding, knowledge, etc.), not model-output benchmarks.

---

## Project mappings

This initiative is implemented entirely in this repository.

- **agent-sdk** — `git@github.com:bipa-app/agent-sdk.git`
  - local: `~/work/trees/agent-sdk/main`
  - strategy: `worktree`, base `main`
  - Rust 2024 workspace. Gates: `cargo check --all-targets`, `cargo fmt`, `cargo clippy -- -D warnings`, `cargo test`. SQLX offline builds (`.sqlx`); after any `sqlx::query!` change run `scripts/postgres18-dev.sh prepare` + `scripts/sqlite-dev.sh prepare`.

## Linear delivery — Server SDK · Phases 10–12

Tracked in the **Server SDK** Linear project (team ENG). The three workstreams map to three milestones; each tracking card links back to this document.

| Milestone | Workstream | Tracking | Children |
|---|---|---|---|
| Phase 10: Harness Correctness — Async & Durability Hardening | A1 | ENG-8700 | ENG-8703 (A · cancel LLM/stream/compaction + `Cancelled`), ENG-8704 (B · cancel async/listen + timeout + kill-on-drop), ENG-8705 (C · panic isolation), ENG-8706 (D · durable commit + wakeup), ENG-8707 (E · idempotency + back-pressure + retry budget) |
| Phase 11: Robustness Test Suite & Provider Determinism | A2 | ENG-8701 | ENG-8708 (A · test substrate + Postgres-in-CI), ENG-8709 (B · journal conformance battery), ENG-8710 (C · cancel/lifecycle matrix), ENG-8711 (D · durability/replay/DST/loom), ENG-8712 (E · provider determinism + streaming tests) |
| Phase 12: Simplification, DX & Public Open-Source Release | B + C | ENG-8702 | ENG-8713 (A · ergonomics + façade), ENG-8714 (B · feature-gating + deps), ENG-8715 (C · CLI + examples), ENG-8716 (D · MIT relicense + scrub + secret scan), ENG-8717 (E · crates.io packaging), ENG-8718 (F · CI/release automation), ENG-8719 (G · publish + go public) |

**Dependency gating (Linear `Blocked by`):** Phase 10 → Phase 11 → Phase 12 at the tracking level. Cross-phase: 11·C ← 10·A/B/D; 11·D ← 10·D/E; 11·B/C/D/E ← 11·A. Within Phase 12: 12·E ← 12·B+12·D; 12·F ← 12·E; 12·G ← 12·D+12·E+12·F.

---

## 0. Where we are today

`agent-sdk` was public under Apache-2.0, was privatized to improve it, and is
now being re-opened. The workspace is **9 crates, ~160k LOC**, cleanly layered:

| Layer | Crates | LOC | Role |
|---|---|---|---|
| **SDK (in-process)** | `agent-sdk-core`, `agent-sdk-tools`, `agent-sdk-providers`, `agent-sdk` (façade), `agent-sdk-otel`, `agent-sdk-cli` | ~57k | The library most users consume |
| **Durable server/host** | `agent-server`, `agent-service-host`, `agent-service-proto` | ~93k | gRPC + journal + Postgres/SQLite + AMQP orchestration |

**The single best piece of news:** the heavy-dependency boundary is already
clean. A plain `cargo add agent-sdk` (no features) pulls **none** of `sqlx`,
`tonic`, `lapin`, `prost`, or `opentelemetry` — those live only in the
server/host crates. The in-process SDK does not drag in Postgres/AMQP. So
"simplify" is mostly DX and dependency-hygiene, not a re-architecture.

### Direct answers to your three questions

1. **"Is there a dataset or another framework's test suite we can base on and
   pass?"** — Yes, but **not a model-quality benchmark**. The right targets for
   *harness* confidence are:
   - **Anthropic Claude Agent SDK's `run_session_store_conformance`** — a
     shipped, packaged store/journal conformance battery. This is the *single
     most directly borrowable artifact*; port it as a Rust trait test.
   - **LangGraph's checkpointer conformance suite** (one spec, many backends)
     and its interrupt/resume semantics.
   - The **harness edge-case matrix** distilled below from Temporal, OpenAI
     Agents SDK, Mastra, Pydantic AI, and Vercel AI SDK.
   - For tool-calling correctness only, **BFCL** is the one model-benchmark
     with SDK relevance — and only as a *non-blocking* smoke. **Do not** chase
     tau-bench / SWE-bench / GAIA / terminal-bench for an SDK; they conflate
     model behavior with harness plumbing.

2. **"Simplify as much as we can"** — the crate graph is already well-factored;
   the wins are API ergonomics, feature-gating heavy/optional deps, curating
   the façade, swapping a deprecated dependency, and making the CLI able to
   actually run an agent. See **Workstream B**.

3. **"Plan an open release + simple ways to use it"** — relicense to MIT, scrub
   Bipa-internal coupling (and secret-scan the full git history *before* going
   public), fix crates.io packaging, and add release/CI automation. See
   **Workstream C**.

### The verdict in one paragraph

The harness *core* is genuinely strong: a CAS-guarded single-writer-per-thread
journal, FIFO per-thread queue, atomic completed-turn commit, checkpoint-
anchored recovery, cooperative cancellation with the "balanced tool_use/
tool_result history" fix, and deep unit coverage (827 inline tests in
`agent-server` alone). But three things stand between today and a confident
launch: **(A)** real *correctness* gaps in the async/durability machinery (not
just missing tests), **(B)** a test suite that cannot yet *prove* robustness
(no E2E streaming, Postgres never runs in CI, "crashes" are simulated by
dropping in-memory stores), and **(C)** release blockers (proprietary license,
`publish = false` everywhere, Bipa coupling). The plan tackles all three.

---

## Workstream A — Harness robustness (the heart of #1)

Two halves: **A1** fixes correctness gaps in the harness itself; **A2** builds
the test infrastructure and the borrowed edge-case matrix that *proves* it.

### A1 — Correctness gaps to close (do these before claiming "robust")

These are behaviors that are **missing or wrong today**, with file:line
evidence from the code audit. Severity is impact-on-a-real-agent.

| # | Gap | Sev | Fix | Effort | Evidence |
|---|---|---|---|---|---|
| 1 | **Cancel during streaming / before first token is ignored** until the stream completes — tokens keep burning, honored only at next loop-top check | High | Race `process_stream` loop + `call_llm_with_retry` against `token.cancelled()`; thread cancel into `LlmCallParams` | M | `agent_loop/llm.rs:345-464`; `run_loop.rs:1383-1401` |
| 2 | **Cancel during compaction is ignored** — a slow, *destructive* `replace_history` can complete after the user asked to stop | High | Pass token into compaction; race `provider.chat`; check `is_cancelled` before `replace_history` | M | `compactor.rs:433-434` |
| 3 | **Async tools, listen tools, and the async poll loop are not raced against cancel** — the long-running tools where cancel matters most | High | Extend `run_with_cancel` to async/listen paths; synthesize the same balanced "Cancelled by user" result | L | `tool_execution.rs:1201,1312` |
| 4 | **Panic in a tool/LLM future unwinds the whole run**, drops `state_tx` (caller sees `RecvError`), and orphans the `tool_use` — re-creating the unbalanced-history 400 the cancel fix solved | High | `catch_unwind(AssertUnwindSafe)` around tool/LLM futures → commit an error `ToolResult` via the idempotency path | M | no `catch_unwind` anywhere |
| 5 | **Production turn-commit splits events from state** — worker calls `commit_event_batch` (separate tx) not `commit_events_with_outbox`; a crash between state commit and event commit leaves a committed turn **with no persisted events**, and the in-process notify is lost on host death | High | Route the production path through `commit_events_with_outbox` so events+outbox row commit in the same tx as state | L | `commit.rs:167-174,245-252`; `root_turn.rs:675-703` |
| 6 | **Idempotency cache is in-memory only** (`Arc<Mutex<IdempotencyState>>`) — a `SubmitThreadWork`/`ForkThread`/`DecideConfirmation` retried with the same `request_id` *across a restart* is **not deduped** → duplicate turn / double fork / double-applied decision (the classic at-least-once delivery footgun) | High | Persist dedup records in the store keyed by `request_id`, checked atomically inside task admission | L | `grpc.rs:81-85,982-1006,1072-1138` |
| 7 | **No durable cross-process wakeup** — relay/wakeup off by default; durable SQL emitter is a future phase (in-memory only). Multi-process deploys lack durable broker delivery + cross-process wakeup beyond the fallback sweep | High | Implement `emit_in_transaction` for Postgres/SQLite; default-enable (or warn) relay on SQL backends | L | `relay.rs:263-327`; `wakeup.rs:86-95` |
| 8 | **No `Cancelled` (or generic terminal) event in the stream** — streaming consumers get no closing marker and can hang waiting for `Done` | Med | Add `AgentEvent::Cancelled`, emit at every cancellation return site | S | `agent-sdk-core/src/events.rs` |
| 9 | **No per-thread queued-root depth cap and no input-size limit** — a retry storm enqueues unbounded serial work; large base64 attachments are stored inline with no cap | Med | Configurable queue-depth cap (`RESOURCE_EXHAUSTED`); aggregate/per-item input size limit (`invalid_argument`) + explicit tonic `max_decoding_message_size` | S–M | `store.rs:1927-2051`; `grpc.rs` submit handler |
| 10 | **`max_attempts` footgun** — submission template uses `DEFAULT_MAX_ATTEMPTS=1` while policy default is 3; a root admitted with budget 1 that crashes mid-turn is **failed-closed, not retried** | Med | Reconcile to the resolved `definition.policy.max_attempts`; regression test crash→requeue | M | `definition.rs:99`; `grpc.rs:1030,1040-1045` |
| 11 | Per-tool timeout missing; dropped tool future doesn't kill subprocess; detached `run()` handle keeps spending tokens if caller drops the receiver | Low–Med | Configurable per-tool timeout in `run_with_cancel`; document/enforce `kill_on_drop` contract; warn on dropped handle or expose only `run_abortable` | S–M | `tool_execution.rs`; `agent_loop.rs:328` |

> Items **1–7** are the launch-blocking set: they are the difference between
> "cancellation works for one tested case" and "the runtime is correct under
> the asynchronous scenarios users will actually hit."

### A2 — Prove it: test infrastructure + the borrowed edge-case matrix

Today's suite is overwhelmingly **observability conformance** (spans/metrics),
serialized behind a global `TEST_LOCK`, with a basic sequential stub provider
and `sleep(50ms)` timing. The gaps that block confidence:

- **No E2E streaming** — every worker/SDK mock implements only `chat()`, so the
  `chat_stream` + `StreamAccumulator` commit/ordering path runs the default
  single-shot adapter; **17 streaming tests are `#[ignore]`'d** after a refactor.
- **Postgres tests never run in CI** — `cargo test --all-features` with no DB
  service and no `--ignored`; ~40 prod-store tests + host DB tests silently skip.
- **"Crashes" are simulated** by dropping/recreating in-memory stores — never a
  real torn write, never reopen-from-disk, never `sqlite::memory:` → file.
- **No interleaving/property/fault tooling** — no loom, proptest, fail-rs, wiremock.

#### A2.1 — Port the conformance battery (highest-leverage borrow)

Port Anthropic's `run_session_store_conformance` to a reusable Rust trait test
and run it across **every backend** (in-memory, SQLite, Postgres) à la LangGraph:

```rust
// crates/agent-sdk/src/testing.rs  (new public test surface)
pub async fn run_journal_store_conformance<S: JournalStore>(store: S) { /* ... */ }
```

Conformance cases (each maps 1:1 to a borrowed contract):

- append preserves insertion order across multiple appends
- `load()` returns an **independent deep-equal copy** (mutating it doesn't mutate storage)
- `load()` of an unknown key returns `None`, not an error/empty session
- semantic (not byte) equality on roundtrip (Postgres `jsonb` may reorder keys)
- optional methods absent → auto-skip those contracts
- `delete()` of the main key cascades to subkeys; safe no-op for append-only backends
- subkey count vs main-session count separation; `clear()`/empty-store invariants
- **N concurrent `append()` to one key serialize without loss or intra-batch interleave**
- checkpoint roundtrip; `list()` reverse-chronological; **latest-by-insertion-order, not lexicographic id** (LangGraph #6821 regression); pending-writes persisted; parent-checkpoint link; metadata filter; `before+limit` pagination

#### A2.2 — The harness edge-case matrix (grouped by your three concerns)

Borrowed from Claude Agent SDK, OpenAI Agents SDK, LangGraph, Temporal, Mastra,
Pydantic AI, Vercel AI SDK, DBOS/Inngest/Restate. Each row is a test to write.

**Cancellation mid-turn**

| Scenario | Assert | Borrowed from |
|---|---|---|
| Cancel during streaming | further model events stop promptly; a cancellation surfaces, not a normal final output; balanced history | OpenAI Agents SDK |
| Cancel before first token | run ends `Cancelled`, no orphan `tool_use`, no half-written assistant msg | (gap #1) |
| Cancel during compaction | `replace_history` does not complete after cancel | (gap #2) |
| Cancel during parallel observe `join_all` | each child races cancel; batch settles deterministically | (gap, concurrency) |
| Double-cancel & cancel-vs-result race | double-cancel is a no-op; a just-finished real result is **not** turned into "Cancelled by user" nondeterministically | (gap, `select!` bias) |
| **Interrupt-then-drain ordering** | after interrupt, the in-flight turn still emits a terminal result (`error_during_execution`); a new query's messages only appear after the prior turn is drained | Claude Agent SDK |
| Aborted model stream | yields a well-formed (or clearly-incomplete) tool call — never truncated JSON args presented as complete | Pydantic AI |
| AbortSignal forwarded into tools | aborting the run cancels in-flight tool executions, not just top-level generation | Mastra |
| Caller-cancel must not outrun teardown | awaiting the cancel does not return until the background task + channels are torn down (no leaked task) | Pydantic AI #5132 |
| Persistence not gated on clean finish | aborting mid-stream still persists the user message + completed steps (not behind `if(!aborted)`) | Mastra #13984 |
| **Hard-abort / crash-recovery synth path** | `JoinHandle::abort` leaves an orphan `tool_use`; next turn's `synthesize_error_tool_results` produces a balanced request the provider accepts | (currently untested; cancel_mid_tool.rs scopes it out) |
| Panic in a tool | run ends with a structured error, history stays balanced, no `RecvError` | (gap #4) |

**Message start / queuing / ordering**

| Scenario | Assert | Borrowed from |
|---|---|---|
| Concurrent send + receive on one thread | well-formed first message, no interleave corruption, no deadlock | Claude Agent SDK |
| Concurrent submits, same thread | exactly one `Pending`; rest `Queued` FIFO; cross-process single-writer holds (Postgres `FOR UPDATE`) | repo + LangGraph/Temporal |
| **Composite: new input while a turn is streaming + cancel mid-tool** | in-flight turn settles, FIFO preserved, queued message runs next with balanced history, event streams isolated | (highest-risk real-world race; untested) |
| `receive_response` stops at terminal result | iterator ends at `ResultMessage`, doesn't consume into next turn | Claude Agent SDK |
| String prompt → single user message; per-message session tagging | exact encoding/tagging | Claude Agent SDK |
| Concurrent runs on distinct threads | no cross-read/overwrite of journal/checkpoint; each final state == its own event sequence | LangGraph / Temporal |
| Queued-root depth cap + oversized input | back-pressure error, not unbounded growth / transport failure | (gap #9) |

**Durable journal / replay / resume**

| Scenario | Assert | Borrowed from |
|---|---|---|
| **Real on-disk SQLite reopen-and-resume** | write a mid-flight turn/tool/subagent tree → drop store+pool → reopen from the same file → lease-sweep + `recover_thread` resumes to a deterministic terminal state, no orphan, no dup | (core durability promise; only ever tested in-memory) |
| **Fault-injection crash between durable steps** | `fail-rs` panic after broker-ack-before-mark-delivered, and after journal-commit-before-outbox-insert → recovery reclaims/republishes, no lost/dup, no torn projection | DBOS/Temporal + repo doc-comments |
| Resume re-runs idempotently | code before an interrupt re-executes on resume; an upsert is fine, a raw insert duplicates — assert both to prove the re-run semantics | LangGraph interrupts |
| **Temporal-style replay-determinism check** | capture committed-event histories from conformance runs; re-run through current commit/recovery code → byte/semantic-identical durable state; divergence fails CI | Temporal / Restate |
| Exactly-once-effect over at-least-once delivery | duplicate/reordered wakeups → at-most-one execution effect (the `Pending→Running` CAS is the dedup point) | outbox pattern |
| Completed steps replay cached output | after a crash, recovery re-runs but each checkpointed step returns cached output (body invoked 0 extra times) | DBOS / Inngest |
| Cross-restart idempotency | same `request_id` before and after a simulated host restart → exactly one `RootTurn` task | (gap #6) |

#### A2.3 — Tooling, prioritized for low-risk incremental adoption

| Priority | Tool | Covers | Notes |
|---|---|---|---|
| **P1** | **cargo-nextest** | process isolation (needed for `fail-rs`), speed, `--retries` to surface flakiness, test-groups to serialize DB tests | zero code change; add `.config/nextest.toml` |
| **P1** | **Postgres service in CI** | the prod backend's `FOR UPDATE`/`SKIP LOCKED`/txn-atomicity tests that never run today | add `services: postgres:18` + a job running the `#[ignore]`-gated tests |
| **P2** | **proptest + proptest-state-machine** | model-based fuzz of journal lifecycle + outbox idempotency vs a reference model; auto-shrinks | reuse the trait objects `conformance.rs` already exposes |
| **P2** | **tokio `start_paused`** | deterministic virtual time for lease-expiry/heartbeat/fallback-sweep/retry-backoff | already used in 5 host tests — extend everywhere |
| **P2** | **insta** | snapshot the ordered event/message stream + post-resume reconstructed transcript | catches ordering/queueing regressions as diffs |
| **P3** | **fail-rs (failpoints)** | crash between durable steps → idempotent replay | gate behind a `failpoints` cargo feature; run under nextest |
| **P3** | **Lightweight seeded DST** | single-process seeded scheduler over the in-memory store, randomly interleaving workers + injecting expiry/dup-wakeup/cancel/crash; print seed on failure | cheap — no runtime swap; journal authority model is already deterministic |
| **P4** | **loom** | the 2-3 genuinely lock-free hot spots: `WakeupSignal::notify_one` fold, `Pending→Running` CAS, `(worker_id, lease_id)` heartbeat CAS | scope tightly (2-3 threads, `LOOM_MAX_PREEMPTIONS=2..3`); do **not** loom-ify the whole store |
| **P5** | **toxiproxy** (testcontainers) | wire-level partition/latency to Postgres + RabbitMQ to prove outbox reclaim + publisher-confirm under real faults | integration-tier; adopt after P3 is green |
| **P6** | **Jepsen (elle/porcupine)** | linearizability/serializability of the real Postgres path under partitions | reserve for a formal durability audit; the cheap Temporal-style replay check delivers most of the value |

#### A2.4 — Provider determinism (the LLM boundary)

Make provider tests reproducible and key-free, in three tiers:

- **Tier A (unit, always on):** SSE decode of provider events; `tool_use` JSON
  serialization round-trip (snapshot vs exact wire JSON).
- **Tier B (deterministic integration, always on):** **wiremock-rs** mock server
  (custom `Respond` impl for scripted SSE) for happy-path, streamed turns, and
  **429/5xx retry + backoff**; **rvcr** cassettes (recorded, redacted, replayed)
  for large tool/streamed exchanges.
- **Tier C (gated live-smoke, nightly with secrets):** one completion + one
  streamed turn + one tool call per real provider, `#[ignore]` / env-guarded.

**CI tiering:** PR runs Tier A+B (no secrets, no network) + the conformance
battery + SQLite/in-memory durability; a **nightly secrets job** runs Tier C and
(optionally, non-blocking) a BFCL-style tool-call smoke vs a mock provider.

---

## Workstream B — Simplify (the heart of #2)

The crate graph is already clean, so this is DX, dependency hygiene, façade
curation, and the CLI.

### B1 — API ergonomics (the 30-second path)

| Change | Why | Effort |
|---|---|---|
| `agent.ask(thread, text)` / `agent.send(...)` convenience that builds `ToolContext::new(())` + a `CancellationToken` internally and returns assembled text | Collapses the 4-arg + double-await pattern; removes the surprising `RecvError` | M |
| `run()` returns `impl Future`/`async fn` instead of a bare `oneshot::Receiver` | Removes the non-obvious double-step and an error mode unrelated to agent logic | M |
| `pub mod prelude` exporting ~12 newcomer names (`builder`, `AgentConfig`, `AgentInput`, `AgentEvent`, `Tool`, `ToolContext`, `ToolResult`, `ToolTier`, `ToolRegistry`, `DynamicToolName`, `InMemoryEventStore`, `CancellationToken`, `providers::AnthropicProvider`) | Shrinks apparent surface; cleaner docs.rs / autocomplete | M |
| `Provider::from_env()` / `try_from_env()`; accept `impl Into<String>` keys | Kills the most-copied `std::env::var(...).expect(...)` snippet | S |
| Default `Tool::Name = DynamicToolName` (or a `SimpleTool` taking `&str`, or `#[derive(Tool)]`) | Removes the biggest tool-authoring papercut (learning `ToolName`) | M |

### B2 — Dependency hygiene

| Change | Why | Effort |
|---|---|---|
| Gate providers/tools behind cargo features: `anthropic` (default), `openai`, `openai-codex` (pulls `tokio-tungstenite`), `gemini`/`vertex`, `web` (pulls `html2text`), `mcp`, `skills` | Cuts the ~171-crate transitive tree; an Anthropic-only consumer drops WebSocket + HTML deps | L |
| **Replace deprecated `serde_yaml`** (`0.9.34+deprecated`) with `serde_yaml_ng`/`serde_yml`, behind the `skills` feature | Shipping an unmaintained crate as a *mandatory* public dep is a bad look + latent security liability | S |

### B3 — Façade curation

Move server/internal contract types (`TurnSummary`, `TurnOutcome`,
`AuditProvenance`/`ToolAuditRecord*`, `EventAuthority`/`LocalEventAuthority`,
`ContinuationEnvelope`/`CONTINUATION_VERSION`, `ListenExecuteTool`/`Erased*`,
`HostDependencies`, `ExecutionContextFactory`) behind an `advanced`/`server`
submodule so the front page of `agent_sdk::` and docs.rs isn't dominated by
names a normal user never touches. (Effort: M)

### B4 — The CLI can't run an agent

`agent-sdk-cli` (binary `agent-sdk`) today only installs a Langfuse/OTel docker
stack + `doctor`. For an OSS launch, `cargo install agent-sdk-cli && agent-sdk
chat` should *talk to an agent*. **Recommendation:** add a minimal `chat`/`run`
subcommand (Anthropic provider, streams to stdout) — this *is* a "simple way to
use the framework" and the highest-leverage first-impression win. (Effort: L)

### B5 — Crate count (decision)

The audit floated consolidating `core` + `tools` + `providers` into `agent-sdk`
behind features (XL, risky — server/host depend on the sub-crates).
**Recommendation: keep the 9 crates** (they're already cleanly layered) but make
`agent-sdk` the single documented entry point and let `release-plz` handle the
lockstep versioning overhead. Revisit consolidation only if multi-crate release
friction proves painful in practice.

---

## Workstream C — Open release mechanics (the heart of #3)

### C1 — Relicense to MIT

- Replace proprietary `LICENSE` with MIT; set `license = "MIT"` in
  `[workspace.package]`; drop `license-file` and `publish = false`.
- Rewrite `README.md` (drop "private/proprietary", show `cargo add agent-sdk`),
  `CONTRIBUTING.md` ("open to public contributions"), `SECURITY.md` (public
  contact; fix the stale `0.1.x` → `0.8.x`).
- *Provenance note:* the project was historically Apache-2.0. Audit contributor
  provenance and retain any incorporated third-party `NOTICE` files. (You chose
  MIT; the only thing Apache-2.0 added was an explicit patent grant — flagging
  once, not advocating a change.)

### C2 — Scrub Bipa-internal coupling — and secret-scan history FIRST

| Item | Location | Action |
|---|---|---|
| **Run `gitleaks`/`trufflehog` over the FULL git history** | entire repo history | **Critical, do before flipping public** — a previously-private history may contain real keys even if current fixtures are local-only |
| Hardcoded GCP project `bipa-278720` | `agent-sdk-providers/src/impls/vertex.rs:929,937` | replace with `my-project` (already used at 889/905) |
| ~27 `ENG-####` refs; ADRs link `linear.app/bipa`; comments cite private `bipa/master/src/...` paths | `agent-server/src/journal*`, `agent-sdk-core/src/types.rs:1285,1304`, `CHANGELOG.md`, `docs/adr/0001,0002`, `CLAUDE.md:173` | sweep `ENG-[0-9]+` and `bipa/master`; drop private links/lines; rewrite comments |
| `bipa.exchange`, `chave_bipa`, CPF in test fixtures | `observability_conformance.rs:680,691,714`, `observability/payload.rs:511-622`, `privacy/redaction.rs:1085,1090` | replace with `example.com` / generic categories (low urgency) |
| `ssh://git@github.com/bipa-app/...` git deps; "private repository" language | `README.md`, `LANGFUSE.md:103` | rewrite for crates.io install |

### C3 — crates.io packaging

- **Publish in dependency order:** `core` → `tools`/`providers` → `sdk` →
  `otel` → (`server` → `proto` → `host`). `cargo publish --dry-run`/`package`
  each crate in a clean checkout first.
- Convert intra-workspace deps to **path + version** (crates.io rejects
  path-only). Decide `agent-service-proto`: bring it into the workspace with
  protos in-crate (`build.rs` currently reads a sibling dir + `protoc-bin-
  vendored`), **or** keep `proto` + `host` `publish = false` if the gRPC host
  isn't meant for `cargo add`.
- Fill metadata: `description`, `keywords`, `categories`, `repository`,
  `authors`; add `[package.metadata.docs.rs]` (`all-features = true`,
  `rustdoc-args = ["--cfg","docsrs"]`) + `#![cfg_attr(docsrs, feature(doc_cfg))]`
  for feature-gated crates; ship `.sqlx`; **commit `Cargo.lock`** (currently
  gitignored).

### C4 — CI / release automation

| Add | Why |
|---|---|
| Postgres service + a job running the gated DB tests (`scripts/postgres18-dev.sh test-migrations`) | the prod backend never runs in CI today (see A2.3 P1) |
| `cargo-deny` (advisories + license allowlist + banned/dup crates) + `cargo-audit` | MIT redistribution needs a clean, redistributable graph |
| `cargo-semver-checks` | catch SemVer-incompatible API changes pre-publish |
| `rust-version = 1.85` + a pinned-older-toolchain CI job | declare/enforce MSRV |
| Windows + macOS matrix | currently ubuntu-only |
| **`release-plz`** | automated dependency-order publishing, changelog (git-cliff), version bumps, breaking-change detection — the right tool for a 9-crate workspace |
| Fix `claude.yml` | uses an OAuth secret absent on public forks and grants write perms on any comment trigger → gate on `OWNER`/`MEMBER`, reduce permissions, or remove |

---

## Sequencing

Each phase is a set of independently-mergeable PRs. **Gate the public flip on
Phase 0 + Phase 1 + Phase 2 green.**

- **Phase 0 — Pre-flight (private):** full-history secret scan; MIT relicense +
  Bipa scrub PR; CI hardening (nextest, Postgres-in-CI, deny/audit/semver, MSRV,
  commit `Cargo.lock`).
- **Phase 1 — Robustness correctness:** A1 items **1–7** (streaming/compaction/
  async-tool cancel, panic isolation, `Cancelled` event, production outbox
  commit, persisted idempotency, durable wakeup) + items 8–11 as capacity allows.
- **Phase 2 — Prove it:** A2 conformance battery; the edge-case matrix; provider
  wiremock/cassette tiering; fault-injection + seeded DST; re-enable the 17
  ignored streaming tests against a shared streaming-capable provider harness.
- **Phase 3 — Simplify / DX:** features, `prelude`, `ask()`, CLI `chat`,
  `serde_yaml` swap, façade curation, runnable quickstart examples.
- **Phase 4 — Publish:** metadata, `--dry-run` package each crate, docs.rs,
  `release-plz`, flip repo public, `cargo publish` in order, announce.

---

## Definition of Done (launch checklist)

- [ ] Full git history secret-scanned clean; repo can go public
- [ ] MIT `LICENSE` + consistent README/CONTRIBUTING/SECURITY; no Bipa coupling
- [ ] All A1 items 1–7 fixed, each with a regression test
- [ ] `run_journal_store_conformance` passes on in-memory **+ SQLite + Postgres**
- [ ] Harness edge-case matrix implemented; 17 ignored streaming tests re-enabled and green
- [ ] At least one **real reopen-from-disk resume** test + one **fail-rs crash-between-steps** test green
- [ ] Postgres tests run in CI; nextest + cargo-deny/audit/semver-checks in CI; MSRV declared & tested
- [ ] `cargo add agent-sdk` works with the documented quickstart; `agent-sdk chat` runs an agent
- [ ] Every published crate has metadata + docs.rs config; `cargo publish --dry-run` clean for all
- [ ] `release-plz` configured; crates published in dependency order

---

### Appendix — primary sources for the borrowed tests

Claude Agent SDK (`run_session_store_conformance`, `test_streaming_client.py`,
session-storage docs) · LangGraph checkpointer conformance + interrupts (incl.
issue #6821) · OpenAI Agents SDK (runner lifecycle, RunState resume) · Temporal
(replay/determinism, time-skipping, cancellation scopes) · Restate / DBOS /
Inngest (journal-mismatch, idempotency-key, step memoization) · Mastra (#13984
abort-persistence, snapshots) · Pydantic AI (#5132 teardown ordering) · Vercel
AI SDK (abort vs resumable-stream disconnect) · Rust tooling: loom, turmoil/
madsim, proptest, tokio test-util, fail-rs, wiremock-rs, rvcr, cargo-nextest,
release-plz, cargo-deny. (Full URL list in the analysis artifact.)
