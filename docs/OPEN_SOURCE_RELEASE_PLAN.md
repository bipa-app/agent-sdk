# Agent SDK — Open-Source Release Plan & Execution Log

**Status:** Active · **Last updated:** 2026-05-30 · **Linear project:** [Server SDK](https://linear.app/bipa/project/server-sdk-1f17d8469d74) · **Tracking:** Phase 10 `ENG-8700`, Phase 11 `ENG-8701`, Phase 12 `ENG-8702`

This is the plan of record **and** the running execution log for re-open-sourcing the Agent SDK. It is the document an engineer or coding agent reads to pick up any task cold, and the document they update when work lands.

## How to use this document

- **Before starting a task:** read the Linear card, then this plan's task row (§6) and the workstream it belongs to. Linear is the canonical source for task *state*; this file is the canonical source for task *scope, rationale, and dependencies*.
- **When you open or land a PR:** update the task's row in §7 (Status) with the PR link and state, and add a dated line to §8 (Change log). Keep entries terse and factual.
- **When reality diverges from the plan:** edit the relevant section in the same PR that causes the divergence, and note it in §8. Do not let the plan drift from the code.
- **Editing etiquette:** §8 is append-only (newest last). Everything else is living — correct it in place. Keep prose professional and skimmable; preserve `file:line` evidence and acceptance criteria.

---

## 1. Objective

Re-open-source the Agent SDK as a welcoming public project: a correct, well-tested, and ergonomic Rust agent runtime that external developers can adopt with `cargo add agent-sdk`. The SDK was previously public (Apache-2.0), was taken private to mature it, and is now being re-opened.

The release bar is **harness robustness**: the runtime must behave correctly under every asynchronous scenario a real agent (coding, knowledge, or otherwise) encounters — cancellation at any point in a turn, message start/queueing/ordering, and durable journals that persist and replay correctly. This is prioritised over model-output benchmarks.

## 2. Key decisions

| Decision | Choice | Rationale | Date |
|---|---|---|---|
| Release scope | **All 9 crates public** | Ship the full stack, including the durable server/host. | 2026-05-29 |
| Distribution | **Public GitHub repo + crates.io** | `cargo add agent-sdk` is the target on-ramp. | 2026-05-29 |
| License | **MIT** | Maximally permissive; simplest for adopters. Historical releases were Apache-2.0; dual-licensing was considered and declined. | 2026-05-29 |
| Quality bar | **Harness robustness** (async/durability correctness), not model benchmarks | The runtime's correctness under cancellation/queueing/crash is what makes it safe to hand to others. | 2026-05-29 |

## 3. Architecture snapshot

Cargo workspace, 9 crates, ~160k LOC, cleanly layered:

| Layer | Crates | ~LOC | Role |
|---|---|---|---|
| **SDK (in-process)** | `agent-sdk-core`, `agent-sdk-tools`, `agent-sdk-providers`, `agent-sdk` (façade), `agent-sdk-otel`, `agent-sdk-cli` | 57k | The library most users consume |
| **Durable server/host** | `agent-server`, `agent-service-host`, `agent-service-proto` | 93k | gRPC + journal + Postgres/SQLite + AMQP orchestration |

Dependency layering: `agent-sdk-core` → `agent-sdk-tools` / `agent-sdk-providers` → `agent-sdk` → `agent-server` → `agent-service-host`.

**Load-bearing fact:** the heavy-dependency boundary is already clean — a plain `cargo add agent-sdk` (no features) pulls **none** of `sqlx`, `tonic`, `lapin`, `prost`, or `opentelemetry`. Simplification is therefore a DX/packaging effort, not a re-architecture.

## 4. Project mapping

Implemented entirely in this repository.

- **agent-sdk** — `git@github.com:bipa-app/agent-sdk.git`
  - local root: `~/work/trees/agent-sdk` (worktree-per-task; main worktree at `~/work/trees/agent-sdk/main`)
  - strategy: `worktree`, base branch `main`
  - Rust 2024. Gates: `cargo check --all-targets`, `cargo fmt`, `cargo clippy -- -D warnings`, `cargo test` (or `cargo nextest run`). Builds are `SQLX_OFFLINE`; after any `sqlx::query!` change run `scripts/postgres18-dev.sh prepare` + `scripts/sqlite-dev.sh prepare` and commit `.sqlx/`.

## 5. Workstreams

- **A — Harness robustness.** Close async/durability correctness gaps (A1) and build the test suite that proves them (A2). This is the launch gate.
- **B — Simplification & DX.** API ergonomics, dependency feature-gating, façade curation, a CLI that runs an agent.
- **C — Release mechanics.** MIT relicense, Bipa-internal scrub + secret scan, crates.io packaging, CI/release automation, publish.

Workstream A maps to Phases 10–11; Workstreams B + C map to Phase 12.

---

## 6. Phases & tasks

Dependency rule across phases: **Phase 10 → 11 → 12.** Cross-phase and intra-phase blockers are noted per task and enforced as Linear `Blocked by` links.

### Phase 10 — Harness Correctness (Workstream A1) · `ENG-8700`

Behavioural fixes. Each task ships with its own regression test. `file:line` references are entry points, not exhaustive.

| Task | Scope | Evidence | Sev |
|---|---|---|---|
| **A** `ENG-8703` | Honor cancellation during the LLM call (streaming + non-streaming) and during compaction (guard the destructive `replace_history`); add a first-class `AgentEvent::Cancelled`. | `agent_loop/llm.rs:345-464`, `run_loop.rs:1383-1401`, `compactor.rs:433-434`, `agent-sdk-core/src/events.rs` | High |
| **B** `ENG-8704` | Extend the SDK-boundary cancel race to async tools, the `stream_async_tool_progress` poll loop, and listen tools; add a configurable per-tool timeout; document + enforce the subprocess kill-on-drop contract. | `tool_execution.rs:957-994,1201,1312` | High |
| **C** `ENG-8705` | Isolate panics in tool/LLM/compaction futures (`catch_unwind`) → structured error result with balanced history; never unwind the run task or drop `state_tx`. | no `catch_unwind` today | High |
| **D** `ENG-8706` | Make the production turn-commit atomic across **events + state** (route through `commit_events_with_outbox`); implement durable cross-process wakeup (`emit_in_transaction`, Postgres + SQLite). | `commit.rs:167-174,245-252`, `root_turn.rs:675-703`, `relay.rs:263-327`, `wakeup.rs:86-95` | High |
| **E** `ENG-8707` | Persist `request_id` idempotency atomically with admission (survive restart); per-thread queued-root depth cap; `SubmitThreadWork` input-size limit; reconcile `max_attempts` so a mid-turn crash requeues instead of failing closed. | `grpc.rs:81-85,982-1006,1030,1040-1045`, `store.rs:1927-2051`, `definition.rs:99` | High |

**Definition of done (Phase 10):** a cancel issued during streaming, compaction, or an async/listen tool stops work promptly and leaves balanced `tool_use`/`tool_result` history; a panicking tool ends the run as a structured error; a crash between state-commit and event-commit cannot leave a committed turn with zero events; a `request_id` retried across a restart produces exactly one effect.

### Phase 11 — Robustness Test Suite & Provider Determinism (Workstream A2) · `ENG-8701`

Gated on Phase 10 (tests the corrected behaviour). `11·A` and `11·B` can begin in parallel with late Phase 10 work.

| Task | Scope | Blocked by |
|---|---|---|
| **A** `ENG-8708` | Test substrate: `cargo-nextest` (+ test groups), **Postgres-in-CI** running the `#[ignore]`-gated store/host tests, `proptest`/`proptest-state-machine`/`insta`/`tokio start_paused`, and a zero-cost `failpoints` (`fail-rs`) feature. | — |
| **B** `ENG-8709` | Port Anthropic's `run_session_store_conformance` to a reusable Rust trait battery; run across in-memory + SQLite + Postgres; include checkpointer cases (incl. LangGraph #6821 latest-by-insertion-order). | 11·A |
| **C** `ENG-8710` | Cancellation + message-lifecycle edge-case matrix (§6.1), incl. the composite "new input mid-stream + cancel mid-tool" and the hard-abort/crash-recovery synth path. | 11·A, 10·A/B/D |
| **D** `ENG-8711` | Durability suite: real on-disk SQLite reopen-and-resume, `fail-rs` crash-between-steps, Temporal-style replay-determinism, a seeded DST loop, scoped `loom` on the lock-free hot spots. | 11·A, 10·D/E |
| **E** `ENG-8712` | Provider determinism: shared streaming-capable scripted provider harness; re-enable the 17 `#[ignore]`'d streaming tests; `wiremock` + `rvcr` Tier-B tests; gated live-smoke Tier-C; CI tiering. | 11·A |

#### 6.1 Harness edge-case test matrix (spec for 11·C/D/E)

Borrowed from the Claude Agent SDK, OpenAI Agents SDK, LangGraph, Temporal, Mastra, Pydantic AI, Vercel AI SDK, and DBOS/Inngest/Restate.

**Cancellation mid-turn** — cancel during streaming; cancel before first token; cancel during compaction; cancel during the parallel observe `join_all`; double-cancel; cancel-vs-result race determinism; interrupt-then-drain ordering (Claude SDK); aborted stream yields well-formed (not half-parsed) tool calls (Pydantic AI); `AbortSignal` forwarded into tools (Mastra); caller-cancel must not outrun teardown (Pydantic AI #5132); persistence not gated on clean finish (Mastra #13984); hard-abort → `synthesize_error_tool_results` produces a provider-acceptable next turn; panic-in-tool.

**Message start / queueing / ordering** — concurrent send+receive on one thread; concurrent same-thread submit serialises to exactly one `Pending`; composite "new input mid-stream + cancel mid-tool" (in-flight turn settles, FIFO preserved, queued message runs next with balanced history, event streams isolated); `receive_response` stops at the terminal result; distinct-thread isolation; queue-depth/oversize back-pressure.

**Durable journal / replay / resume** — real on-disk SQLite reopen-and-resume of a mid-flight turn/tool/subagent tree; `fail-rs` crash between durable steps → no lost/duplicate work; idempotent resume re-run; replay-determinism check (divergence fails CI); exactly-once-effect over at-least-once delivery; completed steps replay cached output; same `request_id` across restart → one effect.

#### 6.2 Testing toolchain (priority order)

| Pri | Tool | Covers |
|---|---|---|
| P1 | `cargo-nextest` + Postgres-in-CI | process isolation (needed for `fail-rs`), retries surface flakiness, and the prod backend's concurrency/txn tests that never run today |
| P2 | `proptest` + `proptest-state-machine`; `tokio start_paused`; `insta` | model-based journal/outbox invariants; deterministic virtual time; ordering snapshots |
| P3 | `fail-rs`; lightweight seeded DST loop | crash-between-steps; seeded interleavings with replayable seed |
| P4 | `loom` (scoped) | exhaustive interleavings of the wakeup fold + the two CAS sites only |
| P5–P6 | `toxiproxy`; Jepsen (`elle`/`porcupine`) | wire-fault and linearizability proofs — deferred to a formal durability audit |

Provider tests are tiered: **A** unit (SSE decode, tool_use JSON round-trip), **B** deterministic integration (`wiremock` scripted SSE + `rvcr` cassettes), **C** env-gated live-smoke (one round-trip per real provider, nightly/secrets only). Model benchmarks (tau-bench/SWE-bench/GAIA/terminal-bench) are **not** CI gates; an optional non-blocking BFCL-style tool-call smoke vs a mock is the only one with SDK relevance.

### Phase 12 — Simplification, DX & Public Release (Workstreams B + C) · `ENG-8702`

Gated on Phases 10 + 11 green. `12·G` (publish + go public) is the final, irreversible gate.

| Task | Scope | Type | Blocked by |
|---|---|---|---|
| **A** `ENG-8713` | API ergonomics + façade curation: `agent.ask()`/`send()`, `run()` returns `impl Future`, `prelude`, provider `from_env()`, default `Tool::Name`; move server-only types behind `agent_sdk::advanced`. | AFK | 10+11 |
| **B** `ENG-8714` | Dependency hygiene: feature-gate providers/tools (`anthropic` default, `openai`, `openai-codex`, `gemini`/`vertex`, `web`, `mcp`, `skills`); replace deprecated `serde_yaml`. | AFK | 10+11 |
| **C** `ENG-8715` | CLI that runs an agent (`agent-sdk chat`/`run`) + runnable quickstart examples; README quickstart compiles without a key. | AFK | 10+11 |
| **D** `ENG-8716` | MIT relicense + governance rewrite (README/CONTRIBUTING/SECURITY) + Bipa scrub + **full git-history secret scan**. | **HITL** | 10+11 |
| **E** `ENG-8717` | crates.io packaging: flip `publish`, path+version deps, metadata, docs.rs config, commit `Cargo.lock`, resolve `agent-service-proto` packaging. | AFK | 12·B, 12·D |
| **F** `ENG-8718` | CI/release automation: `release-plz`, `cargo-deny`/`cargo-audit`, `cargo-semver-checks`, MSRV (`rust-version = 1.85`) + pinned-toolchain job, cross-platform matrix, fix `claude.yml`. | AFK | 12·E |
| **G** `ENG-8719` | Publish to crates.io in dependency order + flip repo public + announce. | **HITL** | 12·D, 12·E, 12·F |

#### 6.3 Release blockers (spec for 12·D/E/F)

| Blocker | Location | Fix |
|---|---|---|
| Proprietary LICENSE; `publish = false` on all crates; out-of-crate `license-file`; no SPDX | `LICENSE`; `crates/*/Cargo.toml` | MIT `LICENSE`; `license = "MIT"` in `[workspace.package]`; drop `license-file`/`publish=false` |
| Path-only intra-workspace deps; `agent-service-proto` excluded from workspace with sibling-dir `build.rs` | `crates/agent-service-host/Cargo.toml:40`; `crates/agent-service-proto/build.rs` | path + version deps; bring proto in-crate or keep proto+host `publish=false` |
| Missing metadata; no docs.rs config; `Cargo.lock` gitignored | root + per-crate `Cargo.toml`; `.gitignore` | fill metadata; `[package.metadata.docs.rs]` + `doc_cfg`; commit `Cargo.lock` |
| `claude.yml` uses an OAuth secret + write perms on any comment trigger | `.github/workflows/claude.yml` | gate on `OWNER`/`MEMBER`, reduce perms, or remove |
| CI ubuntu-only, no MSRV/release/audit/deny/semver | `.github/workflows/ci.yml` | add the gates in 12·F |

#### 6.4 Bipa-internal scrub (spec for 12·D)

| Item | Location | Action |
|---|---|---|
| **Full git-history secret scan** (do before going public) | entire history | `gitleaks`/`trufflehog`; rotate + rewrite history on any real finding |
| Hardcoded GCP project `bipa-278720` | `crates/agent-sdk-providers/src/impls/vertex.rs:929,937` | replace with `my-project` |
| `ENG-####` refs; `linear.app/bipa` ADR links; `bipa/master/src/...` comments | `agent-server/src/journal*`, `agent-sdk-core/src/types.rs:1285,1304`, `CHANGELOG.md`, `docs/adr/0001,0002`, `CLAUDE.md:173` | sweep + rewrite |
| `bipa.exchange`, `chave_bipa`, CPF fixtures | `observability_conformance.rs:680,691,714`, `observability/payload.rs:511-622`, `privacy/redaction.rs:1085,1090` | replace with `example.com` / generic |
| `ssh://…bipa-app…` git deps; "private repository" language | `README.md`, `LANGFUSE.md:103` | rewrite for crates.io install |

---

## 7. Status at a glance

Linear is canonical for state; this table mirrors it for convenience. Update the row when a task's PR opens/merges.

| Task | Linear | State | PR |
|---|---|---|---|
| Plan & Linear setup | — | Open | [#251](https://github.com/bipa-app/agent-sdk/pull/251) |
| 10·A cancel LLM/stream/compaction | `ENG-8703` | In Review | [#256](https://github.com/bipa-app/agent-sdk/pull/256) |
| 10·B cancel async/listen + timeout | `ENG-8704` | In Review | [#254](https://github.com/bipa-app/agent-sdk/pull/254) |
| 10·C panic isolation | `ENG-8705` | In Review | [#252](https://github.com/bipa-app/agent-sdk/pull/252) |
| 10·D atomic commit + durable wakeup | `ENG-8706` | In Review | [#255](https://github.com/bipa-app/agent-sdk/pull/255) |
| 10·E idempotency + back-pressure + retry | `ENG-8707` | In Progress | — |
| 11·A test substrate + Postgres-in-CI | `ENG-8708` | In Review | [#253](https://github.com/bipa-app/agent-sdk/pull/253) |
| 11·B journal conformance battery | `ENG-8709` | Backlog (blocked) | — |
| 11·C cancel/lifecycle matrix | `ENG-8710` | Backlog (blocked) | — |
| 11·D durability/replay/DST/loom | `ENG-8711` | Backlog (blocked) | — |
| 11·E provider determinism + streaming | `ENG-8712` | Backlog (blocked) | — |
| 12·A ergonomics + façade | `ENG-8713` | Backlog (blocked) | — |
| 12·B feature-gating + deps | `ENG-8714` | Backlog (blocked) | — |
| 12·C CLI + examples | `ENG-8715` | Backlog (blocked) | — |
| 12·D relicense + scrub + secret scan | `ENG-8716` | Backlog (blocked) | — |
| 12·E crates.io packaging | `ENG-8717` | Backlog (blocked) | — |
| 12·F CI/release automation | `ENG-8718` | Backlog (blocked) | — |
| 12·G publish + go public | `ENG-8719` | Backlog (blocked) | — |

## 8. Change log

Append-only; newest last. One line per event: date — what changed.

- 2026-05-29 — Plan drafted from a multi-agent audit of the harness (concurrency, durability, lifecycle, tests, API, release-readiness). Linear milestones Phase 10/11/12 created with tracking cards `ENG-8700/8701/8702` and 17 child cards; dependency links wired. Plan committed (PR #251).
- 2026-05-30 — Launched the unblocked set in parallel worktrees (one per task). Draft PRs opened and cards moved to In Review: 10·A #256, 10·B #254, 10·C #252, 10·D #255, 11·A #253. 10·D and 11·A verified against live Postgres 18. 10·E in progress.

## 9. Definition of Done (launch)

- [ ] Full git history secret-scanned clean; repo can go public
- [ ] MIT `LICENSE`; README/CONTRIBUTING/SECURITY consistent; no Bipa-internal coupling remains
- [ ] All Phase 10 tasks merged, each with a regression test
- [ ] `run_journal_store_conformance` passes on in-memory + SQLite + Postgres
- [ ] Edge-case matrix (§6.1) implemented; the 17 ignored streaming tests re-enabled and green
- [ ] ≥1 real reopen-from-disk resume test and ≥1 `fail-rs` crash-between-steps test green
- [ ] Postgres tests run in CI; nextest + cargo-deny/audit/semver-checks in CI; MSRV declared & tested
- [ ] `cargo add agent-sdk` works with the documented quickstart; `agent-sdk chat` runs an agent
- [ ] Every published crate has metadata + docs.rs config; `cargo publish --dry-run` clean for all
- [ ] Crates published in dependency order; repo public; launch announced

## 10. References

Borrowed-test sources: Claude Agent SDK (`run_session_store_conformance`, `test_streaming_client.py`, session-storage docs) · LangGraph checkpointer conformance + interrupts (incl. #6821) · OpenAI Agents SDK · Temporal (replay/determinism, time-skipping, cancellation scopes) · Restate / DBOS / Inngest (journal-mismatch, idempotency-key, step memoization) · Mastra (#13984, snapshots) · Pydantic AI (#5132) · Vercel AI SDK (abort vs resumable-stream). Tooling: `loom`, `turmoil`/`madsim`, `proptest`, `tokio` test-util, `fail-rs`, `wiremock-rs`, `rvcr`, `cargo-nextest`, `release-plz`, `cargo-deny`.
