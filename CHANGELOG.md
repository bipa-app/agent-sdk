# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `agent_sdk_providers::impls::normalized_strict_schema` and its error,
  `StrictSchemaError` (`#[non_exhaustive]`), behind `openai`/`openai-codex`.
  They return a tool schema as `OpenAI` strict mode receives it, so callers
  can check that their tool inputs accept it.
- **crates.io packaging readiness.** The SDK-surface crates (`agent-sdk-foundation`,
  `agent-sdk-tools`, `agent-sdk-providers`, `agent-sdk`, `agent-sdk-otel`,
  `agent-sdk-cli`) are now publishable: package metadata
  (repository/homepage/authors/keywords/categories), `rust-version`,
  `[package.metadata.docs.rs]` + `docsrs` feature badges, a crate-local
  `agent-sdk` README, and a committed `Cargo.lock`. The internal
  server/host/loom/proto crates stay `publish = false`.
- **Release automation.** `release-plz` config + workflow (release-PR only;
  publish is HITL-gated), `deny.toml` (cargo-deny), and a `release-gates`
  CI workflow running cargo-deny / cargo-audit / cargo-semver-checks, an MSRV
  pinned-toolchain job, and a packaging dry-run. CI gains a Windows + macOS
  cross-platform matrix over the published crates.
- **`ChatRequest::new` + `with_*` builder setters** — an ergonomic alternative
  to the (still-public) struct literal for the common system + messages case.
- **Cookbook** (`crates/agent-sdk/COOKBOOK.md`) and a runnable, offline
  `structured_output` example.
- **Python binding spike** (`bindings/`) — a thin gRPC client over the durable
  `agent-service-host`, with a written feasibility recommendation.
- Typed `AccountRotation` and `AccountPoolExhausted` journal events let hosts
  surface successful OAuth account handoffs and Retry-After-faithful pool
  exhaustion without parsing provider error text.
- **Provider-agnostic embeddings.** `LlmProvider::embed` accepts a bounded typed
  batch and returns validated, input-ordered float vectors. `OpenAIProvider`
  uses `/embeddings`; `GeminiProvider` uses the current `batchEmbedContents`
  request shape. Gateway, fallback, router, credential-refresh, and
  record/replay wrappers propagate the operation, while unsupported providers
  return a typed capability error without fabricating vectors.
- **Confirmation decision guard.** `GrpcTransport::with_confirmation_decision_guard`
  registers a `ConfirmationDecisionGuard` (`agent_service_host::confirmation_guard`)
  that sees each `DecideConfirmation` before anything applies: the parked task,
  the call it awaits (`awaited_call`), the decision and the request's
  `automatic` flag. The caller sets `automatic`, so it stops honest automatic
  clients and is not an authorization boundary. A refusal leaves the task
  parked and fails with `FAILED_PRECONDITION`; it runs before the idempotency
  claim, so a retry is refused the same way. Hosts build a
  `ConfirmationDecisionRequest` (non-exhaustive) with
  `ConfirmationDecisionRequest::new`.
- **`DecideConfirmationRequest.automatic`** (`agent.service.v1`, field 5) marks
  a decision no person made. An unset field encodes nothing, so stored
  idempotency fingerprints are unchanged, but Rust struct literals of
  `DecideConfirmationRequest` must add the field when they bump.
- **Gemini 4 Argon and the current Gemini lineup.** The catalog adds
  `gemini-4-argon` (1,048,576-token window shared by input, thinking and
  output; 65,536 synchronous output; introductory $2 / $10 / $0.10 per 1M
  tokens through 2026-11-30), `gemini-3.8-flash`, `gemini-3.6-flash`,
  `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, and `gemini-3.7-flash`
  (deprecated; Google serves it as 3.8 Flash). Gemini and Vertex refuse a
  `cached_content` reference on Argon, which has no explicit caching, and
  wait up to 15 minutes for its non-streaming responses. New constants
  `MODEL_GEMINI_4_ARGON`, `MODEL_GEMINI_38_FLASH`, `MODEL_GEMINI_36_FLASH`,
  `MODEL_GEMINI_35_FLASH_LITE`.

### Changed
- **GPT-5.6 prices.** `gpt-5.6-sol` (and the `gpt-5.6` alias) is now
  $4 / $0.40 cached / $20 per 1M tokens, OpenAI's promotional rate through at
  least 2026-11-21; `gpt-5.6-terra` $2 / $0.20 / $12; `gpt-5.6-luna`
  $0.20 / $0.02 / $1.20.
- **Gemini catalog.** Rows for the shut-down `gemini-2.0-flash`,
  `gemini-2.0-flash-lite`, `gemini-3.1-flash-lite-preview` and `gemini-3.0-pro`
  are removed; those ids now resolve as unknown models.
  `MODEL_GEMINI_31_FLASH_LITE` names the GA `gemini-3.1-flash-lite`,
  `GeminiProvider::flash_lite` builds `gemini-3.5-flash-lite`, and
  `MODEL_GEMINI_2_FLASH`, `MODEL_GEMINI_2_FLASH_LITE` and `MODEL_GEMINI_3_PRO`
  are deprecated. A Gemini request with provider-default thinking sends the
  serving model's documented default level (`MEDIUM` on Argon, 3.8 Flash and
  3.6 Flash; `MINIMAL` on 3.5 Flash-Lite) instead of always `HIGH`.
- **Breaking: a batch's mutating calls run alone and in model order.** A
  model that emits `[edit, edit, bash "cargo test"]` in one response expects
  the test to see both edits. The durable worker spawned every call of a batch
  as a runnable child, so the command could start, and finish, before either
  edit; the in-process loop did the same for any call whose tool was
  `ToolTier::Observe`, the tier hosts give mutating tools to skip prompts.
  `ToolConcurrency` is a new tool property, independent of the tier:
  `Exclusive` by default, `ReadOnly` for `read`, `grep`, `glob`, `todo_read`,
  `web_search`, and `link_fetch`. `agent_sdk_foundation::batch_order` holds
  the rule both runtimes follow: consecutive read-only calls run together,
  every other call runs alone, and a call starts only after every earlier
  call it could affect or observe has finished. The durable worker records
  each later child's waits in `after_siblings`, and acquisition leases it
  only once every sibling it names is terminal; no write releases anything.
  The FIFO scan reads only rows that wait on nothing, so held rows never
  slow it down. When a tool child settles, the host asks the new
  `AgentTaskStore::ready_held_children` for the siblings it released and
  hands them to the pool with `WakeupSignal::hint_tasks`, waking one
  worker per row instead of the whole pool; a bounded probe over held rows
  is the backstop.
  New calls of a resumed turn also wait for the still-live calls it carries
  over. Subagent spawns, and calls settled before spawn, neither wait nor
  hold anything back, so a coordinator's spawns still fan out together.
  The tool traits gain `concurrency()`, which defaults to `Exclusive`
  (including on the `Erased*Tool` traits), and the derives accept
  `concurrency = ...`. Breaking: `llm::Tool`, `PendingToolCallInfo`,
  `ChildSpawnSpec`, and `AgentTask` gain a public field, and
  `TaskSchemaError` gains `InvalidSiblingWait` and `HeldBySiblings`, and
  `PostgresDurableStore::from_pool` is no longer `const`. A call
  after a durable child that awaits confirmation now waits for that
  decision, so confirmations in one batch are asked one at a time.
  Migration 0026 adds the nullable `after_siblings_json` column and three
  partial indexes over pending rows (`agent_sdk_tasks_ready_fifo_idx`,
  `agent_sdk_tasks_held_fifo_idx`, `agent_sdk_tasks_held_by_parent_idx`)
  on both Postgres and SQLite, and drops `agent_sdk_tasks_runnable_fifo_idx`,
  which the ready index replaces.

  **Rollback and rolling deploys:** an older build refuses to start
  against a database that has migration 0026 applied
  (`VersionMissing(26)`): the durable migrator does not ignore unknown
  applied versions, a policy that predates this change. To roll back,
  revert 0026 by hand first, in one transaction: drop the three indexes,
  recreate `agent_sdk_tasks_runnable_fifo_idx` (`ON agent_sdk_tasks
  (created_at, id) WHERE status = 'pending'`), `ALTER TABLE agent_sdk_tasks
  DROP COLUMN after_siblings_json`, and `DELETE FROM _sqlx_migrations WHERE
  version = 26`. Batches parked at
  that point lose their order. This build loads rows an older build wrote during a mixed
  deploy: it ignores a wait list on any row that is not pending, and it
  leases a pending row left with a wait list once every sibling it names
  has finished.
- **Breaking:** `ToolExecutionStore::get_execution` now takes the thread id
  (`get_execution(&thread_id, tool_call_id)`), and `InMemoryExecutionStore`
  and `SqliteStore` key executions by thread and tool-call id. Providers only
  keep tool-call ids unique within one conversation, so a thread that reused an
  id could read another thread's cached result. Custom stores must key
  executions by thread as well.
- `SqliteStore` schema version 3: opening an older file rebuilds its
  tool-execution table with a `(thread_id, tool_call_id)` key, taking each
  row's thread from its payload, in one transaction that also stamps the
  version. A row with invalid JSON or no string `thread_id` fails the open and
  is named by its `tool_call_id`; the file is left unchanged.
  The upgrade is one-way: an older SDK refuses to open a v3 file, and an
  older process that still has the file open fails every tool-execution
  insert, so stop older processes first and do not roll back across it.
- OpenAI strict mode: a tool schema strict mode cannot express is now sent
  without `strict` on Chat Completions, Responses and Codex instead of being
  rejected with a 400. That covers untyped or annotation-only subschemas
  (`true`, `{}`, `{"description": ...}`) and free-form objects, including
  no-argument tools declared as `{"type": "object", "properties": {}}`. A
  strict response format with such a schema fails locally with the reason.
- **Breaking:** `Usage` gains `request_input_tokens` (the input size of one
  provider request, `None` when unknown), and `CompactionConfig` gains
  `model_aware_threshold`, `model_context_reserve_tokens` and
  `model_context_window_tokens`. Struct literals must set them; prefer
  `..Default::default()` or the `with_*` builders.
- `RecordReplayProvider` never reports a stable request route, in record or
  replay mode, so a recording and its replay make the same compaction
  decisions. `RefreshingProvider::route()` now reports the wrapped
  provider's route.
- **Breaking:** `StreamDelta::Done` gains `served_model`, the
  `LlmProvider::model` of the provider that served the stream, stamped by
  every built-in provider next to `served_route` and forwarded untouched by
  fallback, router, refresh, and record/replay wrappers. Code that builds a
  `Done` sets it (`None` when unknown); patterns without `..` name it. One
  route can serve several models, so the durable worker anchors the measured
  compaction trigger of a provider without a stable request route only on a
  request whose served route and model both match the provider's own.
  Cassettes recorded without the field replay it as `None`.
- **Behaviour change for clients:** a turn's `UserInput` now carries the
  root-turn task id in `emitter_task_id` (`AgentEvent::UserInput`, proto
  `UserInputEvent.emitter_task_id`). Before, only input injected into a
  running turn carried it, so a client that reads its presence as "injected"
  now misclassifies every prompt. Compare it with the `emitter_task_id` of the
  turn's `Start` instead: injected input names a different task. Rows
  journaled before the change stay unstamped.
- **Breaking:** `CompactionResult` gains `published_artifacts`,
  `response_model`, and `served_route`, and is now `#[must_use]`;
  `FailedCompaction` gains `response_model` and `served_route`. Custom
  compactors that build them with struct literals set
  `published_artifacts: PublishedArtifacts::default()` and the identity
  fields to `None` when they make no summarization call. Results keep owning
  their Snapcompact artifacts: `LlmContextCompactor::with_provisional_artifacts`
  opts in to deleting them when the result is dropped before
  `PublishedArtifacts::commit`, which callers run right after their durable
  compaction append succeeds. The agent loop and the durable worker opt in, so
  a compaction that is cancelled, refused, or fails to persist no longer
  leaves orphaned artifacts.
- **Codex streams over HTTP SSE by default.** `OpenAICodexResponsesProvider`
  sends each turn as one HTTP SSE request; the persistent WebSocket transport
  is opt-in via `with_transport(CodexTransport::WebSocket)`. Both transports
  derive prompt-cache affinity (`prompt_cache_key`, `session_id`) from
  `ChatRequest::session_id` and replay the `x-codex-turn-state` sticky-routing
  token on a turn's tool-result follow-ups, never into the next turn. A retry
  of a turn's opening request, or a plain user message appended mid-turn,
  starts without it.
- `OpenAIResponsesProvider` sends `prompt_cache_key` only to the official
  `api.openai.com` base URL, matching the Chat Completions provider. A
  pass-through proxy to `OpenAI` (Azure `OpenAI`, Cloudflare AI Gateway's
  `OpenAI` route) keeps it with `with_prompt_cache_key_forwarding(true)` on
  `OpenAIProvider`, `OpenAIResponsesProvider` or `CloudflareAIGatewayProvider`;
  on `OpenAIProvider` it covers Chat Completions and the Responses reroute.
- The durable worker sends the thread id as `ChatRequest::session_id` on
  every root-turn request, so Codex, Responses and Chat Completions requests
  carry a stable per-thread `prompt_cache_key`. The id is part of the measured
  compaction trigger's request fingerprint: attempts recorded before the
  upgrade never anchor it, so each existing thread's first check afterwards
  uses the full-history estimate.
- Compaction now appends range-addressed lineage entries and rebuilds the
  effective LLM view without deleting committed transcript messages.
- Auto-compaction prunes artifact-backed and superseded read outputs before
  summarization, and threshold decisions use journaled billed usage when present.
- MCP output, shell output, and durable subagent finals now share the lossless
  inline-budget artifact spill path. Hosts can run liveness-aware deterministic
  expiry and per-thread quota sweeps without deleting referenced artifacts.


- MSRV pinned to **1.91** (the verified buildable floor for the published
  crates; the merged MCP transport uses `const Duration::from_mins`).

- **BREAKING — Default-deny payload capture.**
  `ObservabilityStore::capture(...) -> CaptureDecision::Inline` no longer
  unconditionally lands payloads on the LLM span. The SDK now forces every
  `Inline` decision down to `Omit` unless **both** of these are true:

  1. The operator opted in via `OtelConfig::capture_payloads(true)` (which
     `agent-sdk-otel::install_global_provider` propagates onto a process-wide
     gate at startup), and
  2. The store overrode the new
     `ObservabilityStore::acknowledge_pii_redaction()` method to return `true`.

  The trait gains the `acknowledge_pii_redaction` method with a default impl
  returning `false`, so existing implementations keep compiling. Behaviour
  changes silently: any consumer that was relying on `Inline` to land
  `gen_ai.input.messages` / `gen_ai.output.messages` on spans will see the
  attributes disappear after upgrading until the store is updated.
  `CaptureDecision::Reference` is unaffected — externalised payloads always
  pass through. Consumers that inline payloads must update their
  `ObservabilityStore` to attest PII redaction and pair it with a real
  `PayloadRedactor`.

### Fixed

- Gemini and Vertex Gemini count thinking tokens as output. Google reports
  them in `thoughtsTokenCount`, outside `candidatesTokenCount`, and bills
  them at the output rate, but the usage parser read only the candidate
  count, so every thinking model's `output_tokens` and cost came in low.
  `output_tokens` now includes thinking, as it already does for `OpenAI`
  reasoning tokens and Anthropic thinking; input and context-occupancy
  counts are unchanged.
- Gemini and Vertex Gemini accept tool and response schemas whose `type` is
  a list, such as `"type": ["array", "null"]`. Gemini parses these schemas
  as its `Schema` proto, which takes one `type`, so it refused every request
  carrying one with `400 Proto field is not repeating`. The schema
  sanitizer now sends `[T, "null"]` as `type: T, nullable: true` and
  several types as an `anyOf` of single-type branches. It also turns
  `const` into a one-value `enum`, moves `null` out of an `enum` into
  `nullable: true`, drops `enum`s with non-string values,
  and strips `$comment`, `examples`, `exclusiveMinimum`, `exclusiveMaximum`,
  `uniqueItems`, `prefixItems`, `deprecated` and `readOnly`, which Gemini
  rejects in the same way. `default` and `example` values are now sent
  verbatim instead of being sanitized as schemas.
- Built-in tools accept `null` for every optional input field and treat it as
  absent. `OpenAI` strict mode turns optional properties into required
  nullable ones, so GPT models send `"offset": null`; `read` (`offset`),
  `edit` (`replace_all`), `grep` (`recursive`, `case_insensitive`) and
  `ask_user` (`options`, `multi_select`) used to refuse that call and cost a
  round trip.
- A task parked on one call of a multi-call batch now names its own call in
  `ListThreadTasks` and `GetTask`. Every sibling used to name the batch's
  first call, because the snapshot read the shared continuation's
  `awaiting_index`, so a person could be shown one tool while approving
  another.

- **Repeating tool-call ids from OpenAI-compatible gateways** - On routes
  other than `api.openai.com`, the Chat Completions and Responses providers
  replace tool-call ids that only count calls (`bash:0`, `functions.read:3`,
  `call_0`, `0`, or any id shorter than eight characters) and ids that are
  missing or repeat the request history, before streaming or returning them.
  A `<name>:` stem is kept for Kimi's chat template, and every minted id is at
  most 40 characters. `api.openai.com` ids are replaced only when missing or
  repeated.

- **Tool-argument streaming counts as liveness** - The Codex, Responses and
  Chat Completions providers yield `StreamDelta::KeepAlive` while a model
  streams tool-call arguments, and Codex also yields one for SSE comments and
  WebSocket ping/pong. The durable root-turn worker does not treat a
  keep-alive as the stream's first event: before the first event each
  keep-alive restarts the 330s first-event budget, after it the 120s
  inter-event budget.

- **Early resume never re-presents a running call** - A parent resumed while
  tool children still run keeps each running call under its original id:
  the call is answered once with an interim `tool_result`, and its final
  outcome arrives once as a `<late_tool_result>` text block. If the turn ends
  first, by any path (cancel, failure, lease expiry, a fail-closed sweep),
  the next root turn on the thread closes each such call exactly once before
  its first request, with the child's real state: its result if it finished,
  `cancelled` if it was cancelled, or `still_running`, which tells the model
  not to re-issue the call until it is confirmed stopped or finished. This
  holds even when compaction already folded the interim result into history.
  The interim `tool_result` is the only one the server writes with
  `is_error: false` (a tool's result is `None` or `true`), and it carries a
  `kind` discriminator and the task id of the child spawned for that call,
  so tool output that imitates it is never taken for one. Only a
  user-message block laid out exactly as the server renders it counts as a
  late result. Compaction and fork retain the artifact a late result's
  spilled output points at.

  **Rollback:** do not roll the server back to a build without late results
  while any early-resumed parent is parked. Those rows keep already-answered
  calls in `pending_tool_calls`; an older build resuming one emits a second
  `tool_result` for the same call, the provider-request guard refuses it, and
  that turn fails. There is no cheap gate: the server never validates the
  continuation envelope version, and a new task-state kind would need schema
  migrations. Drain or cancel parked parents before rolling back.

- **Workspace compatibility for cached token usage fields** - Updated server and host token usage fixtures, stores, and turn accounting so the new cached-input and cache-creation fields compile across all targets and continue accumulating real provider-reported values.

- **Inline payload capture on LLM spans** - `ObservabilityStore` inline capture decisions now set `gen_ai.input.messages` and `gen_ai.output.messages` on LLM spans, and payload capture failures add a `payload_capture_failed` event without failing the agent run.

- **Cancelled outcomes in observability spans** - Root and turn-level observability instrumentation now records `cancelled` outcomes instead of omitting cancellation states.

### Changed

- **Non-blocking event channel** - The agent loop now uses non-blocking sends for events. If the event channel is full, it logs a warning and waits up to 30 seconds before timing out. If the channel is closed (consumer disconnected), the agent continues processing without blocking. This prevents slow consumers from stalling the LLM stream.

- **Improved stream resilience** - The streaming implementation now detects when the event channel is closed and continues processing the LLM response. This allows the agent to complete even if the consumer disconnects mid-stream.

- **Switched from `tracing` to `log` crate** - The SDK now uses the standard `log` crate instead of `tracing` for logging, making it compatible with any logging framework that implements `log` (e.g., `env_logger`, `simple_logger`, `fern`).

### Added

- **GA observability and operator package for the AMQP latency layer** - Phase 8.7 closes Phase 8 by exposing a programmatic [`MetricsRecorder`](crates/agent-service-host/src/metrics.rs) trait covering relay ticks, reclaim sweeps, backlog observations, wakeup nudges, watch advisories, retention janitor cycles, and lease sweeps. The host crate ships three implementations: `LoggingMetricsRecorder` (default — emits `tracing::info!` events with stable `metric=…` field names), `InMemoryMetricsRecorder` (atomic counters, snapshottable for tests), and `NoopMetricsRecorder`. Adds opt-in `relay.backlog_threshold` configuration with `soft` / `hard` bands; the soft band flips `LatencyLayerHealth` to `Degraded`, the hard band emits a separate alerting log line. Adds the `ga_regression` module that proves duplicate delivery, publish-then-crash recovery, backlog protection, and metrics lifecycle behaviour together. Ships two new operator runbooks under `docs/runbook/` covering the AMQP relay (normal / degraded / retention behaviour, multi-instance fanout topology, recovery procedures) and the health & readiness contract (probe endpoints, Kubernetes integration, alerting matrix).

- **Cached token usage observability** - OpenTelemetry spans now emit `gen_ai.usage.cache_creation.input_tokens` and `gen_ai.usage.cache_read.input_tokens`, and provider usage mappings preserve cache-read and cache-creation token breakdowns.

- **`TurnSummary` server-facing outcome contract** - Every `TurnOutcome` variant except `Error` now carries a structured `TurnSummary` alongside the legacy per-variant fields. The summary captures provider/model provenance, response ID, stop reason, tool-call count, wall-clock duration, and the `TurnOptions` used for the turn. This is the authoritative Phase 1 contract later server phases (journal, workers, transport, storage) build on. Use `TurnOutcome::summary()` to extract it, or pattern-match on the `summary` field directly.

  - `TurnSummary` is fully `Serialize + Deserialize` with a stable `snake_case` JSON wire format so servers can persist it directly as part of durable turn rows.
  - `StopReason` is now `Serialize` and exposes an `as_str()` helper whose discriminants match the serde wire format.
  - `ToolRuntime` is now `Serialize + Deserialize` with `snake_case` variants.
  - `TokenUsage` now derives `PartialEq + Eq`.
  - The `agent-server` crate re-exports `TurnSummary`, `StopReason`, and `AuditProvenance` as part of its convenience surface so server code can import them directly.
  - See the new `server_turn_summary` example and the expanded `agent-server` crate-level docs for the full authoritative-vs-convenience contract table.

- **Optional OpenTelemetry observability** - Added an `otel` feature with observability helpers, payload/context propagation, `ObservabilityStore` hooks, and instrumentation for agent runs, turns, LLM calls, tool execution, subagent invocations, MCP client calls, and context compaction spans.

- **Stream progress logging** - Added periodic debug logging during stream processing to help diagnose issues:
  - `SSE chunk progress` every 10 chunks from the HTTP stream
  - `Stream progress` every 50 deltas processed
  - Warnings when the event channel is full or closed

- **Stream drop detection** - Added a drop guard that logs (to both `log::error!` and `stderr`) when the SSE stream is dropped before completion, helping diagnose task cancellation issues.

## [0.4.0] - 2025-01-27

### Added

- **`AsyncTool` trait** - New trait for long-running operations with progress streaming. Async tools have two phases: `execute()` starts the operation and returns immediately, `check_status()` streams progress updates until completion. Includes typed progress stages via `ProgressStage` trait.

- **`ToolExecutionStore` trait** - Write-ahead logging for tool execution tracking. Enables idempotency and crash recovery by persisting tool execution state before running tools. Includes `InMemoryExecutionStore` default implementation.

- **Extended thinking support** - `ThinkingConfig` for configuring Anthropic's extended thinking feature. Set `budget_tokens` to enable the model to think through complex problems before responding.

- **`ToolProgress` event** - New event type for async tool progress updates, containing stage, message, and optional data.

- **`ErasedToolStatus`** - Type-erased version of `ToolStatus` for use in the agent loop.

- **`ToolOutcome` enum** - Result type for async tools with `Success`, `Failed`, and `InProgress` variants.

### Fixed

- **Store timestamp handling** - Fixed timestamp serialization/deserialization issues in message stores.

### Changed

- **`ToolRegistry` now supports async tools** - New `register_async()` method and `get_async()`, `all_async()`, `is_async()` accessors for async tool management.

## [0.3.0] - 2025-01-23

### Breaking Changes

- **Removed `ToolTier::RequiresPin`** - The SDK now only has `Observe` and `Confirm` tiers. Applications that need PIN verification should implement this at the application layer using hooks.

- **Removed `ToolDecision::RequiresPin`** - Hooks no longer return this variant. Use `RequiresConfirmation` or `Block` instead.

- **Removed `PendingAction`** - Applications now manage the pending action lifecycle externally.

- **Removed `ToolRequiresPin` event** - This event is no longer emitted by the agent loop.

- **Tool trait now requires `type Name` associated type** - All `Tool` implementations must specify a tool name type:
  ```rust
  impl Tool<()> for MyTool {
      type Name = DynamicToolName;  // or your custom ToolName type
      fn name(&self) -> DynamicToolName { DynamicToolName::new("my_tool") }
      // ...
  }
  ```

- **`agent.run()` now returns a tuple** - The return type is `(mpsc::Receiver<AgentEvent>, impl Future<Output = AgentRunState>)` instead of just a receiver.

- **Input is now wrapped in `AgentInput`** - Use `AgentInput::Text(...)` instead of passing a plain string.

- **`AgentContinuation` is now a concrete type** - Previously used `Box<dyn Any>` for encapsulation. Now exposes all fields publicly. The continuation is boxed in enum variants for efficiency (`Box<AgentContinuation>`).

- **New `PendingToolCallInfo` struct** - Public type representing pending tool calls, used within `AgentContinuation`.

### Added

- **Typed tool names** - Tool names are now strongly typed via the `ToolName` trait and associated type:
  - `PrimitiveToolName` enum for SDK's built-in tools
  - `DynamicToolName` for runtime/MCP tools
  - `tool_name_to_string()` and `tool_name_from_str()` helpers

- **`display_name()` method on Tool trait** - Tools can now provide a human-readable display name for UIs.

- **Yield/Resume pattern** - Agent can yield execution when a tool requires confirmation via `AgentRunState::AwaitingConfirmation`, then resume with `AgentInput::Resume`.

- **Single-turn execution** - New `run_turn()` method for external orchestration (e.g., message queues). Returns `TurnOutcome` indicating whether more turns are needed.

- **`AgentRunState` enum** - New return type indicating agent completion status: `Done`, `Error`, or `AwaitingConfirmation`.

- **`AgentContinuation`** - Concrete state for resuming agent execution after yielding. Contains fields like `thread_id`, `turn`, `turn_usage`, `pending_tool_calls`, etc.

- **Streaming support** - LLM providers now support streaming responses via `stream_chat_completion`.

### Fixed

- **Token usage tracking in resume cases** - Previously, token usage was zeroed when resuming after tool confirmation. Now properly tracks usage from the LLM call that generated the tool calls.

### Changed

- **Tool trait no longer requires `#[async_trait]`** - Rust 1.75+ native async traits are used instead.

- **Refactored agent loop** - Internal refactoring into smaller helper functions for better maintainability.

## [0.1.0] - 2025-01-15

### Added

- **Agent Loop**: Core orchestration for LLM conversations with tool calling
  - Event-driven streaming architecture
  - Configurable turn limits and retry behavior
  - Thread-based conversation management

- **LLM Abstraction**: Provider-agnostic interface for chat completions
  - `LlmProvider` trait for implementing custom providers
  - Built-in Anthropic provider (Claude models)
  - OpenAI provider implementation
  - Google Gemini provider implementation

- **Tool System**: Define and register custom tools
  - `Tool` trait for implementing tools
  - `ToolRegistry` for managing available tools
  - JSON schema validation for tool inputs
  - Tool tiers for security classification (Observe, Confirm, RequiresPin)

- **Lifecycle Hooks**: Pre/post tool execution hooks
  - `AgentHooks` trait for custom hook implementations
  - `AllowAllHooks` for development/testing
  - `LoggingHooks` for observability
  - `ToolDecision` for controlling tool execution

- **Environment Abstraction**: File and command execution
  - `Environment` trait for file system operations
  - `InMemoryFileSystem` for testing and sandboxing
  - `LocalFileSystem` for production use
  - Command execution with output capture

- **Primitive Tools**: Built-in tools for file operations
  - `ReadTool` - Read file contents
  - `WriteTool` - Create/overwrite files
  - `EditTool` - Make targeted edits to files
  - `GlobTool` - Find files by pattern
  - `GrepTool` - Search file contents
  - `BashTool` - Execute shell commands

- **Persistence**: Trait-based storage
  - `MessageStore` for conversation history
  - `StateStore` for agent state
  - `InMemoryStore` default implementation

- **Capabilities**: Security model for agent operations
  - `AgentCapabilities` for fine-grained permission control
  - Read-only, write, and exec toggles

- **Subagent System**: Nested agent execution
  - Real-time progress events during execution

- **MCP Support**: Model Context Protocol integration

- **Web Tools**: Fetch and search capabilities

### Security

- `#[forbid(unsafe_code)]` enforced across the codebase
- Capability-based security model
- Tool tier system for operation classification
