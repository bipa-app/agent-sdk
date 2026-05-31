# Observability Inventory

> **Purpose:** A baseline inventory of what `agent-sdk` emits today and the
> concrete gaps that remain to close. Use this doc to verify that an
> instrumentation change actually fills the gap it claims to fill — tick the
> inventory rather than re-deriving it from the source.
>
> **Scope:** SDK-side instrumentation. A few attributes are tagged
> `[host-only-but-should-move]` where the host application sets them today but
> the helper should eventually ship from the SDK so the host can drop its
> override.
>
> **Spec references:**
> - OTel GenAI spans — <https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/>
> - OTel GenAI metrics — <https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/>
> - Langfuse OTel mapping — <https://langfuse.com/integrations/native/opentelemetry>

---

## 1. Spans we emit today

All spans are emitted under the `agent-sdk` `InstrumentationScope` (name +
version pulled from `CARGO_PKG_NAME` / `CARGO_PKG_VERSION` in
[`spans.rs`][spans-tracer]). Every emission site is gated on
`#[cfg(feature = "otel")]`; with the feature off the SDK is silent.

| Span name | Kind | Started in | Ended in | Required attrs (set) | Recommended attrs we miss |
| --- | --- | --- | --- | --- | --- |
| `invoke_agent` (root, mode=`loop`) | Internal | [`run_loop.rs:1646`][run-loop-start] → [`instrument::start_root_span`][start-root] | [`run_loop.rs:1693`][run-loop-end] → [`instrument::end_root_span`][end-root] | `gen_ai.operation.name=invoke_agent`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.conversation.id`, `agent_sdk.provider.id`, `agent_sdk.run.mode`, `agent_sdk.input.kind`, `agent_sdk.config.streaming`, `agent_sdk.tools.count`, `agent_sdk.config.max_turns?` (set on close: `agent_sdk.total_turns`, `gen_ai.usage.*`, `agent_sdk.outcome`) | `server.address` / `server.port`, `gen_ai.agent.name`, `langfuse.observation.type=agent`, `langfuse.trace.{name,input,output,tags,public,metadata.*}`, `langfuse.session.id`, `langfuse.user.id`, `error.type` on failure outcomes |
| `invoke_agent` (root, mode=`single_turn`) | Internal | [`run_loop.rs:1839`][single-turn-start] → `instrument::start_root_span` | [`run_loop.rs:1889`-block][single-turn-end] → `instrument::end_root_span` | same as above | same as above |
| `agent.turn` | Internal | [`turn.rs:1523`][record-turn] → `record_turn_span` | same call site (post-hoc span) | `agent_sdk.turn.number`, `agent_sdk.turn.{input,output,cache_*}_tokens`, `agent_sdk.turn.stop_reason`, `agent_sdk.turn.had_tool_calls` | **Span is started + ended after the work; no live timing.** Missing `gen_ai.operation.name`, `gen_ai.conversation.id`, `agent_sdk.turn.tool_call_count` (already on the type but not attached here). See A6/A7. |
| `chat <model>` | Client | [`turn.rs:582`][chat-start] → `spans::start_client_span` | [`turn.rs:763`][chat-end] → `finish_llm_span` | `gen_ai.operation.name=chat`, `gen_ai.provider.name`, `gen_ai.request.model`, `agent_sdk.llm.streaming`, `agent_sdk.provider.id`, `gen_ai.request.max_output_tokens?` (on close: `gen_ai.response.{id,model,finish_reasons}`, `gen_ai.usage.*`, `agent_sdk.llm.{had_tool_calls,text_output_present,thinking_present}`, `error.type` on error) | `server.address` / `server.port`, `gen_ai.response.time_to_first_chunk`, `gen_ai.usage.reasoning.output_tokens`, `gen_ai.request.{temperature, top_p, top_k, frequency_penalty, presence_penalty, stop_sequences, seed, choice.count, stream}`, `langfuse.observation.type=generation`, `langfuse.observation.{level, status_message, usage_details, cost_details, model.name, prompt.name, prompt.version}`, `gen_ai.input.messages` / `gen_ai.output.messages` / `gen_ai.system_instructions` (only set when an `ObservabilityStore` opts in — see §3) |
| `execute_tool` | Internal | [`tool_execution.rs:149`][exec-tool-start] → `start_tool_span` | [`tool_execution.rs:195`][exec-tool-end] → `finish_tool_span` | `gen_ai.operation.name=execute_tool`, `gen_ai.tool.name`, `gen_ai.tool.call.id`, `agent_sdk.tool.{display_name?, tier?, kind?}` (on close: `agent_sdk.tool.outcome`, `agent_sdk.tool.duration_ms?`, `agent_sdk.tool.confirmation_required?`, `error.type` on error) | `gen_ai.tool.call.arguments` (REDACTED), `gen_ai.tool.call.result` (REDACTED), `gen_ai.tool.description`, `gen_ai.tool.type`, `langfuse.observation.type=tool`, `langfuse.observation.{level, status_message}` |
| `agent.context_compaction` | Internal | [`turn.rs:370`][compaction-start] → `start_compaction_span` | [`turn.rs:408 / 422`][compaction-end] → `finish_compaction_span_{success,error}` | `agent_sdk.compaction.trigger` (on close: `agent_sdk.compaction.{original_count,new_count,original_tokens,new_tokens}`, `agent_sdk.outcome`, `error.type` on error) | `gen_ai.operation.name`, `gen_ai.conversation.id`, `langfuse.observation.type=chain`, `langfuse.observation.{level, status_message}` |
| `invoke_agent` (subagent variant) | Internal | [`subagent.rs:818`][subagent-emit] → `emit_subagent_observability` | same call site (post-hoc span) | `gen_ai.operation.name=invoke_agent`, `gen_ai.agent.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `agent_sdk.run.mode=loop`, `agent_sdk.outcome`, `agent_sdk.total_turns`, `gen_ai.usage.{input,output}_tokens`, `error.type` on error | **Span is started + ended after the work; no live timing.** Missing `gen_ai.conversation.id`, `gen_ai.usage.cache_*`, `langfuse.observation.type=agent`, parent-link to caller's `invoke_agent`. See A6/A7. |
| `mcp.initialize` | Client | [`mcp/client.rs:87`][mcp-init-start] → `start_mcp_span` | [`mcp/client.rs:92`][mcp-init-end] → `finish_mcp_span` | `mcp.server.name` | `gen_ai.operation.name`, MCP protocol version, `langfuse.observation.type=tool`, `error.type` is implicit via `Status::error` only |
| `mcp.tools/list` | Client | [`mcp/client.rs:149`][mcp-list-start] → `start_mcp_span` | [`mcp/client.rs:163`][mcp-list-end] → `finish_mcp_span` | `mcp.server.name`, `mcp.tools.count?` (set after success) | same as above |
| `mcp.tools/call` | Client | [`mcp/client.rs:199`][mcp-call-start] → `start_mcp_span_with_attrs` | [`mcp/client.rs:210`][mcp-call-end] → `finish_mcp_call_tool_span` | `mcp.server.name`, `gen_ai.tool.name` | `gen_ai.tool.call.{id, arguments, result}` (REDACTED), `gen_ai.tool.description`, `gen_ai.tool.type`, `langfuse.observation.type=tool` |

### Known structural issues (flagged for A6/A7)

1. `agent.turn` and the subagent `invoke_agent` are emitted **post-hoc** —
   started and ended in `record_turn_span` / `emit_subagent_observability`
   *after* the work has completed. They therefore record a near-zero duration
   and have no live parent stack while the work runs. Work item A6 (span
   events) or A7 (links) should restructure these to wrap their work properly.
2. The subagent `invoke_agent` span does **not** declare a parent link back to
   the caller's root `invoke_agent`, even though `subagent.rs` runs inside the
   parent's tokio task. Without an explicit
   `with_remote_span_context`/`add_link` it relies on `Context::current()`
   propagation, which is fragile across the parent's `tokio::spawn`.
3. `agent.context_compaction` does not trigger a child `chat <model>` span for
   the summarisation LLM call performed by `LlmContextCompactor`. The
   compaction LLM round-trip is therefore invisible in traces today.

---

## 2. Attributes we already set

All SDK attribute *constants* live in
[`crates/agent-sdk/src/observability/attrs.rs`][attrs-rs] — that file is the
canonical source. To avoid drift this section deliberately does **not** copy
the constants; instead it groups them by category, with one-line summaries and
a pointer back into `attrs.rs`.

| Bucket | Where the constants live | What they describe |
| --- | --- | --- |
| GenAI core (operation, provider, model, conversation, agent) | [`attrs.rs:10–18`][attrs-genai-core] | OTel-semconv keys identifying *who* is calling *which* model under *which* conversation. |
| GenAI usage (input / output / cache tokens) | [`attrs.rs:20–24`][attrs-genai-usage] | OTel-semconv token-accounting keys. Cached input + cache-creation are present; **`gen_ai.usage.reasoning.output_tokens` is missing**. |
| GenAI tool (name, call id, description) | [`attrs.rs:26–28`][attrs-genai-tool] | `GEN_AI_TOOL_DESCRIPTION` constant exists but is **never set on a span** today (see §3). |
| GenAI payload (system instructions, input messages, output messages) | [`attrs.rs:30–32`][attrs-genai-payload] | Set conditionally — only when an `ObservabilityStore` returns `CaptureDecision::Inline` from `record_payload_on_span`. Default = omit. |
| SDK run / config | [`attrs.rs:36–43`][attrs-sdk-run] | Provider id, run mode (`loop` / `single_turn`), input kind, streaming flag, max-turns config, tools count, total turns, outcome. |
| SDK turn-level | [`attrs.rs:45–53`][attrs-sdk-turn] | Turn number, resumed flag, had-tool-calls, tool-call count, stop reason, per-turn token deltas. |
| SDK LLM call | [`attrs.rs:55–58`][attrs-sdk-llm] | Streaming flag (separately from config), tool-call/text/thinking presence flags. |
| SDK tool call | [`attrs.rs:60–65`][attrs-sdk-tool] | Display name, tier (observe/confirm), kind (sync/async/listen), confirmation-required flag, outcome bucket, duration ms. |
| SDK compaction | [`attrs.rs:67–71`][attrs-sdk-compact] | Original/new message + token counts, trigger (`threshold` / `overflow`). |
| Observability payload references | [`attrs.rs:73–76`][attrs-sdk-otel] | When the store externalises a payload, this is the reference the span carries. |
| Error | [`attrs.rs:80`][attrs-error] | `error.type` — set on `chat`, `execute_tool`, `agent.context_compaction`, `agent.turn`. **Missing on root `invoke_agent`** (status is set, but no typed error tag — see §3). |

### Inline attributes set *outside* `attrs.rs`

Three keys are set inline in [`crates/agent-sdk/src/mcp/client.rs`][mcp-client]
without a constant in `attrs.rs`. Work item A6 should lift these into
`attrs.rs` so all attribute keys live in one place:

- `mcp.server.name` — every MCP span.
- `mcp.tools.count` — `mcp.tools/list` only, after success.
- `gen_ai.tool.name` — `mcp.tools/call` only (overlaps with the existing
  `GEN_AI_TOOL_NAME` constant; just needs to use the constant).

### Capture decision flow (payload attributes)

`gen_ai.input.messages` / `gen_ai.output.messages` / `gen_ai.system_instructions`
go through [`spans::record_payload_on_span` → `apply_capture_decision`][record-payload]:

- `CaptureDecision::Inline` → JSON serialised onto the span attribute.
- `CaptureDecision::Reference(ref)` → the matching `agent_sdk.observability.*_ref`
  string is set instead.
- `CaptureDecision::Omit` → nothing is written.

A run with no `ObservabilityStore` therefore emits **none** of the payload
attributes by default, regardless of the `otel` feature flag.

---

## 3. Attributes we owe (per OTel GenAI semconv & Langfuse mapping)

Each row identifies one missing attribute, the spans where it should appear,
which work item resolves it, and where the fix lives.

| Missing attribute | Affected spans | Why we owe it | Fix lives in | Card |
| --- | --- | --- | --- | --- |
| `gen_ai.input.messages` (default-on, redacted) | `chat <model>` | Currently gated behind opt-in store. Spec lists it as recommended on every chat span, with PII redaction. | `[SDK]` | C2 (default-deny + acknowledge_pii_redaction switches the gate) |
| `gen_ai.output.messages` (default-on, redacted) | `chat <model>` | same | `[SDK]` | C2 |
| `gen_ai.system_instructions` (default-on, redacted) | `chat <model>` | same | `[SDK]` | C2 |
| `gen_ai.tool.call.arguments` (REDACTED) | `execute_tool`, `mcp.tools/call` | Required by GenAI spec for tool spans. Never set today. | `[SDK]` | A6 (instrumentation) + C1 (mandatory redactor on egress) |
| `gen_ai.tool.call.result` (REDACTED) | `execute_tool`, `mcp.tools/call` | same | `[SDK]` | A6 + C1 |
| `gen_ai.tool.description` | `execute_tool`, `mcp.tools/call` | Constant exists in `attrs.rs` but is never set. | `[SDK]` | A6 |
| `gen_ai.tool.type` | `execute_tool`, `mcp.tools/call` | Required by spec. Maps to `function` / `extension` / `datastore`. | `[SDK]` | A6 |
| `server.address` / `server.port` | `invoke_agent` (root + subagent), `chat <model>`, `mcp.*` | OTel semconv requires it on `Client` spans; recommended on `Internal` for filtering. | `[SDK]` | A6 |
| `gen_ai.response.time_to_first_chunk` | `chat <model>` | Required for streaming TTFC histogram (B2). Span attr seeds the metric. | `[SDK]` | A6 + B2 |
| `gen_ai.usage.reasoning.output_tokens` | `chat <model>`, root `invoke_agent` | Reasoning models report this separately; today we collapse it into `output_tokens`. | `[SDK]` | A6 |
| `gen_ai.request.temperature` | `chat <model>` | Recommended on every chat span. | `[SDK]` | A6 |
| `gen_ai.request.top_p` | `chat <model>` | same | `[SDK]` | A6 |
| `gen_ai.request.top_k` | `chat <model>` | same | `[SDK]` | A6 |
| `gen_ai.request.frequency_penalty` | `chat <model>` | same | `[SDK]` | A6 |
| `gen_ai.request.presence_penalty` | `chat <model>` | same | `[SDK]` | A6 |
| `gen_ai.request.stop_sequences` | `chat <model>` | same | `[SDK]` | A6 |
| `gen_ai.request.seed` | `chat <model>` | same | `[SDK]` | A6 |
| `gen_ai.request.choice.count` | `chat <model>` | same | `[SDK]` | A6 |
| `gen_ai.request.stream` (boolean) | `chat <model>` | We have `agent_sdk.llm.streaming`; spec wants the canonical `gen_ai.request.stream`. | `[SDK]` | A6 |
| `error.type` on root span error outcome | `invoke_agent` (root) | `instrument::end_root_span` calls `set_span_error("agent_error", …)` only when `outcome=="error"`; the typed-error vocabulary does not match the spec's bucketed values (`timeout`, `rate_limit`, `cancelled`, `tool_failure`, …). | `[SDK]` | A6 |
| `langfuse.observation.type` (`agent`, `generation`, `tool`, `chain`) | every span | Langfuse routes by this hint; today it is only set by a host application via custom processors. | `[SDK]` | A4 (Langfuse helpers) |
| `langfuse.trace.name` | root `invoke_agent` | Set by the host application today. Should ship from the SDK so the host can drop the override. | `[host-only-but-should-move]` → `[SDK]` | A5 (`RunOptions`) + A4 |
| `langfuse.trace.input` | root `invoke_agent` | same | `[host-only-but-should-move]` → `[SDK]` | A5 + A4 |
| `langfuse.trace.output` | root `invoke_agent` | same | `[host-only-but-should-move]` → `[SDK]` | A5 + A4 |
| `langfuse.trace.tags` | root `invoke_agent` | same | `[host-only-but-should-move]` → `[SDK]` | A5 + A4 |
| `langfuse.trace.public` | root `invoke_agent` | same | `[host-only-but-should-move]` → `[SDK]` | A5 + A4 |
| `langfuse.trace.metadata.*` | root `invoke_agent` | same | `[host-only-but-should-move]` → `[SDK]` | A5 + A4 |
| `langfuse.session.id` (via baggage) | every span | A host application lifts `langfuse.session.id` out of `OTel` baggage today. The helper itself should live in the SDK. | `[host-only-but-should-move]` → `[SDK]` | A3 (baggage) + A4 |
| `langfuse.user.id` (via baggage) | every span | same | `[host-only-but-should-move]` → `[SDK]` | A3 + A4 |
| `langfuse.observation.level` | every span | Spec: `DEBUG` / `DEFAULT` / `WARNING` / `ERROR`. | `[SDK]` | A4 |
| `langfuse.observation.status_message` | every span | Free-form failure context for Langfuse status panels. | `[SDK]` | A4 |
| `langfuse.observation.usage_details` | `chat <model>`, root `invoke_agent` | Required for Langfuse cost tables. | `[SDK]` | A4 |
| `langfuse.observation.cost_details` | `chat <model>`, root `invoke_agent` | same | `[SDK]` | A4 |
| `langfuse.observation.model.name` | `chat <model>` | Pricing dimension; differs from `gen_ai.response.model` when proxies rewrite it. | `[SDK]` | A4 |
| `langfuse.observation.prompt.name` | `chat <model>` | For Langfuse prompt-management linkage. | `[SDK]` | A4 / A5 |
| `langfuse.observation.prompt.version` | `chat <model>` | same | `[SDK]` | A4 / A5 |

---

## 4. Metrics we owe (zero today)

`grep`ing `crates/agent-sdk/src` for `opentelemetry::metrics`, `histogram(`,
`counter(`, `meter(` returns no application instruments — the SDK emits **zero
metrics** at the moment. Every row below is therefore both a new instrument
and a new wiring site.

### 4.1 GenAI client metrics (OTel-semconv mandatory)

| Instrument | Type | Required attributes | Where it fires | Fix lives in | Card |
| --- | --- | --- | --- | --- | --- |
| `gen_ai.client.token.usage` | Histogram (tokens) | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.token.type` (`input` / `output`) | `chat <model>` close path | `[SDK]` | B1 |
| `gen_ai.client.operation.duration` | Histogram (s) | same set + `error.type` on failure | `chat <model>` close path | `[SDK]` | B1 |
| `gen_ai.client.operation.time_to_first_chunk` | Histogram (s) | same set | streaming `chat <model>` only | `[SDK]` | B2 |
| `gen_ai.client.operation.time_per_output_chunk` | Histogram (s) | same set | streaming `chat <model>` only | `[SDK]` | B2 |

### 4.2 SDK-specific metrics

| Instrument | Type | Attributes | Where | Fix lives in | Card |
| --- | --- | --- | --- | --- | --- |
| `agent_sdk.turns.duration` | Histogram (s) | `gen_ai.provider.name`, `agent_sdk.turn.stop_reason` | `agent.turn` close | `[SDK]` | B3 |
| `agent_sdk.runs.outcome` | Counter | `agent_sdk.outcome`, `gen_ai.provider.name`, `agent_sdk.run.mode` | root `invoke_agent` close | `[SDK]` | B3 |
| `agent_sdk.tools.execution.duration` | Histogram (ms) | `gen_ai.tool.name`, `agent_sdk.tool.tier`, `agent_sdk.tool.outcome` | `execute_tool` close | `[SDK]` | B3 |
| `agent_sdk.tools.execution.count` | Counter | same set | `execute_tool` close | `[SDK]` | B3 |
| `agent_sdk.context.compaction` | Counter | `agent_sdk.compaction.trigger`, `agent_sdk.outcome` | `agent.context_compaction` close | `[SDK]` | B3 |
| `agent_sdk.context.compaction.tokens_saved` | Histogram (tokens) | `agent_sdk.compaction.trigger` | `agent.context_compaction` close | `[SDK]` | B3 |
| `agent_sdk.subagent.invocations` | Counter | `gen_ai.agent.name`, `agent_sdk.outcome` | `emit_subagent_observability` | `[SDK]` | B3 |
| `agent_sdk.mcp.requests.duration` | Histogram (s) | `mcp.server.name`, `mcp.method` (`initialize` / `tools/list` / `tools/call`), `error.type` on failure | `mcp.*` close paths | `[SDK]` | B3 |
| `agent_sdk.llm.retries` | Counter | `gen_ai.provider.name`, `gen_ai.request.model`, retry classification | `chat <model>` (today only emits a span event per retry) | `[SDK]` | B3 |

### 4.3 Agent-server metrics

These live in the host crate (`agent-server`) but are listed here so the
inventory covers the whole observability surface.

| Instrument | Type | Attributes | Fix lives in | Card |
| --- | --- | --- | --- | --- |
| `agent_server.workers.active` | UpDownCounter | `worker.kind` | `[agent-server]` | B4 |
| `agent_server.tasks.acquired` | Counter | `task.kind` | `[agent-server]` | B4 |
| `agent_server.tasks.execution.duration` | Histogram (s) | `task.kind`, `task.outcome` | `[agent-server]` | B4 |
| `agent_server.tasks.lease.expired` | Counter | `task.kind` | `[agent-server]` | B4 |
| `agent_server.journal.commit.duration` | Histogram (s) | `journal.kind` | `[agent-server]` | B4 |
| `agent_server.journal.outbox.depth` | UpDownCounter | `journal.kind` | `[agent-server]` | B4 |
| `agent_server.relay.tick` | Counter | `relay.outcome` | `[agent-server]` | B4 |
| `agent_server.wakeup.fallback_sweep` | Counter | (none) | `[agent-server]` | B4 |
| `agent_server.thread_events_watch.lag_ms` | Histogram (ms) | `watch.kind` | `[agent-server]` | B4 |
| `agent_server.tool_audit.outcome` | Counter | `tool.tier`, `audit.outcome` | `[agent-server]` | B4 |

### 4.4 Agent-service-host metrics

| Instrument | Type | Attributes | Fix lives in | Card |
| --- | --- | --- | --- | --- |
| `rpc.server.duration` (per OTel RPC semconv) | Histogram (s) | `rpc.system=grpc`, `rpc.service`, `rpc.method`, `rpc.grpc.status_code` | `[agent-service-host]` | B5 |
| `db.pool.connections.active` | UpDownCounter | `pool.name`, `db.system` | `[agent-service-host]` | B5 |
| `db.pool.connections.idle` | UpDownCounter | `pool.name`, `db.system` | `[agent-service-host]` | B5 |
| `db.client.connections.create_time` | Histogram (s) | `pool.name`, `db.system` | `[agent-service-host]` | B5 |
| AMQP queue depth | UpDownCounter | `messaging.destination`, `messaging.system=rabbitmq` | `[agent-service-host]` | B5 |
| AMQP publish duration | Histogram (s) | same set + `messaging.operation=publish` | `[agent-service-host]` | B5 |
| AMQP consume duration | Histogram (s) | same set + `messaging.operation=process` | `[agent-service-host]` | B5 |

---

## 5. Work item map

| Gap | Resolved by | Type |
| --- | --- | --- |
| OTel bootstrap helper | A2 | New API |
| Baggage propagation | A3 | New API |
| Langfuse helpers | A4 | New API |
| `RunOptions` for trace metadata | A5 | New API |
| Span events (LLM lifecycle, retries, payload-capture) | A6 | Instrumentation |
| Span links for replay (resume → original run, subagent → parent) | A7 | Instrumentation |
| GenAI client metrics (token usage, op duration) | B1 | Metrics |
| Streaming TTFC / TPOC | B2 | Metrics |
| Tool / subagent / compaction / MCP / retry metrics | B3 | Metrics |
| Server-side metrics (workers, tasks, journal, relay, watch, audit) | B4 | Metrics |
| Host gRPC + DB + AMQP metrics | B5 | Metrics |
| Mandatory redactor on observability boundary | C1 | Privacy |
| Default-deny payload capture + `acknowledge_pii_redaction` | C2 | Privacy |
| Baggage allow-list | C3 | Privacy |
| Langfuse stack lift (compose) | D1 | Dev env |
| Grafana stack | D2 | Dev env |
| Starter dashboard JSON | D3 | Dev env |
| Host adoption (`agent-service-host` wiring) | E1 | Adoption |
| Downstream host cut-over to SDK-shipped helpers | E2 | Adoption |
| Tests | F1 | Tests |
| Examples | F2 | Examples |
| Docs | F3 | Docs |

Every gap row in §3 and §4 ends in a work-item ID from this table, so a
reviewer can read this doc once and verify a change is complete by ticking
off the rows tagged with that work item.

---

## Appendix A · Source-of-truth pointers

- Span constants & helpers: [`crates/agent-sdk/src/observability/attrs.rs`][attrs-rs]
- Span construction: [`crates/agent-sdk/src/observability/spans.rs`][spans-rs]
- Root span lifecycle: [`crates/agent-sdk/src/observability/instrument.rs`][instrument-rs]
- Capture types & store trait: [`crates/agent-sdk/src/observability/types.rs`][types-rs]
- Payload conversion + redactor: [`crates/agent-sdk/src/observability/payload.rs`][payload-rs]
- Context propagation across `tokio::spawn`: [`crates/agent-sdk/src/observability/context.rs`][context-rs]
- Provider name normalisation: [`crates/agent-sdk/src/observability/provider_name.rs`][provider-name-rs]
- LLM span emission: [`crates/agent-sdk/src/agent_loop/turn.rs`][turn-rs] (search `cfg(feature = "otel")`)
- Tool span emission: [`crates/agent-sdk/src/agent_loop/tool_execution.rs`][tool-exec-rs]
- Root span emission: [`crates/agent-sdk/src/agent_loop/run_loop.rs`][run-loop-rs]
- Subagent span emission: [`crates/agent-sdk/src/subagent.rs`][subagent-rs]
- MCP span emission: [`crates/agent-sdk/src/mcp/client.rs`][mcp-client]

[attrs-rs]: ../../src/observability/attrs.rs
[spans-rs]: ../../src/observability/spans.rs
[spans-tracer]: ../../src/observability/spans.rs#L11-L23
[instrument-rs]: ../../src/observability/instrument.rs
[types-rs]: ../../src/observability/types.rs
[payload-rs]: ../../src/observability/payload.rs
[context-rs]: ../../src/observability/context.rs
[provider-name-rs]: ../../src/observability/provider_name.rs
[turn-rs]: ../../src/agent_loop/turn.rs
[tool-exec-rs]: ../../src/agent_loop/tool_execution.rs
[run-loop-rs]: ../../src/agent_loop/run_loop.rs
[subagent-rs]: ../../src/subagent.rs
[mcp-client]: ../../src/mcp/client.rs

[attrs-genai-core]: ../../src/observability/attrs.rs#L10-L18
[attrs-genai-usage]: ../../src/observability/attrs.rs#L20-L24
[attrs-genai-tool]: ../../src/observability/attrs.rs#L26-L28
[attrs-genai-payload]: ../../src/observability/attrs.rs#L30-L32
[attrs-sdk-run]: ../../src/observability/attrs.rs#L36-L43
[attrs-sdk-turn]: ../../src/observability/attrs.rs#L45-L53
[attrs-sdk-llm]: ../../src/observability/attrs.rs#L55-L58
[attrs-sdk-tool]: ../../src/observability/attrs.rs#L60-L65
[attrs-sdk-compact]: ../../src/observability/attrs.rs#L67-L71
[attrs-sdk-otel]: ../../src/observability/attrs.rs#L73-L76
[attrs-error]: ../../src/observability/attrs.rs#L80
[start-root]: ../../src/observability/instrument.rs#L15-L49
[end-root]: ../../src/observability/instrument.rs#L52-L83
[record-payload]: ../../src/observability/spans.rs#L55-L109
[run-loop-start]: ../../src/agent_loop/run_loop.rs#L1645-L1659
[run-loop-end]: ../../src/agent_loop/run_loop.rs#L1670-L1699
[single-turn-start]: ../../src/agent_loop/run_loop.rs#L1838-L1852
[single-turn-end]: ../../src/agent_loop/run_loop.rs#L1865-L1900
[chat-start]: ../../src/agent_loop/turn.rs#L562-L583
[chat-end]: ../../src/agent_loop/turn.rs#L684-L764
[record-turn]: ../../src/agent_loop/turn.rs#L1511-L1582
[exec-tool-start]: ../../src/agent_loop/tool_execution.rs#L104-L150
[exec-tool-end]: ../../src/agent_loop/tool_execution.rs#L152-L196
[compaction-start]: ../../src/agent_loop/turn.rs#L366-L374
[compaction-end]: ../../src/agent_loop/turn.rs#L402-L423
[subagent-emit]: ../../src/subagent.rs#L806-L852
[mcp-init-start]: ../../src/mcp/client.rs#L85-L99
[mcp-init-end]: ../../src/mcp/client.rs#L91-L92
[mcp-list-start]: ../../src/mcp/client.rs#L147-L167
[mcp-list-end]: ../../src/mcp/client.rs#L163
[mcp-call-start]: ../../src/mcp/client.rs#L194-L213
[mcp-call-end]: ../../src/mcp/client.rs#L210
