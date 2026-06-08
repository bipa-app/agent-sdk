# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.2](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-v0.9.1...agent-sdk-v0.9.2) - 2026-06-05

### Other

- release v0.9.2 ([#290](https://github.com/bipa-app/agent-sdk/pull/290))

## [0.9.0](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-v0.8.0...agent-sdk-v0.9.0) - 2026-06-02

### Added

- *(api)* [**breaking**] mark public wire/streaming enums #[non_exhaustive] + serde(other) StopReason (ENG-8734, Phase 14·A) ([#279](https://github.com/bipa-app/agent-sdk/pull/279))
- *(macros)* macro-based tool ergonomics — #[derive(Tool)] + schema derivation (ENG-8729) ([#274](https://github.com/bipa-app/agent-sdk/pull/274))
- *(obs)* one coherent trace per daemon turn + inbound gRPC trace propagation ([#275](https://github.com/bipa-app/agent-sdk/pull/275))
- *(mcp)* broaden MCP — streamable-HTTP/SSE transport, OAuth, resources/prompts, protocol negotiation (ENG-8727) ([#270](https://github.com/bipa-app/agent-sdk/pull/270))
- *(tools)* typed tool-arg validation via TypedTool + self-correction (ENG-8726) ([#271](https://github.com/bipa-app/agent-sdk/pull/271))
- *(structured-output)* schema-validated structured output (ENG-8725) ([#272](https://github.com/bipa-app/agent-sdk/pull/272))
- *(agent-sdk)* feature-gate providers/tools & replace deprecated serde_yaml (ENG-8714) ([#267](https://github.com/bipa-app/agent-sdk/pull/267))
- *(agent-loop)* cancel + timeout for async/listen tools (ENG-8704) ([#254](https://github.com/bipa-app/agent-sdk/pull/254))
- *(agent-loop)* isolate tool/LLM panics into structured errors (ENG-8705) ([#252](https://github.com/bipa-app/agent-sdk/pull/252))
- *(events)* add UserInput as first-class committed AgentEvent ([#245](https://github.com/bipa-app/agent-sdk/pull/245))
- *(observability)* add local Grafana + Tempo + Prometheus stack (Phase 9 · D2, ENG-8301) ([#242](https://github.com/bipa-app/agent-sdk/pull/242))
- *(observability)* lift local Langfuse stack + ship agent-sdk-cli (Phase 9 · D1, ENG-8300) ([#241](https://github.com/bipa-app/agent-sdk/pull/241))
- *(observability)* default-deny payload capture + acknowledge_pii_redaction (ENG-8298) ([#235](https://github.com/bipa-app/agent-sdk/pull/235))
- *(agent-server)* host-driven auto-compaction (proactive + recovery) ([#236](https://github.com/bipa-app/agent-sdk/pull/236))
- *(observability)* tool/subagent/MCP/compaction metric assertions and skip-on-expansion guard ([#232](https://github.com/bipa-app/agent-sdk/pull/232))
- *(observability)* record streaming TTFC and time-per-output-chunk ([#231](https://github.com/bipa-app/agent-sdk/pull/231))
- *(observability)* span links for replay & subagent-of relations (ENG-8292) ([#229](https://github.com/bipa-app/agent-sdk/pull/229))
- *(observability)* GenAI client metrics + SDK-specific meters (ENG-8293) ([#230](https://github.com/bipa-app/agent-sdk/pull/230))
- *(observability)* span events for streaming, retries, lifecycle (ENG-8291) ([#228](https://github.com/bipa-app/agent-sdk/pull/228))
- *(observability)* RunOptions for per-run Langfuse trace metadata (ENG-8290) ([#226](https://github.com/bipa-app/agent-sdk/pull/226))
- *(observability)* native Langfuse attribute helpers (ENG-8289) ([#225](https://github.com/bipa-app/agent-sdk/pull/225))
- *(observability)* propagate Langfuse baggage onto every SDK span (ENG-8288) ([#224](https://github.com/bipa-app/agent-sdk/pull/224))
- *(otel)* add agent-sdk-otel bootstrap helpers (ENG-8287) ([#222](https://github.com/bipa-app/agent-sdk/pull/222))
- *(events)* emit AutoRetryStart/End during transient LLM retries ([#213](https://github.com/bipa-app/agent-sdk/pull/213))
- *(agent-server)* stream TextDelta/ThinkingDelta events during root turns ([#209](https://github.com/bipa-app/agent-sdk/pull/209))
- PII redaction layer — baseline detectors + observability & journal wiring ([#207](https://github.com/bipa-app/agent-sdk/pull/207))
- *(providers)* add ToolChoice to ChatRequest for forced tool calling ([#204](https://github.com/bipa-app/agent-sdk/pull/204))
- *(otel)* track cached token usage breakdown ([#120](https://github.com/bipa-app/agent-sdk/pull/120))

### Fixed

- *(features)* compile with --no-default-features + add CI feature matrix ([#278](https://github.com/bipa-app/agent-sdk/pull/278))
- *(mcp)* bound pending-request map — remove entry on timeout (ENG-8736, Phase 14·C) ([#277](https://github.com/bipa-app/agent-sdk/pull/277))
- *(obs)* instrument daemon worker LLM + tool boundaries ([#269](https://github.com/bipa-app/agent-sdk/pull/269))
- *(agent-loop)* race tool execution against cancel token at SDK boundary ([#247](https://github.com/bipa-app/agent-sdk/pull/247))
- *(compaction)* chain-aware split selection + recovery from projection ([#237](https://github.com/bipa-app/agent-sdk/pull/237))
- *(agent-server/worker)* fix child retry budget, suspension idempotency, and dropped text blocks ([#141](https://github.com/bipa-app/agent-sdk/pull/141))

### Other

- Phase 13 · D — Distribution: crates.io readiness + release automation + binding spike (ENG-8728) ([#273](https://github.com/bipa-app/agent-sdk/pull/273))
- *(pre-public)* revert migration-comment scrub + neutralize internal Linear link ([#268](https://github.com/bipa-app/agent-sdk/pull/268))
- Phase 12 · D — MIT relicense + governance + Bipa scrub + full-history secret scan (ENG-8716) ([#265](https://github.com/bipa-app/agent-sdk/pull/265))
- Phase 12 · C — CLI can run an agent + quickstart examples (ENG-8715) ([#264](https://github.com/bipa-app/agent-sdk/pull/264))
- Phase 12 · A — API ergonomics + façade curation (ENG-8713) ([#266](https://github.com/bipa-app/agent-sdk/pull/266))
- *(agent-sdk)* cancellation + message-lifecycle edge-case matrix (Phase 11 · C, ENG-8710) ([#259](https://github.com/bipa-app/agent-sdk/pull/259))
- Phase 10 · A — Cancellation completeness: LLM/streaming/compaction + Cancelled event (ENG-8703) ([#256](https://github.com/bipa-app/agent-sdk/pull/256))
- *(observability)* conformance + privacy integration tests (Phase 9 · F1, ENG-8305) ([#246](https://github.com/bipa-app/agent-sdk/pull/246))
- *(privacy)* lift RedactionPolicy into agent-sdk-foundation (ENG-8286) ([#221](https://github.com/bipa-app/agent-sdk/pull/221))
- *(observability)* add Phase 9 inventory & gap report (ENG-8285) ([#220](https://github.com/bipa-app/agent-sdk/pull/220))
- *(loop)* run consecutive Observe-tier tool calls in parallel ([#201](https://github.com/bipa-app/agent-sdk/pull/201))
- bumping old dependencies to free bipa main of duplicated stuff ([#200](https://github.com/bipa-app/agent-sdk/pull/200))
- *(tools)* run built-in SDK tools through the v2 durable runtime ([#194](https://github.com/bipa-app/agent-sdk/pull/194))
- Convert Agent SDK repo to private distribution ([#193](https://github.com/bipa-app/agent-sdk/pull/193))
- summary-only parent subagent visibility ([#181](https://github.com/bipa-app/agent-sdk/pull/181))
- add gRPC local daemon transport ([#169](https://github.com/bipa-app/agent-sdk/pull/169))
- sdk/v2 ([#117](https://github.com/bipa-app/agent-sdk/pull/117))
