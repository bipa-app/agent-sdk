# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.13.1](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-foundation-v0.13.0...agent-sdk-foundation-v0.13.1) - 2026-07-15

### Other

- release v0.13.1 ([#385](https://github.com/bipa-app/agent-sdk/pull/385))

## [0.13.0](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-foundation-v0.12.0...agent-sdk-foundation-v0.13.0) - 2026-07-14

### Added

- *(foundation,host)* [**breaking**] stamp emitter task identity on lifecycle events; causal follower close gate ([#363](https://github.com/bipa-app/agent-sdk/pull/363)) ([#369](https://github.com/bipa-app/agent-sdk/pull/369))

## [0.12.0](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-foundation-v0.11.8...agent-sdk-foundation-v0.12.0) - 2026-07-12

### Added

- *(agent-loop)* [**breaking**] usage/cost budgets, parallel-tool cap, guardrail-hook wiring, reminders, run_stream ([#312](https://github.com/bipa-app/agent-sdk/pull/312))

## [0.11.6](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-foundation-v0.11.5...agent-sdk-foundation-v0.11.6) - 2026-07-09

### Fixed

- *(gemini)* preserve nested schema property names in tool/response conversion ([#340](https://github.com/bipa-app/agent-sdk/pull/340))

## [0.11.3](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-foundation-v0.11.2...agent-sdk-foundation-v0.11.3) - 2026-07-01

### Other

- release v0.11.2 ([#328](https://github.com/bipa-app/agent-sdk/pull/328))

## [0.11.2](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-foundation-v0.11.1...agent-sdk-foundation-v0.11.2) - 2026-07-01

### Other

- release v0.11.1 ([#325](https://github.com/bipa-app/agent-sdk/pull/325))

## [0.11.1](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-foundation-v0.11.0...agent-sdk-foundation-v0.11.1) - 2026-06-18

### Fixed

- *(loop)* durably backfill orphaned tool_use instead of 400ing ([#323](https://github.com/bipa-app/agent-sdk/pull/323))

## [0.11.0](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-foundation-v0.10.0...agent-sdk-foundation-v0.11.0) - 2026-06-14

### Added

- *(providers)* rate-limit Retry-After, prompt-cache control, failover, streaming-structured output, record/replay provider ([#311](https://github.com/bipa-app/agent-sdk/pull/311))

### Fixed

- *(foundation)* use non-deprecated time parse_borrowed ([#320](https://github.com/bipa-app/agent-sdk/pull/320))

## [0.10.0](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-foundation-v0.9.2...agent-sdk-foundation-v0.10.0) - 2026-06-11

### Added

- [**breaking**] deep-review fix sweep — 272 findings across all crates ([#306](https://github.com/bipa-app/agent-sdk/pull/306))

### Other

- release v0.9.3 ([#307](https://github.com/bipa-app/agent-sdk/pull/307))

## [0.9.2](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-foundation-v0.9.1...agent-sdk-foundation-v0.9.2) - 2026-06-05

### Other

- release v0.9.2 ([#290](https://github.com/bipa-app/agent-sdk/pull/290))

## [0.9.0](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-foundation-v0.1.0-alpha.3...agent-sdk-foundation-v0.9.0) - 2026-06-02

### Added

- *(api)* [**breaking**] mark public wire/streaming enums #[non_exhaustive] + serde(other) StopReason (ENG-8734, Phase 14·A) ([#279](https://github.com/bipa-app/agent-sdk/pull/279))
- *(structured-output)* schema-validated structured output (ENG-8725) ([#272](https://github.com/bipa-app/agent-sdk/pull/272))
- *(agent-loop)* cancel + timeout for async/listen tools (ENG-8704) ([#254](https://github.com/bipa-app/agent-sdk/pull/254))
- *(events)* add UserInput as first-class committed AgentEvent ([#245](https://github.com/bipa-app/agent-sdk/pull/245))
- *(observability)* RunOptions for per-run Langfuse trace metadata (ENG-8290) ([#226](https://github.com/bipa-app/agent-sdk/pull/226))
- *(events)* emit AutoRetryStart/End during transient LLM retries ([#213](https://github.com/bipa-app/agent-sdk/pull/213))
- PII redaction layer — baseline detectors + observability & journal wiring ([#207](https://github.com/bipa-app/agent-sdk/pull/207))
- *(providers)* add ToolChoice to ChatRequest for forced tool calling ([#204](https://github.com/bipa-app/agent-sdk/pull/204))
- *(otel)* track cached token usage breakdown ([#120](https://github.com/bipa-app/agent-sdk/pull/120))
- *(agent-server)* Phase 6.2 atomic event commit rules for root and tool task transitions ([#154](https://github.com/bipa-app/agent-sdk/pull/154))
- *(agent-server/worker)* Phase 4.4 tool-boundary suspension and child task dispatch ([#139](https://github.com/bipa-app/agent-sdk/pull/139))
- *(agent-server/worker)* Phase 4.1 AgentDefinition resolution and root worker bootstrapping (ENG-7934) ([#134](https://github.com/bipa-app/agent-sdk/pull/134))

### Fixed

- address PR #151 code review comments ([#153](https://github.com/bipa-app/agent-sdk/pull/153))
- *(agent-server/worker)* fix child retry budget, suspension idempotency, and dropped text blocks ([#141](https://github.com/bipa-app/agent-sdk/pull/141))

### Other

- Phase 13 · D — Distribution: crates.io readiness + release automation + binding spike (ENG-8728) ([#273](https://github.com/bipa-app/agent-sdk/pull/273))
- *(pre-public)* revert migration-comment scrub + neutralize internal Linear link ([#268](https://github.com/bipa-app/agent-sdk/pull/268))
- Phase 12 · D — MIT relicense + governance + Bipa scrub + full-history secret scan (ENG-8716) ([#265](https://github.com/bipa-app/agent-sdk/pull/265))
- Phase 10 · A — Cancellation completeness: LLM/streaming/compaction + Cancelled event (ENG-8703) ([#256](https://github.com/bipa-app/agent-sdk/pull/256))
- Phase 11 · A — Test substrate & CI: nextest, Postgres-in-CI, proptest/start_paused/insta, failpoints (ENG-8708) ([#253](https://github.com/bipa-app/agent-sdk/pull/253))
- *(privacy)* lift RedactionPolicy into agent-sdk-foundation (ENG-8286) ([#221](https://github.com/bipa-app/agent-sdk/pull/221))
- Convert Agent SDK repo to private distribution ([#193](https://github.com/bipa-app/agent-sdk/pull/193))
- summary-only parent subagent visibility ([#181](https://github.com/bipa-app/agent-sdk/pull/181))
- add gRPC local daemon transport ([#169](https://github.com/bipa-app/agent-sdk/pull/169))
- sdk/v2 ([#117](https://github.com/bipa-app/agent-sdk/pull/117))
