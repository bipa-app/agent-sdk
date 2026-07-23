# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.14.0](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-providers-v0.13.1...agent-sdk-providers-v0.14.0) - 2026-07-23

### Fixed

- *(codex)* stream collected chat responses ([#393](https://github.com/bipa-app/agent-sdk/pull/393))
- *(compaction)* recover OpenAI Responses threads from context overflow ([#389](https://github.com/bipa-app/agent-sdk/pull/389))

### Other

- Persist terminal reasons and live subagent usage ([#394](https://github.com/bipa-app/agent-sdk/pull/394))
- persist turn attempt evidence ([#392](https://github.com/bipa-app/agent-sdk/pull/392))

## [0.13.1](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-providers-v0.13.0...agent-sdk-providers-v0.13.1) - 2026-07-14

### Other

- add the missing #380 entry to the 0.13.0 changelogs ([#382](https://github.com/bipa-app/agent-sdk/pull/382))

## [0.13.0](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-providers-v0.12.0...agent-sdk-providers-v0.13.0) - 2026-07-14

### Added

- *(providers)* [**breaking**] carry Retry-After through streaming errors + parse body-embedded retry hints ([#366](https://github.com/bipa-app/agent-sdk/pull/366))
- *(sdk)* budget consults dynamic model catalog with static fallback ([#356](https://github.com/bipa-app/agent-sdk/pull/356)) ([#368](https://github.com/bipa-app/agent-sdk/pull/368))

### Fixed

- *(sdk,providers)* bill reasoning at max-rate; take max of complete candidate estimates ([#380](https://github.com/bipa-app/agent-sdk/pull/380))

## [0.12.0](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-providers-v0.11.8...agent-sdk-providers-v0.12.0) - 2026-07-12

### Fixed

- *(anthropic,server)* bound silent streams so stalled connections retry instead of hanging ([#357](https://github.com/bipa-app/agent-sdk/pull/357))
- *(openai-codex)* preserve atomic streamed tool calls ([#353](https://github.com/bipa-app/agent-sdk/pull/353))

## [0.11.8](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-providers-v0.11.7...agent-sdk-providers-v0.11.8) - 2026-07-11

### Other

- update Cargo.toml dependencies

## [0.11.7](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-providers-v0.11.6...agent-sdk-providers-v0.11.7) - 2026-07-11

### Fixed

- *(openai-codex)* parse non-terminal response statuses in stream events ([#347](https://github.com/bipa-app/agent-sdk/pull/347))

### Added

- *(openai)* add GPT-5.6 Sol, Terra, and Luna factories; exact reasoning, API-surface, tool, storage, and prompt-cache controls; and surface-scoped capability metadata

### Fixed

- *(openai)* preserve encrypted Responses reasoning items across tool turns, surface refusals and incomplete responses accurately, normalize strict schemas, retain output-item ordering, and reject truncated streams instead of committing partial success
- *(openai)* serialize Chat Completions `reasoning_effort` as a scalar, remove the implicit 4K output cap for known models, map cache-write usage, and honor explicit GPT-5.6 cache breakpoints

## [0.11.6](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-providers-v0.11.5...agent-sdk-providers-v0.11.6) - 2026-07-09

### Fixed

- *(gemini)* preserve nested schema property names in tool/response conversion ([#340](https://github.com/bipa-app/agent-sdk/pull/340))

## [0.11.5](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-providers-v0.11.4...agent-sdk-providers-v0.11.5) - 2026-07-09

### Added

- wire task wakeup + interleaved-thinking beta on API-key auth (Fixes 1, 3, 8) ([#338](https://github.com/bipa-app/agent-sdk/pull/338))

## [0.11.4](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-providers-v0.11.3...agent-sdk-providers-v0.11.4) - 2026-07-08

### Fixed

- *(anthropic)* surface mid-stream SSE `error` events instead of dropping them ([#334](https://github.com/bipa-app/agent-sdk/pull/334))

## [0.11.3](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-providers-v0.11.2...agent-sdk-providers-v0.11.3) - 2026-07-07

### Other

- release v0.11.3 ([#329](https://github.com/bipa-app/agent-sdk/pull/329))

## [0.11.2](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-providers-v0.11.1...agent-sdk-providers-v0.11.2) - 2026-07-01

### Added

- *(model-capabilities)* add claude-sonnet-5 ([#326](https://github.com/bipa-app/agent-sdk/pull/326))

## [0.11.0](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-providers-v0.10.0...agent-sdk-providers-v0.11.0) - 2026-06-14

### Added

- *(providers)* dynamic model discovery + third-party capability/pricing feed ([#317](https://github.com/bipa-app/agent-sdk/pull/317))
- *(providers)* rate-limit Retry-After, prompt-cache control, failover, streaming-structured output, record/replay provider ([#311](https://github.com/bipa-app/agent-sdk/pull/311))

### Fixed

- *(providers)* stop Codex WebSocket stalls in WS-hostile networks ([#318](https://github.com/bipa-app/agent-sdk/pull/318))

## [0.10.0](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-providers-v0.9.2...agent-sdk-providers-v0.10.0) - 2026-06-11

### Added

- [**breaking**] deep-review fix sweep — 272 findings across all crates ([#306](https://github.com/bipa-app/agent-sdk/pull/306))

### Fixed

- *(providers)* preserve streamed openai usage ([#305](https://github.com/bipa-app/agent-sdk/pull/305))

### Other

- release v0.9.3 ([#307](https://github.com/bipa-app/agent-sdk/pull/307))

## [0.9.2](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-providers-v0.9.1...agent-sdk-providers-v0.9.2) - 2026-06-09

### Added

- *(providers)* add Claude Fable 5 model support ([#297](https://github.com/bipa-app/agent-sdk/pull/297))

### Fixed

- *(providers)* streaming reasoning fallback + multi-turn reasoning echo-back + MiniMax cache price ([#291](https://github.com/bipa-app/agent-sdk/pull/291))

## [0.9.1](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-providers-v0.9.0...agent-sdk-providers-v0.9.1) - 2026-06-04

### Added

- *(providers)* register open-model capabilities + reasoning-response fallback ([#288](https://github.com/bipa-app/agent-sdk/pull/288))

## [0.9.0](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-providers-v0.8.0...agent-sdk-providers-v0.9.0) - 2026-06-02

### Added

- *(api)* [**breaking**] mark public wire/streaming enums #[non_exhaustive] + serde(other) StopReason (ENG-8734, Phase 14·A) ([#279](https://github.com/bipa-app/agent-sdk/pull/279))
- *(structured-output)* schema-validated structured output (ENG-8725) ([#272](https://github.com/bipa-app/agent-sdk/pull/272))
- *(agent-sdk)* feature-gate providers/tools & replace deprecated serde_yaml (ENG-8714) ([#267](https://github.com/bipa-app/agent-sdk/pull/267))
- *(providers)* register Claude Opus 4.8; correct Opus 4.7/4.8 pricing ([#250](https://github.com/bipa-app/agent-sdk/pull/250))
- *(agent-server)* stream TextDelta/ThinkingDelta events during root turns ([#209](https://github.com/bipa-app/agent-sdk/pull/209))
- *(providers)* add ToolChoice to ChatRequest for forced tool calling ([#204](https://github.com/bipa-app/agent-sdk/pull/204))
- *(providers)* RefreshingProvider wrapper with 401 retry ([#192](https://github.com/bipa-app/agent-sdk/pull/192))
- *(otel)* track cached token usage breakdown ([#120](https://github.com/bipa-app/agent-sdk/pull/120))
- *(agent-server/worker)* Phase 4.4 tool-boundary suspension and child task dispatch ([#139](https://github.com/bipa-app/agent-sdk/pull/139))

### Fixed

- *(features)* compile with --no-default-features + add CI feature matrix ([#278](https://github.com/bipa-app/agent-sdk/pull/278))
- *(providers)* register Claude Opus 4.7 with adaptive-thinking requirement ([#202](https://github.com/bipa-app/agent-sdk/pull/202))

### Other

- Phase 13 · D — Distribution: crates.io readiness + release automation + binding spike (ENG-8728) ([#273](https://github.com/bipa-app/agent-sdk/pull/273))
- Phase 12 · D — MIT relicense + governance + Bipa scrub + full-history secret scan (ENG-8716) ([#265](https://github.com/bipa-app/agent-sdk/pull/265))
- Phase 12 · A — API ergonomics + façade curation (ENG-8713) ([#266](https://github.com/bipa-app/agent-sdk/pull/266))
- Phase 11 · E — Provider determinism & CI tiering; re-enable streaming tests (ENG-8712) ([#262](https://github.com/bipa-app/agent-sdk/pull/262))
- bumping old dependencies to free bipa main of duplicated stuff ([#200](https://github.com/bipa-app/agent-sdk/pull/200))
- Convert Agent SDK repo to private distribution ([#193](https://github.com/bipa-app/agent-sdk/pull/193))
- sdk/v2 ([#117](https://github.com/bipa-app/agent-sdk/pull/117))
