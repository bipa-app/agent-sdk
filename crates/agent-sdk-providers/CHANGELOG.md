# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
