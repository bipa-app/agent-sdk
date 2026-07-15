# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.13.1](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-tools-v0.13.0...agent-sdk-tools-v0.13.1) - 2026-07-15

### Other

- release v0.13.1 ([#385](https://github.com/bipa-app/agent-sdk/pull/385))

## [0.13.0](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-tools-v0.12.0...agent-sdk-tools-v0.13.0) - 2026-07-14

### Other

- release v0.13.0 ([#378](https://github.com/bipa-app/agent-sdk/pull/378))

## [0.12.0](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-tools-v0.11.8...agent-sdk-tools-v0.12.0) - 2026-07-12

### Added

- *(agent-loop)* [**breaking**] usage/cost budgets, parallel-tool cap, guardrail-hook wiring, reminders, run_stream ([#312](https://github.com/bipa-app/agent-sdk/pull/312))

## [0.11.6](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-tools-v0.11.5...agent-sdk-tools-v0.11.6) - 2026-07-10

### Other

- release v0.11.6 ([#341](https://github.com/bipa-app/agent-sdk/pull/341))

## [0.11.3](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-tools-v0.11.2...agent-sdk-tools-v0.11.3) - 2026-07-01

### Other

- release v0.11.2 ([#328](https://github.com/bipa-app/agent-sdk/pull/328))

## [0.11.2](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-tools-v0.11.1...agent-sdk-tools-v0.11.2) - 2026-07-01

### Other

- release v0.11.1 ([#325](https://github.com/bipa-app/agent-sdk/pull/325))

## [0.10.0](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-tools-v0.9.2...agent-sdk-tools-v0.10.0) - 2026-06-11

### Added

- [**breaking**] deep-review fix sweep — 272 findings across all crates ([#306](https://github.com/bipa-app/agent-sdk/pull/306))

### Other

- release v0.9.3 ([#307](https://github.com/bipa-app/agent-sdk/pull/307))

## [0.9.2](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-tools-v0.9.1...agent-sdk-tools-v0.9.2) - 2026-06-05

### Other

- release v0.9.2 ([#290](https://github.com/bipa-app/agent-sdk/pull/290))

## [0.9.0](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-tools-v0.8.0...agent-sdk-tools-v0.9.0) - 2026-06-02

### Added

- *(api)* [**breaking**] mark public wire/streaming enums #[non_exhaustive] + serde(other) StopReason (ENG-8734, Phase 14·A) ([#279](https://github.com/bipa-app/agent-sdk/pull/279))
- *(macros)* macro-based tool ergonomics — #[derive(Tool)] + schema derivation (ENG-8729) ([#274](https://github.com/bipa-app/agent-sdk/pull/274))
- *(tools)* typed tool-arg validation via TypedTool + self-correction (ENG-8726) ([#271](https://github.com/bipa-app/agent-sdk/pull/271))
- *(agent-loop)* cancel + timeout for async/listen tools (ENG-8704) ([#254](https://github.com/bipa-app/agent-sdk/pull/254))
- *(agent-server/worker)* Phase 4.4 tool-boundary suspension and child task dispatch ([#139](https://github.com/bipa-app/agent-sdk/pull/139))

### Fixed

- *(tools)* sort `to_llm_tools` by name for stable Anthropic prompt cache ([#205](https://github.com/bipa-app/agent-sdk/pull/205))

### Other

- Phase 13 · D — Distribution: crates.io readiness + release automation + binding spike (ENG-8728) ([#273](https://github.com/bipa-app/agent-sdk/pull/273))
- Phase 12 · D — MIT relicense + governance + Bipa scrub + full-history secret scan (ENG-8716) ([#265](https://github.com/bipa-app/agent-sdk/pull/265))
- Phase 12 · A — API ergonomics + façade curation (ENG-8713) ([#266](https://github.com/bipa-app/agent-sdk/pull/266))
- Convert Agent SDK repo to private distribution ([#193](https://github.com/bipa-app/agent-sdk/pull/193))
- sdk/v2 ([#117](https://github.com/bipa-app/agent-sdk/pull/117))
