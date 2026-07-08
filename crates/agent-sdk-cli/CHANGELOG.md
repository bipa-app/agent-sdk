# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.4](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-cli-v0.11.3...agent-sdk-cli-v0.11.4) - 2026-07-08

### Other

- release v0.11.4 ([#335](https://github.com/bipa-app/agent-sdk/pull/335))

## [0.11.3](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-cli-v0.11.2...agent-sdk-cli-v0.11.3) - 2026-07-07

### Other

- release v0.11.3 ([#329](https://github.com/bipa-app/agent-sdk/pull/329))

## [0.11.2](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-cli-v0.11.1...agent-sdk-cli-v0.11.2) - 2026-07-01

### Other

- update Cargo.lock dependencies

## [0.11.0](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-cli-v0.10.0...agent-sdk-cli-v0.11.0) - 2026-06-14

### Other

- update Cargo.lock dependencies

## [0.10.0](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-cli-v0.9.2...agent-sdk-cli-v0.10.0) - 2026-06-11

### Added

- [**breaking**] deep-review fix sweep — 272 findings across all crates ([#306](https://github.com/bipa-app/agent-sdk/pull/306))

### Other

- release v0.9.3 ([#307](https://github.com/bipa-app/agent-sdk/pull/307))

## [0.9.2](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-cli-v0.9.1...agent-sdk-cli-v0.9.2) - 2026-06-09

### Added

- *(providers)* add Claude Fable 5 model support — `--model fable` alias ([#297](https://github.com/bipa-app/agent-sdk/pull/297))

### Fixed

- *(cli)* exclude the agent-sdk binary from rustdoc (doc collision) ([#292](https://github.com/bipa-app/agent-sdk/pull/292))

## [0.9.0](https://github.com/bipa-app/agent-sdk/compare/agent-sdk-cli-v0.8.0...agent-sdk-cli-v0.9.0) - 2026-06-02

### Added

- *(agent-sdk)* feature-gate providers/tools & replace deprecated serde_yaml (ENG-8714) ([#267](https://github.com/bipa-app/agent-sdk/pull/267))
- *(observability)* lift local Langfuse stack + ship agent-sdk-cli (Phase 9 · D1, ENG-8300) ([#241](https://github.com/bipa-app/agent-sdk/pull/241))

### Other

- Phase 13 · D — Distribution: crates.io readiness + release automation + binding spike (ENG-8728) ([#273](https://github.com/bipa-app/agent-sdk/pull/273))
- Phase 12 · D — MIT relicense + governance + Bipa scrub + full-history secret scan (ENG-8716) ([#265](https://github.com/bipa-app/agent-sdk/pull/265))
- Phase 12 · C — CLI can run an agent + quickstart examples (ENG-8715) ([#264](https://github.com/bipa-app/agent-sdk/pull/264))
