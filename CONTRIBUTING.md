# Contributing to Agent SDK

Thanks for your interest in contributing! Agent SDK is an open-source
project and we welcome bug reports, feature requests, documentation
improvements, and code contributions from the community. This document
describes the workflow for working on the SDK.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## Getting Started

1. Fork the repository and clone your fork
2. Create a feature branch from `main`
3. Make your changes
4. Run the quality checks (see below)
5. Commit your changes
6. Push your branch to your fork
7. Open a pull request against `main` for review

For larger changes, please open an issue first to discuss the approach
before investing significant effort.

## Development Setup

### Prerequisites

- Rust 1.91+ (2024 edition; workspace MSRV, verified by CI)
- Cargo

### Building

This is a Cargo workspace. All commands run from the repo root:

```bash
cargo build
```

### Local Postgres 18 and SQLx

The Postgres-backed service-host code uses `sqlx` compile-time checked
queries and commits `.sqlx` metadata for offline builds. The repo ships
with a local Postgres 18 Docker Compose setup in `compose.yml` and a
helper script at `scripts/postgres18-dev.sh`.

The helper respects `SQLX_DEV_CARGO_HOME` if you need to point SQLx prep
at a specific Cargo cache; otherwise it falls back to the current
`CARGO_HOME` or a temporary cache.

Bring up the local database:

```bash
scripts/postgres18-dev.sh up
scripts/postgres18-dev.sh wait
```

Refresh SQLx metadata against a fresh local database. The `.sqlx` cache holds
entries for **both** the Postgres and SQLite backends, and `cargo sqlx prepare`
wipes `.sqlx/` before writing — so you must refresh both backends, in order:

```bash
# 1. Regenerate the Postgres entries (this DROPS the SQLite entries).
scripts/postgres18-dev.sh prepare

# 2. Regenerate the SQLite entries. This script backs up and merges the
#    existing Postgres entries, so run it AFTER the Postgres prepare.
scripts/sqlite-dev.sh prepare
```

> **Warning:** running `scripts/postgres18-dev.sh prepare` on its own leaves the
> SQLite query cache empty and breaks `SQLX_OFFLINE` builds for the SQLite
> backend. Always follow it with `scripts/sqlite-dev.sh prepare`.

Validate that the current migration bundle applies cleanly and that the
Postgres store tests pass against the local database:

```bash
scripts/postgres18-dev.sh test-migrations
```

Normal Cargo builds run with `SQLX_OFFLINE=true` via
`.cargo/config.toml`, so if you change a compile-time checked query you
must refresh the `.sqlx` metadata before submitting the change.

### Running Tests

```bash
cargo test
```

## Code Quality

Before submitting a pull request, ensure your code passes all quality checks:

```bash
# Type check
cargo check --all-targets

# Format code
cargo fmt

# Lint (must pass with no warnings)
cargo clippy -- -D warnings

# Run tests
cargo test
```

All of these checks run in CI and must pass for a PR to be merged.

## Coding Guidelines

### Modern Rust Module System

Use the modern Rust 2018+ module style. **Do not use `mod.rs` files.**

```
src/
├── lib.rs              # mod llm;
├── llm.rs              # pub mod types; + main code
└── llm/
    └── types.rs
```

### Error Handling

Use `anyhow` for error handling:

```rust
use anyhow::{Result, Context, bail, ensure};

pub async fn my_function() -> anyhow::Result<Output> {
    let data = fetch_data().await.context("failed to fetch")?;
    ensure!(!data.is_empty(), "data cannot be empty");
    Ok(output)
}
```

**Never use `unwrap()` or `expect()`** - always propagate errors with `?` and add context.

### Self-Documenting Code

Prefer self-documenting code over comments:

```rust
// Avoid: redundant comment
/// Find a user by their ID.
pub async fn find_by_id(...) { }

// Good: name is clear, no comment needed
pub async fn find_by_id(...) { }

// Good: comment explains non-obvious behavior
/// Returns None if the thread was archived more than 24 hours ago.
pub async fn find_active_thread(...) { }
```

### Testing

- Test behavior, not implementation details
- Never change test expectations to make tests pass - fix the code
- Skip trivial tests where Rust's type system provides sufficient guarantees

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn feature_works() -> anyhow::Result<()> {
        let store = InMemoryStore::new();
        let result = my_function(&store).await?;
        assert_eq!(result.status, ExpectedStatus);
        Ok(())
    }
}
```

### Clippy

Never bypass clippy with `#[allow(...)]` - fix the code instead:

```rust
// Bad: bypassing clippy
#[allow(clippy::too_many_arguments)]
pub fn create(a: A, b: B, c: C, d: D, e: E, f: F) { }

// Good: use a struct
pub struct CreateParams { pub a: A, pub b: B, /* ... */ }
pub fn create(params: CreateParams) { }
```

## Signed Commits

**All commits must be GPG-signed and verified.** Configure signing before you
start:

```bash
git config user.signingkey <YOUR_KEY_ID>
git config commit.gpgsign true   # or set globally
```

Then add the public key to your GitHub account (Settings → SSH and GPG keys) so
commits show as **Verified**. Sign an individual commit with `git commit -S`;
re-sign existing commits on a branch with:

```bash
git rebase --exec 'git commit --amend --no-edit -S' <base>
git push --force-with-lease
```

If your tooling commits non-interactively (some sandboxes disable signing even
when `commit.gpgsign = true`), pass `-S` explicitly and make sure your
`gpg-agent` has the key passphrase cached.

## Pull Request Process

All pull requests target the **`main`** branch.

1. Ensure all quality checks pass
2. Update documentation if you're changing public APIs
3. Add tests for new functionality
4. Keep PRs focused - one feature or fix per PR
5. Write clear commit messages that explain the "why"
6. Sign every commit (see [Signed Commits](#signed-commits)) — commits must be Verified
7. Reference any related issues in the PR description

## Reporting Issues

When reporting issues, please open a GitHub issue and include:

- A clear description of the problem
- Steps to reproduce
- Expected behavior
- Actual behavior
- Rust version (`rustc --version`)
- Operating system

## Feature Requests

Feature requests are welcome. Open a GitHub issue describing:

- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

## License

By contributing to this repository, you agree that your contributions
will be licensed under the [MIT License](LICENSE) that covers the
project. Only contribute code, assets, or documentation that you have
the right to submit under that license; do not add third-party material
unless its license is compatible and you retain any required notices.
