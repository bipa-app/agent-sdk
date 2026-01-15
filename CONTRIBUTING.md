# Contributing to Agent SDK

Thank you for your interest in contributing to Agent SDK! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/agent-sdk.git`
3. Create a feature branch: `git checkout -b my-feature`
4. Make your changes
5. Run the quality checks (see below)
6. Commit your changes
7. Push to your fork and submit a pull request

## Development Setup

### Prerequisites

- Rust 1.85+ (2024 edition)
- Cargo

### Building

```bash
cargo build
```

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

## Pull Request Process

1. Ensure all quality checks pass
2. Update documentation if you're changing public APIs
3. Add tests for new functionality
4. Keep PRs focused - one feature or fix per PR
5. Write clear commit messages that explain the "why"
6. Reference any related issues in the PR description

## Reporting Issues

When reporting issues, please include:

- A clear description of the problem
- Steps to reproduce
- Expected behavior
- Actual behavior
- Rust version (`rustc --version`)
- Operating system

## Feature Requests

Feature requests are welcome! Please open an issue describing:

- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

## License

By contributing to Agent SDK, you agree that your contributions will be licensed under the Apache License, Version 2.0.
