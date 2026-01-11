# CLAUDE.md - Agent SDK Quick Reference

This file provides essential patterns for working with this Rust codebase.

## Essential Patterns

### Modern Rust Module System (CRITICAL)

**DO NOT use `mod.rs` files!** Use the modern Rust 2018+ module style:

```
src/
├── lib.rs              # mod llm;
├── llm.rs              # pub mod types; pub mod router; + main code
└── llm/
    ├── types.rs
    └── router.rs
```

The `foo.rs` file serves as the module root for `foo/` directory. No `mod.rs` needed!

### Error Handling

```rust
use anyhow::{Result, Context, bail, ensure};

// Standard return type
pub async fn my_function() -> anyhow::Result<Output> {
    // Add context to errors
    let data = fetch_data().await.context("failed to fetch")?;

    // Early exit with ensure
    ensure!(!data.is_empty(), "data cannot be empty");

    // Bail for errors
    if invalid {
        bail!("invalid state");
    }

    Ok(output)
}
```

### Testing Philosophy

1. **Test behavior, not coverage** - focus on user-facing functionality
2. **Never change test expectations to make tests pass** - fix the code
3. **Skip trivial tests** - Rust's type system catches many bugs at compile time

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn feature_works() -> anyhow::Result<()> {
        // Setup
        let store = InMemoryStore::new();

        // Execute
        let result = my_function(&store).await?;

        // Assert
        assert_eq!(result.status, ExpectedStatus);

        Ok(())
    }
}
```

### Code Quality Workflow

**Before every commit:**

```bash
cargo check --all-targets    # Type check
cargo fmt                    # Format
cargo clippy -- -D warnings  # Lint (must pass)
cargo test                   # Tests (must pass)
```

### Self-Documenting Code

Functions should be self-documenting - avoid redundant comments:

```rust
// BAD: Comment restates function name
/// Find a user by their ID.
pub async fn find_by_id(...)

// GOOD: No comment needed, name is clear
pub async fn find_by_id(...)

// GOOD: Comment explains non-obvious behavior
/// Returns None if the thread was archived more than 24 hours ago.
pub async fn find_active_thread(...)
```

## Project Structure

```
src/
├── lib.rs              # Public API exports
├── agent_loop.rs       # Core agent orchestration
├── events.rs           # AgentEvent enum
├── types.rs            # Core types (ThreadId, Config, etc.)
├── tools.rs            # Tool trait and registry
├── hooks.rs            # Lifecycle hooks
├── stores.rs           # Persistence traits
├── environment.rs      # File/command abstraction
├── capabilities.rs     # Security model
├── filesystem.rs       # LocalFileSystem, InMemoryFileSystem
├── llm.rs              # LLM module root
├── llm/
│   ├── types.rs        # Message, ChatRequest, etc.
│   └── router.rs       # Model routing
├── primitive_tools.rs  # Primitive tools module root
├── primitive_tools/
│   ├── read.rs
│   ├── write.rs
│   ├── edit.rs
│   ├── bash.rs
│   ├── glob.rs
│   └── grep.rs
├── providers.rs        # Providers module root
└── providers/
    ├── anthropic.rs
    └── anthropic/
        └── data.rs
```

## Clippy Rules

**Never bypass clippy** with `#[allow(...)]` - fix the code instead:

```rust
// BAD: Bypassing clippy
#[allow(clippy::too_many_arguments)]
pub fn create(a: A, b: B, c: C, d: D, e: E, f: F) { }

// GOOD: Use a struct
pub struct CreateParams { pub a: A, pub b: B, /* ... */ }
pub fn create(params: CreateParams) { }
```

## Async Patterns

```rust
// Async function
pub async fn fetch_data(id: &str) -> Result<Data> {
    let response = client.get(url).await?;
    Ok(response.json().await?)
}

// Async trait (with async-trait crate)
#[async_trait]
pub trait DataFetcher {
    async fn fetch(&self, id: &str) -> Result<Data>;
}
```

## Key Conventions

1. **Use `anyhow::Result`** for all fallible functions
2. **Use `time` crate** (not `chrono`) for datetime handling
3. **Use `tracing`** for logging
4. **Prefer inline full paths** for clarity: `llm::Message::user("hi")`
5. **No unsafe code** - `#[forbid(unsafe_code)]` is enforced
6. **Never use `unwrap()` or `expect()`** - Always propagate errors with `?` and add context:

```rust
// BAD: Panics without context
let value = some_result.unwrap();
let value = some_result.expect("something failed");
let lock = mutex.lock().unwrap();

// GOOD: Propagate errors with context
let value = some_result.context("failed to get value")?;

// For RwLock/Mutex: convert poison error to anyhow error
let lock = self.data.read().ok().context("lock poisoned")?;
let lock = self.data.write().ok().context("lock poisoned")?;

// For Option types
let value = some_option.context("value was None")?;
```

This rule applies to both production code AND tests. In tests, return `Result<()>` and use `?`.
