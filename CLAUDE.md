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

## Repository Layout

This is a Cargo workspace. New crates go under `crates/`.

```
Cargo.toml              # Virtual workspace manifest (no [package])
crates/
├── agent-sdk/          # Main SDK crate (published)
│   ├── Cargo.toml      # Inherits workspace deps, lints, metadata
│   ├── src/
│   │   ├── lib.rs
│   │   ├── agent_loop.rs
│   │   ├── llm.rs
│   │   ├── llm/
│   │   │   ├── types.rs
│   │   │   └── router.rs
│   │   ├── providers.rs
│   │   └── ...
│   └── examples/
└── (future crates added here during Phase 0 extraction)
```

### Workspace Conventions

* **Shared dependency versions** are declared in `[workspace.dependencies]` and
  referenced from member crates via `{ workspace = true }`.
* **Lint policy** lives in `[workspace.lints]` and members opt in with
  `[lints] workspace = true`.
* **Common package metadata** (`edition`, `license`, `repository`) is shared
  through `[workspace.package]` and inherited with `field.workspace = true`.

## sdk/v2 Rewrite Workflow

All rewrite work targets the **`sdk/v2`** branch.

* Base branch for every rewrite PR: **`sdk/v2`**
* New crates go under `crates/`
* The workspace root `Cargo.toml` is a virtual manifest — no `[package]` section

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

## sqlx and Postgres Development (CRITICAL)

### Always use typed sqlx macros

**Use `sqlx::query!` and `sqlx::query_as!` macros** for all Postgres queries in
`agent-service-host`. These macros provide compile-time SQL validation and type
checking. Never use the untyped `sqlx::query(...)` string form.

```rust
// BAD: untyped query — no compile-time checking
let row: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM agent_sdk_tasks WHERE thread_id = $1")
    .bind(thread_key(thread_id))
    .fetch_one(&self.pool)
    .await?;

// GOOD: typed macro — validated at compile time
let record = sqlx::query!(
    r"SELECT COUNT(*) AS cnt FROM agent_sdk_tasks WHERE thread_id = $1",
    thread_key(thread_id),
)
.fetch_one(&self.pool)
.await?;
let count = record.cnt.unwrap_or(0);
```

### Local Postgres via Docker Compose

A Postgres 18 instance is available via `compose.yml`:

```bash
# Start Postgres
scripts/postgres18-dev.sh up

# Wait until healthy
scripts/postgres18-dev.sh wait

# Connection URL
scripts/postgres18-dev.sh url
# → postgres://agent_sdk:agent_sdk@127.0.0.1:55432/agent_sdk
```

### sqlx Offline Cache

Normal builds use `SQLX_OFFLINE=true` (set in `.cargo/config.toml`). The
`.sqlx/` directory holds cached query metadata so builds work without a live
database.

**After adding or changing any `sqlx::query!` / `sqlx::query_as!` call, you must
refresh the offline cache:**

```bash
# Apply migrations + regenerate .sqlx/ cache
scripts/postgres18-dev.sh prepare

# Or manually:
scripts/postgres18-dev.sh wait
SQLX_OFFLINE=false DATABASE_URL="postgres://agent_sdk:agent_sdk@127.0.0.1:55432/agent_sdk" \
  cargo sqlx prepare --workspace -- -p agent-service-host --all-targets
```

### Running Postgres Integration Tests

```bash
# Full cycle: migrate + run store tests against real Postgres
scripts/postgres18-dev.sh test-migrations
```
