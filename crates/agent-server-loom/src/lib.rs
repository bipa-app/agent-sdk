//! Internal, test-only crate hosting scoped `loom` concurrency models
//! for `agent-server`'s lock-free journal hot spots.
//!
//! The crate has no public surface. Its sole purpose is to compile and
//! run the `loom` models in `tests/loom_models.rs` under
//! `RUSTFLAGS="--cfg loom"` against a minimal dependency closure
//! (`loom` only). See that file and the crate manifest for the full
//! rationale on why these models live here rather than inside
//! `agent-server` (short version: `--cfg loom` poisons the whole build
//! graph, and `tokio` disables `net` under loom, so any crate pulling in
//! the async networking stack cannot be built under loom).
//!
//! Run the models with:
//!
//! ```bash
//! RUSTFLAGS="--cfg loom" LOOM_MAX_PREEMPTIONS=3 \
//!   cargo test -p agent-server-loom --test loom_models
//! ```
//!
//! Without `--cfg loom` the test compiles to an empty module and runs no
//! models, so a normal `cargo test` is a fast no-op for this crate.

#[cfg(test)]
mod tests {
    /// Sentinel so a normal (non-loom) `cargo nextest run` over this
    /// crate is not an empty test binary (nextest treats zero tests as an
    /// error). The real coverage lives in `tests/loom_models.rs` and runs
    /// only under `RUSTFLAGS="--cfg loom"`.
    #[test]
    fn crate_compiles_and_hosts_loom_models() {
        // The real coverage lives in `tests/loom_models.rs`, gated behind
        // `#![cfg(loom)]` and run under `RUSTFLAGS="--cfg loom"`. This
        // sentinel just keeps the normal (non-loom) test binary non-empty
        // so `cargo nextest run` does not report zero tests for the crate.
        assert_eq!(env!("CARGO_PKG_NAME"), "agent-server-loom");
    }
}
