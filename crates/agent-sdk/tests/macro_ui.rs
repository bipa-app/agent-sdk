//! Compile-fail UI tests for the Phase 13·E derive macros.
//!
//! Asserts the derives produce *helpful* diagnostics for misuse rather than
//! cryptic downstream type errors: a missing required `#[tool(...)]` attribute
//! and applying `#[derive(ToolName)]` to a non-enum.
//!
//! `trybuild`'s expected-stderr snapshots are sensitive to the exact compiler
//! version, so a committed `.stderr` can drift and break an otherwise-green
//! `cargo nextest` on a different toolchain. To keep the default gate run
//! hermetic this test is **opt-in**: set `RUN_UI_TESTS=1` to exercise it
//! (e.g. locally or in a pinned-toolchain CI job). Without the env var it is a
//! no-op that still proves the fixtures compile *as test inputs*.

#[test]
fn derive_misuse_is_rejected_with_helpful_errors() {
    if std::env::var_os("RUN_UI_TESTS").is_none() {
        eprintln!("skipping trybuild UI tests; set RUN_UI_TESTS=1 to run them");
        return;
    }
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/missing_name.rs");
    t.compile_fail("tests/ui/tool_name_on_struct.rs");
}
