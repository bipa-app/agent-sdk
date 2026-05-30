//! Crash-injection failpoints for durability testing.
//!
//! Phase 11 needs to inject failures *between* the discrete durable
//! steps of the commit path (e.g. "panic after the journal commit but
//! before the outbox insert") to prove that recovery is idempotent —
//! no lost or duplicated work. This module wires tikv's
//! [`fail-rs`](https://docs.rs/fail) registry, but **only** when the
//! crate is built with the `failpoints` feature.
//!
//! ## Zero-cost in production
//!
//! Without the `failpoints` feature:
//!
//! * `fail-rs` is not a dependency at all (it is `optional = true` and
//!   pulled in only by the `failpoints` feature).
//! * The `fail_point!` macro expands to nothing — the call sites in
//!   the commit path compile away entirely.
//!
//! With the `failpoints` feature the macro forwards to
//! `fail::fail_point!`, whose behaviour is driven at runtime by the
//! process-global registry (via `fail::cfg` or the `FAILPOINTS`
//! environment variable).
//!
//! ## Process isolation is mandatory
//!
//! The `fail-rs` registry is **process-global**. Two tests configuring
//! the same named failpoint in one process would clobber each other, so
//! every failpoint test must run under `cargo-nextest` (one process per
//! test) and is serialized by the `failpoints` test-group in
//! `.config/nextest.toml`.

/// Inject a configured failure at a named point, or compile to nothing.
///
/// With the `failpoints` feature this forwards to `fail::fail_point!`;
/// without it the invocation expands to an empty statement so production
/// builds pay nothing.
///
/// ```ignore
/// // In the durable commit path:
/// fail_point!("commit.before_event_commit");
/// ```
///
/// A test then arms it (under nextest, so the registry is process-local):
///
/// ```ignore
/// fail::cfg("commit.before_event_commit", "panic").unwrap();
/// ```
#[cfg(feature = "failpoints")]
#[macro_export]
macro_rules! fail_point {
    ($name:expr) => {{
        $crate::__fail_reexport::fail_point!($name);
    }};
    ($name:expr, $cond:expr) => {{
        $crate::__fail_reexport::fail_point!($name, $cond);
    }};
    ($name:expr, $e:expr, $($args:tt)*) => {{
        $crate::__fail_reexport::fail_point!($name, $e, $($args)*);
    }};
}

/// No-op expansion of `fail_point!` for builds without the
/// `failpoints` feature. The argument expressions are not evaluated.
#[cfg(not(feature = "failpoints"))]
#[macro_export]
macro_rules! fail_point {
    ($($arg:tt)*) => {{}};
}

#[cfg(test)]
mod tests {
    /// Without the `failpoints` feature the macro must be a true no-op:
    /// it expands to nothing and never evaluates its arguments. This
    /// test compiles and passes in default builds, proving zero-cost.
    #[cfg(not(feature = "failpoints"))]
    #[test]
    fn fail_point_is_a_noop_without_feature() {
        use std::cell::Cell;

        let side_effect = Cell::new(0);
        // If the macro evaluated its arguments, the closure would run and
        // bump the cell. It must not — the whole invocation compiles away.
        fail_point!("test.noop", {
            side_effect.set(side_effect.get() + 1);
            String::from("unused")
        });
        assert_eq!(
            side_effect.get(),
            0,
            "fail_point! must not evaluate arguments"
        );
    }

    /// With the `failpoints` feature the macro is wired to the live
    /// `fail-rs` registry. We arm a panic failpoint and confirm it
    /// fires when the named point is reached.
    ///
    /// MUST run under nextest (one process per test): the registry is
    /// process-global. The `failpoints` nextest test-group serializes it.
    #[cfg(feature = "failpoints")]
    #[test]
    fn fail_point_fires_when_armed() {
        let _scenario = fail::FailScenario::setup();
        fail::cfg("test.armed_panic", "panic").expect("configure failpoint");

        let fired = std::panic::catch_unwind(|| {
            fail_point!("test.armed_panic");
        })
        .is_err();

        assert!(fired, "armed failpoint must panic when reached");
    }

    /// With the feature on but the failpoint left unconfigured, the
    /// call site is inert — proving failpoints are opt-in per name.
    #[cfg(feature = "failpoints")]
    #[test]
    fn fail_point_is_inert_when_not_configured() {
        let _scenario = fail::FailScenario::setup();
        // Not configured → reaching it does nothing.
        fail_point!("test.unconfigured");
    }
}
