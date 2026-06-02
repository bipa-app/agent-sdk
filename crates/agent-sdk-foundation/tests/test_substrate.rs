//! Worked examples of the Phase 11 test substrate.
//!
//! This file is intentionally small: it exists to prove that each new
//! testing tool compiles and runs in the workspace, and to give the
//! rest of Phase 11 a copy-paste-able pattern for each. It asserts only
//! framework-level invariants of `agent-sdk-foundation` types — no runtime
//! behaviour is exercised or changed.
//!
//! Tools demonstrated:
//!
//! * `proptest` — property-based fuzzing of an invariant.
//! * `proptest-state-machine` — model-based testing against a reference
//!   model (here: a plain integer mirroring `SequenceCounter`).
//! * `insta` — snapshot of a serialized event so ordering/shape
//!   regressions show up as a reviewable diff.
//!
//! `tokio` `start_paused` virtual time is also part of this substrate, but it
//! is only meaningful against time-dependent logic, so it is exercised in the
//! `agent-service-host` scheduling/lease tests (extended in Phase 11·D) rather
//! than demonstrated here against bare `tokio` primitives.

use agent_sdk_foundation::{AgentEvent, SequenceCounter, ThreadId};

// ── proptest: property-based fuzzing ─────────────────────────────────
//
// Invariant under test: `SequenceCounter::next()` is strictly monotonic
// and contiguous regardless of the starting offset. proptest generates
// many (offset, count) pairs and auto-shrinks any counterexample.

proptest::proptest! {
    #[test]
    fn sequence_counter_is_contiguous_from_any_offset(
        offset in 0u64..1_000_000,
        count in 0usize..256,
    ) {
        let counter = SequenceCounter::with_offset(offset);
        let drawn: Vec<u64> = (0..count).map(|_| counter.next()).collect();

        for (i, value) in drawn.iter().enumerate() {
            let expected = offset + u64::try_from(i).expect("index fits u64");
            proptest::prop_assert_eq!(*value, expected);
        }
    }
}

// ── proptest-state-machine: model-based testing ──────────────────────
//
// We model `SequenceCounter` against a trivial reference model (a `u64`)
// and let the framework drive a random sequence of transitions, checking
// the system-under-test against the model after each one. The only
// transition is `Next`, whose postcondition is "returns the model's
// current value, then both advance by one".

mod state_machine {
    use agent_sdk_foundation::SequenceCounter;
    use proptest::prelude::*;
    use proptest_state_machine::{ReferenceStateMachine, StateMachineTest};

    #[derive(Clone, Debug)]
    enum Transition {
        Next,
    }

    // Reference model: the next value the counter is expected to yield.
    struct CounterModel;

    impl ReferenceStateMachine for CounterModel {
        type State = u64;
        type Transition = Transition;

        fn init_state() -> BoxedStrategy<Self::State> {
            Just(0u64).boxed()
        }

        fn transitions(_state: &Self::State) -> BoxedStrategy<Self::Transition> {
            Just(Transition::Next).boxed()
        }

        fn apply(state: Self::State, transition: &Self::Transition) -> Self::State {
            match transition {
                Transition::Next => state + 1,
            }
        }
    }

    // System under test: the real counter plus a shadow of the value the
    // model said we should next observe.
    struct CounterSut {
        counter: SequenceCounter,
        expected_next: u64,
    }

    impl StateMachineTest for CounterSut {
        type SystemUnderTest = Self;
        type Reference = CounterModel;

        fn init_test(_ref_state: &u64) -> Self::SystemUnderTest {
            Self {
                counter: SequenceCounter::new(),
                expected_next: 0,
            }
        }

        fn apply(
            mut state: Self::SystemUnderTest,
            _ref_state: &u64,
            transition: Transition,
        ) -> Self::SystemUnderTest {
            match transition {
                Transition::Next => {
                    let got = state.counter.next();
                    assert_eq!(
                        got, state.expected_next,
                        "counter diverged from reference model",
                    );
                    state.expected_next += 1;
                }
            }
            state
        }
    }

    proptest_state_machine::prop_state_machine! {
        #[test]
        fn sequence_counter_matches_reference_model(
            sequential 1..50 => CounterSut
        );
    }
}

// ── insta: snapshot testing ──────────────────────────────────────────
//
// Snapshot the serialized form of an event so any change to the wire
// shape or field ordering surfaces as a reviewable diff. The accepted
// snapshot lives in `tests/snapshots/` and is committed alongside this
// file.

#[test]
fn agent_event_start_serializes_to_stable_shape() {
    let event = AgentEvent::Start {
        thread_id: ThreadId::from_string("thread-fixture-001"),
        turn: 1,
    };

    insta::assert_json_snapshot!(event);
}
