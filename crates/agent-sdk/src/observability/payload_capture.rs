//! Process-wide payload-capture gate.
//!
//! [`ObservabilityStore::capture`] may say
//! [`CaptureDecision::Inline`] all it wants, but the SDK forces
//! every `Inline` decision to [`CaptureDecision::Omit`] unless
//! **both** of these are true:
//!
//! 1. The operator has opted in by setting
//!    `OtelConfig::capture_payloads = true`. The
//!    `agent-sdk-otel::install_global_provider` flips this gate via
//!    [`set_enabled`] when its config says so.
//! 2. The store has explicitly attested PII safety by overriding
//!    [`ObservabilityStore::acknowledge_pii_redaction`] to return
//!    `true`. Default returns `false`.
//!
//! [`CaptureDecision::Reference`] passes through the gate
//! untouched: externalized references never leak the underlying
//! payload to a span, so they are always safe.
//!
//! [`ObservabilityStore::capture`]: super::types::ObservabilityStore::capture
//! [`CaptureDecision::Inline`]: super::types::CaptureDecision::Inline
//! [`CaptureDecision::Omit`]: super::types::CaptureDecision::Omit
//! [`CaptureDecision::Reference`]: super::types::CaptureDecision::Reference
//! [`ObservabilityStore::acknowledge_pii_redaction`]: super::types::ObservabilityStore::acknowledge_pii_redaction

use std::sync::atomic::{AtomicBool, Ordering};

use super::types::{CaptureDecision, CaptureResult, ObservabilityStore};

/// Latches once we have warned about an attesting store that left the
/// default noop redactor in place, so the warning does not spam every LLM
/// round-trip.
static NOOP_REDACTOR_WARNED: AtomicBool = AtomicBool::new(false);

/// Process-wide gate. Defaults to `false` (closed) so payloads are
/// dropped from spans by default, even when an `ObservabilityStore`
/// returns `Inline`.
static ENABLED: AtomicBool = AtomicBool::new(false);

/// Flip the global payload-capture gate.
///
/// Production code calls this exactly once, indirectly, by passing
/// `capture_payloads = true` to
/// `agent_sdk_otel::OtelConfig::capture_payloads(true)`. Tests
/// drive the gate directly and **must** restore the previous value
/// before returning so adjacent tests are not poisoned.
pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::Release);
}

/// Public alias for [`set_enabled`] re-exported under
/// [`crate::observability`] as `set_payload_capture_enabled`. Same
/// thread-safety + reset-in-tests contract.
pub fn set_payload_capture_enabled(enabled: bool) {
    set_enabled(enabled);
}

/// Read the current value of the gate.
#[must_use]
pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Acquire)
}

/// Public alias for [`is_enabled`] re-exported under
/// [`crate::observability`] as `is_payload_capture_enabled`.
#[must_use]
pub fn is_payload_capture_enabled() -> bool {
    is_enabled()
}

/// Apply the gate to a [`CaptureResult`].
///
/// Returns the result unchanged when the gate is open **and** `store` has
/// attested PII redaction; otherwise downgrades every `Inline` decision to
/// `Omit`. `Reference` and `Omit` decisions are preserved in either case.
///
/// Attestation is the store's promise that inline payloads are PII-safe — it
/// may redact them internally before persisting. As a defense-in-depth signal
/// we additionally warn (once) when an attesting store leaves the default
/// noop [`PayloadRedactor`](super::payload::PayloadRedactor), since that is a
/// common misconfiguration that lets unredacted payloads reach spans; we warn
/// rather than suppress so we never silently drop telemetry the operator
/// explicitly enabled and the store explicitly vouched for.
pub(crate) fn gate(store: &dyn ObservabilityStore, result: CaptureResult) -> CaptureResult {
    if is_enabled() && store.acknowledge_pii_redaction() {
        if store.redactor().is_noop() {
            warn_attesting_store_has_noop_redactor();
        }
        return result;
    }
    CaptureResult {
        system_instructions: downgrade(result.system_instructions),
        input_messages: downgrade(result.input_messages),
        output_messages: downgrade(result.output_messages),
    }
}

/// Warn (once) that a store attested PII redaction but installs the default
/// noop redactor, so its `Inline` payloads may reach spans unredacted unless
/// the store redacts them internally.
fn warn_attesting_store_has_noop_redactor() {
    if NOOP_REDACTOR_WARNED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        log::warn!(
            "observability store attested PII redaction (acknowledge_pii_redaction = true) but \
             installs the default noop PayloadRedactor; its inline payloads may carry unredacted \
             PII onto spans. Install a non-noop PayloadRedactor, or ensure the store redacts \
             internally before attesting."
        );
    }
}

fn downgrade(decision: CaptureDecision) -> CaptureDecision {
    match decision {
        // `Reference` is always safe — the store has externalised
        // the payload behind an opaque ID, no PII reaches the span.
        // Pass it through unchanged.
        CaptureDecision::Reference(r) => CaptureDecision::Reference(r),
        // `Inline` is the dangerous one we need to suppress, and
        // `Omit` is already the safe default; both collapse to
        // `Omit` to keep the function total.
        CaptureDecision::Inline | CaptureDecision::Omit => CaptureDecision::Omit,
    }
}

#[cfg(test)]
mod tests {
    use super::super::payload::PayloadRedactor;
    use super::super::types::{CaptureDecision, CaptureResult, ObservabilityStore, PayloadBundle};
    use super::*;
    use agent_sdk_foundation::privacy::BaselineDetector;
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};

    /// Tests share the global `ENABLED` flag; serialise them so
    /// concurrent runners don't read each other's transient state.
    static GATE_LOCK: Mutex<()> = Mutex::new(());

    fn lock_gate() -> std::sync::MutexGuard<'static, ()> {
        match GATE_LOCK.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        }
    }

    /// Restore the gate to whatever value it had before a test
    /// started so adjacent tests are not poisoned by the leaked
    /// state.
    struct GateGuard {
        previous: bool,
    }

    impl GateGuard {
        fn enable() -> Self {
            let previous = is_enabled();
            set_enabled(true);
            Self { previous }
        }

        fn disable() -> Self {
            let previous = is_enabled();
            set_enabled(false);
            Self { previous }
        }
    }

    impl Drop for GateGuard {
        fn drop(&mut self) {
            set_enabled(self.previous);
        }
    }

    struct DefaultStore;

    #[async_trait]
    impl ObservabilityStore for DefaultStore {
        async fn capture(&self, _bundle: &PayloadBundle) -> anyhow::Result<CaptureResult> {
            Ok(CaptureResult {
                system_instructions: CaptureDecision::Inline,
                input_messages: CaptureDecision::Inline,
                output_messages: CaptureDecision::Inline,
            })
        }
    }

    struct AttestingStore {
        redactor: PayloadRedactor,
    }

    impl AttestingStore {
        /// Attesting store with a real (non-noop) redactor installed —
        /// the only configuration that may pass `Inline` through.
        fn with_real_redactor() -> anyhow::Result<Self> {
            Ok(Self {
                redactor: PayloadRedactor::new(Arc::new(BaselineDetector::new()?)),
            })
        }

        /// Attesting store that (incorrectly) left the default noop
        /// redactor in place.
        fn with_noop_redactor() -> Self {
            Self {
                redactor: PayloadRedactor::noop(),
            }
        }
    }

    #[async_trait]
    impl ObservabilityStore for AttestingStore {
        async fn capture(&self, _bundle: &PayloadBundle) -> anyhow::Result<CaptureResult> {
            Ok(inline_result())
        }

        fn redactor(&self) -> &PayloadRedactor {
            &self.redactor
        }

        fn acknowledge_pii_redaction(&self) -> bool {
            true
        }
    }

    fn inline_result() -> CaptureResult {
        CaptureResult {
            system_instructions: CaptureDecision::Inline,
            input_messages: CaptureDecision::Inline,
            output_messages: CaptureDecision::Inline,
        }
    }

    fn assert_all_omit(result: &CaptureResult) {
        assert!(matches!(result.system_instructions, CaptureDecision::Omit));
        assert!(matches!(result.input_messages, CaptureDecision::Omit));
        assert!(matches!(result.output_messages, CaptureDecision::Omit));
    }

    fn assert_all_inline(result: &CaptureResult) {
        assert!(matches!(
            result.system_instructions,
            CaptureDecision::Inline
        ));
        assert!(matches!(result.input_messages, CaptureDecision::Inline));
        assert!(matches!(result.output_messages, CaptureDecision::Inline));
    }

    #[test]
    fn default_store_with_gate_open_still_omits_inline() {
        let _g = lock_gate();
        let _gate = GateGuard::enable();
        let result = gate(&DefaultStore, inline_result());
        assert_all_omit(&result);
    }

    #[test]
    fn default_store_with_gate_closed_omits_inline() {
        let _g = lock_gate();
        let _gate = GateGuard::disable();
        let result = gate(&DefaultStore, inline_result());
        assert_all_omit(&result);
    }

    #[test]
    fn attesting_store_with_gate_closed_still_omits_inline() {
        let _g = lock_gate();
        let _gate = GateGuard::disable();
        let result = gate(&AttestingStore::with_noop_redactor(), inline_result());
        assert_all_omit(&result);
    }

    #[test]
    fn attesting_store_with_real_redactor_and_gate_open_passes_inline_through() -> anyhow::Result<()>
    {
        let _g = lock_gate();
        let _gate = GateGuard::enable();
        let store = AttestingStore::with_real_redactor()?;
        let result = gate(&store, inline_result());
        assert_all_inline(&result);
        Ok(())
    }

    #[test]
    fn attesting_store_with_noop_redactor_passes_inline_but_is_a_warned_misconfig() {
        // The store attests `acknowledge_pii_redaction = true` but left the
        // default noop redactor in place. The gate honors the store's
        // explicit attestation (it may redact internally) and passes Inline
        // through, while warning once about the likely misconfiguration —
        // it does not silently suppress operator-enabled, store-vouched
        // telemetry.
        let _g = lock_gate();
        let _gate = GateGuard::enable();
        let result = gate(&AttestingStore::with_noop_redactor(), inline_result());
        assert_all_inline(&result);
    }

    #[test]
    fn references_pass_through_when_gate_is_closed() {
        let _g = lock_gate();
        let _gate = GateGuard::disable();
        let input = CaptureResult {
            system_instructions: CaptureDecision::Reference("sys-1".into()),
            input_messages: CaptureDecision::Reference("in-1".into()),
            output_messages: CaptureDecision::Omit,
        };
        let result = gate(&DefaultStore, input);
        assert!(matches!(
            result.system_instructions,
            CaptureDecision::Reference(ref r) if r == "sys-1"
        ));
        assert!(matches!(
            result.input_messages,
            CaptureDecision::Reference(ref r) if r == "in-1"
        ));
        assert!(matches!(result.output_messages, CaptureDecision::Omit));
    }
}
