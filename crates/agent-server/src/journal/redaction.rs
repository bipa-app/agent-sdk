//! Baseline redaction policy for tool audit records.
//!
//! Tool inputs and outputs may contain sensitive data (passwords, API
//! keys, tokens, connection strings, and — importantly for financial
//! workloads — card PANs, CPFs, CNPJs, emails, phone numbers) that
//! should not be stored in durable audit records without explicit
//! redaction. This module provides:
//!
//! - [`RedactionPolicy`] — configurable redaction rules with three
//!   levels: [`None`](RedactionLevel::None),
//!   [`Baseline`](RedactionLevel::Baseline), and
//!   [`Full`](RedactionLevel::Full).
//! - [`redact_value`] — applies redaction rules to a JSON value,
//!   replacing sensitive keys with a `[REDACTED]` marker and masking
//!   entity PII in string leaves with `[REDACTED:<category>]`.
//! - [`redact_string`] / [`redact_error`] — apply redaction rules to
//!   plain strings.
//!
//! # Baseline policy
//!
//! The [`RedactionPolicy::baseline`] constructor returns a policy
//! that composes two redaction layers:
//!
//! 1. **Structural** — JSON object keys matching sensitive names
//!    (`password`, `secret`, `token`, `api_key`, `authorization`,
//!    `credential`, `cpf`, `cnpj`, etc.) wholesale-redact their
//!    value. String values that *start with* a sensitive prefix
//!    (`Bearer `, `sk-`, `ghp_`, `AKIA…`) are likewise wholesale
//!    redacted.
//! 2. **Entity-level** — a [`PiiDetector`] scans every remaining
//!    string leaf for emails, E.164 phones, credit card PANs (Luhn
//!    validated), Brazilian CPFs and CNPJs (mod-11 validated), Pix
//!    UUID keys, IPv4 addresses, JWTs, and embedded credential
//!    tokens. Detected spans are replaced with
//!    `[REDACTED:<category>]` while the surrounding context stays
//!    intact. This catches PII that leaks into freeform text (e.g.
//!    a PAN mentioned in a tool response) without wrecking
//!    debuggability.
//!
//! The detector defaults to
//! [`BaselineDetector`](agent_sdk_core::privacy::BaselineDetector).
//! Callers can plug in a custom detector by assigning
//! [`RedactionPolicy::detector`] directly.
//!
//! # Serialisation
//!
//! [`RedactionPolicy`] is `Serialize` + `Deserialize`. The detector
//! is skipped on serialize and re-populated with the process-wide
//! baseline on deserialize — policies persisted to disk retain
//! their levels and pattern lists, and the runtime detector is
//! rebound on load.
//!
//! # Usage
//!
//! ```ignore
//! use agent_server::journal::redaction::{RedactionPolicy, redact_value};
//!
//! let policy = RedactionPolicy::baseline();
//! let input = serde_json::json!({
//!     "command": "echo hello",
//!     "api_key": "sk-abc123",
//!     "note": "CPF 111.444.777-35 on file"
//! });
//! let redacted = redact_value(&input, &policy);
//! // redacted["api_key"] == "[REDACTED]"       (sensitive key)
//! // redacted["command"] == "echo hello"       (no PII)
//! // redacted["note"] contains "[REDACTED:cpf]" (entity mask)
//! ```

use agent_sdk_core::privacy::{BaselineDetector, NoopDetector, PiiDetector, mask_spans};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, LazyLock};

/// Redaction marker used for wholesale redaction (sensitive key
/// match or full-string secret prefix). Entity-level masks use
/// `[REDACTED:<category>]` — see [`agent_sdk_core::privacy`].
pub const REDACTED_MARKER: &str = "[REDACTED]";

/// Shared baseline detector. Compiled lazily on first use; cloning
/// the `Arc` is a single atomic inc.
static BASELINE_DETECTOR: LazyLock<Arc<dyn PiiDetector>> = LazyLock::new(|| {
    BaselineDetector::new().map_or_else(
        |_| Arc::new(NoopDetector) as Arc<dyn PiiDetector>,
        |d| Arc::new(d) as Arc<dyn PiiDetector>,
    )
});

/// Shared noop detector.
static NOOP_DETECTOR: LazyLock<Arc<dyn PiiDetector>> =
    LazyLock::new(|| Arc::new(NoopDetector) as Arc<dyn PiiDetector>);

/// Default detector used when a policy is deserialised without an
/// embedded detector (which is always, since the field is
/// `#[serde(skip)]`).
fn default_detector() -> Arc<dyn PiiDetector> {
    BASELINE_DETECTOR.clone()
}

// ─────────────────────────────────────────────────────────────────────
// Redaction level
// ─────────────────────────────────────────────────────────────────────

/// How aggressively to redact a given field category.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RedactionLevel {
    /// No redaction — store full values as-is.
    None,
    /// Redact values whose keys match sensitive patterns.
    Baseline,
    /// Full redaction — all values replaced with [`REDACTED_MARKER`].
    Full,
}

// ─────────────────────────────────────────────────────────────────────
// Redaction policy
// ─────────────────────────────────────────────────────────────────────

/// Configurable redaction rules for tool audit records.
///
/// Each field category (input, output, error) has its own
/// [`RedactionLevel`]. At [`Baseline`](RedactionLevel::Baseline) the
/// policy composes two layers:
///
/// 1. Structural — [`sensitive_key_patterns`](Self::sensitive_key_patterns)
///    triggers wholesale replacement of JSON object values by key
///    name, and [`sensitive_value_prefixes`](Self::sensitive_value_prefixes)
///    does the same for strings that *start with* a known prefix.
/// 2. Entity-level — [`detector`](Self::detector) scans every
///    remaining string leaf for emails, PANs, CPFs, CNPJs, etc. and
///    masks the spans it finds in place.
///
/// The detector is a runtime object not persisted across
/// serialisation; on deserialise it is rebound to the process-wide
/// baseline ([`agent_sdk_core::privacy::BaselineDetector`]).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RedactionPolicy {
    /// Redaction level for tool input values.
    pub input_level: RedactionLevel,
    /// Redaction level for tool output values.
    pub output_level: RedactionLevel,
    /// Redaction level for error detail strings.
    pub error_level: RedactionLevel,
    /// Key substrings that trigger redaction at baseline level.
    /// Case-insensitive matching.
    pub sensitive_key_patterns: Vec<String>,
    /// String patterns in values that trigger redaction at baseline
    /// level (e.g. `"Bearer "`, `"sk-"`). Case-sensitive prefix match.
    pub sensitive_value_prefixes: Vec<String>,
    /// Entity-level PII detector applied at baseline. Defaults to
    /// [`BaselineDetector`]; assign directly to plug in a custom
    /// implementation.
    #[serde(skip, default = "default_detector")]
    pub detector: Arc<dyn PiiDetector>,
}

impl RedactionPolicy {
    /// Baseline redaction policy suitable for production audit logs.
    ///
    /// Redacts JSON object keys that look like credentials and
    /// string values that look like tokens wholesale, and masks
    /// entity-level PII (emails, PANs, CPFs, CNPJs, Pix UUIDs,
    /// E.164 phones, IPs, JWTs) detected anywhere in remaining
    /// string leaves. Preserves non-sensitive structural data for
    /// debugging.
    #[must_use]
    pub fn baseline() -> Self {
        Self {
            input_level: RedactionLevel::Baseline,
            output_level: RedactionLevel::Baseline,
            error_level: RedactionLevel::None,
            sensitive_key_patterns: vec![
                "password".into(),
                "passwd".into(),
                "secret".into(),
                "token".into(),
                "api_key".into(),
                "apikey".into(),
                "authorization".into(),
                "credential".into(),
                "private_key".into(),
                "private".into(),
                "access_key".into(),
                "session".into(),
                "cookie".into(),
                "bearer".into(),
                "ssn".into(),
                "credit_card".into(),
                "cpf".into(),
                "cnh".into(),
                "cnpj".into(),
                "crm".into(),
                "passport".into(),
                "driver_license".into(),
                "social_security".into(),
                "social_security_number".into(),
            ],
            sensitive_value_prefixes: vec![
                "Bearer ".into(),
                "sk-".into(),
                "pk-".into(),
                "xox".into(),
                "ghp_".into(),
                "gho_".into(),
                "github_pat_".into(),
                "AKIA".into(),
            ],
            detector: default_detector(),
        }
    }

    /// No-redaction policy — stores all values as-is.
    ///
    /// Suitable only for development and testing. Never use in
    /// production audit logs.
    #[must_use]
    pub fn none() -> Self {
        Self {
            input_level: RedactionLevel::None,
            output_level: RedactionLevel::None,
            error_level: RedactionLevel::None,
            sensitive_key_patterns: Vec::new(),
            sensitive_value_prefixes: Vec::new(),
            detector: NOOP_DETECTOR.clone(),
        }
    }

    /// Full-redaction policy — replaces all input/output/error content.
    ///
    /// Suitable for high-security environments where no tool data
    /// should be stored in audit logs.
    #[must_use]
    pub fn full() -> Self {
        Self {
            input_level: RedactionLevel::Full,
            output_level: RedactionLevel::Full,
            error_level: RedactionLevel::Full,
            sensitive_key_patterns: Vec::new(),
            sensitive_value_prefixes: Vec::new(),
            detector: NOOP_DETECTOR.clone(),
        }
    }

    /// Check whether a JSON key matches any sensitive key pattern
    /// (case-insensitive substring match).
    #[must_use]
    fn is_sensitive_key(&self, key: &str) -> bool {
        let lower = key.to_lowercase();
        self.sensitive_key_patterns
            .iter()
            .any(|pattern| lower.contains(pattern.as_str()))
    }

    /// Check whether a string value matches any sensitive value prefix.
    #[must_use]
    fn is_sensitive_value(&self, value: &str) -> bool {
        self.sensitive_value_prefixes
            .iter()
            .any(|prefix| value.starts_with(prefix.as_str()))
    }
}

impl Default for RedactionPolicy {
    fn default() -> Self {
        Self::baseline()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Redaction functions
// ─────────────────────────────────────────────────────────────────────

/// Apply redaction rules to a JSON value based on the given policy's
/// input level.
///
/// - [`None`](RedactionLevel::None): returns the value unchanged.
/// - [`Baseline`](RedactionLevel::Baseline): recursively walks JSON
///   objects and redacts values whose keys match sensitive patterns,
///   or string values that match sensitive value prefixes.
/// - [`Full`](RedactionLevel::Full): returns `json!("[REDACTED]")`.
#[must_use]
pub fn redact_value(value: &serde_json::Value, policy: &RedactionPolicy) -> serde_json::Value {
    apply_redaction(value, policy.input_level, policy)
}

/// Apply redaction rules to a string value based on the given policy's
/// output level.
///
/// - [`None`](RedactionLevel::None): returns the string unchanged.
/// - [`Baseline`](RedactionLevel::Baseline): wholesale-masks if the
///   string matches any sensitive value prefix; otherwise applies
///   entity detection and masks individual PII spans
///   (`[REDACTED:<category>]`) while preserving surrounding context.
/// - [`Full`](RedactionLevel::Full): returns `"[REDACTED]"`.
#[must_use]
pub fn redact_string(value: &str, policy: &RedactionPolicy) -> String {
    match policy.output_level {
        RedactionLevel::None => value.to_owned(),
        RedactionLevel::Baseline => baseline_redact_str(value, policy),
        RedactionLevel::Full => REDACTED_MARKER.to_owned(),
    }
}

/// Apply redaction rules to an error string based on the given policy's
/// error level. Same semantics as [`redact_string`], but gated by
/// [`RedactionPolicy::error_level`].
#[must_use]
pub fn redact_error(value: &str, policy: &RedactionPolicy) -> String {
    match policy.error_level {
        RedactionLevel::None => value.to_owned(),
        RedactionLevel::Baseline => baseline_redact_str(value, policy),
        RedactionLevel::Full => REDACTED_MARKER.to_owned(),
    }
}

/// Shared baseline redaction for a plain string: prefix-match first
/// (wholesale), then entity detection (span-level).
fn baseline_redact_str(value: &str, policy: &RedactionPolicy) -> String {
    if policy.is_sensitive_value(value) {
        return REDACTED_MARKER.to_owned();
    }
    let spans = policy.detector.detect(value);
    if spans.is_empty() {
        value.to_owned()
    } else {
        mask_spans(value, &spans)
    }
}

/// Internal recursive redaction for JSON values.
fn apply_redaction(
    value: &serde_json::Value,
    level: RedactionLevel,
    policy: &RedactionPolicy,
) -> serde_json::Value {
    match level {
        RedactionLevel::None => value.clone(),
        RedactionLevel::Full => serde_json::json!(REDACTED_MARKER),
        RedactionLevel::Baseline => redact_baseline(value, policy),
    }
}

/// Baseline redaction: recursively walk JSON and redact sensitive
/// keys (wholesale), sensitive value prefixes (wholesale), and any
/// entity-level PII detected within remaining string leaves
/// (span-level).
fn redact_baseline(value: &serde_json::Value, policy: &RedactionPolicy) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut redacted = serde_json::Map::new();
            for (key, val) in map {
                if policy.is_sensitive_key(key) {
                    redacted.insert(key.clone(), serde_json::json!(REDACTED_MARKER));
                } else {
                    redacted.insert(key.clone(), redact_baseline(val, policy));
                }
            }
            serde_json::Value::Object(redacted)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(|v| redact_baseline(v, policy)).collect())
        }
        serde_json::Value::String(s) => {
            if policy.is_sensitive_value(s) {
                return serde_json::json!(REDACTED_MARKER);
            }
            let spans = policy.detector.detect(s);
            if spans.is_empty() {
                value.clone()
            } else {
                serde_json::Value::String(mask_spans(s, &spans))
            }
        }
        _ => value.clone(),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── RedactionLevel ──────────────────────────────────────────

    #[test]
    fn redaction_level_round_trips_through_json() -> anyhow::Result<()> {
        for level in [
            RedactionLevel::None,
            RedactionLevel::Baseline,
            RedactionLevel::Full,
        ] {
            let json = serde_json::to_string(&level)?;
            let back: RedactionLevel = serde_json::from_str(&json)?;
            assert_eq!(back, level);
        }
        Ok(())
    }

    // ── RedactionPolicy construction ────────────────────────────

    #[test]
    fn baseline_policy_has_expected_defaults() {
        let policy = RedactionPolicy::baseline();
        assert_eq!(policy.input_level, RedactionLevel::Baseline);
        assert_eq!(policy.output_level, RedactionLevel::Baseline);
        assert_eq!(policy.error_level, RedactionLevel::None);
        assert!(!policy.sensitive_key_patterns.is_empty());
        assert!(!policy.sensitive_value_prefixes.is_empty());
    }

    #[test]
    fn none_policy_has_no_redaction() {
        let policy = RedactionPolicy::none();
        assert_eq!(policy.input_level, RedactionLevel::None);
        assert_eq!(policy.output_level, RedactionLevel::None);
        assert_eq!(policy.error_level, RedactionLevel::None);
    }

    #[test]
    fn full_policy_redacts_everything() {
        let policy = RedactionPolicy::full();
        assert_eq!(policy.input_level, RedactionLevel::Full);
        assert_eq!(policy.output_level, RedactionLevel::Full);
        assert_eq!(policy.error_level, RedactionLevel::Full);
    }

    #[test]
    fn policy_round_trips_through_json() -> anyhow::Result<()> {
        let policy = RedactionPolicy::baseline();
        let json = serde_json::to_string(&policy)?;
        let back: RedactionPolicy = serde_json::from_str(&json)?;
        assert_eq!(back.input_level, policy.input_level);
        assert_eq!(
            back.sensitive_key_patterns.len(),
            policy.sensitive_key_patterns.len(),
        );
        Ok(())
    }

    // ── redact_value: none level ────────────────────────────────

    #[test]
    fn none_level_preserves_all_values() {
        let policy = RedactionPolicy::none();
        let input = serde_json::json!({
            "password": "secret123",
            "api_key": "sk-abc",
            "normal": "hello",
        });
        let result = redact_value(&input, &policy);
        assert_eq!(result, input);
    }

    // ── redact_value: full level ────────────────────────────────

    #[test]
    fn full_level_redacts_entire_value() {
        let policy = RedactionPolicy::full();
        let input = serde_json::json!({
            "command": "echo hello",
            "data": [1, 2, 3],
        });
        let result = redact_value(&input, &policy);
        assert_eq!(result, serde_json::json!(REDACTED_MARKER));
    }

    // ── redact_value: baseline level ────────────────────────────

    #[test]
    fn baseline_redacts_sensitive_keys() {
        let policy = RedactionPolicy::baseline();
        let input = serde_json::json!({
            "command": "echo hello",
            "password": "secret123",
            "api_key": "sk-abc",
            "normal_field": "visible",
        });
        let result = redact_value(&input, &policy);

        assert_eq!(result["command"], "echo hello");
        assert_eq!(result["password"], REDACTED_MARKER);
        assert_eq!(result["api_key"], REDACTED_MARKER);
        assert_eq!(result["normal_field"], "visible");
    }

    #[test]
    fn baseline_redacts_case_insensitively() {
        let policy = RedactionPolicy::baseline();
        let input = serde_json::json!({
            "Password": "secret",
            "API_KEY": "key",
            "Authorization": "Bearer xyz",
        });
        let result = redact_value(&input, &policy);

        assert_eq!(result["Password"], REDACTED_MARKER);
        assert_eq!(result["API_KEY"], REDACTED_MARKER);
        assert_eq!(result["Authorization"], REDACTED_MARKER);
    }

    #[test]
    fn baseline_redacts_sensitive_value_prefixes() {
        let policy = RedactionPolicy::baseline();
        let input = serde_json::json!({
            "header": "Bearer eyJ...",
            "key": "sk-abc123",
            "normal": "just a string",
        });
        let result = redact_value(&input, &policy);

        assert_eq!(result["header"], REDACTED_MARKER);
        assert_eq!(result["key"], REDACTED_MARKER);
        assert_eq!(result["normal"], "just a string");
    }

    #[test]
    fn baseline_recurses_into_nested_objects() {
        let policy = RedactionPolicy::baseline();
        let input = serde_json::json!({
            "config": {
                "api_key": "sk-nested",
                "endpoint": "https://example.com",
            },
            "name": "test",
        });
        let result = redact_value(&input, &policy);

        assert_eq!(result["config"]["api_key"], REDACTED_MARKER);
        assert_eq!(result["config"]["endpoint"], "https://example.com");
        assert_eq!(result["name"], "test");
    }

    #[test]
    fn baseline_recurses_into_arrays() {
        let policy = RedactionPolicy::baseline();
        let input = serde_json::json!([
            {"password": "secret", "name": "test"},
            {"token": "abc", "data": 42},
        ]);
        let result = redact_value(&input, &policy);

        assert_eq!(result[0]["password"], REDACTED_MARKER);
        assert_eq!(result[0]["name"], "test");
        assert_eq!(result[1]["token"], REDACTED_MARKER);
        assert_eq!(result[1]["data"], 42);
    }

    #[test]
    fn baseline_preserves_non_string_values() {
        let policy = RedactionPolicy::baseline();
        let input = serde_json::json!({
            "count": 42,
            "active": true,
            "ratio": 2.72,
            "empty": null,
        });
        let result = redact_value(&input, &policy);
        assert_eq!(result, input);
    }

    // ── redact_string ───────────────────────────────────────────

    #[test]
    fn redact_string_none_preserves() {
        let policy = RedactionPolicy::none();
        assert_eq!(redact_string("Bearer token123", &policy), "Bearer token123");
    }

    #[test]
    fn redact_string_baseline_masks_sensitive() {
        let policy = RedactionPolicy::baseline();
        assert_eq!(redact_string("Bearer token123", &policy), REDACTED_MARKER);
        assert_eq!(redact_string("sk-abc123", &policy), REDACTED_MARKER);
        assert_eq!(
            redact_string("just normal output", &policy),
            "just normal output"
        );
    }

    #[test]
    fn redact_string_full_masks_everything() {
        let policy = RedactionPolicy::full();
        assert_eq!(
            redact_string("totally safe output", &policy),
            REDACTED_MARKER
        );
    }

    // ── redact_error ────────────────────────────────────────────

    #[test]
    fn redact_error_none_preserves() {
        let policy = RedactionPolicy::baseline();
        assert_eq!(
            redact_error("connection timeout after 30s", &policy),
            "connection timeout after 30s"
        );
    }

    #[test]
    fn redact_error_full_masks() {
        let policy = RedactionPolicy::full();
        assert_eq!(
            redact_error("internal error details", &policy),
            REDACTED_MARKER
        );
    }

    // ── Sensitive key detection ─────────────────────────────────

    #[test]
    fn sensitive_key_detection() {
        let policy = RedactionPolicy::baseline();

        // Should match
        assert!(policy.is_sensitive_key("password"));
        assert!(policy.is_sensitive_key("user_password"));
        assert!(policy.is_sensitive_key("api_key"));
        assert!(policy.is_sensitive_key("MY_API_KEY"));
        assert!(policy.is_sensitive_key("Authorization"));
        assert!(policy.is_sensitive_key("session_id"));
        assert!(policy.is_sensitive_key("private_key"));
        assert!(policy.is_sensitive_key("access_key_id"));

        // Should not match — guards against overly-short substring patterns
        assert!(!policy.is_sensitive_key("username"));
        assert!(!policy.is_sensitive_key("command"));
        assert!(!policy.is_sensitive_key("amount"));
        assert!(!policy.is_sensitive_key("path"));
        assert!(!policy.is_sensitive_key("args"));
        assert!(!policy.is_sensitive_key("target"));
        assert!(!policy.is_sensitive_key("author"));
        assert!(!policy.is_sensitive_key("org_id"));
        assert!(!policy.is_sensitive_key("merge"));
    }

    // ── Sensitive value detection ───────────────────────────────

    #[test]
    fn sensitive_value_detection() {
        let policy = RedactionPolicy::baseline();

        // Should match
        assert!(policy.is_sensitive_value("Bearer eyJhbGciOiJIUzI1NiJ9"));
        assert!(policy.is_sensitive_value("sk-abc123def456"));
        assert!(policy.is_sensitive_value("ghp_xxxxxxxxxxxx"));
        assert!(policy.is_sensitive_value("xoxb-token-value"));
        assert!(policy.is_sensitive_value("AKIAIOSFODNN7EXAMPLE"));

        // Should not match
        assert!(!policy.is_sensitive_value("hello world"));
        assert!(!policy.is_sensitive_value("echo test"));
        assert!(!policy.is_sensitive_value("123.45"));
    }

    // ── Edge cases ──────────────────────────────────────────────

    #[test]
    fn redact_empty_object() {
        let policy = RedactionPolicy::baseline();
        let input = serde_json::json!({});
        let result = redact_value(&input, &policy);
        assert_eq!(result, serde_json::json!({}));
    }

    #[test]
    fn redact_empty_array() {
        let policy = RedactionPolicy::baseline();
        let input = serde_json::json!([]);
        let result = redact_value(&input, &policy);
        assert_eq!(result, serde_json::json!([]));
    }

    #[test]
    fn redact_scalar_string() {
        let policy = RedactionPolicy::baseline();
        let input = serde_json::json!("sk-secret");
        let result = redact_value(&input, &policy);
        assert_eq!(result, serde_json::json!(REDACTED_MARKER));
    }

    #[test]
    fn redact_scalar_number() {
        let policy = RedactionPolicy::baseline();
        let input = serde_json::json!(42);
        let result = redact_value(&input, &policy);
        assert_eq!(result, serde_json::json!(42));
    }

    #[test]
    fn deeply_nested_redaction() {
        let policy = RedactionPolicy::baseline();
        let input = serde_json::json!({
            "level1": {
                "level2": {
                    "level3": {
                        "api_key": "sk-deep",
                        "value": "safe",
                    }
                }
            }
        });
        let result = redact_value(&input, &policy);
        assert_eq!(
            result["level1"]["level2"]["level3"]["api_key"],
            REDACTED_MARKER,
        );
        assert_eq!(result["level1"]["level2"]["level3"]["value"], "safe");
    }

    // ── Entity-level detection via the plugged-in PiiDetector ──

    #[test]
    fn baseline_masks_email_in_non_sensitive_string_value() {
        let policy = RedactionPolicy::baseline();
        let input = serde_json::json!({
            "note": "forward to ana.silva+bipa@bipa.exchange please"
        });
        let result = redact_value(&input, &policy);
        let note = result["note"].as_str().expect("note is string");
        assert!(note.contains("[REDACTED:email]"), "got: {note}");
        assert!(!note.contains("ana.silva+bipa@bipa.exchange"));
    }

    #[test]
    fn baseline_masks_cpf_in_freeform_text() {
        let policy = RedactionPolicy::baseline();
        let input = serde_json::json!({
            "description": "confirmou pelo CPF 111.444.777-35 ontem"
        });
        let result = redact_value(&input, &policy);
        let desc = result["description"].as_str().expect("desc is string");
        assert!(desc.contains("[REDACTED:cpf]"), "got: {desc}");
        assert!(!desc.contains("111.444.777-35"));
    }

    #[test]
    fn baseline_masks_cnpj_in_freeform_text() {
        let policy = RedactionPolicy::baseline();
        let input = serde_json::json!({
            "description": "pagar CNPJ 11.222.333/0001-81 até sexta"
        });
        let result = redact_value(&input, &policy);
        let desc = result["description"].as_str().expect("desc is string");
        assert!(desc.contains("[REDACTED:cnpj]"), "got: {desc}");
    }

    #[test]
    fn baseline_masks_luhn_valid_pan_in_tool_output() {
        let policy = RedactionPolicy::baseline();
        let output = "charged card 4111 1111 1111 1111 successfully for 150 BRL";
        let result = redact_string(output, &policy);
        assert!(result.contains("[REDACTED:credit_card]"), "got: {result}");
        assert!(!result.contains("4111 1111 1111 1111"));
    }

    #[test]
    fn baseline_does_not_mask_luhn_invalid_digits() {
        // 16 digits that aren't Luhn-valid — must not be flagged as a PAN.
        let policy = RedactionPolicy::baseline();
        let output = "order 1234 5678 9012 3456 processed";
        let result = redact_string(output, &policy);
        assert!(
            !result.contains("[REDACTED:"),
            "false positive on non-PAN digits: {result}"
        );
    }

    #[test]
    fn baseline_masks_embedded_secret_token() {
        // The wholesale prefix check only fires when the WHOLE string
        // starts with a prefix. Embedded secrets rely on the entity
        // detector's SecretDetector component.
        let policy = RedactionPolicy::baseline();
        let output = "deploy failed: key=sk-abcdefghijklmnopqrstuv rejected";
        let result = redact_string(output, &policy);
        assert!(result.contains("[REDACTED:secret]"), "got: {result}");
    }

    #[test]
    fn baseline_preserves_wholesale_prefix_behaviour() {
        // A string that STARTS with a sensitive prefix still falls
        // into the wholesale `[REDACTED]` path — entity detection
        // does not override that stronger signal.
        let policy = RedactionPolicy::baseline();
        let result = redact_string("sk-abc123def456ghi789jkl", &policy);
        assert_eq!(result, REDACTED_MARKER);
    }

    #[test]
    fn baseline_masks_pii_in_nested_string_leaves() {
        let policy = RedactionPolicy::baseline();
        let input = serde_json::json!({
            "audit_log": [
                {
                    "actor": "system",
                    "details": "user CPF 111.444.777-35 contacted from 192.168.1.100"
                }
            ]
        });
        let result = redact_value(&input, &policy);
        let details = result["audit_log"][0]["details"]
            .as_str()
            .expect("details string");
        assert!(details.contains("[REDACTED:cpf]"), "got: {details}");
        assert!(details.contains("[REDACTED:ip_address]"), "got: {details}");
    }

    #[test]
    fn sensitive_key_match_wins_over_entity_detection() {
        // Values under a sensitive key still get wholesale `[REDACTED]`
        // — not a partial entity mask. Preserves the pre-upgrade
        // contract.
        let policy = RedactionPolicy::baseline();
        let input = serde_json::json!({
            "api_key": "sk-leaky",
            "access_token": "Bearer eyJ..."
        });
        let result = redact_value(&input, &policy);
        assert_eq!(result["api_key"], REDACTED_MARKER);
        assert_eq!(result["access_token"], REDACTED_MARKER);
    }

    #[test]
    fn none_policy_performs_no_entity_detection() {
        let policy = RedactionPolicy::none();
        let input = serde_json::json!({
            "note": "CPF 111.444.777-35 email a@b.co"
        });
        let result = redact_value(&input, &policy);
        assert_eq!(result, input, "none policy must not mutate input");
    }

    #[test]
    fn deserialized_policy_retains_baseline_entity_detection() -> anyhow::Result<()> {
        // The detector field is `#[serde(skip)]`. After a round-trip
        // through JSON, the policy must still perform entity
        // detection via the default BaselineDetector.
        let policy = RedactionPolicy::baseline();
        let json = serde_json::to_string(&policy)?;
        let back: RedactionPolicy = serde_json::from_str(&json)?;
        let result = redact_string("pix para CPF 111.444.777-35 agora", &back);
        assert!(
            result.contains("[REDACTED:cpf]"),
            "deserialized policy stopped detecting CPF: {result}"
        );
        Ok(())
    }

    #[test]
    fn error_level_baseline_masks_entities_in_stack_trace() {
        // Opt-in: callers can flip error_level to Baseline and the
        // detector applies to error strings too.
        let policy = RedactionPolicy {
            error_level: RedactionLevel::Baseline,
            ..RedactionPolicy::baseline()
        };
        let trace = "NotFound: user with CPF 111.444.777-35 missing in table users";
        let result = redact_error(trace, &policy);
        assert!(result.contains("[REDACTED:cpf]"), "got: {result}");
    }
}
