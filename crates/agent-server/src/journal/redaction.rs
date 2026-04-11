//! Baseline redaction policy for tool audit records.
//!
//! Tool inputs and outputs may contain sensitive data (passwords, API
//! keys, tokens, connection strings) that should not be stored in
//! durable audit records without explicit redaction. This module
//! provides:
//!
//! - [`RedactionPolicy`] — configurable redaction rules with three
//!   levels: [`None`](RedactionLevel::None),
//!   [`Baseline`](RedactionLevel::Baseline), and
//!   [`Full`](RedactionLevel::Full).
//! - [`redact_value`] — applies redaction rules to a JSON value,
//!   replacing sensitive keys with a `[REDACTED]` marker.
//! - [`redact_string`] — applies redaction rules to a plain string,
//!   masking patterns that look like secrets.
//!
//! # Baseline policy
//!
//! The [`RedactionPolicy::baseline`] constructor returns a policy that
//! redacts common sensitive patterns:
//!
//! - JSON object keys matching sensitive patterns (`password`, `secret`,
//!   `token`, `api_key`, `authorization`, `credential`, etc.)
//! - String values that look like bearer tokens or API keys
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
//! });
//! let redacted = redact_value(&input, &policy);
//! // redacted["api_key"] == "[REDACTED]"
//! // redacted["command"] == "echo hello"
//! ```

use serde::{Deserialize, Serialize};

/// Redaction marker used to replace sensitive values.
pub const REDACTED_MARKER: &str = "[REDACTED]";

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
/// [`RedactionLevel`], and the policy carries a list of key patterns
/// that the [`Baseline`](RedactionLevel::Baseline) level uses to
/// identify sensitive values.
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
}

impl RedactionPolicy {
    /// Baseline redaction policy suitable for production audit logs.
    ///
    /// Redacts JSON object keys that look like credentials and string
    /// values that look like tokens, while preserving non-sensitive
    /// structural data for debugging.
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
        }
    }

    /// No-redaction policy — stores all values as-is.
    ///
    /// Suitable only for development and testing. Never use in
    /// production audit logs.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            input_level: RedactionLevel::None,
            output_level: RedactionLevel::None,
            error_level: RedactionLevel::None,
            sensitive_key_patterns: Vec::new(),
            sensitive_value_prefixes: Vec::new(),
        }
    }

    /// Full-redaction policy — replaces all input/output/error content.
    ///
    /// Suitable for high-security environments where no tool data
    /// should be stored in audit logs.
    #[must_use]
    pub const fn full() -> Self {
        Self {
            input_level: RedactionLevel::Full,
            output_level: RedactionLevel::Full,
            error_level: RedactionLevel::Full,
            sensitive_key_patterns: Vec::new(),
            sensitive_value_prefixes: Vec::new(),
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
/// - [`Baseline`](RedactionLevel::Baseline): masks the string if it
///   matches any sensitive value prefix.
/// - [`Full`](RedactionLevel::Full): returns `"[REDACTED]"`.
#[must_use]
pub fn redact_string(value: &str, policy: &RedactionPolicy) -> String {
    match policy.output_level {
        RedactionLevel::None => value.to_owned(),
        RedactionLevel::Baseline => {
            if policy.is_sensitive_value(value) {
                REDACTED_MARKER.to_owned()
            } else {
                value.to_owned()
            }
        }
        RedactionLevel::Full => REDACTED_MARKER.to_owned(),
    }
}

/// Apply redaction rules to an error string based on the given policy's
/// error level.
///
/// - [`None`](RedactionLevel::None): returns the string unchanged.
/// - [`Baseline`](RedactionLevel::Baseline): masks the string if it
///   matches any sensitive value prefix.
/// - [`Full`](RedactionLevel::Full): returns `"[REDACTED]"`.
#[must_use]
pub fn redact_error(value: &str, policy: &RedactionPolicy) -> String {
    match policy.error_level {
        RedactionLevel::None => value.to_owned(),
        RedactionLevel::Baseline => {
            if policy.is_sensitive_value(value) {
                REDACTED_MARKER.to_owned()
            } else {
                value.to_owned()
            }
        }
        RedactionLevel::Full => REDACTED_MARKER.to_owned(),
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

/// Baseline redaction: recursively walk JSON and redact sensitive keys
/// and values.
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
                serde_json::json!(REDACTED_MARKER)
            } else {
                value.clone()
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
}
