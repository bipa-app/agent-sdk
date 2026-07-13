//! Fallback parsing of retry hints embedded in provider error bodies.
//!
//! The `Retry-After` header is the primary source of a 429's backoff delay
//! (see [`crate::http::retry_after_from_headers`]), but some providers omit it
//! and state the delay in the error payload instead:
//!
//! * Google (Gemini / Vertex) attaches a `google.rpc.RetryInfo` entry to the
//!   error `details` array, whose `retryDelay` is a proto-JSON duration string
//!   (`"30s"`, `"3.5s"`).
//! * `OpenAI` embeds the delay in prose in the error message
//!   (`"Please try again in 20s."`, `"try again in 250ms"`).
//!
//! Both parsers are strict: they only accept a duration in the position the
//! provider documents it, so an unrelated number elsewhere in the body (a
//! quota value, an organization id) is never mistaken for a delay.

use std::time::Duration;

/// Longest delay these parsers will report.
///
/// Callers clamp the hint against their own retry ceiling anyway; this bound
/// only keeps an absurd or hostile body (`"retryDelay": "1e18s"`) from
/// producing a `Duration` that overflows later arithmetic.
const MAX_HINT: Duration = Duration::from_hours(24);

/// Parse `google.rpc.RetryInfo.retryDelay` out of a Google API error body.
///
/// Shape (429 `RESOURCE_EXHAUSTED`):
///
/// ```json
/// {"error":{"code":429,"details":[
///   {"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"30s"}
/// ]}}
/// ```
///
/// Returns `None` when the body is not JSON, carries no `RetryInfo` detail, or
/// the `retryDelay` is not a proto-JSON duration.
pub fn google_retry_delay(body: &str) -> Option<Duration> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    let details = value.get("error")?.get("details")?.as_array()?;
    details
        .iter()
        .filter(|detail| {
            detail
                .get("@type")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|type_url| type_url.ends_with("google.rpc.RetryInfo"))
        })
        .find_map(|detail| {
            let delay = detail.get("retryDelay")?.as_str()?;
            parse_proto_duration(delay)
        })
}

/// Parse an `OpenAI` rate-limit message's embedded "try again in …" delay.
///
/// The delay is read from the JSON body's `error.message` when the body parses
/// as an API error, and from the raw text otherwise (some gateways return the
/// prose alone). Recognized units are `h`, `m`, `s`, and `ms`, optionally
/// combined (`1m30s`) and optionally fractional (`1.5s`).
pub fn openai_retry_delay(body: &str) -> Option<Duration> {
    const MARKER: &str = "try again in ";

    let message = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("error")?
                .get("message")?
                .as_str()
                .map(str::to_owned)
        });
    let haystack = message.as_deref().unwrap_or(body);

    let start = haystack.to_ascii_lowercase().find(MARKER)? + MARKER.len();
    let token: String = haystack[start..]
        .chars()
        .take_while(|c| !c.is_whitespace())
        .collect();
    // The phrase usually ends a sentence ("… in 20s."); a fractional value keeps
    // its dot because trimming only ever removes trailing characters.
    parse_unit_duration(token.trim_end_matches(['.', ',', ';', ')', '"', '\'']))
}

/// Parse a proto-JSON duration: decimal seconds with a mandatory `s` suffix.
///
/// A zero delay is reported as absent: it carries no information the caller's
/// own backoff does not already have, and honouring it would retry instantly.
fn parse_proto_duration(value: &str) -> Option<Duration> {
    let seconds = value.trim().strip_suffix('s')?;
    let delay = duration_from_secs(parse_non_negative(seconds)?)?;
    (delay > Duration::ZERO).then_some(delay)
}

/// Parse a Go-style duration token: one or more `<number><unit>` pairs.
///
/// A bare number with no unit is rejected — without a unit the value is
/// ambiguous, and accepting it would let unrelated digits become a delay.
fn parse_unit_duration(token: &str) -> Option<Duration> {
    let mut rest = token;
    let mut total = Duration::ZERO;
    while !rest.is_empty() {
        let unit_start = rest.find(|c: char| !c.is_ascii_digit() && c != '.')?;
        let (number, tail) = rest.split_at(unit_start);
        let value = parse_non_negative(number)?;
        let (unit_len, seconds_per_unit) = if tail.starts_with("ms") {
            (2, 0.001)
        } else if tail.starts_with('s') {
            (1, 1.0)
        } else if tail.starts_with('m') {
            (1, 60.0)
        } else if tail.starts_with('h') {
            (1, 3600.0)
        } else {
            return None;
        };
        total = total.checked_add(duration_from_secs(value * seconds_per_unit)?)?;
        rest = &tail[unit_len..];
    }
    (total > Duration::ZERO).then_some(total.min(MAX_HINT))
}

fn parse_non_negative(value: &str) -> Option<f64> {
    // `f64::from_str` accepts "inf", "nan", and a leading sign; none of those
    // are a delay, and each would poison the `Duration` conversion below.
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit() || b == b'.') {
        return None;
    }
    let parsed: f64 = value.parse().ok()?;
    parsed.is_finite().then_some(parsed)
}

fn duration_from_secs(seconds: f64) -> Option<Duration> {
    Duration::try_from_secs_f64(seconds)
        .ok()
        .map(|delay| delay.min(MAX_HINT))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn google_retry_info_whole_and_fractional_seconds() {
        let body = r#"{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","details":[
            {"@type":"type.googleapis.com/google.rpc.QuotaFailure","violations":[{"quotaValue":"60"}]},
            {"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"30s"}
        ]}}"#;
        assert_eq!(google_retry_delay(body), Some(Duration::from_secs(30)));

        let fractional = r#"{"error":{"details":[
            {"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"3.5s"}
        ]}}"#;
        assert_eq!(
            google_retry_delay(fractional),
            Some(Duration::from_millis(3500))
        );
    }

    #[test]
    fn google_retry_info_absent_or_malformed_is_none() {
        // No details array at all.
        assert_eq!(
            google_retry_delay(r#"{"error":{"code":429,"message":"quota exceeded"}}"#),
            None
        );
        // Details present, but no RetryInfo entry — the QuotaFailure's numbers
        // must not be mistaken for a delay.
        assert_eq!(
            google_retry_delay(
                r#"{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.QuotaFailure","violations":[{"quotaValue":"60"}]}]}}"#
            ),
            None
        );
        // RetryInfo present, but the delay is not a proto duration.
        assert_eq!(
            google_retry_delay(
                r#"{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"soon"}]}}"#
            ),
            None
        );
        // Missing unit suffix.
        assert_eq!(
            google_retry_delay(
                r#"{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"30"}]}}"#
            ),
            None
        );
        // Not JSON at all.
        assert_eq!(google_retry_delay("429 Too Many Requests"), None);
        assert_eq!(google_retry_delay(""), None);
    }

    #[test]
    fn openai_message_seconds_millis_and_fractions() {
        let body = r#"{"error":{"message":"Rate limit reached for gpt-5 in organization org-4bqm on requests per min (RPM): Limit 3, Used 3. Please try again in 20s.","type":"rate_limit_error","code":"rate_limit_exceeded"}}"#;
        assert_eq!(openai_retry_delay(body), Some(Duration::from_secs(20)));

        let fractional = r#"{"error":{"message":"Please try again in 1.5s"}}"#;
        assert_eq!(
            openai_retry_delay(fractional),
            Some(Duration::from_millis(1500))
        );

        let millis = r#"{"error":{"message":"Please try again in 250ms."}}"#;
        assert_eq!(openai_retry_delay(millis), Some(Duration::from_millis(250)));

        // Composite Go-style duration, as returned for daily token quotas.
        let composite = r#"{"error":{"message":"Please try again in 6m0s."}}"#;
        assert_eq!(openai_retry_delay(composite), Some(Duration::from_mins(6)));

        // Prose body with no JSON envelope (proxies/gateways).
        assert_eq!(
            openai_retry_delay("Rate limited. Please try again in 2m30s"),
            Some(Duration::from_secs(150))
        );
    }

    #[test]
    fn openai_message_without_a_hint_is_none() {
        // Rate-limit prose, but no retry phrase — the limit numbers must not
        // be read as a delay.
        assert_eq!(
            openai_retry_delay(
                r#"{"error":{"message":"Rate limit reached for gpt-5: Limit 3, Used 3."}}"#
            ),
            None
        );
        // Phrase present, but the value carries no unit.
        assert_eq!(
            openai_retry_delay(r#"{"error":{"message":"Please try again in 20"}}"#),
            None
        );
        // Phrase present, but the value is not a number.
        assert_eq!(
            openai_retry_delay(r#"{"error":{"message":"Please try again in a while."}}"#),
            None
        );
        // Unknown unit.
        assert_eq!(
            openai_retry_delay(r#"{"error":{"message":"Please try again in 5d."}}"#),
            None
        );
        // Zero delay carries no information and would mask the caller's backoff.
        assert_eq!(
            openai_retry_delay(r#"{"error":{"message":"Please try again in 0s."}}"#),
            None
        );
        assert_eq!(openai_retry_delay(""), None);
    }

    #[test]
    fn absurd_hints_are_bounded() {
        assert_eq!(
            google_retry_delay(
                r#"{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"999999999999s"}]}}"#
            ),
            Some(MAX_HINT)
        );
        assert_eq!(
            openai_retry_delay(r#"{"error":{"message":"Please try again in 999999h."}}"#),
            Some(MAX_HINT)
        );
    }
}
