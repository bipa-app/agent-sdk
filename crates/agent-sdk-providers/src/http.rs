//! Small HTTP helpers shared by the provider implementations.

use std::time::Duration;

use agent_sdk_foundation::llm::parse_retry_after;
use reqwest::header::{HeaderMap, RETRY_AFTER};

/// Read and parse a `Retry-After` header off a response's headers.
///
/// Returns the parsed backoff [`Duration`] when the header is present and
/// carries a value this SDK understands (delta-seconds or an IMF-fixdate in
/// the future); `None` otherwise. Call this on a 429 response **before** the
/// body is consumed, since `reqwest`'s `bytes()` takes the response by value.
#[must_use]
pub fn retry_after_from_headers(headers: &HeaderMap) -> Option<Duration> {
    let value = headers.get(RETRY_AFTER)?;
    parse_retry_after(value.to_str().ok()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::HeaderValue;

    #[test]
    fn reads_delta_seconds_header() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_static("42"));
        assert_eq!(
            retry_after_from_headers(&headers),
            Some(Duration::from_secs(42))
        );
    }

    #[test]
    fn missing_header_is_none() {
        assert_eq!(retry_after_from_headers(&HeaderMap::new()), None);
    }
}
