//! `prompt_cache_key` shaping shared by the `OpenAI` Chat Completions,
//! Responses, and Codex providers.

use std::borrow::Cow;

use agent_sdk_foundation::llm::sha256_hex;

/// Longest `prompt_cache_key` the `OpenAI` API accepts.
pub const MAX_PROMPT_CACHE_KEY_LEN: usize = 64;

/// Map a session id onto a `prompt_cache_key` the `OpenAI` API accepts.
///
/// Ids of up to 64 bytes pass through unchanged. Longer ids (thread ids with a
/// UUID suffix easily exceed the limit) become their SHA-256 hex digest, which
/// is exactly 64 characters and the same on every turn, so cache routing still
/// groups one session's requests.
pub fn bounded_prompt_cache_key(session_id: &str) -> Cow<'_, str> {
    if session_id.len() <= MAX_PROMPT_CACHE_KEY_LEN {
        Cow::Borrowed(session_id)
    } else {
        Cow::Owned(sha256_hex(session_id.as_bytes()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LONG_SESSION_ID: &str =
        "eval-bipa-premium-data-query-001-0b4e88a4-6f1c-4c55-9d2a-51c0e4a7f3b9";

    #[test]
    fn keys_within_the_limit_pass_through_unchanged() {
        let at_limit = "k".repeat(MAX_PROMPT_CACHE_KEY_LEN);
        for session_id in ["thread-42", at_limit.as_str()] {
            let key = bounded_prompt_cache_key(session_id);
            assert!(matches!(key, Cow::Borrowed(_)), "{session_id}");
            assert_eq!(key, session_id);
        }
    }

    #[test]
    fn long_keys_are_bounded_deterministic_and_distinct() {
        assert!(LONG_SESSION_ID.len() > MAX_PROMPT_CACHE_KEY_LEN);
        let key = bounded_prompt_cache_key(LONG_SESSION_ID);
        assert!(
            key.len() <= MAX_PROMPT_CACHE_KEY_LEN,
            "got {} chars",
            key.len()
        );
        assert_ne!(key, LONG_SESSION_ID);
        assert_eq!(key, bounded_prompt_cache_key(LONG_SESSION_ID));

        let sibling = LONG_SESSION_ID.replace("-001-", "-002-");
        assert_ne!(key, bounded_prompt_cache_key(&sibling));

        let just_over = "k".repeat(MAX_PROMPT_CACHE_KEY_LEN + 1);
        assert!(bounded_prompt_cache_key(&just_over).len() <= MAX_PROMPT_CACHE_KEY_LEN);
    }
}
