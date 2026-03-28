//! Primitive tools that work with the Environment abstraction.
//!
//! These tools provide basic file and command operations:
//! - `ReadTool` - Read file contents
//! - `WriteTool` - Write/create files
//! - `EditTool` - Edit existing files with string replacement
//! - `GlobTool` - Find files by pattern
//! - `GrepTool` - Search file contents
//! - `BashTool` - Execute shell commands
//!
//! All tools respect `AgentCapabilities` for security.

mod bash;
mod edit;
mod glob;
mod grep;
mod read;
mod write;

pub use bash::BashTool;
pub use edit::EditTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use read::ReadTool;
pub use write::WriteTool;

use crate::{AgentCapabilities, Environment};
use serde::Deserialize;
use serde::de::{self, Deserializer};
use std::fmt::Display;
use std::str::FromStr;
use std::sync::Arc;

/// Context for primitive tools that need environment access
pub struct PrimitiveToolContext<E: Environment> {
    pub environment: Arc<E>,
    pub capabilities: AgentCapabilities,
}

impl<E: Environment> PrimitiveToolContext<E> {
    #[must_use]
    pub const fn new(environment: Arc<E>, capabilities: AgentCapabilities) -> Self {
        Self {
            environment,
            capabilities,
        }
    }
}

impl<E: Environment> Clone for PrimitiveToolContext<E> {
    fn clone(&self) -> Self {
        Self {
            environment: Arc::clone(&self.environment),
            capabilities: self.capabilities.clone(),
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum StringOrU64 {
    Number(u64),
    String(String),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum StringOrUsize {
    Number(usize),
    String(String),
}

fn parse_numeric_string<T>(value: &str) -> Result<T, String>
where
    T: FromStr,
    T::Err: Display,
{
    value
        .trim()
        .parse::<T>()
        .map_err(|error| format!("invalid numeric string '{value}': {error}"))
}

pub(super) fn deserialize_optional_u64_from_string_or_int<'de, D>(
    deserializer: D,
) -> Result<Option<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    match Option::<StringOrU64>::deserialize(deserializer)? {
        None => Ok(None),
        Some(StringOrU64::Number(value)) => Ok(Some(value)),
        Some(StringOrU64::String(value)) => parse_numeric_string(&value)
            .map(Some)
            .map_err(de::Error::custom),
    }
}

/// Truncate a string to at most `max_bytes` without splitting a multi-byte
/// UTF-8 character. Returns the original string when it already fits.
pub(crate) fn truncate_str(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

pub(super) fn deserialize_usize_from_string_or_int<'de, D>(
    deserializer: D,
) -> Result<usize, D::Error>
where
    D: Deserializer<'de>,
{
    match StringOrUsize::deserialize(deserializer)? {
        StringOrUsize::Number(value) => Ok(value),
        StringOrUsize::String(value) => parse_numeric_string(&value).map_err(de::Error::custom),
    }
}

#[cfg(test)]
mod tests {
    use super::truncate_str;

    #[test]
    fn test_truncate_str_ascii_fits() {
        assert_eq!(truncate_str("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_str_ascii_exact() {
        assert_eq!(truncate_str("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_str_ascii_truncated() {
        assert_eq!(truncate_str("hello world", 5), "hello");
    }

    #[test]
    fn test_truncate_str_multibyte_emoji() {
        let s = "Hello 🎉 world";
        // "Hello " is 6 bytes, emoji is 4 bytes, so cutting at 8 would
        // land inside the emoji. The helper must back up to byte 6.
        let result = truncate_str(s, 8);
        assert_eq!(result, "Hello ");
    }

    #[test]
    fn test_truncate_str_cjk() {
        let s = "漢字テスト";
        // Each CJK char is 3 bytes. Truncating at 7 should give 2 chars (6 bytes).
        let result = truncate_str(s, 7);
        assert_eq!(result, "漢字");
    }

    #[test]
    fn test_truncate_str_zero_max() {
        assert_eq!(truncate_str("hello", 0), "");
    }

    #[test]
    fn test_truncate_str_empty() {
        assert_eq!(truncate_str("", 10), "");
    }
}
