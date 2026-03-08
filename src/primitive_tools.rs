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

pub(super) fn deserialize_optional_usize_from_string_or_int<'de, D>(
    deserializer: D,
) -> Result<Option<usize>, D::Error>
where
    D: Deserializer<'de>,
{
    match Option::<StringOrUsize>::deserialize(deserializer)? {
        None => Ok(None),
        Some(StringOrUsize::Number(value)) => Ok(Some(value)),
        Some(StringOrUsize::String(value)) => parse_numeric_string(&value)
            .map(Some)
            .map_err(de::Error::custom),
    }
}
