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
