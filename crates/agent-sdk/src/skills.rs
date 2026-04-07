//! Skills system for loading agent behavior from markdown files.
//!
//! Skills allow you to define agent behavior, system prompts, and tool configurations
//! in markdown files with YAML frontmatter.
//!
//! # Skill File Format
//!
//! ```markdown
//! ---
//! name: code-review
//! description: Review code for quality and security
//! tools: [read, grep, glob]
//! denied_tools: [bash, write]
//! ---
//!
//! # Code Review Skill
//!
//! You are an expert code reviewer...
//! ```
//!
//! # Example
//!
//! ```ignore
//! use agent_sdk::skills::{FileSkillLoader, SkillLoader};
//!
//! let loader = FileSkillLoader::new("./skills");
//! let skill = loader.load("code-review").await?;
//!
//! let agent = builder()
//!     .provider(provider)
//!     .with_skill(skill)
//!     .build();
//! ```

pub mod loader;
pub mod parser;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A loaded skill definition.
///
/// Skills contain:
/// - A system prompt that defines agent behavior
/// - Tool configurations (which tools are available/denied)
/// - Optional metadata for custom extensions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Skill {
    /// Unique identifier for the skill.
    pub name: String,

    /// Human-readable description of what the skill does.
    pub description: String,

    /// The system prompt content (markdown body after frontmatter).
    pub system_prompt: String,

    /// List of tool names that should be enabled for this skill.
    /// If empty, all registered tools are available.
    pub tools: Vec<String>,

    /// Optional list of tools explicitly allowed (whitelist).
    /// If set, only these tools are available.
    pub allowed_tools: Option<Vec<String>>,

    /// Optional list of tools explicitly denied (blacklist).
    /// These tools will be filtered out even if in `tools` list.
    pub denied_tools: Option<Vec<String>>,

    /// Additional metadata from frontmatter.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Skill {
    /// Create a new skill with the given name and system prompt.
    #[must_use]
    pub fn new(name: impl Into<String>, system_prompt: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            system_prompt: system_prompt.into(),
            tools: Vec::new(),
            allowed_tools: None,
            denied_tools: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Set the list of tools.
    #[must_use]
    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.tools = tools;
        self
    }

    /// Set the allowed tools whitelist.
    #[must_use]
    pub fn with_allowed_tools(mut self, tools: Vec<String>) -> Self {
        self.allowed_tools = Some(tools);
        self
    }

    /// Set the denied tools blacklist.
    #[must_use]
    pub fn with_denied_tools(mut self, tools: Vec<String>) -> Self {
        self.denied_tools = Some(tools);
        self
    }

    /// Check if a tool is allowed by this skill.
    ///
    /// Returns true if:
    /// - The tool is not in `denied_tools`, AND
    /// - Either `allowed_tools` is None, or the tool is in `allowed_tools`
    #[must_use]
    pub fn is_tool_allowed(&self, tool_name: &str) -> bool {
        // Check denied list first
        if let Some(ref denied) = self.denied_tools
            && denied.iter().any(|t| t == tool_name)
        {
            return false;
        }

        // Check allowed list if set
        if let Some(ref allowed) = self.allowed_tools {
            return allowed.iter().any(|t| t == tool_name);
        }

        // No whitelist, tool is allowed
        true
    }
}

pub use loader::{FileSkillLoader, SkillLoader};
pub use parser::parse_skill_file;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skill_builder() {
        let skill = Skill::new("test", "You are a test assistant.")
            .with_description("A test skill")
            .with_tools(vec!["read".into(), "write".into()])
            .with_denied_tools(vec!["bash".into()]);

        assert_eq!(skill.name, "test");
        assert_eq!(skill.description, "A test skill");
        assert_eq!(skill.system_prompt, "You are a test assistant.");
        assert_eq!(skill.tools, vec!["read", "write"]);
        assert_eq!(skill.denied_tools, Some(vec!["bash".into()]));
    }

    #[test]
    fn test_is_tool_allowed_no_restrictions() {
        let skill = Skill::new("test", "prompt");

        assert!(skill.is_tool_allowed("read"));
        assert!(skill.is_tool_allowed("write"));
        assert!(skill.is_tool_allowed("bash"));
    }

    #[test]
    fn test_is_tool_allowed_with_denied() {
        let skill = Skill::new("test", "prompt").with_denied_tools(vec!["bash".into()]);

        assert!(skill.is_tool_allowed("read"));
        assert!(skill.is_tool_allowed("write"));
        assert!(!skill.is_tool_allowed("bash"));
    }

    #[test]
    fn test_is_tool_allowed_with_whitelist() {
        let skill =
            Skill::new("test", "prompt").with_allowed_tools(vec!["read".into(), "grep".into()]);

        assert!(skill.is_tool_allowed("read"));
        assert!(skill.is_tool_allowed("grep"));
        assert!(!skill.is_tool_allowed("write"));
        assert!(!skill.is_tool_allowed("bash"));
    }

    #[test]
    fn test_is_tool_allowed_denied_takes_precedence() {
        let skill = Skill::new("test", "prompt")
            .with_allowed_tools(vec!["read".into(), "bash".into()])
            .with_denied_tools(vec!["bash".into()]);

        assert!(skill.is_tool_allowed("read"));
        assert!(!skill.is_tool_allowed("bash")); // Denied takes precedence
    }
}
