//! Skill file parser for markdown with YAML frontmatter.
//!
//! This parser supports skill files from multiple coding agents:
//! - Claude Code style (YAML frontmatter with markdown body)
//! - Cursor style (similar YAML frontmatter)
//! - Amp style (may include `system_prompt` in frontmatter)
//! - Codex style (may use `id` instead of `name`)
//!
//! The parser handles common field name variations:
//! - `name`, `id`, `title` -> name
//! - `description`, `desc`, `summary` -> description
//! - `system_prompt`, `prompt`, `instructions` -> can be in frontmatter
//! - `tools`, `allowed_tools`, `denied_tools` -> tool configuration

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::HashMap;

use super::Skill;

/// Frontmatter structure parsed from YAML.
///
/// Supports multiple naming conventions for compatibility with
/// Claude Code, Cursor, Amp, and Codex skill formats.
#[derive(Debug, Deserialize)]
pub struct SkillFrontmatter {
    /// Skill name - supports `name`, `id`, or `title`.
    #[serde(alias = "id", alias = "title")]
    pub name: Option<String>,

    /// Skill description - supports `description`, `desc`, or `summary`.
    #[serde(default, alias = "desc", alias = "summary")]
    pub description: Option<String>,

    /// System prompt in frontmatter (Amp style).
    /// If present, overrides the markdown body.
    #[serde(default, alias = "prompt", alias = "instructions")]
    pub system_prompt: Option<String>,

    /// List of tools to enable (optional).
    #[serde(default)]
    pub tools: Vec<String>,

    /// Whitelist of allowed tools (optional).
    /// Also supports `enabled_tools` alias.
    #[serde(default, alias = "enabled_tools")]
    pub allowed_tools: Option<Vec<String>>,

    /// Blacklist of denied tools (optional).
    /// Also supports `disabled_tools` or `blocked_tools` alias.
    #[serde(default, alias = "disabled_tools", alias = "blocked_tools")]
    pub denied_tools: Option<Vec<String>>,

    /// Additional metadata fields.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Parse a skill file content into frontmatter and body.
///
/// The file format is:
/// ```text
/// ---
/// name: skill-name
/// description: Optional description
/// tools: [tool1, tool2]
/// ---
///
/// # Markdown content here
///
/// This becomes the system prompt.
/// ```
///
/// # Compatibility
///
/// This parser supports multiple skill file formats:
/// - **Claude Code**: Standard YAML frontmatter + markdown body
/// - **Cursor**: Similar format, may use `title` instead of `name`
/// - **Amp**: May include `system_prompt` or `instructions` in frontmatter
/// - **Codex**: May use `id` instead of `name`
///
/// # Errors
///
/// Returns an error if:
/// - The file doesn't start with `---`
/// - The YAML frontmatter is invalid
/// - Required fields are missing (must have `name`, `id`, or `title`)
pub fn parse_skill_file(content: &str) -> Result<Skill> {
    let content = content.trim();

    // Check for frontmatter delimiter
    if !content.starts_with("---") {
        bail!("Skill file must start with YAML frontmatter (---)");
    }

    // Find the closing delimiter
    let after_first = &content[3..];
    let end_index = after_first
        .find("---")
        .context("Missing closing frontmatter delimiter (---)")?;

    let yaml_content = &after_first[..end_index].trim();
    let body = after_first[end_index + 3..].trim();

    // Parse YAML frontmatter
    let frontmatter: SkillFrontmatter =
        serde_yaml::from_str(yaml_content).context("Failed to parse YAML frontmatter")?;

    // Name is required (can come from name, id, or title via aliases)
    let name = frontmatter
        .name
        .context("Skill must have a 'name', 'id', or 'title' field")?;

    // System prompt: prefer frontmatter field, fall back to body
    let system_prompt = frontmatter
        .system_prompt
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| body.to_string());

    // Extra fields are already serde_json::Value from the flatten
    let metadata: HashMap<String, serde_json::Value> = frontmatter.extra;

    Ok(Skill {
        name,
        description: frontmatter.description.unwrap_or_default(),
        system_prompt,
        tools: frontmatter.tools,
        allowed_tools: frontmatter.allowed_tools,
        denied_tools: frontmatter.denied_tools,
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_skill() -> Result<()> {
        let content = "---
name: test-skill
description: A test skill
---

You are a helpful assistant.
";

        let skill = parse_skill_file(content)?;

        assert_eq!(skill.name, "test-skill");
        assert_eq!(skill.description, "A test skill");
        assert_eq!(skill.system_prompt, "You are a helpful assistant.");
        assert!(skill.tools.is_empty());
        assert!(skill.allowed_tools.is_none());
        assert!(skill.denied_tools.is_none());

        Ok(())
    }

    #[test]
    fn test_parse_skill_with_tools() -> Result<()> {
        let content = "---
name: code-review
description: Review code for quality
tools:
  - read
  - grep
  - glob
denied_tools:
  - bash
  - write
---

# Code Review

You are an expert code reviewer.

## Guidelines

1. Check for security issues
2. Look for performance problems
";

        let skill = parse_skill_file(content)?;

        assert_eq!(skill.name, "code-review");
        assert_eq!(skill.description, "Review code for quality");
        assert_eq!(skill.tools, vec!["read", "grep", "glob"]);
        assert_eq!(skill.denied_tools, Some(vec!["bash".into(), "write".into()]));
        assert!(skill.system_prompt.contains("# Code Review"));
        assert!(skill.system_prompt.contains("## Guidelines"));

        Ok(())
    }

    #[test]
    fn test_parse_skill_with_allowed_tools() -> Result<()> {
        let content = "---
name: restricted
allowed_tools:
  - read
  - grep
---

Only read operations allowed.
";

        let skill = parse_skill_file(content)?;

        assert_eq!(skill.name, "restricted");
        assert_eq!(
            skill.allowed_tools,
            Some(vec!["read".into(), "grep".into()])
        );

        Ok(())
    }

    #[test]
    fn test_parse_skill_with_extra_metadata() -> Result<()> {
        let content = "---
name: custom
version: \"1.0\"
author: test
custom_field: 42
---

Custom skill.
";

        let skill = parse_skill_file(content)?;

        assert_eq!(skill.name, "custom");
        assert_eq!(skill.metadata.get("version").and_then(|v| v.as_str()), Some("1.0"));
        assert_eq!(skill.metadata.get("author").and_then(|v| v.as_str()), Some("test"));
        assert_eq!(skill.metadata.get("custom_field").and_then(serde_json::Value::as_i64), Some(42));

        Ok(())
    }

    #[test]
    fn test_parse_missing_frontmatter() {
        let content = "No frontmatter here";
        let result = parse_skill_file(content);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must start with"));
    }

    #[test]
    fn test_parse_missing_closing_delimiter() {
        let content = "---
name: broken
";
        let result = parse_skill_file(content);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("closing frontmatter"));
    }

    #[test]
    fn test_parse_invalid_yaml() {
        let content = "---
name: [invalid yaml
---

Body
";
        let result = parse_skill_file(content);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_missing_name() {
        let content = "---
description: No name field
---

Body
";
        let result = parse_skill_file(content);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_body() -> Result<()> {
        let content = "---
name: minimal
---
";

        let skill = parse_skill_file(content)?;

        assert_eq!(skill.name, "minimal");
        assert!(skill.system_prompt.is_empty());

        Ok(())
    }

    #[test]
    fn test_parse_preserves_markdown_formatting() -> Result<()> {
        let content = r#"---
name: formatted
---

# Header

- List item 1
- List item 2

```rust
fn main() {
    println!("Hello");
}
```

**Bold** and *italic* text.
"#;

        let skill = parse_skill_file(content)?;

        assert!(skill.system_prompt.contains("# Header"));
        assert!(skill.system_prompt.contains("- List item 1"));
        assert!(skill.system_prompt.contains("```rust"));
        assert!(skill.system_prompt.contains("**Bold**"));

        Ok(())
    }

    // ==========================================
    // Compatibility tests for other skill formats
    // ==========================================

    #[test]
    fn test_parse_with_id_instead_of_name() -> Result<()> {
        // Codex-style: uses `id` instead of `name`
        let content = "---
id: codex-skill
description: A Codex-style skill
---

Codex instructions here.
";

        let skill = parse_skill_file(content)?;

        assert_eq!(skill.name, "codex-skill");
        assert_eq!(skill.description, "A Codex-style skill");

        Ok(())
    }

    #[test]
    fn test_parse_with_title_instead_of_name() -> Result<()> {
        // Cursor-style: uses `title` instead of `name`
        let content = "---
title: cursor-skill
summary: A Cursor-style skill
---

Cursor instructions here.
";

        let skill = parse_skill_file(content)?;

        assert_eq!(skill.name, "cursor-skill");
        assert_eq!(skill.description, "A Cursor-style skill");

        Ok(())
    }

    #[test]
    fn test_parse_with_system_prompt_in_frontmatter() -> Result<()> {
        // Amp-style: system_prompt in frontmatter
        let content = "---
name: amp-skill
system_prompt: This is the system prompt from frontmatter.
---

This body is ignored when system_prompt is in frontmatter.
";

        let skill = parse_skill_file(content)?;

        assert_eq!(skill.name, "amp-skill");
        assert_eq!(
            skill.system_prompt,
            "This is the system prompt from frontmatter."
        );

        Ok(())
    }

    #[test]
    fn test_parse_with_instructions_alias() -> Result<()> {
        // Alternative: uses `instructions` for system prompt
        let content = "---
name: instructions-skill
instructions: Use these instructions.
---

Body ignored.
";

        let skill = parse_skill_file(content)?;

        assert_eq!(skill.system_prompt, "Use these instructions.");

        Ok(())
    }

    #[test]
    fn test_parse_with_enabled_disabled_tools() -> Result<()> {
        // Alternative tool naming
        let content = "---
name: tool-aliases
enabled_tools:
  - read
  - grep
disabled_tools:
  - bash
---

Body content.
";

        let skill = parse_skill_file(content)?;

        assert_eq!(
            skill.allowed_tools,
            Some(vec!["read".into(), "grep".into()])
        );
        assert_eq!(skill.denied_tools, Some(vec!["bash".into()]));

        Ok(())
    }

    #[test]
    fn test_parse_empty_system_prompt_in_frontmatter_uses_body() -> Result<()> {
        // If system_prompt is empty in frontmatter, use body
        let content = "---
name: empty-prompt
system_prompt: \"\"
---

This body should be used.
";

        let skill = parse_skill_file(content)?;

        assert_eq!(skill.system_prompt, "This body should be used.");

        Ok(())
    }
}
