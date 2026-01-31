//! Skill loader implementations.

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use std::path::{Path, PathBuf};

use super::{Skill, parser::parse_skill_file};

/// Trait for loading skills from various sources.
#[async_trait]
pub trait SkillLoader: Send + Sync {
    /// Load a skill by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the skill cannot be found or loaded.
    async fn load(&self, name: &str) -> Result<Skill>;

    /// List all available skill names.
    ///
    /// # Errors
    ///
    /// Returns an error if the skill list cannot be retrieved.
    async fn list(&self) -> Result<Vec<String>>;

    /// Check if a skill exists.
    async fn exists(&self, name: &str) -> bool {
        self.load(name).await.is_ok()
    }
}

/// File-based skill loader.
///
/// Loads skills from markdown files in a directory. Each skill file should be
/// named `{skill-name}.md` and contain YAML frontmatter.
///
/// # Example
///
/// ```ignore
/// let loader = FileSkillLoader::new("./skills");
/// let skill = loader.load("code-review").await?;
/// ```
pub struct FileSkillLoader {
    base_path: PathBuf,
}

impl FileSkillLoader {
    /// Create a new file-based skill loader.
    ///
    /// # Arguments
    ///
    /// * `base_path` - Directory containing skill files
    #[must_use]
    pub fn new(base_path: impl Into<PathBuf>) -> Self {
        Self {
            base_path: base_path.into(),
        }
    }

    /// Get the base path for skill files.
    #[must_use]
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Get the file path for a skill by name.
    fn skill_path(&self, name: &str) -> PathBuf {
        self.base_path.join(format!("{name}.md"))
    }
}

#[async_trait]
impl SkillLoader for FileSkillLoader {
    async fn load(&self, name: &str) -> Result<Skill> {
        let path = self.skill_path(name);

        if !path.exists() {
            bail!("Skill file not found: {}", path.display());
        }

        let content = tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("Failed to read skill file: {}", path.display()))?;

        let skill = parse_skill_file(&content)
            .with_context(|| format!("Failed to parse skill file: {}", path.display()))?;

        // Verify the parsed name matches the filename
        if skill.name != name {
            log::warn!(
                "Skill name '{}' in file doesn't match filename '{}'",
                skill.name,
                name
            );
        }

        Ok(skill)
    }

    async fn list(&self) -> Result<Vec<String>> {
        if !self.base_path.exists() {
            return Ok(Vec::new());
        }

        let mut entries = tokio::fs::read_dir(&self.base_path)
            .await
            .with_context(|| {
                format!(
                    "Failed to read skills directory: {}",
                    self.base_path.display()
                )
            })?;

        let mut skills = Vec::new();

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            if path.extension().is_some_and(|ext| ext == "md")
                && let Some(name) = path.file_stem().and_then(|s| s.to_str())
            {
                skills.push(name.to_string());
            }
        }

        skills.sort();
        Ok(skills)
    }
}

/// In-memory skill loader for testing.
///
/// Stores skills in memory rather than loading from files.
#[derive(Default)]
pub struct InMemorySkillLoader {
    skills: std::sync::RwLock<std::collections::HashMap<String, Skill>>,
}

impl InMemorySkillLoader {
    /// Create a new empty in-memory skill loader.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a skill to the loader.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal lock is poisoned.
    pub fn add(&self, skill: Skill) -> Result<()> {
        self.skills
            .write()
            .ok()
            .context("lock poisoned")?
            .insert(skill.name.clone(), skill);
        Ok(())
    }

    /// Remove a skill from the loader.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal lock is poisoned.
    pub fn remove(&self, name: &str) -> Result<Option<Skill>> {
        let mut skills = self.skills.write().ok().context("lock poisoned")?;
        Ok(skills.remove(name))
    }
}

#[async_trait]
impl SkillLoader for InMemorySkillLoader {
    async fn load(&self, name: &str) -> Result<Skill> {
        let skills = self.skills.read().ok().context("lock poisoned")?;
        skills
            .get(name)
            .cloned()
            .with_context(|| format!("Skill not found: {name}"))
    }

    async fn list(&self) -> Result<Vec<String>> {
        let mut names: Vec<_> = self
            .skills
            .read()
            .ok()
            .context("lock poisoned")?
            .keys()
            .cloned()
            .collect();
        names.sort();
        Ok(names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_file_loader_load() -> Result<()> {
        let dir = TempDir::new()?;
        let skill_path = dir.path().join("test-skill.md");

        let mut file = std::fs::File::create(&skill_path)?;
        writeln!(
            file,
            "---
name: test-skill
description: A test skill
---

You are a test assistant."
        )?;

        let loader = FileSkillLoader::new(dir.path());
        let skill = loader.load("test-skill").await?;

        assert_eq!(skill.name, "test-skill");
        assert_eq!(skill.description, "A test skill");
        assert!(skill.system_prompt.contains("test assistant"));

        Ok(())
    }

    #[tokio::test]
    async fn test_file_loader_load_not_found() {
        let dir = TempDir::new().unwrap();
        let loader = FileSkillLoader::new(dir.path());

        let result = loader.load("nonexistent").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_file_loader_list() -> Result<()> {
        let dir = TempDir::new()?;

        // Create some skill files
        for name in ["alpha", "beta", "gamma"] {
            let path = dir.path().join(format!("{name}.md"));
            let mut file = std::fs::File::create(&path)?;
            writeln!(
                file,
                "---
name: {name}
---

Content"
            )?;
        }

        // Create a non-skill file
        let _ = std::fs::File::create(dir.path().join("readme.txt"))?;

        let loader = FileSkillLoader::new(dir.path());
        let skills = loader.list().await?;

        assert_eq!(skills, vec!["alpha", "beta", "gamma"]);

        Ok(())
    }

    #[tokio::test]
    async fn test_file_loader_list_empty_dir() -> Result<()> {
        let dir = TempDir::new()?;
        let loader = FileSkillLoader::new(dir.path());

        let skills = loader.list().await?;
        assert!(skills.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_file_loader_list_nonexistent_dir() -> Result<()> {
        let loader = FileSkillLoader::new("/nonexistent/path");
        let skills = loader.list().await?;
        assert!(skills.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_file_loader_exists() -> Result<()> {
        let dir = TempDir::new()?;
        let skill_path = dir.path().join("exists.md");

        let mut file = std::fs::File::create(&skill_path)?;
        writeln!(
            file,
            "---
name: exists
---

Content"
        )?;

        let loader = FileSkillLoader::new(dir.path());

        assert!(loader.exists("exists").await);
        assert!(!loader.exists("not-exists").await);

        Ok(())
    }

    #[tokio::test]
    async fn test_in_memory_loader() -> Result<()> {
        let loader = InMemorySkillLoader::new();

        loader.add(Skill::new("skill1", "Prompt 1").with_description("First skill"))?;
        loader.add(Skill::new("skill2", "Prompt 2").with_description("Second skill"))?;

        let first = loader.load("skill1").await?;
        assert_eq!(first.name, "skill1");
        assert_eq!(first.description, "First skill");

        let skill_names = loader.list().await?;
        assert_eq!(skill_names, vec!["skill1", "skill2"]);

        loader.remove("skill1")?;
        assert!(!loader.exists("skill1").await);

        Ok(())
    }

    #[tokio::test]
    async fn test_in_memory_loader_not_found() {
        let loader = InMemorySkillLoader::new();
        let result = loader.load("nonexistent").await;
        assert!(result.is_err());
    }
}
