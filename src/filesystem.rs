//! Filesystem implementations for the Environment trait.
//!
//! Provides:
//! - `LocalFileSystem` - Standard filesystem operations using `std::fs`
//! - `InMemoryFileSystem` - In-memory filesystem for testing

use crate::environment::{Environment, ExecResult, FileEntry, GrepMatch};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

/// Local filesystem implementation using `std::fs`
pub struct LocalFileSystem {
    root: PathBuf,
}

impl LocalFileSystem {
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn resolve(&self, path: &str) -> PathBuf {
        if Path::new(path).is_absolute() {
            PathBuf::from(path)
        } else {
            self.root.join(path)
        }
    }
}

#[async_trait]
impl Environment for LocalFileSystem {
    async fn read_file(&self, path: &str) -> Result<String> {
        let path = self.resolve(path);
        tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("Failed to read file: {}", path.display()))
    }

    async fn read_file_bytes(&self, path: &str) -> Result<Vec<u8>> {
        let path = self.resolve(path);
        tokio::fs::read(&path)
            .await
            .with_context(|| format!("Failed to read file: {}", path.display()))
    }

    async fn write_file(&self, path: &str, content: &str) -> Result<()> {
        let path = self.resolve(path);
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&path, content)
            .await
            .with_context(|| format!("Failed to write file: {}", path.display()))
    }

    async fn write_file_bytes(&self, path: &str, content: &[u8]) -> Result<()> {
        let path = self.resolve(path);
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&path, content)
            .await
            .with_context(|| format!("Failed to write file: {}", path.display()))
    }

    async fn list_dir(&self, path: &str) -> Result<Vec<FileEntry>> {
        let path = self.resolve(path);
        let mut entries = Vec::new();
        let mut dir = tokio::fs::read_dir(&path)
            .await
            .with_context(|| format!("Failed to read directory: {}", path.display()))?;

        while let Some(entry) = dir.next_entry().await? {
            let metadata = entry.metadata().await?;
            entries.push(FileEntry {
                name: entry.file_name().to_string_lossy().to_string(),
                path: entry.path().to_string_lossy().to_string(),
                is_dir: metadata.is_dir(),
                size: if metadata.is_file() {
                    Some(metadata.len())
                } else {
                    None
                },
            });
        }

        Ok(entries)
    }

    async fn exists(&self, path: &str) -> Result<bool> {
        let path = self.resolve(path);
        Ok(tokio::fs::try_exists(&path).await.unwrap_or(false))
    }

    async fn is_dir(&self, path: &str) -> Result<bool> {
        let path = self.resolve(path);
        Ok(tokio::fs::metadata(&path)
            .await
            .map(|m| m.is_dir())
            .unwrap_or(false))
    }

    async fn is_file(&self, path: &str) -> Result<bool> {
        let path = self.resolve(path);
        Ok(tokio::fs::metadata(&path)
            .await
            .map(|m| m.is_file())
            .unwrap_or(false))
    }

    async fn create_dir(&self, path: &str) -> Result<()> {
        let path = self.resolve(path);
        tokio::fs::create_dir_all(&path)
            .await
            .with_context(|| format!("Failed to create directory: {}", path.display()))
    }

    async fn delete_file(&self, path: &str) -> Result<()> {
        let path = self.resolve(path);
        tokio::fs::remove_file(&path)
            .await
            .with_context(|| format!("Failed to delete file: {}", path.display()))
    }

    async fn delete_dir(&self, path: &str, recursive: bool) -> Result<()> {
        let path = self.resolve(path);
        if recursive {
            tokio::fs::remove_dir_all(&path)
                .await
                .with_context(|| format!("Failed to delete directory: {}", path.display()))
        } else {
            tokio::fs::remove_dir(&path)
                .await
                .with_context(|| format!("Failed to delete directory: {}", path.display()))
        }
    }

    async fn grep(&self, pattern: &str, path: &str, recursive: bool) -> Result<Vec<GrepMatch>> {
        let path = self.resolve(path);
        let regex = regex::Regex::new(pattern).context("Invalid regex pattern")?;
        let mut matches = Vec::new();

        if path.is_file() {
            self.grep_file(&path, &regex, &mut matches).await?;
        } else if path.is_dir() {
            self.grep_dir(&path, &regex, recursive, &mut matches)
                .await?;
        }

        Ok(matches)
    }

    async fn glob(&self, pattern: &str) -> Result<Vec<String>> {
        let pattern_path = self.resolve(pattern);
        let pattern_str = pattern_path.to_string_lossy();

        let paths: Vec<String> = glob::glob(&pattern_str)
            .context("Invalid glob pattern")?
            .filter_map(std::result::Result::ok)
            .map(|p| p.to_string_lossy().to_string())
            .collect();

        Ok(paths)
    }

    async fn exec(&self, command: &str, timeout_ms: Option<u64>) -> Result<ExecResult> {
        use std::process::Stdio;
        use tokio::process::Command;

        let timeout = std::time::Duration::from_millis(timeout_ms.unwrap_or(120_000));

        let output = tokio::time::timeout(
            timeout,
            Command::new("sh")
                .arg("-c")
                .arg(command)
                .current_dir(&self.root)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output(),
        )
        .await
        .context("Command timed out")?
        .context("Failed to execute command")?;

        Ok(ExecResult {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code().unwrap_or(-1),
        })
    }

    fn root(&self) -> &str {
        self.root.to_str().unwrap_or("/")
    }
}

impl LocalFileSystem {
    async fn grep_file(
        &self,
        path: &Path,
        regex: &regex::Regex,
        matches: &mut Vec<GrepMatch>,
    ) -> Result<()> {
        let content = tokio::fs::read_to_string(path).await?;
        for (line_num, line) in content.lines().enumerate() {
            if let Some(m) = regex.find(line) {
                matches.push(GrepMatch {
                    path: path.to_string_lossy().to_string(),
                    line_number: line_num + 1,
                    line_content: line.to_string(),
                    match_start: m.start(),
                    match_end: m.end(),
                });
            }
        }
        Ok(())
    }

    async fn grep_dir(
        &self,
        start_dir: &Path,
        regex: &regex::Regex,
        recursive: bool,
        matches: &mut Vec<GrepMatch>,
    ) -> Result<()> {
        // Use an iterative approach with explicit queue to avoid stack overflow
        let mut dirs_to_process = vec![start_dir.to_path_buf()];

        while let Some(dir) = dirs_to_process.pop() {
            let Ok(mut entries) = tokio::fs::read_dir(&dir).await else {
                continue; // Skip directories we can't read
            };

            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                let Ok(metadata) = entry.metadata().await else {
                    continue;
                };

                if metadata.is_file() {
                    // Skip binary files (simple heuristic)
                    if let Ok(content) = tokio::fs::read(&path).await
                        && content.iter().take(1024).any(|&b| b == 0)
                    {
                        continue; // Skip binary
                    }
                    let _ = self.grep_file(&path, regex, matches).await;
                } else if metadata.is_dir() && recursive {
                    dirs_to_process.push(path);
                }
            }
        }
        Ok(())
    }
}

/// In-memory filesystem for testing
pub struct InMemoryFileSystem {
    root: String,
    files: RwLock<HashMap<String, Vec<u8>>>,
    dirs: RwLock<std::collections::HashSet<String>>,
}

impl InMemoryFileSystem {
    #[must_use]
    pub fn new(root: impl Into<String>) -> Self {
        let root = root.into();
        let dirs = RwLock::new({
            let mut set = std::collections::HashSet::new();
            set.insert(root.clone());
            set
        });
        Self {
            root,
            files: RwLock::new(HashMap::new()),
            dirs,
        }
    }

    fn normalize_path(&self, path: &str) -> String {
        if path.starts_with('/') {
            path.to_string()
        } else {
            format!("{}/{}", self.root.trim_end_matches('/'), path)
        }
    }

    fn parent_dir(path: &str) -> Option<String> {
        Path::new(path)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
    }
}

#[async_trait]
impl Environment for InMemoryFileSystem {
    async fn read_file(&self, path: &str) -> Result<String> {
        let path = self.normalize_path(path);
        self.files
            .read()
            .ok()
            .context("lock poisoned")?
            .get(&path)
            .map(|bytes| String::from_utf8_lossy(bytes).to_string())
            .ok_or_else(|| anyhow::anyhow!("File not found: {path}"))
    }

    async fn read_file_bytes(&self, path: &str) -> Result<Vec<u8>> {
        let path = self.normalize_path(path);
        self.files
            .read()
            .ok()
            .context("lock poisoned")?
            .get(&path)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("File not found: {path}"))
    }

    async fn write_file(&self, path: &str, content: &str) -> Result<()> {
        self.write_file_bytes(path, content.as_bytes()).await
    }

    async fn write_file_bytes(&self, path: &str, content: &[u8]) -> Result<()> {
        let path = self.normalize_path(path);

        // Create parent directories
        if let Some(parent) = Self::parent_dir(&path) {
            self.create_dir(&parent).await?;
        }

        self.files
            .write()
            .ok()
            .context("lock poisoned")?
            .insert(path, content.to_vec());
        Ok(())
    }

    async fn list_dir(&self, path: &str) -> Result<Vec<FileEntry>> {
        let path = self.normalize_path(path);
        let prefix = format!("{}/", path.trim_end_matches('/'));
        let mut entries = Vec::new();

        // Check if directory exists and collect file entries
        {
            let dirs = self.dirs.read().ok().context("lock poisoned")?;
            if !dirs.contains(&path) {
                anyhow::bail!("Directory not found: {path}");
            }

            // Find subdirectories
            for dir_path in dirs.iter() {
                if dir_path.starts_with(&prefix) && dir_path != &path {
                    let relative = &dir_path[prefix.len()..];
                    if !relative.contains('/') {
                        entries.push(FileEntry {
                            name: relative.to_string(),
                            path: dir_path.clone(),
                            is_dir: true,
                            size: None,
                        });
                    }
                }
            }
        }

        // Find files in this directory
        {
            let files = self.files.read().ok().context("lock poisoned")?;
            for (file_path, content) in files.iter() {
                if file_path.starts_with(&prefix) {
                    let relative = &file_path[prefix.len()..];
                    if !relative.contains('/') {
                        entries.push(FileEntry {
                            name: relative.to_string(),
                            path: file_path.clone(),
                            is_dir: false,
                            size: Some(content.len() as u64),
                        });
                    }
                }
            }
        }

        Ok(entries)
    }

    async fn exists(&self, path: &str) -> Result<bool> {
        let path = self.normalize_path(path);
        let in_files = self
            .files
            .read()
            .ok()
            .context("lock poisoned")?
            .contains_key(&path);
        let in_dirs = self
            .dirs
            .read()
            .ok()
            .context("lock poisoned")?
            .contains(&path);
        Ok(in_files || in_dirs)
    }

    async fn is_dir(&self, path: &str) -> Result<bool> {
        let path = self.normalize_path(path);
        Ok(self
            .dirs
            .read()
            .ok()
            .context("lock poisoned")?
            .contains(&path))
    }

    async fn is_file(&self, path: &str) -> Result<bool> {
        let path = self.normalize_path(path);
        Ok(self
            .files
            .read()
            .ok()
            .context("lock poisoned")?
            .contains_key(&path))
    }

    async fn create_dir(&self, path: &str) -> Result<()> {
        let path = self.normalize_path(path);

        // Collect all parent directories first
        let mut current = String::new();
        let dirs_to_create: Vec<String> = path
            .split('/')
            .filter(|p| !p.is_empty())
            .map(|part| {
                current = format!("{current}/{part}");
                current.clone()
            })
            .collect();

        // Insert all directories at once
        for dir in dirs_to_create {
            self.dirs.write().ok().context("lock poisoned")?.insert(dir);
        }

        Ok(())
    }

    async fn delete_file(&self, path: &str) -> Result<()> {
        let path = self.normalize_path(path);
        self.files
            .write()
            .ok()
            .context("lock poisoned")?
            .remove(&path)
            .ok_or_else(|| anyhow::anyhow!("File not found: {path}"))?;
        Ok(())
    }

    async fn delete_dir(&self, path: &str, recursive: bool) -> Result<()> {
        let path = self.normalize_path(path);
        let prefix = format!("{}/", path.trim_end_matches('/'));

        // Check if directory exists
        if !self
            .dirs
            .read()
            .ok()
            .context("lock poisoned")?
            .contains(&path)
        {
            anyhow::bail!("Directory not found: {path}");
        }

        if recursive {
            // Remove all files and subdirs
            self.files
                .write()
                .ok()
                .context("lock poisoned")?
                .retain(|k, _| !k.starts_with(&prefix));
            self.dirs
                .write()
                .ok()
                .context("lock poisoned")?
                .retain(|k| !k.starts_with(&prefix) && k != &path);
        } else {
            // Check if empty first
            let has_files = self
                .files
                .read()
                .ok()
                .context("lock poisoned")?
                .keys()
                .any(|k| k.starts_with(&prefix));
            let has_subdirs = self
                .dirs
                .read()
                .ok()
                .context("lock poisoned")?
                .iter()
                .any(|k| k.starts_with(&prefix) && k != &path);

            if has_files || has_subdirs {
                anyhow::bail!("Directory not empty: {path}");
            }

            self.dirs
                .write()
                .ok()
                .context("lock poisoned")?
                .remove(&path);
        }

        Ok(())
    }

    async fn grep(&self, pattern: &str, path: &str, recursive: bool) -> Result<Vec<GrepMatch>> {
        let path = self.normalize_path(path);
        let regex = regex::Regex::new(pattern).context("Invalid regex pattern")?;
        let mut matches = Vec::new();

        // Determine if path is a file or directory
        let is_file = self
            .files
            .read()
            .ok()
            .context("lock poisoned")?
            .contains_key(&path);
        let is_dir = self
            .dirs
            .read()
            .ok()
            .context("lock poisoned")?
            .contains(&path);

        if is_file {
            // Search single file - clone content to release lock early
            let content = self
                .files
                .read()
                .ok()
                .context("lock poisoned")?
                .get(&path)
                .cloned();
            if let Some(content) = content {
                let content = String::from_utf8_lossy(&content);
                for (line_num, line) in content.lines().enumerate() {
                    if let Some(m) = regex.find(line) {
                        matches.push(GrepMatch {
                            path: path.clone(),
                            line_number: line_num + 1,
                            line_content: line.to_string(),
                            match_start: m.start(),
                            match_end: m.end(),
                        });
                    }
                }
            }
        } else if is_dir {
            // Search directory - collect files to search first
            let prefix = format!("{}/", path.trim_end_matches('/'));
            let files_to_search: Vec<_> = {
                let files = self.files.read().ok().context("lock poisoned")?;
                files
                    .iter()
                    .filter(|(file_path, _)| {
                        if recursive {
                            file_path.starts_with(&prefix)
                        } else {
                            file_path.starts_with(&prefix)
                                && !file_path[prefix.len()..].contains('/')
                        }
                    })
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            };

            for (file_path, content) in files_to_search {
                let content = String::from_utf8_lossy(&content);
                for (line_num, line) in content.lines().enumerate() {
                    if let Some(m) = regex.find(line) {
                        matches.push(GrepMatch {
                            path: file_path.clone(),
                            line_number: line_num + 1,
                            line_content: line.to_string(),
                            match_start: m.start(),
                            match_end: m.end(),
                        });
                    }
                }
            }
        }

        Ok(matches)
    }

    async fn glob(&self, pattern: &str) -> Result<Vec<String>> {
        let pattern = self.normalize_path(pattern);

        // Simple glob matching
        let regex_pattern = pattern
            .replace("**", "\x00")
            .replace('*', "[^/]*")
            .replace('\x00', ".*")
            .replace('?', ".");
        let regex =
            regex::Regex::new(&format!("^{regex_pattern}$")).context("Invalid glob pattern")?;

        // Collect matches from files and dirs - release locks as early as possible
        let mut matches: Vec<String> = self
            .files
            .read()
            .ok()
            .context("lock poisoned")?
            .keys()
            .filter(|p| regex.is_match(p))
            .cloned()
            .collect();

        matches.extend(
            self.dirs
                .read()
                .ok()
                .context("lock poisoned")?
                .iter()
                .filter(|p| regex.is_match(p))
                .cloned(),
        );

        matches.sort();
        matches.dedup();
        Ok(matches)
    }

    fn root(&self) -> &str {
        &self.root
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_in_memory_write_and_read() -> Result<()> {
        let fs = InMemoryFileSystem::new("/workspace");

        fs.write_file("test.txt", "Hello, World!").await?;
        let content = fs.read_file("test.txt").await?;

        assert_eq!(content, "Hello, World!");
        Ok(())
    }

    #[tokio::test]
    async fn test_in_memory_exists() -> Result<()> {
        let fs = InMemoryFileSystem::new("/workspace");

        assert!(!fs.exists("test.txt").await?);
        fs.write_file("test.txt", "content").await?;
        assert!(fs.exists("test.txt").await?);
        Ok(())
    }

    #[tokio::test]
    async fn test_in_memory_directories() -> Result<()> {
        let fs = InMemoryFileSystem::new("/workspace");

        fs.create_dir("src/lib").await?;
        assert!(fs.is_dir("src").await?);
        assert!(fs.is_dir("src/lib").await?);
        assert!(!fs.is_file("src").await?);
        Ok(())
    }

    #[tokio::test]
    async fn test_in_memory_list_dir() -> Result<()> {
        let fs = InMemoryFileSystem::new("/workspace");

        fs.write_file("file1.txt", "content1").await?;
        fs.write_file("file2.txt", "content2").await?;
        fs.create_dir("subdir").await?;

        let entries = fs.list_dir("/workspace").await?;
        assert_eq!(entries.len(), 3);

        let names: Vec<_> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"file1.txt"));
        assert!(names.contains(&"file2.txt"));
        assert!(names.contains(&"subdir"));
        Ok(())
    }

    #[tokio::test]
    async fn test_in_memory_grep() -> Result<()> {
        let fs = InMemoryFileSystem::new("/workspace");

        fs.write_file("test.rs", "fn main() {\n    println!(\"Hello\");\n}")
            .await?;

        let matches = fs.grep("println", "/workspace", true).await?;
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].line_number, 2);
        assert!(matches[0].line_content.contains("println"));
        Ok(())
    }

    #[tokio::test]
    async fn test_in_memory_glob() -> Result<()> {
        let fs = InMemoryFileSystem::new("/workspace");

        fs.write_file("src/main.rs", "fn main() {}").await?;
        fs.write_file("src/lib.rs", "pub mod foo;").await?;
        fs.write_file("tests/test.rs", "// test").await?;

        let matches = fs.glob("/workspace/src/*.rs").await?;
        assert_eq!(matches.len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn test_in_memory_delete() -> Result<()> {
        let fs = InMemoryFileSystem::new("/workspace");

        fs.write_file("test.txt", "content").await?;
        assert!(fs.exists("test.txt").await?);

        fs.delete_file("test.txt").await?;
        assert!(!fs.exists("test.txt").await?);
        Ok(())
    }
}
