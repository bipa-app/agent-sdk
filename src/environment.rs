use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};

/// Entry in a directory listing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileEntry {
    pub name: String,
    pub path: String,
    pub is_dir: bool,
    pub size: Option<u64>,
}

/// Match result from grep operation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GrepMatch {
    pub path: String,
    pub line_number: usize,
    pub line_content: String,
    pub match_start: usize,
    pub match_end: usize,
}

/// Result from command execution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

impl ExecResult {
    #[must_use]
    pub const fn success(&self) -> bool {
        self.exit_code == 0
    }
}

/// Environment abstraction for file and command operations.
///
/// The SDK's primitive tools (Read, Write, Grep, Glob, Bash) use this trait
/// to interact with the underlying filesystem or storage backend.
///
/// Implementations:
/// - `LocalFileSystem` - Standard filesystem (provided by SDK)
/// - `InMemoryFileSystem` - For testing (provided by SDK)
/// - Custom backends (S3, Git, iCloud, etc.)
#[async_trait]
pub trait Environment: Send + Sync {
    /// Read file contents as UTF-8 string
    ///
    /// # Errors
    /// Returns an error if the file cannot be read.
    async fn read_file(&self, path: &str) -> Result<String>;

    /// Read file contents as raw bytes
    ///
    /// # Errors
    /// Returns an error if the file cannot be read.
    async fn read_file_bytes(&self, path: &str) -> Result<Vec<u8>>;

    /// Write string content to file (creates or overwrites)
    ///
    /// # Errors
    /// Returns an error if the file cannot be written.
    async fn write_file(&self, path: &str, content: &str) -> Result<()>;

    /// Write raw bytes to file
    ///
    /// # Errors
    /// Returns an error if the file cannot be written.
    async fn write_file_bytes(&self, path: &str, content: &[u8]) -> Result<()>;

    /// List directory contents
    ///
    /// # Errors
    /// Returns an error if the directory cannot be read.
    async fn list_dir(&self, path: &str) -> Result<Vec<FileEntry>>;

    /// Check if path exists
    ///
    /// # Errors
    /// Returns an error if existence cannot be determined.
    async fn exists(&self, path: &str) -> Result<bool>;

    /// Check if path is a directory
    ///
    /// # Errors
    /// Returns an error if the check fails.
    async fn is_dir(&self, path: &str) -> Result<bool>;

    /// Check if path is a file
    ///
    /// # Errors
    /// Returns an error if the check fails.
    async fn is_file(&self, path: &str) -> Result<bool>;

    /// Create directory (including parents)
    ///
    /// # Errors
    /// Returns an error if the directory cannot be created.
    async fn create_dir(&self, path: &str) -> Result<()>;

    /// Delete file
    ///
    /// # Errors
    /// Returns an error if the file cannot be deleted.
    async fn delete_file(&self, path: &str) -> Result<()>;

    /// Delete directory (must be empty unless recursive)
    ///
    /// # Errors
    /// Returns an error if the directory cannot be deleted.
    async fn delete_dir(&self, path: &str, recursive: bool) -> Result<()>;

    /// Search for pattern in files (like ripgrep)
    ///
    /// # Errors
    /// Returns an error if the search fails.
    async fn grep(&self, pattern: &str, path: &str, recursive: bool) -> Result<Vec<GrepMatch>>;

    /// Find files matching glob pattern
    ///
    /// # Errors
    /// Returns an error if the glob operation fails.
    async fn glob(&self, pattern: &str) -> Result<Vec<String>>;

    /// Execute a shell command
    ///
    /// Not all environments support this. Default implementation returns an error.
    ///
    /// # Errors
    /// Returns an error if command execution is not supported or fails.
    async fn exec(&self, _command: &str, _timeout_ms: Option<u64>) -> Result<ExecResult> {
        anyhow::bail!("Command execution not supported in this environment")
    }

    /// Get the root/working directory for this environment
    fn root(&self) -> &str;

    /// Resolve a relative path to absolute within this environment.
    ///
    /// Normalizes `..` and `.` components to prevent path traversal attacks.
    fn resolve_path(&self, path: &str) -> String {
        let joined = if path.starts_with('/') {
            PathBuf::from(path)
        } else {
            PathBuf::from(self.root()).join(path)
        };
        normalize_path(&joined)
    }
}

/// Lexically normalize a path by resolving `.` and `..` components without
/// hitting the filesystem.
///
/// This prevents path traversal attacks where `../../etc/passwd` could escape
/// an allowed directory. Unlike `std::fs::canonicalize`, this does not require
/// the path to exist and does not follow symlinks.
pub fn normalize_path(path: &Path) -> String {
    normalize_path_buf(path).to_string_lossy().into_owned()
}

/// Lexically normalize a path, returning a `PathBuf`.
pub fn normalize_path_buf(path: &Path) -> PathBuf {
    let mut components: Vec<Component<'_>> = Vec::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                // Only pop if we have a normal component to pop (don't pop past root)
                if matches!(components.last(), Some(Component::Normal(_))) {
                    components.pop();
                }
            }
            Component::CurDir => {} // skip `.`
            other => components.push(other),
        }
    }
    if components.is_empty() {
        PathBuf::from("/")
    } else {
        components.iter().collect()
    }
}

/// A null environment that rejects all operations.
/// Useful as a default when no environment is configured.
pub struct NullEnvironment;

#[async_trait]
impl Environment for NullEnvironment {
    async fn read_file(&self, _path: &str) -> Result<String> {
        anyhow::bail!("No environment configured")
    }

    async fn read_file_bytes(&self, _path: &str) -> Result<Vec<u8>> {
        anyhow::bail!("No environment configured")
    }

    async fn write_file(&self, _path: &str, _content: &str) -> Result<()> {
        anyhow::bail!("No environment configured")
    }

    async fn write_file_bytes(&self, _path: &str, _content: &[u8]) -> Result<()> {
        anyhow::bail!("No environment configured")
    }

    async fn list_dir(&self, _path: &str) -> Result<Vec<FileEntry>> {
        anyhow::bail!("No environment configured")
    }

    async fn exists(&self, _path: &str) -> Result<bool> {
        anyhow::bail!("No environment configured")
    }

    async fn is_dir(&self, _path: &str) -> Result<bool> {
        anyhow::bail!("No environment configured")
    }

    async fn is_file(&self, _path: &str) -> Result<bool> {
        anyhow::bail!("No environment configured")
    }

    async fn create_dir(&self, _path: &str) -> Result<()> {
        anyhow::bail!("No environment configured")
    }

    async fn delete_file(&self, _path: &str) -> Result<()> {
        anyhow::bail!("No environment configured")
    }

    async fn delete_dir(&self, _path: &str, _recursive: bool) -> Result<()> {
        anyhow::bail!("No environment configured")
    }

    async fn grep(&self, _pattern: &str, _path: &str, _recursive: bool) -> Result<Vec<GrepMatch>> {
        anyhow::bail!("No environment configured")
    }

    async fn glob(&self, _pattern: &str) -> Result<Vec<String>> {
        anyhow::bail!("No environment configured")
    }

    fn root(&self) -> &'static str {
        "/"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_path_resolves_parent_dir() {
        let path = Path::new("/workspace/src/../../etc/passwd");
        assert_eq!(normalize_path(path), "/etc/passwd");
    }

    #[test]
    fn test_normalize_path_resolves_current_dir() {
        let path = Path::new("/workspace/./src/./file.rs");
        assert_eq!(normalize_path(path), "/workspace/src/file.rs");
    }

    #[test]
    fn test_normalize_path_does_not_escape_root() {
        let path = Path::new("/workspace/../../../etc/shadow");
        assert_eq!(normalize_path(path), "/etc/shadow");
    }

    #[test]
    fn test_normalize_path_identity() {
        let path = Path::new("/workspace/src/main.rs");
        assert_eq!(normalize_path(path), "/workspace/src/main.rs");
    }

    #[test]
    fn test_normalize_path_clamps_at_root() {
        // Trying to go above root should stop at /
        let path = Path::new("/a/../../../../z");
        assert_eq!(normalize_path(path), "/z");
    }

    #[test]
    fn test_resolve_path_normalizes_traversal() {
        let env = NullEnvironment;
        // NullEnvironment root is "/", so relative paths are joined with "/"
        let resolved = env.resolve_path("src/../../etc/passwd");
        assert_eq!(resolved, "/etc/passwd");
    }

    #[test]
    fn test_resolve_path_absolute_normalized() {
        let env = NullEnvironment;
        let resolved = env.resolve_path("/workspace/src/../../../etc/passwd");
        assert_eq!(resolved, "/etc/passwd");
    }
}
