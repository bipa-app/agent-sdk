use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
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

    /// Resolve an input path to an absolute path **clamped to [`root`](Environment::root).**
    ///
    /// This is the path-policy boundary for primitive tools: it interprets the
    /// model-supplied `path` against the environment root and guarantees the
    /// result can never lexically escape that root. Relative inputs are joined
    /// to the root; absolute inputs are accepted only when they already fall
    /// inside the root, otherwise they (and any `..` traversal) are clamped
    /// back inside it. So with `root = /workspace`, both `../../etc/passwd`
    /// and `/etc/passwd` resolve to a path under `/workspace`, never to the
    /// host's `/etc/passwd`.
    ///
    /// # Security
    ///
    /// The clamp is **lexical** (it does not touch the filesystem) so it cannot
    /// detect a symlink *inside* the root that points outside it. Callers that
    /// resolve against a real filesystem and need a symlink-proof boundary must
    /// use [`resolve_within_root_secure`], which canonicalizes the path and
    /// verifies containment after following links.
    fn resolve_path(&self, path: &str) -> String {
        resolve_within_root(self.root(), path)
    }
}

/// Resolve `path` against `root`, clamping the result so it can never lexically
/// escape `root`.
///
/// Relative inputs are joined to `root`. Absolute inputs are kept as-is **only**
/// when they already resolve inside `root`; any input that would escape (via a
/// leading `/` outside the root, or `..` traversal) is re-interpreted as
/// root-relative so the result stays within `root`. The returned string is
/// lexically normalized (`.` / `..` resolved).
///
/// This is a lexical boundary and does **not** follow symlinks — see
/// [`resolve_within_root_secure`] for a symlink-proof check against a real
/// filesystem.
#[must_use]
pub fn resolve_within_root(root: &str, path: &str) -> String {
    let root_norm = normalize_path_buf(Path::new(root));
    let joined = if path.starts_with('/') {
        PathBuf::from(path)
    } else {
        root_norm.join(path)
    };
    let normalized = normalize_path_buf(&joined);
    if normalized == root_norm || normalized.starts_with(&root_norm) {
        normalized.to_string_lossy().into_owned()
    } else {
        // The input escaped the root. Clamp it back inside by treating it as
        // strictly root-relative: leading separators and any `..` that would
        // climb above the root are dropped.
        clamp_to_root(&root_norm, path)
            .to_string_lossy()
            .into_owned()
    }
}

/// Re-interpret `path` as strictly relative to `root_norm`, dropping leading
/// separators and refusing to let `..` climb above the root boundary.
fn clamp_to_root(root_norm: &Path, path: &str) -> PathBuf {
    let root_components: Vec<Component<'_>> = root_norm.components().collect();
    let mut stack: Vec<Component<'_>> = root_components.clone();
    for component in Path::new(path).components() {
        match component {
            // Inputs are always interpreted relative to the root, so a leading
            // `/` (or Windows prefix) never resets the resolution.
            Component::Prefix(_) | Component::RootDir | Component::CurDir => {}
            Component::ParentDir => {
                if stack.len() > root_components.len() {
                    stack.pop();
                }
            }
            normal @ Component::Normal(_) => stack.push(normal),
        }
    }
    stack.iter().collect()
}

/// Lexically normalize a path by resolving `.` and `..` components without
/// hitting the filesystem.
///
/// This collapses `.` / `..` segments but only clamps at the **filesystem
/// root** (`/`); it is unaware of any environment root, so on its own it does
/// **not** confine a path to an allowed directory — `/workspace/../../etc` still
/// normalizes to `/etc`. Use [`resolve_within_root`] to clamp to an environment
/// root, and [`resolve_within_root_secure`] when symlinks must also be defeated.
/// Unlike [`std::fs::canonicalize`], this does not require the path to exist and
/// does not follow symlinks.
#[must_use]
pub fn normalize_path(path: &Path) -> String {
    normalize_path_buf(path).to_string_lossy().into_owned()
}

/// Lexically normalize a path, returning a `PathBuf`.
///
/// See [`normalize_path`] for the security caveats: this is a lexical helper
/// that clamps only at the filesystem root and does not follow symlinks.
#[must_use]
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

/// Resolve `path` against `root` on a real filesystem, following symlinks and
/// verifying the final target is contained within `root`.
///
/// Unlike [`resolve_within_root`] (a purely lexical clamp), this canonicalizes
/// the deepest existing ancestor of the target — resolving any symlinks along
/// the way — and rejects the path if the resolved location escapes the
/// canonicalized `root`. This is the symlink-proof check that
/// `LocalFileSystem`-backed tools should use before reading or writing, so a
/// link such as `workspace/evil -> /etc` cannot be used to step outside the
/// sandbox.
///
/// # Errors
/// Returns an error if `root` cannot be canonicalized, if an existing ancestor
/// cannot be canonicalized, or if the resolved path escapes `root`.
pub fn resolve_within_root_secure(root: &Path, path: &str) -> Result<PathBuf> {
    let canonical_root = std::fs::canonicalize(root)
        .with_context(|| format!("failed to canonicalize environment root {}", root.display()))?;
    let clamped = clamp_to_root(&normalize_path_buf(root), path);
    let resolved = canonicalize_deepest_existing(&clamped)?;
    ensure!(
        resolved == canonical_root || resolved.starts_with(&canonical_root),
        "path {} escapes the environment root {} after resolving symlinks",
        resolved.display(),
        canonical_root.display(),
    );
    Ok(resolved)
}

/// Canonicalize the deepest existing ancestor of `path` (resolving symlinks)
/// and re-append the non-existent tail, so a not-yet-created file still has its
/// real parent resolved.
fn canonicalize_deepest_existing(path: &Path) -> Result<PathBuf> {
    let mut existing = path.to_path_buf();
    let mut tail: Vec<OsString> = Vec::new();
    while !existing.exists() {
        let Some(name) = existing.file_name().map(ToOwned::to_owned) else {
            break;
        };
        tail.push(name);
        if !existing.pop() {
            break;
        }
    }
    let mut resolved = if existing.as_os_str().is_empty() {
        PathBuf::from("/")
    } else {
        std::fs::canonicalize(&existing)
            .with_context(|| format!("failed to canonicalize {}", existing.display()))?
    };
    for name in tail.into_iter().rev() {
        resolved.push(name);
    }
    Ok(resolved)
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
    fn test_normalize_path_lexical_clamps_only_at_filesystem_root() {
        // `normalize_path` is a *lexical* helper: it clamps at the filesystem
        // root (`/`) but is unaware of any environment root, so it deliberately
        // does NOT confine the path to `/workspace`. Root-confinement is the
        // job of `resolve_within_root` / `resolve_path`.
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

    #[test]
    fn resolve_within_root_keeps_paths_already_inside_root() {
        // An absolute path that already lives under the root passes through
        // unchanged (after lexical normalization).
        assert_eq!(
            resolve_within_root("/workspace", "/workspace/src/main.rs"),
            "/workspace/src/main.rs"
        );
        // Relative paths are joined to the root.
        assert_eq!(
            resolve_within_root("/workspace", "src/main.rs"),
            "/workspace/src/main.rs"
        );
    }

    #[test]
    fn resolve_within_root_clamps_parent_traversal() {
        // `..` may not climb above the root: it is clamped back inside.
        assert_eq!(
            resolve_within_root("/workspace", "../../etc/passwd"),
            "/workspace/etc/passwd"
        );
        assert_eq!(
            resolve_within_root("/workspace", "src/../../../../etc/passwd"),
            "/workspace/etc/passwd"
        );
    }

    #[test]
    fn resolve_within_root_clamps_absolute_escape() {
        // An absolute path outside the root is re-rooted, never allowed out.
        assert_eq!(
            resolve_within_root("/workspace", "/etc/passwd"),
            "/workspace/etc/passwd"
        );
    }

    #[test]
    fn resolve_within_root_does_not_confuse_sibling_prefixes() {
        // `/workspace-evil` must not be treated as inside `/workspace`.
        assert_eq!(
            resolve_within_root("/workspace", "/workspace-evil/secret"),
            "/workspace/workspace-evil/secret"
        );
    }

    #[cfg(unix)]
    #[test]
    fn resolve_within_root_secure_rejects_symlink_escape() -> Result<()> {
        use std::os::unix::fs::symlink;

        let nanos = time::OffsetDateTime::now_utc().unix_timestamp_nanos();
        let base =
            std::env::temp_dir().join(format!("agent-sdk-secpath-{}-{nanos}", std::process::id()));
        let root = base.join("workspace");
        let outside = base.join("outside");
        std::fs::create_dir_all(&root)?;
        std::fs::create_dir_all(&outside)?;
        std::fs::write(outside.join("secret.txt"), b"top secret")?;

        // A symlink inside the root that points outside it. A purely lexical
        // check (`resolve_within_root`) would accept `link/secret.txt` as
        // contained; the secure resolver must reject it.
        symlink(&outside, root.join("link"))?;

        let escape = resolve_within_root_secure(&root, "link/secret.txt");
        assert!(
            escape.is_err(),
            "symlink escape must be rejected, got {escape:?}"
        );

        // A genuinely-inside path resolves cleanly.
        std::fs::write(root.join("inside.txt"), b"ok")?;
        let inside = resolve_within_root_secure(&root, "inside.txt")?;
        assert!(inside.starts_with(std::fs::canonicalize(&root)?));

        // A not-yet-created file under the root is allowed (parent resolved).
        let new_file = resolve_within_root_secure(&root, "subdir/new.txt")?;
        assert!(new_file.starts_with(std::fs::canonicalize(&root)?));

        let _ = std::fs::remove_dir_all(&base);
        Ok(())
    }
}
