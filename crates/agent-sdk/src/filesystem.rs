//! Filesystem implementations for the Environment trait.
//!
//! Provides:
//! - `LocalFileSystem` - Standard filesystem operations using `std::fs`
//! - `InMemoryFileSystem` - In-memory filesystem for testing

use crate::environment::{
    self, Environment, ExecResult, ExecSinkSpec, ExecStreamCapture, ExecStreamResult, FileEntry,
    GrepMatch, HARD_MAX_EXEC_CAPTURE_WINDOW_BYTES, HARD_MAX_EXEC_SPOOL_BYTES_PER_STREAM,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::RwLock;
use tokio::process::Command;

const MAX_EXEC_OUTPUT_BYTES: usize = 1024 * 1024;

/// Maximum size of a file that recursive grep will read into memory. Larger
/// files (typically build artifacts / media) are skipped rather than loaded.
const MAX_GREP_FILE_BYTES: u64 = 16 * 1024 * 1024;

pub fn create_private_exec_spool() -> Result<File> {
    let path = std::env::temp_dir().join(format!(
        ".agent-sdk-exec-{}.spool",
        uuid::Uuid::new_v4().as_simple()
    ));
    let mut options = OpenOptions::new();
    options.read(true).write(true).create_new(true);

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_SHARE_READ_WRITE_DELETE: u32 = 0x1 | 0x2 | 0x4;
        const FILE_FLAG_DELETE_ON_CLOSE: u32 = 0x0400_0000;
        options
            .share_mode(FILE_SHARE_READ_WRITE_DELETE)
            .custom_flags(FILE_FLAG_DELETE_ON_CLOSE);
    }

    let file = options
        .open(&path)
        .context("failed to create private process spool")?;

    #[cfg(not(windows))]
    std::fs::remove_file(&path).context("failed to unlink private process spool")?;

    Ok(file)
}

async fn capture_process_stream<R>(
    mut reader: R,
    spool: File,
    head_bytes: usize,
    tail_bytes: usize,
    max_bytes: u64,
    stream: &'static str,
    kill: tokio::sync::mpsc::UnboundedSender<()>,
) -> Result<ExecStreamCapture>
where
    R: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut spool = tokio::fs::File::from_std(spool);
    let mut head = Vec::with_capacity(head_bytes);
    let mut tail = VecDeque::with_capacity(tail_bytes);
    let mut chunk = [0_u8; 8192];
    let mut total_bytes = 0_u64;
    let mut failure = None;

    loop {
        let read = match reader.read(&mut chunk).await {
            Ok(read) => read,
            Err(error) => {
                let _ = kill.send(());
                return Err(
                    anyhow::Error::new(error).context(format!("failed to read process {stream}"))
                );
            }
        };
        if read == 0 {
            break;
        }

        total_bytes = total_bytes
            .checked_add(u64::try_from(read).context("process stream length overflowed u64")?)
            .context("process stream length overflowed u64")?;

        let window_offset = if head.len() < head_bytes {
            let take = (head_bytes - head.len()).min(read);
            head.extend_from_slice(&chunk[..take]);
            take
        } else {
            0
        };
        if tail_bytes > 0 && window_offset < read {
            tail.extend(&chunk[window_offset..read]);
            if tail.len() > tail_bytes {
                tail.drain(..tail.len() - tail_bytes);
            }
        }

        if failure.is_none() {
            if total_bytes > max_bytes {
                failure = Some(anyhow::anyhow!(
                    "{stream} exceeded the {max_bytes}-byte process spool limit; command was terminated and output was not returned"
                ));
                let _ = kill.send(());
            } else if let Err(error) = spool.write_all(&chunk[..read]).await {
                failure = Some(
                    anyhow::Error::new(error)
                        .context(format!("failed to write private {stream} spool")),
                );
                let _ = kill.send(());
            }
        }
    }

    if let Some(error) = failure {
        return Err(error);
    }
    spool
        .flush()
        .await
        .with_context(|| format!("failed to flush private {stream} spool"))?;

    Ok(ExecStreamCapture {
        head,
        tail: tail.into(),
        total_bytes,
        spool: spool.into_std().await,
    })
}

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
        let joined = if Path::new(path).is_absolute() {
            PathBuf::from(path)
        } else {
            self.root.join(path)
        };
        environment::normalize_path_buf(&joined)
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
        Ok(tokio::fs::metadata(&path).await.is_ok_and(|m| m.is_dir()))
    }

    async fn is_file(&self, path: &str) -> Result<bool> {
        let path = self.resolve(path);
        Ok(tokio::fs::metadata(&path).await.is_ok_and(|m| m.is_file()))
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
        let streamed = self
            .exec_streamed(
                command,
                timeout_ms,
                ExecSinkSpec {
                    stdout: create_private_exec_spool()?,
                    stderr: create_private_exec_spool()?,
                    head_bytes: MAX_EXEC_OUTPUT_BYTES,
                    tail_bytes: 0,
                    max_bytes_per_stream: MAX_EXEC_OUTPUT_BYTES as u64,
                },
            )
            .await?;
        let stdout = streamed
            .stdout
            .complete_bytes()
            .context("stdout exceeded the bounded legacy process capture")?;
        let stderr = streamed
            .stderr
            .complete_bytes()
            .context("stderr exceeded the bounded legacy process capture")?;
        Ok(ExecResult {
            stdout: String::from_utf8(stdout).context(
                "stdout was not valid UTF-8; binary output was withheld without lossy conversion",
            )?,
            stderr: String::from_utf8(stderr).context(
                "stderr was not valid UTF-8; binary output was withheld without lossy conversion",
            )?,
            exit_code: streamed.exit_code,
        })
    }

    async fn exec_streamed(
        &self,
        command: &str,
        timeout_ms: Option<u64>,
        sinks: ExecSinkSpec,
    ) -> Result<ExecStreamResult> {
        anyhow::ensure!(
            sinks.head_bytes <= HARD_MAX_EXEC_CAPTURE_WINDOW_BYTES
                && sinks.tail_bytes <= HARD_MAX_EXEC_CAPTURE_WINDOW_BYTES,
            "process capture windows exceed the {HARD_MAX_EXEC_CAPTURE_WINDOW_BYTES}-byte hard limit"
        );
        let max_bytes_per_stream = sinks
            .max_bytes_per_stream
            .min(HARD_MAX_EXEC_SPOOL_BYTES_PER_STREAM);

        let timeout = std::time::Duration::from_millis(timeout_ms.unwrap_or(120_000));
        let mut child = Command::new("sh")
            .arg("-c")
            .arg(command)
            .current_dir(&self.root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .context("Failed to execute command")?;

        let stdout = child.stdout.take().context("missing stdout pipe")?;
        let stderr = child.stderr.take().context("missing stderr pipe")?;
        let (kill_tx, mut kill_rx) = tokio::sync::mpsc::unbounded_channel();
        let kill_keepalive = kill_tx.clone();
        let stdout_capture = capture_process_stream(
            stdout,
            sinks.stdout,
            sinks.head_bytes,
            sinks.tail_bytes,
            max_bytes_per_stream,
            "stdout",
            kill_tx.clone(),
        );
        let stderr_capture = capture_process_stream(
            stderr,
            sinks.stderr,
            sinks.head_bytes,
            sinks.tail_bytes,
            max_bytes_per_stream,
            "stderr",
            kill_tx,
        );
        let supervisor = async move {
            let _kill_keepalive = kill_keepalive;
            tokio::select! {
                status = child.wait() => Ok(Some(status?)),
                _ = kill_rx.recv() => {
                    let _ = child.start_kill();
                    Ok(Some(child.wait().await?))
                }
                () = tokio::time::sleep(timeout) => {
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                    Ok::<_, std::io::Error>(None)
                }
            }
        };

        let (stdout, stderr, status) = tokio::join!(stdout_capture, stderr_capture, supervisor);
        let status = status.context("Failed to wait for command")?;
        let Some(status) = status else {
            anyhow::bail!("Command timed out after {}ms", timeout.as_millis());
        };

        Ok(ExecStreamResult {
            stdout: stdout?,
            stderr: stderr?,
            exit_code: status.code().unwrap_or(-1),
        })
    }

    fn root(&self) -> &str {
        self.root.to_str().unwrap_or_else(|| {
            log::error!(
                "LocalFileSystem root path contains invalid UTF-8: {}",
                self.root.to_string_lossy()
            );
            "/"
        })
    }
}

impl LocalFileSystem {
    async fn grep_file(
        &self,
        path: &Path,
        regex: &regex::Regex,
        matches: &mut Vec<GrepMatch>,
    ) -> Result<()> {
        // Read the file once as bytes and scan the same buffer (an earlier
        // version read it twice: once to sniff for binary, once to grep).
        let content = tokio::fs::read(path).await?;
        Self::grep_bytes(path, &content, regex, matches);
        Ok(())
    }

    /// Scan an already-loaded buffer for regex matches, skipping files that look
    /// binary (a NUL byte in the first 1 KiB).
    fn grep_bytes(path: &Path, content: &[u8], regex: &regex::Regex, matches: &mut Vec<GrepMatch>) {
        if content.iter().take(1024).any(|&b| b == 0) {
            return; // Skip binary
        }
        let text = String::from_utf8_lossy(content);
        for (line_num, line) in text.lines().enumerate() {
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
                    // Skip oversized files (likely build artifacts / media)
                    // before loading them — bounds memory for huge files that
                    // would otherwise be read in full only to be skipped.
                    if metadata.len() > MAX_GREP_FILE_BYTES {
                        continue;
                    }
                    // Read the file exactly once; `grep_bytes` does the binary
                    // sniff and the scan over the same buffer.
                    if let Ok(content) = tokio::fs::read(&path).await {
                        Self::grep_bytes(&path, &content, regex, matches);
                    }
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

        // Escape regex metacharacters (except the glob wildcards `*`/`?` and the
        // char-class delimiters `[`/`]`) so a literal `.`/`(`/`+` matches
        // literally instead of acting as a regex operator. `[`/`]` pass through
        // so a balanced class like `[abc]` works as a glob char class, while an
        // unbalanced `[` makes `Regex::new` fail and surfaces as an "Invalid
        // glob pattern" error (matching real glob implementations).
        let mut escaped = String::with_capacity(pattern.len());
        for c in pattern.chars() {
            match c {
                '.' | '+' | '^' | '$' | '(' | ')' | '{' | '}' | '|' | '\\' => {
                    escaped.push('\\');
                    escaped.push(c);
                }
                _ => escaped.push(c),
            }
        }

        // Simple glob matching
        let regex_pattern = escaped
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

    fn unique_temp_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        std::env::temp_dir().join(format!("agent_sdk_fs_{tag}_{}_{nanos}", std::process::id()))
    }

    fn assert_zero_spool(file: &mut File, expected_bytes: u64) -> Result<()> {
        use std::io::{Read, Seek, SeekFrom};

        file.seek(SeekFrom::Start(0))?;
        let mut total = 0_u64;
        let mut chunk = [1_u8; 8192];
        loop {
            let read = file.read(&mut chunk)?;
            if read == 0 {
                break;
            }
            assert!(chunk[..read].iter().all(|byte| *byte == 0));
            total += read as u64;
        }
        assert_eq!(total, expected_bytes);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn test_private_exec_spool_is_anonymous_and_owner_only() -> Result<()> {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};

        let spool = create_private_exec_spool()?;
        assert_eq!(spool.metadata()?.permissions().mode() & 0o777, 0o600);
        assert_eq!(spool.metadata()?.nlink(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_exec_timeout_kills_child_no_leak() -> Result<()> {
        let tmp = unique_temp_dir("exec_leak");
        tokio::fs::create_dir_all(&tmp).await?;
        let marker = tmp.join("marker.txt");

        let fs = LocalFileSystem::new(tmp.clone());
        // The child sleeps past the timeout, then would create the marker.
        // A leaked (un-killed) child creates it; a reclaimed one never does.
        let command = format!("sleep 1; touch '{}'", marker.display());
        let result = fs.exec(&command, Some(50)).await;

        let Err(error) = result else {
            anyhow::bail!("exec should have timed out");
        };
        let rendered = format!("{error:#}");
        assert!(rendered.contains("timed out"), "got: {rendered}");

        // Wait well past the child's sleep; the marker must never appear.
        tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
        assert!(
            !tokio::fs::try_exists(&marker).await.unwrap_or(false),
            "child process leaked: marker created after timeout"
        );

        let _ = tokio::fs::remove_dir_all(&tmp).await;
        Ok(())
    }

    #[tokio::test]
    async fn test_exec_rejects_large_output_without_unbounded_capture() -> Result<()> {
        let tmp = unique_temp_dir("exec_output");
        tokio::fs::create_dir_all(&tmp).await?;
        let fs = LocalFileSystem::new(tmp.clone());

        let error = fs
            .exec(
                "dd if=/dev/zero bs=1000000 count=2 2>/dev/null",
                Some(10_000),
            )
            .await
            .err()
            .context("oversized output must fail closed")?;
        assert!(format!("{error:#}").contains("exceeded"));
        let _ = tokio::fs::remove_dir_all(&tmp).await;
        Ok(())
    }

    #[tokio::test]
    async fn test_exec_streams_multimegabyte_raw_output_into_bounded_windows() -> Result<()> {
        const STDOUT_BYTES: u64 = 2 * 1024 * 1024;
        const STDERR_BYTES: u64 = 3 * 1024 * 1024;
        const WINDOW_BYTES: usize = 4096;

        let tmp = unique_temp_dir("exec_streamed_output");
        tokio::fs::create_dir_all(&tmp).await?;
        let fs = LocalFileSystem::new(tmp.clone());
        let mut result = fs
            .exec_streamed(
                "dd if=/dev/zero bs=1048576 count=2 2>/dev/null; \
                 dd if=/dev/zero bs=1048576 count=3 1>&2 2>/dev/null",
                Some(10_000),
                ExecSinkSpec {
                    stdout: create_private_exec_spool()?,
                    stderr: create_private_exec_spool()?,
                    head_bytes: WINDOW_BYTES,
                    tail_bytes: WINDOW_BYTES,
                    max_bytes_per_stream: 4 * 1024 * 1024,
                },
            )
            .await?;

        assert_eq!(result.exit_code, 0);
        assert_eq!(result.stdout.total_bytes, STDOUT_BYTES);
        assert_eq!(result.stderr.total_bytes, STDERR_BYTES);
        assert_eq!(
            result.stdout.head.len() + result.stdout.tail.len(),
            WINDOW_BYTES * 2
        );
        assert_eq!(
            result.stderr.head.len() + result.stderr.tail.len(),
            WINDOW_BYTES * 2
        );
        assert_eq!(result.stdout.spool.metadata()?.len(), STDOUT_BYTES);
        assert_eq!(result.stderr.spool.metadata()?.len(), STDERR_BYTES);
        assert_zero_spool(&mut result.stdout.spool, STDOUT_BYTES)?;
        assert_zero_spool(&mut result.stderr.spool, STDERR_BYTES)?;

        let _ = tokio::fs::remove_dir_all(&tmp).await;
        Ok(())
    }

    #[tokio::test]
    async fn test_exec_spool_cap_terminates_child_without_marker_leak() -> Result<()> {
        let tmp = unique_temp_dir("exec_spool_cap");
        tokio::fs::create_dir_all(&tmp).await?;
        let marker = tmp.join("marker.txt");
        let fs = LocalFileSystem::new(tmp.clone());
        let command = format!(
            "dd if=/dev/zero bs=65536 count=2 2>/dev/null; sleep 1; touch '{}'",
            marker.display()
        );
        let error = fs
            .exec_streamed(
                &command,
                Some(10_000),
                ExecSinkSpec {
                    stdout: create_private_exec_spool()?,
                    stderr: create_private_exec_spool()?,
                    head_bytes: 1024,
                    tail_bytes: 1024,
                    max_bytes_per_stream: 16 * 1024,
                },
            )
            .await
            .err()
            .context("spool cap must fail explicitly")?;

        assert!(format!("{error:#}").contains("spool limit"));
        tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
        assert!(
            !tokio::fs::try_exists(&marker).await.unwrap_or(false),
            "child process leaked after spool cap"
        );

        let _ = tokio::fs::remove_dir_all(&tmp).await;
        Ok(())
    }

    #[tokio::test]
    async fn test_exec_rejects_non_utf8_without_lossy_conversion() -> Result<()> {
        let tmp = unique_temp_dir("exec_binary");
        tokio::fs::create_dir_all(&tmp).await?;
        let fs = LocalFileSystem::new(tmp.clone());
        let error = fs
            .exec("printf '\\377\\376'", Some(10_000))
            .await
            .err()
            .context("binary output must fail closed")?;
        assert!(format!("{error:#}").contains("not valid UTF-8"));
        let _ = tokio::fs::remove_dir_all(&tmp).await;
        Ok(())
    }

    #[tokio::test]
    async fn test_local_grep_skips_binary_and_matches_text() -> Result<()> {
        let tmp = unique_temp_dir("grep");
        tokio::fs::create_dir_all(&tmp).await?;
        tokio::fs::write(tmp.join("code.rs"), "fn TODO_here() {}\nlet x = 1;\n").await?;
        // Binary file that also contains the search term but starts with a NUL.
        let mut binary = vec![0u8, 1, 2, 3];
        binary.extend_from_slice(b"TODO_here\n");
        tokio::fs::write(tmp.join("blob.bin"), &binary).await?;

        let fs = LocalFileSystem::new(tmp.clone());
        let matches = fs.grep("TODO_here", ".", true).await?;

        assert_eq!(matches.len(), 1, "binary file must be skipped: {matches:?}");
        assert!(matches[0].path.ends_with("code.rs"));

        let _ = tokio::fs::remove_dir_all(&tmp).await;
        Ok(())
    }

    #[tokio::test]
    async fn test_in_memory_glob_escapes_regex_metacharacters() -> Result<()> {
        let fs = InMemoryFileSystem::new("/workspace");
        fs.write_file("main.rs", "x").await?;
        fs.write_file("mainXrs", "x").await?;

        // `.` in the pattern must be literal: `mainXrs` must NOT match `*.rs`.
        let matches = fs.glob("/workspace/*.rs").await?;
        assert_eq!(matches, vec!["/workspace/main.rs".to_string()]);

        // A pattern with regex metacharacters must not yield an invalid regex
        // (previously `+`/`[` would either over-match or fail the whole call).
        fs.write_file("a+b.txt", "x").await?;
        let plus = fs.glob("/workspace/a+b.txt").await?;
        assert_eq!(plus, vec!["/workspace/a+b.txt".to_string()]);

        Ok(())
    }
}
