//! Session-scoped artifact spill storage for oversized tool output.
//!
//! Truncation must never destroy bytes. Every tool result that would exceed
//! the shared inline output budget is mirrored **byte-identical** to a file
//! under a per-thread artifacts directory, and the inline text is replaced by
//! a bounded head + tail window plus a recovery footer:
//!
//! ```text
//! <head>
//! [... N bytes elided ...]
//! <tail>
//! [raw output: artifact://<id>]
//! ```
//!
//! The model recovers the full stream with the `read` tool:
//! `artifact://<id>` (windowed by `offset`/`limit`) or
//! `artifact://<id>:<start>-<end>` line selectors.
//!
//! # Layout and identity
//!
//! Artifacts live as `<dir>/<id>.<tool>.log` where `<id>` is a store-local
//! monotonically increasing integer. The store scans the directory on first
//! allocation, so a resumed thread continues numbering instead of clobbering
//! prior artifacts, and files are created with `create_new` so an ID can
//! never overwrite existing content even across store instances.
//!
//! Exactly one [`ArtifactStore`] instance must own a directory at a time for
//! IDs to be race-free; [`ArtifactStorage`] multiplexes per-thread stores off
//! one root and guarantees that within a process.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use agent_sdk_foundation::types::{ThreadId, ToolResult};
use anyhow::{Context, Result, anyhow};

/// The shared inline output budget: one knob for every tool kind.
///
/// A tool result whose `output` exceeds this many bytes is spilled to the
/// artifact store and replaced inline by a head + tail window and the
/// recovery footer.
pub const DEFAULT_INLINE_OUTPUT_BUDGET_BYTES: usize = 50 * 1024;

/// Smallest accepted inline budget. Below this the head/tail windows and the
/// elision marker + footer no longer fit meaningfully.
const MIN_INLINE_OUTPUT_BUDGET_BYTES: usize = 1024;

/// URI scheme under which spilled artifacts are addressable.
pub const ARTIFACT_URI_SCHEME: &str = "artifact://";

/// Recovery URI for a spilled artifact.
#[must_use]
pub fn artifact_uri(id: u64) -> String {
    format!("{ARTIFACT_URI_SCHEME}{id}")
}

/// The recovery footer spliced into a capped inline result.
#[must_use]
pub fn artifact_footer(id: u64) -> String {
    format!("[raw output: {ARTIFACT_URI_SCHEME}{id}]")
}

/// A spilled artifact: its store-local ID and the backing file.
#[derive(Clone, Debug)]
pub struct SavedArtifact {
    /// Store-local monotonic ID, addressable as `artifact://<id>`.
    pub id: u64,
    /// Backing file (`<dir>/<id>.<tool>.log`).
    pub path: PathBuf,
}

/// Per-thread spill store for oversized tool output.
///
/// See the [module docs](self) for layout and identity rules.
#[derive(Debug)]
pub struct ArtifactStore {
    dir: PathBuf,
    inline_budget: usize,
    /// Next ID to try; `None` until the first allocation scans `dir`.
    next_id: Mutex<Option<u64>>,
}

impl ArtifactStore {
    /// A store rooted at `dir` with the default inline budget.
    ///
    /// The directory is created lazily on first spill.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            inline_budget: DEFAULT_INLINE_OUTPUT_BUDGET_BYTES,
            next_id: Mutex::new(None),
        }
    }

    /// Override the shared inline output budget (clamped to a sane floor).
    #[must_use]
    pub fn with_inline_budget(mut self, bytes: usize) -> Self {
        self.inline_budget = bytes.max(MIN_INLINE_OUTPUT_BUDGET_BYTES);
        self
    }

    /// The directory artifacts are written to.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The shared inline output budget in bytes.
    #[must_use]
    pub const fn inline_budget(&self) -> usize {
        self.inline_budget
    }

    /// Persist `content` as a new artifact for `tool_name`.
    ///
    /// The file is created with `create_new`, so an allocated ID can never
    /// overwrite an existing artifact; on a name collision the ID is bumped
    /// and retried.
    ///
    /// # Errors
    /// Returns an error when the directory cannot be created or the file
    /// cannot be written. The caller's original content is untouched.
    pub fn save(&self, tool_name: &str, content: &str) -> Result<SavedArtifact> {
        std::fs::create_dir_all(&self.dir)
            .with_context(|| format!("creating artifacts dir {}", self.dir.display()))?;
        let file_stem_tool = sanitize_tool_name(tool_name);
        // Allocate the ID and create the (empty) file under the lock, then
        // release it before streaming the content so a large write never
        // blocks sibling allocations.
        let (mut file, id, path) = {
            let mut next_id = self
                .next_id
                .lock()
                .map_err(|_| anyhow!("artifact ID allocator lock poisoned"))?;
            let mut id = match *next_id {
                Some(id) => id,
                None => scan_next_id(&self.dir)
                    .with_context(|| format!("scanning artifacts dir {}", self.dir.display()))?,
            };
            loop {
                let path = self.dir.join(format!("{id}.{file_stem_tool}.log"));
                match std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&path)
                {
                    Ok(file) => {
                        *next_id = Some(id + 1);
                        drop(next_id);
                        break (file, id, path);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                        id += 1;
                    }
                    Err(error) => {
                        return Err(error)
                            .with_context(|| format!("creating artifact {}", path.display()));
                    }
                }
            }
        };
        file.write_all(content.as_bytes())
            .with_context(|| format!("writing artifact {}", path.display()))?;
        Ok(SavedArtifact { id, path })
    }

    /// Resolve an artifact ID to its backing file.
    ///
    /// # Errors
    /// Returns an error naming the available IDs when `id` has no backing
    /// file (or the directory does not exist yet).
    pub fn resolve(&self, id: u64) -> Result<PathBuf> {
        let mut available = Vec::new();
        for (found_id, path) in list_artifacts(&self.dir)? {
            if found_id == id {
                return Ok(path);
            }
            available.push(found_id);
        }
        available.sort_unstable();
        available.dedup();
        if available.is_empty() {
            Err(anyhow!("artifact {id} not found: no artifacts exist yet"))
        } else {
            let listed: Vec<String> = available.iter().take(20).map(u64::to_string).collect();
            Err(anyhow!(
                "artifact {id} not found; available IDs: {}",
                listed.join(", ")
            ))
        }
    }

    /// Enforce the shared inline budget on a tool result.
    ///
    /// Over-budget output is spilled byte-identical to the store and the
    /// inline text is replaced by a bounded head + tail window, an elision
    /// marker, and the [`artifact_footer`] recovery URI.
    ///
    /// # Errors
    /// Returns an error when the spill write fails; `result` is left
    /// untouched so no bytes are destroyed by a failed spill. Callers that
    /// must bound the result anyway can apply their own lossy fallback.
    pub fn apply_inline_budget(
        &self,
        result: &mut ToolResult,
        tool_name: &str,
    ) -> Result<Option<SavedArtifact>> {
        if result.output.len() <= self.inline_budget {
            return Ok(None);
        }
        let saved = self.save(tool_name, &result.output)?;
        result.output = cap_inline_output(&result.output, self.inline_budget, saved.id);
        Ok(Some(saved))
    }
}

/// Process-wide multiplexer handing out one [`ArtifactStore`] per thread.
///
/// Stores live under `<root>/<thread_id>/` and are cached so every caller in
/// the process shares one allocator per directory (the ID race-freedom
/// contract from the [module docs](self)).
#[derive(Debug)]
pub struct ArtifactStorage {
    root: PathBuf,
    inline_budget: usize,
    stores: Mutex<HashMap<String, Arc<ArtifactStore>>>,
}

impl ArtifactStorage {
    /// A storage root with the default inline budget.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            inline_budget: DEFAULT_INLINE_OUTPUT_BUDGET_BYTES,
            stores: Mutex::new(HashMap::new()),
        }
    }

    /// Override the shared inline budget applied to every per-thread store.
    #[must_use]
    pub fn with_inline_budget(mut self, bytes: usize) -> Self {
        self.inline_budget = bytes.max(MIN_INLINE_OUTPUT_BUDGET_BYTES);
        self
    }

    /// The storage root directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The store for `thread_id`, creating and caching it on first use.
    ///
    /// # Errors
    /// Returns an error when the store cache lock is poisoned.
    pub fn for_thread(&self, thread_id: &ThreadId) -> Result<Arc<ArtifactStore>> {
        let key = sanitize_path_component(&thread_id.0);
        Ok(Arc::clone(
            self.stores
                .lock()
                .map_err(|_| anyhow!("artifact storage cache lock poisoned"))?
                .entry(key)
                .or_insert_with_key(|key| {
                    Arc::new(
                        ArtifactStore::new(self.root.join(key))
                            .with_inline_budget(self.inline_budget),
                    )
                }),
        ))
    }
}

/// Bounded inline replacement for an over-budget output: head + elision
/// marker + tail + recovery footer, all within `budget` bytes.
fn cap_inline_output(full: &str, budget: usize, artifact_id: u64) -> String {
    let footer = artifact_footer(artifact_id);
    // 60% head + 25% tail; the remainder is headroom for the marker and
    // footer, which are ~100 bytes against a >=1 KiB budget.
    let head_budget = budget * 3 / 5;
    let tail_budget = budget / 4;
    let head_end = floor_char_boundary(full, head_budget);
    let tail_start = ceil_char_boundary(full, full.len().saturating_sub(tail_budget));
    if tail_start <= head_end {
        // Unreachable with the clamped minimum budget (head + tail windows
        // cover < 85% of a string that is strictly larger than the budget),
        // but guard so a future budget change cannot panic or duplicate
        // bytes.
        return format!("{}\n{footer}", &full[..head_end]);
    }
    let elided = tail_start - head_end;
    format!(
        "{head}\n[... {elided} bytes elided ...]\n{tail}\n{footer}",
        head = &full[..head_end],
        tail = &full[tail_start..],
    )
}

/// Next free ID: one past the highest `<id>.` file in `dir` (0 when the
/// directory is empty or missing).
fn scan_next_id(dir: &Path) -> Result<u64> {
    let max = list_artifacts(dir)?.into_iter().map(|(id, _)| id).max();
    Ok(max.map_or(0, |max| max + 1))
}

/// Every `(id, path)` pair in `dir` whose file name starts with `<digits>.`.
/// A missing directory lists as empty.
fn list_artifacts(dir: &Path) -> Result<Vec<(u64, PathBuf)>> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(error).with_context(|| format!("reading artifacts dir {}", dir.display()));
        }
    };
    let mut artifacts = Vec::new();
    for entry in entries {
        let entry = entry.with_context(|| format!("reading an entry in {}", dir.display()))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some((id_part, _)) = name.split_once('.') else {
            continue;
        };
        if let Ok(id) = id_part.parse::<u64>() {
            artifacts.push((id, entry.path()));
        }
    }
    Ok(artifacts)
}

/// Restrict a tool name to a safe file-stem component.
fn sanitize_tool_name(tool_name: &str) -> String {
    let cleaned: String = tool_name
        .chars()
        .take(64)
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "tool".to_string()
    } else {
        cleaned
    }
}

/// Restrict a thread ID to a safe directory component.
fn sanitize_path_component(component: &str) -> String {
    let cleaned: String = component
        .chars()
        .take(128)
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "thread".to_string()
    } else {
        cleaned
    }
}

/// Largest byte index `<= idx` that lands on a UTF-8 char boundary.
const fn floor_char_boundary(s: &str, idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    let mut idx = idx;
    while !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Smallest byte index `>= idx` that lands on a UTF-8 char boundary.
const fn ceil_char_boundary(s: &str, idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    let mut idx = idx;
    while !s.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store() -> (tempfile::TempDir, ArtifactStore) {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = ArtifactStore::new(dir.path().join("artifacts"));
        (dir, store)
    }

    #[test]
    fn under_budget_output_is_untouched() -> Result<()> {
        let (_dir, store) = temp_store();
        let original = "x".repeat(store.inline_budget());
        let mut result = ToolResult::success(original.clone());
        let saved = store.apply_inline_budget(&mut result, "bash")?;
        assert!(saved.is_none(), "at-budget output must not spill");
        assert_eq!(result.output, original);
        assert!(!store.dir().exists(), "no spill file may be created");
        Ok(())
    }

    #[test]
    fn one_byte_over_budget_spills() -> Result<()> {
        let (_dir, store) = temp_store();
        let original = "x".repeat(store.inline_budget() + 1);
        let mut result = ToolResult::success(original.clone());
        let saved = store
            .apply_inline_budget(&mut result, "bash")?
            .expect("over-budget output must spill");
        assert_eq!(std::fs::read_to_string(&saved.path)?, original);
        assert!(result.output.len() <= store.inline_budget());
        assert!(result.output.contains(&artifact_footer(saved.id)));
        Ok(())
    }

    #[test]
    fn spill_file_is_byte_identical_for_multi_megabyte_output() -> Result<()> {
        let (_dir, store) = temp_store();
        // 5 MB of varied multi-byte content: catches any lossy re-encoding
        // or window arithmetic touching the persisted stream.
        let chunk = "line-Ω-你好-0123456789 abcdefghijklmnopqrstuvwxyz\n";
        let original = chunk.repeat(5 * 1024 * 1024 / chunk.len() + 1);
        assert!(original.len() > 5 * 1024 * 1024);
        let mut result = ToolResult::error(original.clone());
        let saved = store
            .apply_inline_budget(&mut result, "bash")?
            .expect("must spill");
        let on_disk = std::fs::read(&saved.path)?;
        assert_eq!(on_disk, original.as_bytes(), "spill must be byte-identical");
        assert!(result.output.len() <= store.inline_budget());
        assert!(result.output.starts_with(&original[..1024]));
        assert!(result.output.contains("bytes elided"));
        assert!(result.output.ends_with(&artifact_footer(saved.id)));
        Ok(())
    }

    #[test]
    fn footer_and_uri_formats_are_pinned() {
        assert_eq!(artifact_uri(7), "artifact://7");
        assert_eq!(artifact_footer(7), "[raw output: artifact://7]");
    }

    #[test]
    fn capped_inline_respects_utf8_boundaries() {
        // Multi-byte chars positioned to straddle both window edges.
        let full = "é".repeat(4096);
        let capped = cap_inline_output(&full, MIN_INLINE_OUTPUT_BUDGET_BYTES, 0);
        assert!(capped.contains("bytes elided"));
        assert!(capped.ends_with(&artifact_footer(0)));
    }

    #[test]
    fn ids_are_monotonic_within_a_store() -> Result<()> {
        let (_dir, store) = temp_store();
        let first = store.save("bash", "one")?;
        let second = store.save("grep", "two")?;
        assert_eq!(first.id, 0);
        assert_eq!(second.id, 1);
        Ok(())
    }

    #[test]
    fn allocation_resumes_after_existing_artifacts() -> Result<()> {
        let (_dir, store) = temp_store();
        std::fs::create_dir_all(store.dir())?;
        std::fs::write(store.dir().join("7.bash.log"), "prior")?;
        let saved = store.save("bash", "next")?;
        assert_eq!(saved.id, 8, "scan must continue past existing IDs");
        assert_eq!(
            std::fs::read_to_string(store.dir().join("7.bash.log"))?,
            "prior"
        );
        Ok(())
    }

    #[test]
    fn save_never_overwrites_a_colliding_id() -> Result<()> {
        let (_dir, store) = temp_store();
        let first = store.save("bash", "first")?;
        // A second allocator over the same dir (the misuse the docs forbid)
        // still cannot clobber: create_new forces it onto a fresh ID.
        let rival = ArtifactStore::new(store.dir());
        let second = rival.save("bash", "second")?;
        assert_ne!(first.id, second.id);
        assert_eq!(std::fs::read_to_string(&first.path)?, "first");
        assert_eq!(std::fs::read_to_string(&second.path)?, "second");
        Ok(())
    }

    #[test]
    fn concurrent_saves_get_unique_ids() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = Arc::new(ArtifactStore::new(dir.path().join("artifacts")));
        let handles: Vec<_> = (0..16)
            .map(|i| {
                let store = Arc::clone(&store);
                std::thread::spawn(move || store.save("bash", &format!("content-{i}")))
            })
            .collect();
        let mut ids = Vec::new();
        for handle in handles {
            let saved = handle
                .join()
                .map_err(|_| anyhow!("save thread panicked"))??;
            ids.push(saved.id);
        }
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 16, "every concurrent save must get its own ID");
        Ok(())
    }

    #[test]
    fn resolve_finds_saved_artifacts_and_names_available_ids() -> Result<()> {
        let (_dir, store) = temp_store();
        let saved = store.save("bash", "content")?;
        assert_eq!(store.resolve(saved.id)?, saved.path);
        let error = store.resolve(99).expect_err("missing ID must error");
        assert!(error.to_string().contains("available IDs: 0"));
        let empty = ArtifactStore::new(store.dir().join("nope"));
        let error = empty.resolve(0).expect_err("empty store must error");
        assert!(error.to_string().contains("no artifacts exist yet"));
        Ok(())
    }

    #[test]
    fn storage_hands_out_one_store_per_thread() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let storage = ArtifactStorage::new(dir.path().join("artifacts"));
        let thread = ThreadId::from_string("thread-a");
        let first = storage.for_thread(&thread)?;
        let second = storage.for_thread(&thread)?;
        assert!(
            Arc::ptr_eq(&first, &second),
            "same thread shares one allocator"
        );
        let other = storage.for_thread(&ThreadId::from_string("thread-b"))?;
        assert_ne!(first.dir(), other.dir());
        Ok(())
    }

    #[test]
    fn spill_failure_leaves_output_untouched() {
        // Root the store at a path that cannot be a directory.
        let dir = tempfile::tempdir().expect("tempdir");
        let blocker = dir.path().join("blocker");
        std::fs::write(&blocker, "file").expect("write blocker");
        let store = ArtifactStore::new(blocker.join("artifacts"));
        let original = "x".repeat(store.inline_budget() + 1);
        let mut result = ToolResult::success(original.clone());
        let outcome = store.apply_inline_budget(&mut result, "bash");
        assert!(outcome.is_err(), "spill into a non-directory must fail");
        assert_eq!(
            result.output, original,
            "failed spill must not destroy bytes"
        );
    }
}
