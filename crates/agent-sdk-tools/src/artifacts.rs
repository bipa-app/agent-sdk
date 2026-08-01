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
//! Allocation is serialized by a directory-local file lock, so IDs remain
//! unique across processes and tool names. [`ArtifactStorage`] additionally
//! shares active per-thread stores within one process.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::{OsStr, OsString};
use std::io::{Read, Seek, SeekFrom, Write};
#[cfg(unix)]
use std::os::unix::fs::DirBuilderExt as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, SystemTime};

use agent_sdk_foundation::types::{ThreadId, ToolResult, ToolResultArtifact};
use anyhow::{Context, Result, anyhow};
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::ambient_authority;
use cap_std::fs::{Dir, OpenOptions};
#[cfg(unix)]
use cap_std::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use sha2::{Digest, Sha256};

/// The shared inline output budget: one knob for every tool kind.
///
const DEFAULT_MAX_ARTIFACT_BYTES_PER_THREAD: u64 = 512 * 1024 * 1024;
const STALE_PARTIAL_MAX_AGE: Duration = Duration::from_hours(24);

const ARTIFACT_ALLOCATOR_MARKER: &str = ".id-watermark";
const ARTIFACT_ALLOCATOR_RECORD_BYTES: usize = 21;
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

/// Enforce the shared transcript budget with a truthful bounded fail-closed
/// result when lossless artifact persistence is unavailable.
pub fn enforce_inline_budget(
    result: &mut ToolResult,
    store: Option<&ArtifactStore>,
    tool_name: &str,
) -> Option<SavedArtifact> {
    let budget = store.map_or(
        DEFAULT_INLINE_OUTPUT_BUDGET_BYTES,
        ArtifactStore::inline_budget,
    );
    // Host-provided/custom tools do not own spill provenance. Clear every
    // caller claim first, then re-mint only a byte-exact canonical claim
    // produced by the SDK's streaming spill path.
    let claimed_artifact = result.artifact.take();
    if result.output.len() <= budget {
        if let (Some(store), Some(artifact)) = (store, claimed_artifact)
            && let Ok(mut file) = store.resolve(artifact.id)
            && let Ok(metadata) = file.metadata()
            && canonical_streamed_inline_output_matches(
                &mut file,
                metadata.len(),
                &result.output,
                budget,
                artifact.id,
            )
            .unwrap_or(false)
        {
            result.artifact = Some(artifact);
        }
        return None;
    }
    let original_bytes = result.output.len();
    result.data = None;
    if let Some(store) = store {
        return match store.apply_inline_budget(result, tool_name) {
            Ok(saved) => saved,
            Err(error) => {
                log::warn!("artifact spill failed for tool {tool_name}: {error:#}");
                result.success = false;
                result.output = format!(
                    "Tool output was {original_bytes} bytes, but lossless artifact persistence \
                     failed. The output was not placed in the transcript; re-run the tool with \
                     narrower output."
                );
                None
            }
        };
    }
    result.success = false;
    result.output = format!(
        "Tool output was {original_bytes} bytes, but no artifact store is configured. \
         The output was not placed in the transcript; configure artifact storage or \
         re-run the tool with narrower output."
    );
    None
}

/// A spilled artifact: its store-local ID and the backing file.
#[derive(Clone, Debug)]
pub struct SavedArtifact {
    /// Store-local monotonic ID, addressable as `artifact://<id>`.
    pub id: u64,
    /// Backing file (`<dir>/<id>.<tool>.log`).
    pub path: PathBuf,
}

/// Per-thread facts that make artifacts ineligible for retention.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ArtifactThreadRetention {
    /// A live durable task owns this thread, so the whole directory is protected.
    pub live: bool,
    /// Artifact IDs still named by durable session state.
    pub referenced_ids: BTreeSet<u64>,
}

/// Durable liveness and reference snapshot supplied by the host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactRetentionSnapshot {
    /// Fail closed when the host could not establish durable ownership.
    pub protect_unknown_threads: bool,
    /// Entries keyed by [`artifact_thread_key`].
    pub threads: BTreeMap<String, ArtifactThreadRetention>,
}

impl Default for ArtifactRetentionSnapshot {
    fn default() -> Self {
        Self {
            protect_unknown_threads: true,
            threads: BTreeMap::new(),
        }
    }
}

/// Deterministic per-thread expiry and quota policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArtifactRetentionPolicy {
    /// Remove unreferenced artifacts strictly older than this age.
    pub max_age: Option<Duration>,
    /// Remove oldest unreferenced artifacts until each thread is at or below this size.
    pub max_bytes_per_thread: Option<u64>,
}

/// Observable outcome of one artifact retention sweep.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArtifactSweepReport {
    /// Files removed by expiry or quota.
    pub files_removed: usize,
    /// Bytes reclaimed from removed files.
    pub bytes_removed: u64,
    /// Files retained because they were live, referenced, unknown, or unreadable.
    pub protected_files: usize,
    /// Threads left above quota because only protected files remained.
    pub threads_over_quota: usize,
}

/// Per-thread spill store for oversized tool output.
///
/// See the [module docs](self) for layout and identity rules.
#[derive(Debug)]
pub struct ArtifactStore {
    dir: PathBuf,
    dir_handle: Result<Arc<Dir>, String>,
    inline_budget: usize,
    max_bytes_per_thread: u64,
    operation_gate: Mutex<()>,
    activity_lock: Mutex<Option<std::fs::File>>,
    acquisitions: AtomicU64,
}

impl ArtifactStore {
    /// A store rooted at `dir` with the default inline budget.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        let dir = dir.into();
        let dir_handle = open_confined_dir(&dir, true)
            .map(Arc::new)
            .map_err(|error| format!("{error:#}"));
        Self::from_capability(dir, dir_handle)
    }

    const fn from_capability(dir: PathBuf, dir_handle: Result<Arc<Dir>, String>) -> Self {
        Self {
            dir,
            dir_handle,
            inline_budget: DEFAULT_INLINE_OUTPUT_BUDGET_BYTES,
            max_bytes_per_thread: DEFAULT_MAX_ARTIFACT_BYTES_PER_THREAD,
            operation_gate: Mutex::new(()),
            activity_lock: Mutex::new(None),
            acquisitions: AtomicU64::new(0),
        }
    }

    /// Override the shared inline output budget (clamped to a sane floor).
    #[must_use]
    pub fn with_inline_budget(mut self, bytes: usize) -> Self {
        self.inline_budget = bytes.max(MIN_INLINE_OUTPUT_BUDGET_BYTES);
        self
    }

    /// The hard per-thread artifact admission quota.
    #[must_use]
    pub const fn max_bytes_per_thread(&self) -> u64 {
        self.max_bytes_per_thread
    }

    /// The directory artifacts are written to.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }
    /// Override the hard per-thread artifact admission quota.
    #[must_use]
    pub fn with_max_bytes_per_thread(mut self, bytes: u64) -> Self {
        self.max_bytes_per_thread = bytes.max(self.inline_budget as u64);
        self
    }

    /// The shared inline output budget in bytes.
    #[must_use]
    pub const fn inline_budget(&self) -> usize {
        self.inline_budget
    }
    fn cap_dir(&self) -> Result<Arc<Dir>> {
        self.dir_handle
            .as_ref()
            .map(Arc::clone)
            .map_err(|error| anyhow!("{error}"))
    }

    fn mark_acquired(&self) -> Result<()> {
        let _operation_guard = self
            .operation_gate
            .lock()
            .map_err(|_| anyhow!("artifact operation lock poisoned"))?;
        let mut activity_lock = self
            .activity_lock
            .lock()
            .map_err(|_| anyhow!("artifact activity lock state poisoned"))?;
        if activity_lock.is_none() {
            let dir = self.cap_dir()?;
            let file = open_activity_lock(&dir)?;
            file.lock_shared()
                .context("locking active artifact store")?;
            *activity_lock = Some(file);
        }
        drop(activity_lock);
        self.acquisitions.fetch_add(1, Ordering::Relaxed);
        Ok(())
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
        self.save_streamed(tool_name, &mut content.as_bytes())
    }

    /// Persist a byte stream as a new artifact without buffering it in memory.
    ///
    /// # Errors
    /// Returns an error when reading the source, staging, quota admission, or
    /// crash-safe publication fails.
    pub fn save_streamed(&self, tool_name: &str, source: &mut dyn Read) -> Result<SavedArtifact> {
        let _operation_guard = self
            .operation_gate
            .lock()
            .map_err(|_| anyhow!("artifact operation lock poisoned"))?;
        let dir = self.cap_dir()?;
        let _allocation_lock = lock_artifact_directory(&dir)?;
        let existing_bytes = artifact_bytes(&dir)?;
        let (mut temp, temp_name) = loop {
            let sequence = self.acquisitions.fetch_add(1, Ordering::Relaxed);
            let temp_name = format!(".partial-{}-{sequence}.log", std::process::id());
            let mut options = OpenOptions::new();
            options
                .write(true)
                .create_new(true)
                .follow(FollowSymlinks::No);
            #[cfg(unix)]
            options.mode(0o600);
            match dir.open_with(&temp_name, &options) {
                Ok(file) => break (file, temp_name),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error).context("creating staged artifact"),
            }
        };
        let mut published_name = None;
        let result = (|| -> Result<(u64, String)> {
            let mut content_bytes = 0_u64;
            let mut buffer = vec![0_u8; 1024 * 1024];
            loop {
                let read = source
                    .read(&mut buffer)
                    .context("reading streamed artifact source")?;
                if read == 0 {
                    break;
                }
                content_bytes = content_bytes
                    .checked_add(u64::try_from(read).context("artifact chunk exceeds u64")?)
                    .context("artifact content length exceeds u64")?;
                anyhow::ensure!(
                    existing_bytes
                        .checked_add(content_bytes)
                        .is_some_and(|total| total <= self.max_bytes_per_thread),
                    "artifact storage quota exceeded"
                );
                temp.write_all(&buffer[..read])
                    .context("writing staged artifact")?;
            }
            temp.sync_all().context("syncing staged artifact")?;
            drop(temp);

            let file_stem_tool = sanitize_tool_name(tool_name);
            // Reserve durably before publication. A failed publish burns an ID
            // rather than allowing a later artifact to impersonate it.
            let mut id = reserve_artifact_id(&dir)?;
            let name = loop {
                let name = format!("{id}.{file_stem_tool}.log");
                match dir.hard_link(&temp_name, &dir, &name) {
                    Ok(()) => break name,
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                        id = reserve_artifact_id(&dir)?;
                    }
                    Err(error) => return Err(error).context("publishing staged artifact"),
                }
            };
            published_name = Some(name.clone());
            dir.remove_file(&temp_name)
                .context("removing staged artifact")?;
            sync_directory(&dir)
                .with_context(|| format!("syncing artifacts dir {}", self.dir.display()))?;
            Ok((id, name))
        })();
        match result {
            Ok((id, name)) => Ok(SavedArtifact {
                id,
                path: self.dir.join(name),
            }),
            Err(error) => {
                let _ = dir.remove_file(&temp_name);
                if let Some(name) = published_name {
                    let _ = dir.remove_file(name);
                }
                Err(error)
            }
        }
    }

    /// Open an artifact ID through a stable directory capability.
    ///
    /// # Errors
    /// Returns an error naming the available IDs when `id` has no regular
    /// backing file (or the directory does not exist yet).
    pub fn resolve(&self, id: u64) -> Result<std::fs::File> {
        let _operation_guard = self
            .operation_gate
            .lock()
            .map_err(|_| anyhow!("artifact operation lock poisoned"))?;
        let dir = self.cap_dir().map_err(|error| {
            log::warn!("artifact store unavailable while resolving {id}: {error:#}");
            anyhow!("artifact {id} is unavailable")
        })?;
        let mut available = Vec::new();
        let mut matching = Vec::new();
        for (found_id, name) in list_artifacts(&dir)? {
            if found_id == id {
                matching.push(name);
            } else {
                available.push(found_id);
            }
        }
        anyhow::ensure!(
            matching.len() <= 1,
            "artifact {id} is ambiguous: multiple backing files exist"
        );
        if let Some(name) = matching.pop() {
            let mut options = OpenOptions::new();
            options.read(true).follow(FollowSymlinks::No);
            let file = dir
                .open_with(&name, &options)
                .with_context(|| format!("opening artifact {id}"))?;
            let metadata = file
                .metadata()
                .with_context(|| format!("inspecting artifact {id}"))?;
            anyhow::ensure!(metadata.is_file(), "artifact {id} is not a regular file");
            return Ok(file.into_std());
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
    /// Recover a legacy footer claim only when it byte-for-byte matches this
    /// store's canonical bounded rendering of the claimed durable artifact.
    ///
    /// This is the compatibility seam for journal rows written before typed
    /// `ToolResultArtifact` provenance existed. Syntax alone is never trusted.
    ///
    /// # Errors
    /// Returns an error only when a present artifact cannot be inspected.
    pub fn verified_legacy_inline_artifact_id(&self, inline: &str) -> Result<Option<u64>> {
        const PREFIX: &str = "[raw output: artifact://";
        let mut remaining = inline;
        while let Some(start) = remaining.find(PREFIX) {
            let digits = &remaining[start + PREFIX.len()..];
            let Some(end) = digits.find(']') else {
                return Ok(None);
            };
            let candidate = &digits[..end];
            remaining = &digits[end + 1..];
            if candidate.is_empty() || !candidate.bytes().all(|byte| byte.is_ascii_digit()) {
                continue;
            }
            let Ok(id) = candidate.parse::<u64>() else {
                continue;
            };
            let Ok(mut file) = self.resolve(id) else {
                continue;
            };
            let total_bytes = file
                .metadata()
                .with_context(|| format!("inspecting legacy artifact {id}"))?
                .len();
            let streamed_matches = canonical_streamed_inline_output_matches(
                &mut file,
                total_bytes,
                inline,
                self.inline_budget,
                id,
            )?;
            let inline_matches = if streamed_matches {
                false
            } else {
                canonical_inline_output_matches(
                    &mut file,
                    total_bytes,
                    inline,
                    self.inline_budget,
                    id,
                )
                // Invalid UTF-8 cannot be a standard string-backed spill. It
                // may still be a streamed spill (checked above); otherwise
                // the claim is simply unverified rather than a sweep outage.
                .unwrap_or(false)
            };
            if streamed_matches || inline_matches {
                return Ok(Some(id));
            }
        }
        Ok(None)
    }

    /// Enforce the shared inline budget on a tool result.
    ///
    /// Over-budget output is spilled byte-identical to the store and the
    /// inline text is replaced by a bounded head + tail window, an elision
    /// marker, and the [`artifact_footer`] recovery URI.
    ///
    /// # Errors
    /// Returns an error when the spill write fails; `result` is left
    /// untouched so callers can propagate the failure without destroying
    /// output bytes.
    pub fn apply_inline_budget(
        &self,
        result: &mut ToolResult,
        tool_name: &str,
    ) -> Result<Option<SavedArtifact>> {
        // Never trust caller-supplied provenance. Only this spill boundary may
        // attach an artifact ID, and only after a successful durable write.
        result.artifact = None;
        if result.output.len() <= self.inline_budget {
            return Ok(None);
        }
        let saved = self.save(tool_name, &result.output)?;
        result.data = None;
        result.output = cap_inline_output(&result.output, self.inline_budget, saved.id);
        result.artifact = Some(ToolResultArtifact { id: saved.id });
        Ok(Some(saved))
    }
}

/// Process-wide multiplexer handing out one [`ArtifactStore`] per active thread.
///
/// Stores live under `<root>/<thread_id>/` and are weakly cached: overlapping
/// callers share one allocator, while a completed operation releases its
/// strong handle so retention can reclaim stale output without waiting for a
/// process restart.
#[derive(Debug)]
pub struct ArtifactStorage {
    root: PathBuf,
    root_handle: Result<Arc<Dir>, String>,
    inline_budget: usize,
    max_bytes_per_thread: u64,
    stores: Mutex<HashMap<String, Weak<ArtifactStore>>>,
}
/// Existing artifact directories fenced before the host reads durable liveness.
#[derive(Debug)]
pub struct ArtifactSweepFence {
    threads: Vec<FencedArtifactThread>,
}

#[derive(Debug)]
struct FencedArtifactThread {
    key: String,
    dir: Arc<Dir>,
    _store: Arc<ArtifactStore>,
    activity_lock: Option<std::fs::File>,
    allocation_lock: Option<std::fs::File>,
}

impl ArtifactSweepFence {
    /// Apply deterministic retention using a snapshot captured after this
    /// fence was acquired.
    ///
    /// # Errors
    /// Returns an error when a fenced directory cannot be inspected, pruned,
    /// or synchronized.
    pub fn sweep(
        self,
        policy: ArtifactRetentionPolicy,
        snapshot: &ArtifactRetentionSnapshot,
        now: SystemTime,
    ) -> Result<ArtifactSweepReport> {
        if policy.max_age.is_none() && policy.max_bytes_per_thread.is_none() {
            return Ok(ArtifactSweepReport::default());
        }
        let mut report = ArtifactSweepReport::default();
        for thread in self.threads {
            let durable = snapshot.threads.get(&thread.key);
            if thread.activity_lock.is_none()
                || thread.allocation_lock.is_none()
                || durable.is_some_and(|state| state.live)
                || (durable.is_none() && snapshot.protect_unknown_threads)
            {
                report.protected_files += list_artifacts(&thread.dir)?.len();
                continue;
            }
            sweep_thread_dir(&thread.dir, durable, policy, now, &mut report)?;
            sync_directory(&thread.dir).context("syncing artifact directory after sweep")?;
        }
        Ok(report)
    }
}

impl ArtifactStorage {
    /// A storage root with the default inline budget.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let root_handle = open_confined_dir(&root, true)
            .map(Arc::new)
            .map_err(|error| format!("{error:#}"));
        Self {
            root,
            root_handle,
            inline_budget: DEFAULT_INLINE_OUTPUT_BUDGET_BYTES,
            max_bytes_per_thread: DEFAULT_MAX_ARTIFACT_BYTES_PER_THREAD,
            stores: Mutex::new(HashMap::new()),
        }
    }

    /// Override the shared inline budget applied to every per-thread store.
    #[must_use]
    pub fn with_inline_budget(mut self, bytes: usize) -> Self {
        self.inline_budget = bytes.max(MIN_INLINE_OUTPUT_BUDGET_BYTES);
        self
    }
    /// Override the hard per-thread artifact admission quota.
    #[must_use]
    pub fn with_max_bytes_per_thread(mut self, bytes: u64) -> Self {
        self.max_bytes_per_thread = bytes.max(self.inline_budget as u64);
        self
    }

    /// The storage root directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The store for `thread_id`, creating and weakly caching it on first use.
    ///
    /// The caller must hold the returned handle until any recovery footer is
    /// durably committed; retention treats a live strong handle as in-flight.
    ///
    /// # Errors
    /// Returns an error when the store cache lock is poisoned.
    pub fn for_thread(&self, thread_id: &ThreadId) -> Result<Arc<ArtifactStore>> {
        let key = artifact_thread_key(thread_id);
        let cached = {
            let stores = self
                .stores
                .lock()
                .map_err(|_| anyhow!("artifact storage cache lock poisoned"))?;
            let cached = stores.get(&key).and_then(Weak::upgrade);
            drop(stores);
            cached
        };
        if let Some(store) = cached {
            store.mark_acquired()?;
            return Ok(store);
        }

        let root = self
            .root_handle
            .as_ref()
            .map(Arc::clone)
            .map_err(|error| anyhow!("{error}"))?;
        let (thread_key, thread_dir) = {
            let _root_lock = lock_artifact_directory(&root)?;
            let thread_key = select_thread_directory_key(&root, thread_id, &key)?;
            let thread_dir = open_child_dir(&root, &thread_key)?;
            (thread_key, thread_dir)
        };
        let candidate = Arc::new(
            ArtifactStore::from_capability(self.root.join(thread_key), Ok(thread_dir))
                .with_inline_budget(self.inline_budget)
                .with_max_bytes_per_thread(self.max_bytes_per_thread),
        );
        let store = {
            let mut stores = self
                .stores
                .lock()
                .map_err(|_| anyhow!("artifact storage cache lock poisoned"))?;
            stores.get(&key).and_then(Weak::upgrade).unwrap_or_else(|| {
                stores.insert(key, Arc::downgrade(&candidate));
                candidate
            })
        };
        store.mark_acquired()?;
        Ok(store)
    }

    /// Copy the source artifacts referenced at a fork boundary into the
    /// unexposed destination while preserving IDs and the allocator fence.
    ///
    /// Re-driving a partial copy is idempotent only when every colliding ID is
    /// byte-identical. Any conflicting destination bytes fail closed.
    ///
    /// # Errors
    /// Returns an error on quota overflow, ambiguous/colliding IDs, or any
    /// durability failure.
    pub fn copy_thread_artifacts(
        &self,
        source_thread_id: &ThreadId,
        destination_thread_id: &ThreadId,
        artifact_ids: &BTreeSet<u64>,
    ) -> Result<()> {
        anyhow::ensure!(
            source_thread_id != destination_thread_id,
            "cannot copy an artifact namespace onto itself"
        );
        let source = self.for_thread(source_thread_id)?;
        let destination = self.for_thread(destination_thread_id)?;
        let source_first = source.dir <= destination.dir;
        let (_first_operation, _second_operation) = if source_first {
            (
                source
                    .operation_gate
                    .lock()
                    .map_err(|_| anyhow!("source artifact operation lock poisoned"))?,
                destination
                    .operation_gate
                    .lock()
                    .map_err(|_| anyhow!("destination artifact operation lock poisoned"))?,
            )
        } else {
            (
                destination
                    .operation_gate
                    .lock()
                    .map_err(|_| anyhow!("destination artifact operation lock poisoned"))?,
                source
                    .operation_gate
                    .lock()
                    .map_err(|_| anyhow!("source artifact operation lock poisoned"))?,
            )
        };
        let source_dir = source.cap_dir()?;
        let destination_dir = destination.cap_dir()?;
        let (_first_allocation, _second_allocation) = if source_first {
            (
                lock_artifact_directory(&source_dir)?,
                lock_artifact_directory(&destination_dir)?,
            )
        } else {
            (
                lock_artifact_directory(&destination_dir)?,
                lock_artifact_directory(&source_dir)?,
            )
        };

        let source_next = ensure_allocator_next(&source_dir)?;
        let destination_next = ensure_allocator_next(&destination_dir)?;
        if destination_next < source_next {
            // Seed the destination fence before publishing any copied ID.
            persist_allocator_next(&destination_dir, source_next)?;
        }
        let source_artifacts = unique_artifacts_by_id(&source_dir)?;
        let destination_artifacts = unique_artifacts_by_id(&destination_dir)?;
        let mut admitted_bytes = artifact_bytes(&destination_dir)?;
        for (id, source_name) in source_artifacts {
            if !artifact_ids.contains(&id) {
                continue;
            }
            if let Some(destination_name) = destination_artifacts.get(&id) {
                anyhow::ensure!(
                    artifact_files_equal(
                        &source_dir,
                        &source_name,
                        &destination_dir,
                        destination_name,
                    )?,
                    "destination artifact {id} collides with different bytes"
                );
                continue;
            }
            let source_len = artifact_file_len(&source_dir, &source_name)?;
            admitted_bytes = admitted_bytes
                .checked_add(source_len)
                .context("forked artifact quota byte count overflow")?;
            anyhow::ensure!(
                admitted_bytes <= destination.max_bytes_per_thread,
                "forked artifact namespace exceeds destination quota"
            );
            source_dir
                .hard_link(&source_name, &destination_dir, &source_name)
                .with_context(|| format!("copying fork artifact {id}"))?;
        }
        sync_directory(&destination_dir).context("syncing forked artifact namespace")
    }

    /// Fence all existing thread directories before the host captures its
    /// durable liveness/reference snapshot. New writers take a shared activity
    /// lock; this pass holds the exclusive lock through deletion.
    ///
    /// # Errors
    /// Returns an error when the storage root, activity locks, or allocation
    /// locks cannot be inspected or acquired.
    pub fn begin_sweep(&self) -> Result<ArtifactSweepFence> {
        let root = self
            .root_handle
            .as_ref()
            .map(Arc::clone)
            .map_err(|error| anyhow!("{error}"))?;
        let mut thread_dirs = Vec::new();
        for entry in root
            .entries()
            .with_context(|| format!("reading artifact root {}", self.root.display()))?
        {
            let entry =
                entry.with_context(|| format!("reading an entry in {}", self.root.display()))?;
            let file_type = entry.file_type().context("reading artifact entry type")?;
            if !file_type.is_dir() || file_type.is_symlink() {
                continue;
            }
            let name = entry.file_name();
            let Some(key) = name.to_str().map(str::to_owned) else {
                continue;
            };
            let dir = Arc::new(
                root.open_dir_nofollow(&name)
                    .with_context(|| format!("opening artifact thread directory {key}"))?,
            );
            thread_dirs.push((key, dir));
        }
        thread_dirs.sort_by(|left, right| left.0.cmp(&right.0));

        let stores_for_sweep = {
            let mut stores = self
                .stores
                .lock()
                .map_err(|_| anyhow!("artifact storage cache lock poisoned"))?;
            stores.retain(|_, store| store.strong_count() > 0);
            let mut selected = Vec::new();
            for (key, dir) in thread_dirs {
                let existing = stores.get(&key).and_then(Weak::upgrade);
                let (store, was_active) = existing.map_or_else(
                    || {
                        let store = Arc::new(
                            ArtifactStore::from_capability(
                                self.root.join(&key),
                                Ok(Arc::clone(&dir)),
                            )
                            .with_inline_budget(self.inline_budget)
                            .with_max_bytes_per_thread(self.max_bytes_per_thread),
                        );
                        stores.insert(key.clone(), Arc::downgrade(&store));
                        (store, false)
                    },
                    |store| (store, true),
                );
                selected.push((key, dir, store, was_active));
            }
            drop(stores);
            selected
        };

        let mut threads = Vec::with_capacity(stores_for_sweep.len());
        for (key, dir, store, was_active) in stores_for_sweep {
            let activity_lock = if was_active {
                None
            } else {
                try_lock_inactive_artifact_store(&dir)?
            };
            let allocation_lock = if activity_lock.is_some() {
                Some(lock_artifact_directory(&dir)?)
            } else {
                None
            };
            if allocation_lock.is_some() {
                ensure_allocator_next(&dir)?;
            }
            threads.push(FencedArtifactThread {
                key,
                dir,
                _store: store,
                activity_lock,
                allocation_lock,
            });
        }
        Ok(ArtifactSweepFence { threads })
    }
}

/// Stable directory key used by [`ArtifactStorage`] and host retention snapshots.
#[must_use]
pub fn artifact_thread_key(thread_id: &ThreadId) -> String {
    format!("t-{:x}", Sha256::digest(thread_id.0.as_bytes()))
}

fn sanitize_path_component(component: &str) -> String {
    let cleaned: String = component
        .chars()
        .take(128)
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "thread".to_owned()
    } else {
        cleaned
    }
}

fn legacy_artifact_thread_key(thread_id: &ThreadId) -> String {
    sanitize_path_component(&thread_id.0)
}

fn select_thread_directory_key(
    root: &Dir,
    thread_id: &ThreadId,
    hashed_key: &str,
) -> Result<String> {
    let legacy_key = legacy_artifact_thread_key(thread_id);
    if legacy_key == hashed_key {
        return Ok(hashed_key.to_owned());
    }
    let legacy_metadata = match root.symlink_metadata(&legacy_key) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(hashed_key.to_owned());
        }
        Err(error) => return Err(error).context("inspecting legacy artifact thread directory"),
    };
    anyhow::ensure!(
        legacy_metadata.is_dir() && !legacy_metadata.file_type().is_symlink(),
        "legacy artifact thread path is not a real directory"
    );
    match root.symlink_metadata(hashed_key) {
        Ok(_) => {
            return Err(anyhow!(
                "legacy and hashed artifact thread directories both exist; refusing to hide recoverable output"
            ));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).context("inspecting hashed artifact thread directory"),
    }

    let legacy_dir = root
        .open_dir_nofollow(&legacy_key)
        .context("opening legacy artifact thread directory")?;
    let Some(_activity_lock) = try_lock_inactive_artifact_store(&legacy_dir)? else {
        return Ok(legacy_key);
    };
    let _allocation_lock = lock_artifact_directory(&legacy_dir)?;
    root.rename(&legacy_key, root, hashed_key)
        .context("migrating legacy artifact thread directory")?;
    sync_directory(root).context("syncing migrated artifact root")?;
    Ok(hashed_key.to_owned())
}

#[derive(Debug)]
struct RetentionEntry {
    id: u64,
    name: OsString,
    len: u64,
    modified: SystemTime,
}

fn sweep_thread_dir(
    dir: &Dir,
    durable: Option<&ArtifactThreadRetention>,
    policy: ArtifactRetentionPolicy,
    now: SystemTime,
    report: &mut ArtifactSweepReport,
) -> Result<()> {
    sweep_stale_partials(dir, now, report)?;
    let referenced = durable.map(|state| &state.referenced_ids);
    let mut total_bytes = 0_u64;
    let mut candidates = Vec::new();
    for (id, name) in list_artifacts(dir)? {
        let metadata = match dir.symlink_metadata(&name) {
            Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => metadata,
            Ok(_) => continue,
            Err(_) => {
                report.protected_files += 1;
                continue;
            }
        };
        total_bytes = total_bytes.saturating_add(metadata.len());
        if referenced.is_some_and(|ids| ids.contains(&id)) {
            report.protected_files += 1;
            continue;
        }
        let Ok(modified) = metadata.modified() else {
            report.protected_files += 1;
            continue;
        };
        candidates.push(RetentionEntry {
            id,
            name,
            len: metadata.len(),
            modified: modified.into_std(),
        });
    }
    candidates.sort_by(|left, right| {
        (left.modified, left.id, &left.name).cmp(&(right.modified, right.id, &right.name))
    });

    let cutoff = policy.max_age.and_then(|age| now.checked_sub(age));
    for candidate in &candidates {
        if cutoff.is_some_and(|cutoff| candidate.modified < cutoff)
            && remove_retention_candidate(dir, candidate, report)
        {
            total_bytes = total_bytes.saturating_sub(candidate.len);
        }
    }

    if let Some(max_bytes) = policy.max_bytes_per_thread {
        for candidate in &candidates {
            if total_bytes <= max_bytes {
                break;
            }
            if remove_retention_candidate(dir, candidate, report) {
                total_bytes = total_bytes.saturating_sub(candidate.len);
            }
        }
        if total_bytes > max_bytes {
            report.threads_over_quota += 1;
        }
    }
    Ok(())
}

fn remove_retention_candidate(
    dir: &Dir,
    candidate: &RetentionEntry,
    report: &mut ArtifactSweepReport,
) -> bool {
    match dir.remove_file(&candidate.name) {
        Ok(()) => {
            report.files_removed += 1;
            report.bytes_removed = report.bytes_removed.saturating_add(candidate.len);
            true
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(_) => {
            report.protected_files += 1;
            false
        }
    }
}

/// Bounded inline replacement for an over-budget output: head + elision
/// marker + tail + recovery footer, all within `budget` bytes.
///
/// Public so readers of pre-provenance journals can authenticate a legacy
/// footer by reproducing the exact spill boundary output byte-for-byte.
#[must_use]
pub fn cap_inline_output(full: &str, budget: usize, artifact_id: u64) -> String {
    let footer = artifact_footer(artifact_id);
    let head_budget = budget * 3 / 5;
    let tail_budget = budget / 4;
    let head_end = floor_char_boundary(full, head_budget);
    let tail_start = ceil_char_boundary(full, full.len().saturating_sub(tail_budget));
    if tail_start <= head_end {
        return format!("{}\n{footer}", &full[..head_end]);
    }
    let elided = tail_start - head_end;
    format!(
        "{head}\n[... {elided} bytes elided ...]\n{tail}\n{footer}",
        head = &full[..head_end],
        tail = &full[tail_start..],
    )
}

/// Compare a legacy inline spill boundary with the durable artifact without
/// buffering or scanning the artifact body.
///
/// The comparison reproduces [`cap_inline_output`] from bounded head and tail
/// windows. At most roughly 85% of `budget` is read and allocated regardless
/// of artifact size.
///
/// # Errors
/// Returns an error when seeking, reading, sizing, or UTF-8 validation fails.
pub fn canonical_inline_output_matches<R: Read + Seek>(
    reader: &mut R,
    total_bytes: u64,
    inline: &str,
    budget: usize,
    artifact_id: u64,
) -> Result<bool> {
    let total = usize::try_from(total_bytes).context("artifact size exceeds usize")?;
    let head_budget = budget.saturating_mul(3) / 5;
    let tail_budget = budget / 4;

    let head_read_len = total.min(head_budget.saturating_add(1));
    let mut head = vec![0_u8; head_read_len];
    reader
        .seek(SeekFrom::Start(0))
        .context("seeking artifact head")?;
    reader
        .read_exact(&mut head)
        .context("reading artifact head")?;
    let head_end = if head_budget >= total {
        total
    } else {
        let mut boundary = head_budget;
        while boundary > 0 && head[boundary] & 0b1100_0000 == 0b1000_0000 {
            boundary -= 1;
        }
        boundary
    };

    let tail_nominal = total.saturating_sub(tail_budget);
    let tail_len = total.saturating_sub(tail_nominal);
    let mut tail = vec![0_u8; tail_len];
    reader
        .seek(SeekFrom::Start(
            u64::try_from(tail_nominal).context("artifact tail offset exceeds u64")?,
        ))
        .context("seeking artifact tail")?;
    reader
        .read_exact(&mut tail)
        .context("reading artifact tail")?;
    let mut tail_offset = 0;
    while tail_offset < tail.len() && tail[tail_offset] & 0b1100_0000 == 0b1000_0000 {
        tail_offset += 1;
    }
    let tail_start = tail_nominal.saturating_add(tail_offset);

    let head = std::str::from_utf8(&head[..head_end]).context("artifact head is not UTF-8")?;
    let footer = artifact_footer(artifact_id);
    if tail_start <= head_end {
        return Ok(format!("{head}\n{footer}") == inline);
    }
    let tail = std::str::from_utf8(&tail[tail_offset..]).context("artifact tail is not UTF-8")?;
    let elided = tail_start - head_end;
    Ok(format!("{head}\n[... {elided} bytes elided ...]\n{tail}\n{footer}") == inline)
}

/// Compare the bounded output produced by [`cap_inline_from_windows`] with its
/// durable artifact while reading only bounded head and tail windows.
///
/// The bash capture path converts each raw window with `from_utf8_lossy`
/// before applying its display budget. Reproduce that order exactly so invalid
/// UTF-8 remains verifiable without weakening the byte-exact artifact check.
///
/// # Errors
/// Returns an error when seeking, reading, or sizing fails.
pub fn canonical_streamed_inline_output_matches<R: Read + Seek>(
    reader: &mut R,
    total_bytes: u64,
    inline: &str,
    budget: usize,
    artifact_id: u64,
) -> Result<bool> {
    let total = usize::try_from(total_bytes).context("artifact size exceeds usize")?;
    let window_bytes = budget.max(MIN_INLINE_OUTPUT_BUDGET_BYTES);
    let head_len = total.min(window_bytes);
    let mut head = vec![0_u8; head_len];
    reader
        .seek(SeekFrom::Start(0))
        .context("seeking streamed artifact head")?;
    reader
        .read_exact(&mut head)
        .context("reading streamed artifact head")?;

    let tail_len = total.min(window_bytes);
    let tail_start = total.saturating_sub(tail_len);
    let mut tail = vec![0_u8; tail_len];
    reader
        .seek(SeekFrom::Start(
            u64::try_from(tail_start).context("streamed artifact tail offset exceeds u64")?,
        ))
        .context("seeking streamed artifact tail")?;
    reader
        .read_exact(&mut tail)
        .context("reading streamed artifact tail")?;

    let head = String::from_utf8_lossy(&head);
    let tail = String::from_utf8_lossy(&tail);
    Ok(cap_inline_from_windows(&head, &tail, total_bytes, budget, artifact_id) == inline)
}

/// Render already-captured head and tail windows with the canonical footer.
#[must_use]
pub fn cap_inline_from_windows(
    head: &str,
    tail: &str,
    total_bytes: u64,
    budget: usize,
    artifact_id: u64,
) -> String {
    let budget = budget.max(MIN_INLINE_OUTPUT_BUDGET_BYTES);
    let footer = artifact_footer(artifact_id);
    let structural_bytes = footer.len() + 64;
    let payload_budget = budget.saturating_sub(structural_bytes);
    let head_budget = payload_budget * 3 / 4;
    let tail_budget = payload_budget.saturating_sub(head_budget);
    let head_end = floor_char_boundary(head, head.len().min(head_budget));
    let tail_start = ceil_char_boundary(tail, tail.len().saturating_sub(tail_budget));
    let kept_head = &head[..head_end];
    let kept_tail = &tail[tail_start..];
    let retained = u64::try_from(kept_head.len().saturating_add(kept_tail.len()))
        .map_or(u64::MAX, |bytes| bytes);
    let elided = total_bytes.saturating_sub(retained);
    format!("{kept_head}\n[... {elided} bytes elided ...]\n{kept_tail}\n{footer}")
}

/// Flush directory entry updates through a syncable capability-derived handle.
///
/// On Unix, a [`Dir`] may wrap an `O_PATH` descriptor that rejects `fsync`;
/// reopening `.` produces a real readable directory descriptor. On Windows,
/// directory handles require backup semantics, and flushing requires write
/// access. Both paths stay relative to the existing capability.
fn sync_directory(dir: &Dir) -> Result<()> {
    #[cfg(unix)]
    {
        let mut options = OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No);
        dir.open_with(".", &options)
            .context("opening directory for sync")?
            .sync_all()
            .context("syncing directory")?;
        Ok(())
    }
    #[cfg(windows)]
    {
        use cap_std::fs::OpenOptionsExt as _;
        use windows_sys::Win32::Storage::FileSystem::{
            FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT, FILE_SHARE_READ,
            FILE_SHARE_WRITE,
        };

        let mut options = OpenOptions::new();
        options
            .read(true)
            .write(true)
            .follow(FollowSymlinks::No)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE);
        dir.open_with(".", &options)
            .context("opening directory for metadata flush")?
            .sync_all()
            .context("flushing directory metadata")?;
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        Err(anyhow!(
            "durable artifact directory metadata sync is unsupported on this platform"
        ))
    }
}

/// Open `path` as a stable capability while rejecting a symlink at its final
/// component. Relative operations through the returned handle cannot escape.
fn open_confined_dir(path: &Path, create: bool) -> Result<Dir> {
    let parent = path
        .parent()
        .context("artifact directory must have a parent")?;
    let name = path
        .file_name()
        .context("artifact directory must have a final component")?;
    if create {
        let mut builder = std::fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        builder.mode(0o700);
        builder
            .create(parent)
            .with_context(|| format!("creating artifact parent {}", parent.display()))?;
    }
    let parent_dir = Dir::open_ambient_dir(parent, ambient_authority())
        .with_context(|| format!("opening artifact parent {}", parent.display()))?;
    match parent_dir.symlink_metadata(name) {
        Ok(metadata) => {
            anyhow::ensure!(
                metadata.is_dir() && !metadata.file_type().is_symlink(),
                "artifact directory {} is not a real directory",
                path.display()
            );
        }
        Err(error) if create && error.kind() == std::io::ErrorKind::NotFound => {
            let builder = cap_std::fs::DirBuilder::new();
            #[cfg(unix)]
            let mut builder = builder;
            #[cfg(unix)]
            builder.mode(0o700);
            parent_dir
                .create_dir_with(name, &builder)
                .with_context(|| format!("creating artifact directory {}", path.display()))?;
        }
        Err(error) => {
            return Err(error)
                .with_context(|| format!("inspecting artifact directory {}", path.display()));
        }
    }
    let dir = parent_dir
        .open_dir_nofollow(name)
        .with_context(|| format!("opening artifact directory {}", path.display()))?;
    #[cfg(unix)]
    dir.set_permissions(".", cap_std::fs::Permissions::from_mode(0o700))
        .with_context(|| format!("securing artifact directory {}", path.display()))?;
    Ok(dir)
}

fn open_child_dir(parent: &Dir, name: &str) -> Result<Arc<Dir>> {
    match parent.symlink_metadata(name) {
        Ok(metadata) => anyhow::ensure!(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "artifact child is not a real directory"
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let builder = cap_std::fs::DirBuilder::new();
            #[cfg(unix)]
            let mut builder = builder;
            #[cfg(unix)]
            builder.mode(0o700);
            match parent.create_dir_with(name, &builder) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error).context("creating artifact child directory"),
            }
        }
        Err(error) => return Err(error).context("inspecting artifact child directory"),
    }
    let dir = parent
        .open_dir_nofollow(name)
        .context("opening artifact child directory")?;
    #[cfg(unix)]
    dir.set_permissions(".", cap_std::fs::Permissions::from_mode(0o700))
        .context("securing artifact child directory")?;
    Ok(Arc::new(dir))
}
fn open_lock_file(dir: &Dir, name: &str) -> Result<std::fs::File> {
    let mut options = OpenOptions::new();
    options
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .follow(FollowSymlinks::No);
    #[cfg(unix)]
    options.mode(0o600);
    dir.open_with(name, &options)
        .with_context(|| format!("opening artifact lock {name}"))
        .map(cap_std::fs::File::into_std)
}

fn open_activity_lock(dir: &Dir) -> Result<std::fs::File> {
    open_lock_file(dir, ".activity.lock")
}

fn lock_artifact_directory(dir: &Dir) -> Result<std::fs::File> {
    let file = open_lock_file(dir, ".allocation.lock")?;
    file.lock().context("locking artifact allocation")?;
    Ok(file)
}

fn try_lock_inactive_artifact_store(dir: &Dir) -> Result<Option<std::fs::File>> {
    let file = open_activity_lock(dir)?;
    match file.try_lock() {
        Ok(()) => Ok(Some(file)),
        Err(std::fs::TryLockError::WouldBlock) => Ok(None),
        Err(std::fs::TryLockError::Error(error)) => {
            Err(error).context("locking inactive artifact store")
        }
    }
}

/// Next free ID: one past the highest `<id>.` file in `dir`.
fn scan_next_id(dir: &Dir) -> Result<u64> {
    let max = list_artifacts(dir)?.into_iter().map(|(id, _)| id).max();
    max.map_or(Ok(0), |max| {
        max.checked_add(1).context("artifact ID space exhausted")
    })
}

/// Read the latest complete fixed-width allocator record without scanning an
/// attacker-sized marker. A non-empty marker with no valid record fails closed.
fn read_allocator_next(dir: &Dir) -> Result<Option<u64>> {
    let mut options = OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    let mut file = match dir.open_with(ARTIFACT_ALLOCATOR_MARKER, &options) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).context("opening artifact allocator marker"),
    };
    let len = file
        .metadata()
        .context("inspecting artifact allocator marker")?
        .len();
    if len == 0 {
        anyhow::bail!("artifact allocator marker is empty");
    }
    let tail_len = usize::try_from(len.min((ARTIFACT_ALLOCATOR_RECORD_BYTES * 3) as u64))
        .context("allocator marker tail exceeds usize")?;
    file.seek(SeekFrom::End(
        -i64::try_from(tail_len).context("allocator marker tail exceeds i64")?,
    ))
    .context("seeking artifact allocator marker")?;
    let mut tail = vec![0_u8; tail_len];
    file.read_exact(&mut tail)
        .context("reading artifact allocator marker")?;
    let tail = std::str::from_utf8(&tail).context("artifact allocator marker is not UTF-8")?;
    tail.lines()
        .rev()
        .find_map(|line| {
            (line.len() == ARTIFACT_ALLOCATOR_RECORD_BYTES - 1
                && line.bytes().all(|byte| byte.is_ascii_digit()))
            .then(|| line.parse::<u64>().ok())
            .flatten()
        })
        .map(Some)
        .ok_or_else(|| anyhow!("artifact allocator marker has no complete record"))
}

/// Append and fsync one allocator high-watermark record.
fn persist_allocator_next(dir: &Dir, next: u64) -> Result<()> {
    let record = format!("{next:020}\n");
    anyhow::ensure!(
        record.len() == ARTIFACT_ALLOCATOR_RECORD_BYTES,
        "artifact allocator ID exceeds fixed-width marker"
    );
    let mut options = OpenOptions::new();
    options
        .write(true)
        .append(true)
        .create(true)
        .follow(FollowSymlinks::No);
    #[cfg(unix)]
    options.mode(0o600);
    let mut marker = dir
        .open_with(ARTIFACT_ALLOCATOR_MARKER, &options)
        .context("opening artifact allocator marker for append")?;
    marker
        .write_all(record.as_bytes())
        .context("appending artifact allocator marker")?;
    marker
        .sync_all()
        .context("syncing artifact allocator marker")?;
    sync_directory(dir).context("syncing artifact allocator directory")
}

/// Migrate a legacy directory once and reconcile the durable watermark with
/// any externally restored artifacts.
fn ensure_allocator_next(dir: &Dir) -> Result<u64> {
    let scanned = scan_next_id(dir)?;
    match read_allocator_next(dir) {
        Ok(Some(marked)) => {
            let next = marked.max(scanned);
            if next != marked {
                persist_allocator_next(dir, next)?;
            }
            Ok(next)
        }
        Ok(None) => {
            persist_allocator_next(dir, scanned)?;
            Ok(scanned)
        }
        Err(error) => Err(error),
    }
}

fn reserve_artifact_id(dir: &Dir) -> Result<u64> {
    let id = ensure_allocator_next(dir)?;
    let next = id.checked_add(1).context("artifact ID space exhausted")?;
    persist_allocator_next(dir, next)?;
    Ok(id)
}

/// Every regular artifact-looking entry in `dir`, represented only by its
/// handle-relative name so later metadata and deletion remain confined.
fn list_artifacts(dir: &Dir) -> Result<Vec<(u64, OsString)>> {
    let mut artifacts = Vec::new();
    for entry in dir.entries().context("reading artifact directory")? {
        let entry = entry.context("reading artifact directory entry")?;
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        let Some((id_part, _)) = name_str.split_once('.') else {
            continue;
        };
        if let Ok(id) = id_part.parse::<u64>() {
            artifacts.push((id, name));
        }
    }
    Ok(artifacts)
}
fn artifact_file_len(dir: &Dir, name: &OsStr) -> Result<u64> {
    let metadata = dir
        .symlink_metadata(name)
        .context("inspecting artifact file")?;
    anyhow::ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "artifact namespace contains a non-regular entry"
    );
    Ok(metadata.len())
}

fn unique_artifacts_by_id(dir: &Dir) -> Result<BTreeMap<u64, OsString>> {
    let mut by_id = BTreeMap::new();
    for (id, name) in list_artifacts(dir)? {
        artifact_file_len(dir, &name)?;
        anyhow::ensure!(
            by_id.insert(id, name).is_none(),
            "artifact {id} is ambiguous: multiple backing files exist"
        );
    }
    Ok(by_id)
}

fn artifact_files_equal(
    left_dir: &Dir,
    left_name: &OsStr,
    right_dir: &Dir,
    right_name: &OsStr,
) -> Result<bool> {
    if artifact_file_len(left_dir, left_name)? != artifact_file_len(right_dir, right_name)? {
        return Ok(false);
    }
    let mut options = OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    let mut left = left_dir
        .open_with(left_name, &options)
        .context("opening source artifact for comparison")?;
    let mut right = right_dir
        .open_with(right_name, &options)
        .context("opening destination artifact for comparison")?;
    let mut left_buffer = [0_u8; 64 * 1024];
    let mut right_buffer = [0_u8; 64 * 1024];
    loop {
        let left_read = left
            .read(&mut left_buffer)
            .context("reading source artifact for comparison")?;
        let right_read = right
            .read(&mut right_buffer)
            .context("reading destination artifact for comparison")?;
        if left_read != right_read || left_buffer[..left_read] != right_buffer[..right_read] {
            return Ok(false);
        }
        if left_read == 0 {
            return Ok(true);
        }
    }
}

fn artifact_bytes(dir: &Dir) -> Result<u64> {
    let mut total = 0_u64;
    for entry in dir
        .entries()
        .context("reading artifact directory for quota")?
    {
        let entry = entry.context("reading artifact directory entry for quota")?;
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        let is_artifact = name_str
            .split_once('.')
            .is_some_and(|(id, _)| id.parse::<u64>().is_ok());
        if !is_artifact && !is_partial_name(name_str) {
            continue;
        }
        let metadata = dir
            .symlink_metadata(&name)
            .context("inspecting artifact for quota")?;
        anyhow::ensure!(
            metadata.is_file() && !metadata.file_type().is_symlink(),
            "artifact quota encountered a non-regular entry"
        );
        total = total
            .checked_add(metadata.len())
            .context("artifact quota byte count overflow")?;
    }
    Ok(total)
}

fn is_partial_name(name: &str) -> bool {
    name.starts_with(".partial-") || name.starts_with(".partial.")
}

fn sweep_stale_partials(
    dir: &Dir,
    now: SystemTime,
    report: &mut ArtifactSweepReport,
) -> Result<()> {
    let cutoff = now
        .checked_sub(STALE_PARTIAL_MAX_AGE)
        .context("partial artifact retention cutoff underflow")?;
    for entry in dir
        .entries()
        .context("reading artifact directory for partial sweep")?
    {
        let entry = entry.context("reading partial artifact entry")?;
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        if !is_partial_name(name_str) {
            continue;
        }
        let metadata = match dir.symlink_metadata(&name) {
            Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => metadata,
            Ok(_) | Err(_) => {
                report.protected_files += 1;
                continue;
            }
        };
        let Ok(modified) = metadata.modified() else {
            report.protected_files += 1;
            continue;
        };
        if modified.into_std() < cutoff {
            match dir.remove_file(&name) {
                Ok(()) => {
                    report.files_removed += 1;
                    report.bytes_removed = report.bytes_removed.saturating_add(metadata.len());
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(_) => report.protected_files += 1,
            }
        } else {
            report.protected_files += 1;
        }
    }
    Ok(())
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

    const CHILD_ROOT_ENV: &str = "AGENT_SDK_ARTIFACT_TEST_ROOT";
    const CHILD_INDEX_ENV: &str = "AGENT_SDK_ARTIFACT_TEST_INDEX";

    fn wait_for_path(path: &Path) -> Result<()> {
        for _ in 0..500 {
            if path.exists() {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        anyhow::bail!("timed out waiting for {}", path.display())
    }

    #[test]
    fn multiprocess_allocator_child() -> Result<()> {
        let Some(root) = std::env::var_os(CHILD_ROOT_ENV).map(PathBuf::from) else {
            return Ok(());
        };
        let Some(index) = std::env::var_os(CHILD_INDEX_ENV) else {
            return Ok(());
        };
        let index = index
            .into_string()
            .map_err(|_| anyhow!("child index is not UTF-8"))?;
        let store = ArtifactStore::new(root.clone());
        let saved = store.save("mcp", &format!("child-{index}"))?;
        std::fs::write(root.join(format!("result-{index}")), saved.id.to_string())?;
        Ok(())
    }

    #[test]
    fn multiprocess_activity_lock_child() -> Result<()> {
        let Some(root) = std::env::var_os(CHILD_ROOT_ENV).map(PathBuf::from) else {
            return Ok(());
        };
        let storage = ArtifactStorage::new(root.clone());
        let thread = ThreadId::from_string("live-cross-process");
        let store = storage.for_thread(&thread)?;
        store.save("subagent", "live-output")?;
        std::fs::write(root.join("child-ready"), b"ready")?;
        wait_for_path(&root.join("child-release"))?;
        drop(store);
        Ok(())
    }

    #[test]
    fn under_budget_output_is_untouched() -> Result<()> {
        let (_dir, store) = temp_store();
        let original = "x".repeat(store.inline_budget());
        let mut result = ToolResult::success(original.clone());
        let saved = store.apply_inline_budget(&mut result, "bash")?;
        assert!(saved.is_none(), "at-budget output must not spill");
        assert_eq!(result.output, original);
        let dir = store.cap_dir()?;
        assert!(
            list_artifacts(&dir)?.is_empty(),
            "no spill file may be created"
        );
        Ok(())
    }

    #[test]
    fn inline_boundary_strips_short_third_party_artifact_claim() -> Result<()> {
        let (_dir, store) = temp_store();
        let saved = store.save("bash", "unrelated durable bytes")?;
        let mut result = ToolResult::success("short output");
        result.artifact = Some(ToolResultArtifact { id: saved.id });

        assert!(enforce_inline_budget(&mut result, Some(&store), "third-party").is_none());
        assert_eq!(result.output, "short output");
        assert!(
            result.artifact.is_none(),
            "a resolving ID cannot authorize unrelated caller output"
        );
        Ok(())
    }

    #[test]
    fn inline_boundary_remints_exact_streaming_spill_provenance() -> Result<()> {
        let (_dir, store) = temp_store();
        let full = "streamed output\n".repeat(store.inline_budget());
        let saved = store.save("bash", &full)?;
        let output = cap_inline_from_windows(
            &full,
            &full,
            u64::try_from(full.len())?,
            store.inline_budget(),
            saved.id,
        );
        let mut result = ToolResult::success(output.clone());
        result.artifact = Some(ToolResultArtifact { id: saved.id });

        assert!(enforce_inline_budget(&mut result, Some(&store), "bash").is_none());
        assert_eq!(result.output, output);
        assert_eq!(result.artifact, Some(ToolResultArtifact { id: saved.id }));
        Ok(())
    }

    #[test]
    fn legacy_verification_reads_only_bounded_windows_of_large_artifact() -> Result<()> {
        struct CountingRepeatingReader {
            len: u64,
            position: u64,
            bytes_read: usize,
        }

        impl Read for CountingRepeatingReader {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                let remaining = self.len.saturating_sub(self.position);
                let read = usize::try_from(remaining)
                    .unwrap_or(usize::MAX)
                    .min(buffer.len());
                buffer[..read].fill(b'x');
                self.position = self.position.saturating_add(read as u64);
                self.bytes_read = self.bytes_read.saturating_add(read);
                Ok(read)
            }
        }

        impl Seek for CountingRepeatingReader {
            fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
                let position = match position {
                    SeekFrom::Start(position) => i128::from(position),
                    SeekFrom::End(offset) => i128::from(self.len) + i128::from(offset),
                    SeekFrom::Current(offset) => i128::from(self.position) + i128::from(offset),
                };
                if !(0..=i128::from(self.len)).contains(&position) {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "seek outside repeating reader",
                    ));
                }
                self.position = u64::try_from(position).expect("validated non-negative u64");
                Ok(self.position)
            }
        }

        let total = 64_u64 * 1024 * 1024;
        let budget = DEFAULT_INLINE_OUTPUT_BUDGET_BYTES;
        let artifact_id = 7;
        let head_budget = budget * 3 / 5;
        let tail_budget = budget / 4;
        let elided = usize::try_from(total)? - head_budget - tail_budget;
        let canonical = format!(
            "{}\n[... {elided} bytes elided ...]\n{}\n{}",
            "x".repeat(head_budget),
            "x".repeat(tail_budget),
            artifact_footer(artifact_id),
        );
        let max_read = head_budget + 1 + tail_budget;
        let mut reader = CountingRepeatingReader {
            len: total,
            position: 0,
            bytes_read: 0,
        };
        assert!(canonical_inline_output_matches(
            &mut reader,
            total,
            &canonical,
            budget,
            artifact_id,
        )?);
        assert!(reader.bytes_read <= max_read);

        let forged = canonical.replacen('x', "y", 1);
        reader.bytes_read = 0;
        assert!(!canonical_inline_output_matches(
            &mut reader,
            total,
            &forged,
            budget,
            artifact_id,
        )?);
        assert!(
            reader.bytes_read <= max_read,
            "forged valid-ID footer must not trigger an artifact-sized read"
        );
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
        let mut resolved = store.resolve(saved.id)?;
        let mut content = String::new();
        resolved.read_to_string(&mut content)?;
        assert_eq!(content, "content");
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
    fn sweep_keeps_references_and_applies_expiry_then_quota() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let storage = ArtifactStorage::new(dir.path().join("artifacts"));
        let thread = ThreadId::from_string("retained-thread");
        let thread_key = artifact_thread_key(&thread);
        let thread_dir = storage.root().join(&thread_key);
        std::fs::create_dir_all(&thread_dir)?;
        let referenced = thread_dir.join("0.bash.log");
        let expired = thread_dir.join("1.mcp.log");
        let over_quota = thread_dir.join("2.subagent.log");
        for path in [&referenced, &expired, &over_quota] {
            std::fs::write(path, b"12345678")?;
        }
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
        std::fs::File::options()
            .write(true)
            .open(&referenced)?
            .set_modified(now - Duration::from_secs(100))?;
        std::fs::File::options()
            .write(true)
            .open(&expired)?
            .set_modified(now - Duration::from_secs(100))?;
        std::fs::File::options()
            .write(true)
            .open(&over_quota)?
            .set_modified(now - Duration::from_secs(1))?;

        let snapshot = ArtifactRetentionSnapshot {
            protect_unknown_threads: false,
            threads: BTreeMap::from([(
                thread_key,
                ArtifactThreadRetention {
                    live: false,
                    referenced_ids: BTreeSet::from([0]),
                },
            )]),
        };
        let report = storage.begin_sweep()?.sweep(
            ArtifactRetentionPolicy {
                max_age: Some(Duration::from_secs(10)),
                max_bytes_per_thread: Some(4),
            },
            &snapshot,
            now,
        )?;

        assert_eq!(report.files_removed, 2);
        assert_eq!(report.bytes_removed, 16);
        assert_eq!(report.threads_over_quota, 1);
        assert!(referenced.exists());
        assert!(!expired.exists());
        assert!(!over_quota.exists());
        Ok(())
    }

    #[cfg(any(target_os = "linux", windows))]
    fn publish_and_sweep_use_syncable_directory_handles() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let storage = ArtifactStorage::new(dir.path().join("artifacts"));
        let thread = ThreadId::from_string("directory-sync");
        let key = artifact_thread_key(&thread);
        let store = storage.for_thread(&thread)?;
        let saved = store.save("mcp", "recoverable")?;
        assert_eq!(std::fs::read_to_string(&saved.path)?, "recoverable");
        drop(store);

        let report = storage.begin_sweep()?.sweep(
            ArtifactRetentionPolicy {
                max_age: Some(Duration::ZERO),
                max_bytes_per_thread: None,
            },
            &ArtifactRetentionSnapshot {
                protect_unknown_threads: false,
                threads: BTreeMap::from([(key, ArtifactThreadRetention::default())]),
            },
            SystemTime::now() + Duration::from_secs(1),
        )?;

        assert_eq!(report.files_removed, 1);
        assert!(!saved.path.exists());
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_publish_and_sweep_use_syncable_directory_handles() -> Result<()> {
        publish_and_sweep_use_syncable_directory_handles()
    }

    #[cfg(windows)]
    #[test]
    fn windows_publish_and_sweep_flush_directory_metadata() -> Result<()> {
        publish_and_sweep_use_syncable_directory_handles()
    }

    #[test]
    fn quota_ties_remove_lower_artifact_id_first() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let storage = ArtifactStorage::new(dir.path().join("artifacts"));
        let thread = ThreadId::from_string("quota-thread");
        let thread_key = artifact_thread_key(&thread);
        let thread_dir = storage.root().join(&thread_key);
        std::fs::create_dir_all(&thread_dir)?;
        let modified = SystemTime::UNIX_EPOCH + Duration::from_secs(100);
        for id in 0..3 {
            let path = thread_dir.join(format!("{id}.mcp.log"));
            std::fs::write(&path, b"1234")?;
            std::fs::File::options()
                .write(true)
                .open(path)?
                .set_modified(modified)?;
        }

        let report = storage.begin_sweep()?.sweep(
            ArtifactRetentionPolicy {
                max_age: None,
                max_bytes_per_thread: Some(8),
            },
            &ArtifactRetentionSnapshot {
                protect_unknown_threads: false,
                threads: BTreeMap::from([(thread_key, ArtifactThreadRetention::default())]),
            },
            modified + Duration::from_secs(1),
        )?;

        assert_eq!(report.files_removed, 1);
        assert!(!thread_dir.join("0.mcp.log").exists());
        assert!(thread_dir.join("1.mcp.log").exists());
        assert!(thread_dir.join("2.mcp.log").exists());
        Ok(())
    }

    #[test]
    fn sweep_serializes_with_live_store_creation_and_fails_closed() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let storage = ArtifactStorage::new(dir.path().join("artifacts"));
        let live = storage.for_thread(&ThreadId::from_string("live-thread"))?;
        let live_artifact = live.save("mcp", "live")?;
        let unknown_dir = storage.root().join("unknown-thread");
        std::fs::create_dir_all(&unknown_dir)?;
        let unknown_artifact = unknown_dir.join("0.mcp.log");
        std::fs::write(&unknown_artifact, "unknown")?;

        let report = storage.begin_sweep()?.sweep(
            ArtifactRetentionPolicy {
                max_age: Some(Duration::ZERO),
                max_bytes_per_thread: Some(1),
            },
            &ArtifactRetentionSnapshot {
                protect_unknown_threads: true,
                threads: BTreeMap::new(),
            },
            SystemTime::now(),
        )?;

        assert_eq!(report.files_removed, 0);
        assert_eq!(report.protected_files, 2);
        assert!(live_artifact.path.exists());
        assert!(unknown_artifact.exists());
        Ok(())
    }

    #[test]
    fn concurrent_sweep_never_deletes_a_new_live_spill() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let storage = Arc::new(ArtifactStorage::new(dir.path().join("artifacts")));
        let thread = ThreadId::from_string("racing-thread");
        let thread_dir = storage.root().join(artifact_thread_key(&thread));
        std::fs::create_dir_all(&thread_dir)?;
        std::fs::write(thread_dir.join("0.old.log"), "stale")?;
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let snapshot = ArtifactRetentionSnapshot {
            protect_unknown_threads: false,
            threads: BTreeMap::from([(
                artifact_thread_key(&thread),
                ArtifactThreadRetention::default(),
            )]),
        };

        let sweep_storage = Arc::clone(&storage);
        let sweep_barrier = Arc::clone(&barrier);
        let sweep = std::thread::spawn(move || -> Result<ArtifactSweepReport> {
            sweep_barrier.wait();
            sweep_storage.begin_sweep()?.sweep(
                ArtifactRetentionPolicy {
                    max_age: Some(Duration::ZERO),
                    max_bytes_per_thread: Some(1),
                },
                &snapshot,
                SystemTime::now(),
            )
        });

        let writer_storage = Arc::clone(&storage);
        let writer_barrier = Arc::clone(&barrier);
        let writer = std::thread::spawn(move || -> Result<(SavedArtifact, Arc<ArtifactStore>)> {
            writer_barrier.wait();
            let store = writer_storage.for_thread(&thread)?;
            let saved = store.save("mcp", "live-bytes")?;
            Ok((saved, store))
        });

        barrier.wait();
        let _report = sweep
            .join()
            .map_err(|_| anyhow!("artifact sweep thread panicked"))??;
        let (saved, _store_guard) = writer
            .join()
            .map_err(|_| anyhow!("artifact writer thread panicked"))??;
        assert_eq!(std::fs::read(saved.path)?, b"live-bytes");
        Ok(())
    }
    #[test]
    fn spilled_result_drops_oversized_structured_payload() -> Result<()> {
        let (_dir, store) = temp_store();
        let raw = "secret".repeat(store.inline_budget());
        let mut result = ToolResult::success(raw.clone());
        result.data = Some(serde_json::json!({"raw": raw}));
        let saved = store.apply_inline_budget(&mut result, "mcp")?;
        assert!(saved.is_some());
        assert!(result.data.is_none());
        Ok(())
    }

    #[test]
    fn quota_rejection_removes_partial_and_preserves_existing_artifact() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = ArtifactStore::new(dir.path().join("artifacts"))
            .with_inline_budget(1024)
            .with_max_bytes_per_thread(1024);
        let first = store.save("mcp", &"a".repeat(800))?;
        let Err(error) = store.save("mcp", &"b".repeat(400)) else {
            anyhow::bail!("quota overflow unexpectedly succeeded");
        };
        assert!(error.to_string().contains("quota exceeded"));
        assert_eq!(std::fs::read(first.path)?, vec![b'a'; 800]);
        let partials = std::fs::read_dir(store.dir())?
            .filter_map(std::result::Result::ok)
            .filter(|entry| is_partial_name(&entry.file_name().to_string_lossy()))
            .count();
        assert_eq!(partials, 0);
        Ok(())
    }

    #[test]
    fn sweep_removes_only_stale_partial_files() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let storage = ArtifactStorage::new(dir.path().join("artifacts"));
        let thread = ThreadId::from_string("partial-thread");
        let key = artifact_thread_key(&thread);
        let thread_dir = storage.root().join(&key);
        std::fs::create_dir_all(&thread_dir)?;
        let stale = thread_dir.join(".partial-1-0.log");
        let fresh = thread_dir.join(".partial-1-1.log");
        std::fs::write(&stale, b"stale")?;
        std::fs::write(&fresh, b"fresh")?;
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(200_000);
        std::fs::File::options()
            .write(true)
            .open(&stale)?
            .set_modified(now - STALE_PARTIAL_MAX_AGE - Duration::from_secs(1))?;
        std::fs::File::options()
            .write(true)
            .open(&fresh)?
            .set_modified(now)?;
        let report = storage.begin_sweep()?.sweep(
            ArtifactRetentionPolicy {
                max_age: Some(Duration::from_secs(1)),
                max_bytes_per_thread: None,
            },
            &ArtifactRetentionSnapshot {
                protect_unknown_threads: false,
                threads: BTreeMap::from([(key, ArtifactThreadRetention::default())]),
            },
            now,
        )?;
        assert_eq!(report.files_removed, 1);
        assert!(!stale.exists());
        assert!(fresh.exists());
        Ok(())
    }

    #[test]
    fn exhausted_artifact_id_space_fails_without_partial() -> Result<()> {
        let (_dir, store) = temp_store();
        std::fs::write(store.dir().join(format!("{}.mcp.log", u64::MAX)), b"last")?;
        let Err(error) = store.save("mcp", "next") else {
            anyhow::bail!("exhausted ID space unexpectedly succeeded");
        };
        assert!(format!("{error:#}").contains("ID space exhausted"));
        let cap_dir = store.cap_dir()?;
        assert_eq!(list_artifacts(&cap_dir)?.len(), 1);
        Ok(())
    }

    #[test]
    fn hostile_thread_ids_map_to_distinct_safe_keys() {
        let ids = [
            "../escape",
            "a/b",
            ".",
            "..",
            "CON",
            "",
            "normal-thread",
            "Thread",
            "thread",
        ];
        let keys: BTreeSet<_> = ids
            .iter()
            .map(|id| artifact_thread_key(&ThreadId::from_string(*id)))
            .collect();
        assert_eq!(keys.len(), ids.len());
        assert!(keys.iter().all(|key| {
            key.len() == 66
                && key.starts_with("t-")
                && key.bytes().skip(2).all(|byte| byte.is_ascii_hexdigit())
        }));
    }

    #[test]
    fn legacy_thread_directory_is_migrated_without_losing_recovery() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let storage = ArtifactStorage::new(dir.path().join("artifacts"));
        let thread = ThreadId::from_string("legacy/thread");
        let legacy_key = legacy_artifact_thread_key(&thread);
        let hashed_key = artifact_thread_key(&thread);
        let legacy_dir = storage.root().join(&legacy_key);
        std::fs::create_dir_all(&legacy_dir)?;
        std::fs::write(legacy_dir.join("7.mcp.log"), b"legacy output")?;

        let store = storage.for_thread(&thread)?;
        assert_eq!(store.dir(), storage.root().join(&hashed_key));
        assert!(!legacy_dir.exists());
        assert!(storage.root().join(&hashed_key).is_dir());
        let mut recovered = String::new();
        store.resolve(7)?.read_to_string(&mut recovered)?;
        assert_eq!(recovered, "legacy output");
        let saved = store.save("mcp", "new output")?;
        assert_eq!(saved.id, 8);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn renamed_root_and_artifact_symlink_cannot_redirect_capability_io() -> Result<()> {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir()?;
        let root = dir.path().join("artifacts");
        let moved = dir.path().join("moved");
        let outside = dir.path().join("outside");
        let store = ArtifactStore::new(&root);
        std::fs::rename(&root, &moved)?;
        std::fs::create_dir(&outside)?;
        symlink(&outside, &root)?;
        let saved = store.save("mcp", "confined")?;
        let mut resolved = store.resolve(saved.id)?;
        let mut content = String::new();
        resolved.read_to_string(&mut content)?;
        assert_eq!(content, "confined");
        assert_eq!(std::fs::read_dir(&outside)?.count(), 0);

        let target = dir.path().join("target");
        std::fs::write(&target, "outside-data")?;
        symlink(&target, moved.join("99.mcp.log"))?;
        assert!(store.resolve(99).is_err());
        Ok(())
    }

    #[test]
    fn multiprocess_allocators_publish_unique_ids() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().join("artifacts");
        std::fs::create_dir(&root)?;
        let executable = std::env::current_exe()?;
        let mut children = Vec::new();
        for index in 0..8 {
            children.push(
                std::process::Command::new(&executable)
                    .args([
                        "--exact",
                        "artifacts::tests::multiprocess_allocator_child",
                        "--nocapture",
                    ])
                    .env(CHILD_ROOT_ENV, &root)
                    .env(CHILD_INDEX_ENV, index.to_string())
                    .spawn()?,
            );
        }
        for mut child in children {
            let status = child.wait()?;
            anyhow::ensure!(status.success(), "allocator child failed: {status}");
        }
        let mut ids = BTreeSet::new();
        for index in 0..8 {
            ids.insert(std::fs::read_to_string(
                root.join(format!("result-{index}")),
            )?);
        }
        assert_eq!(ids.len(), 8);
        assert_eq!(list_artifacts(&open_confined_dir(&root, false)?)?.len(), 8);
        Ok(())
    }

    #[test]
    fn sweep_observes_activity_lock_across_processes() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().join("artifacts");
        std::fs::create_dir(&root)?;
        let executable = std::env::current_exe()?;
        let mut child = std::process::Command::new(executable)
            .args([
                "--exact",
                "artifacts::tests::multiprocess_activity_lock_child",
                "--nocapture",
            ])
            .env(CHILD_ROOT_ENV, &root)
            .spawn()?;
        wait_for_path(&root.join("child-ready"))?;

        let storage = ArtifactStorage::new(&root);
        let key = artifact_thread_key(&ThreadId::from_string("live-cross-process"));
        let snapshot = ArtifactRetentionSnapshot {
            protect_unknown_threads: false,
            threads: BTreeMap::from([(key.clone(), ArtifactThreadRetention::default())]),
        };
        let policy = ArtifactRetentionPolicy {
            max_age: Some(Duration::ZERO),
            max_bytes_per_thread: Some(1),
        };
        let first = storage
            .begin_sweep()?
            .sweep(policy, &snapshot, SystemTime::now())?;
        assert_eq!(first.files_removed, 0);
        assert_eq!(first.protected_files, 1);
        std::fs::write(root.join("child-release"), b"release")?;
        let status = child.wait()?;
        anyhow::ensure!(status.success(), "activity child failed: {status}");
        let second = storage
            .begin_sweep()?
            .sweep(policy, &snapshot, SystemTime::now())?;
        assert_eq!(second.files_removed, 1);
        let thread_dir = open_confined_dir(&root.join(key), false)?;
        assert!(list_artifacts(&thread_dir)?.is_empty());
        Ok(())
    }

    #[test]
    fn allocator_never_reuses_deleted_ids_after_restart() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().join("artifacts");
        let thread = ThreadId::from_string("durable-watermark");
        let storage = ArtifactStorage::new(&root);
        let first = storage.for_thread(&thread)?.save("bash", "first")?;
        assert_eq!(first.id, 0);
        std::fs::remove_file(&first.path)?;
        drop(storage);

        let restarted = ArtifactStorage::new(&root);
        let second = restarted.for_thread(&thread)?.save("bash", "second")?;
        assert_eq!(second.id, 1, "a deleted ID must remain permanently burned");
        Ok(())
    }

    #[test]
    fn legacy_footer_verification_handles_lossy_utf8_but_rejects_forgery() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let storage = ArtifactStorage::new(dir.path().join("artifacts")).with_inline_budget(1_024);
        let source = ThreadId::from_string("legacy-source");
        let destination = ThreadId::from_string("legacy-destination");
        let store = storage.for_thread(&source)?;
        let mut bytes = vec![b'x'; 4_096];
        bytes[100] = 0xff;
        bytes[4_000] = 0xfe;
        let saved = store.save_streamed("bash", &mut std::io::Cursor::new(&bytes))?;
        let head = String::from_utf8_lossy(&bytes[..1_024]);
        let tail = String::from_utf8_lossy(&bytes[bytes.len() - 1_024..]);
        let inline = cap_inline_from_windows(
            &head,
            &tail,
            bytes.len() as u64,
            store.inline_budget(),
            saved.id,
        );

        let mut verifier_file = store.resolve(saved.id)?;
        assert!(
            canonical_streamed_inline_output_matches(
                &mut verifier_file,
                bytes.len() as u64,
                &inline,
                store.inline_budget(),
                saved.id,
            )?,
            "lossy streamed rendering must reproduce exactly"
        );
        assert_eq!(
            store.verified_legacy_inline_artifact_id(&inline)?,
            Some(saved.id)
        );
        assert!(inline.contains('\u{fffd}'));
        let mut reopened = store.resolve(saved.id)?;
        let mut exact = Vec::new();
        reopened.read_to_end(&mut exact)?;
        assert_eq!(
            exact, bytes,
            "durable artifact keeps original invalid bytes"
        );

        let forged = inline.replacen('x', "y", 1);
        assert_eq!(store.verified_legacy_inline_artifact_id(&forged)?, None);
        assert_eq!(
            store.verified_legacy_inline_artifact_id(&artifact_footer(saved.id))?,
            None,
            "footer syntax without the canonical window is not provenance"
        );
        storage.copy_thread_artifacts(&source, &destination, &BTreeSet::from([saved.id]))?;
        let destination_store = storage.for_thread(&destination)?;
        let mut forked = destination_store.resolve(saved.id)?;
        let mut forked_exact = Vec::new();
        forked.read_to_end(&mut forked_exact)?;
        assert_eq!(
            forked_exact, bytes,
            "verified legacy reference survives fork"
        );
        Ok(())
    }
    #[test]
    fn fork_copy_is_reference_scoped_and_seeds_allocator_watermark() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let storage = ArtifactStorage::new(dir.path().join("artifacts"));
        let source = ThreadId::from_string("fork-source");
        let destination = ThreadId::from_string("fork-destination");
        let source_store = storage.for_thread(&source)?;
        let before = source_store.save("bash", "before-boundary")?;
        let after = source_store.save("bash", "post-boundary-secret")?;

        storage.copy_thread_artifacts(&source, &destination, &BTreeSet::from([before.id]))?;
        let destination_store = storage.for_thread(&destination)?;
        let mut copied = String::new();
        destination_store
            .resolve(before.id)?
            .read_to_string(&mut copied)?;
        assert_eq!(copied, "before-boundary");
        assert!(
            destination_store.resolve(after.id).is_err(),
            "an unreferenced post-boundary artifact must not enter the fork"
        );
        let fresh = destination_store.save("bash", "fork-local")?;
        assert!(
            fresh.id > after.id,
            "fork allocator must seed above the source high watermark"
        );
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
