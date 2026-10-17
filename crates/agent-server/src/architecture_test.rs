//! Design §6.6: state changes go through the journal's transitions. Outside
//! `journal/`, every textual call to a split write is counted per file and
//! held to an allowlist that can only shrink.
//!
//! The counted methods are a subset of §6.6's write methods: `set_draft`,
//! `replace_history`, `cancel_tree`, `complete_task` and `insert_batch` are
//! called outside `journal/` today and are not counted yet.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};

/// Store write methods whose calls are counted, in method-call, UFCS
/// (`EventRepository::commit_event(repo, …)`) and bare-call form. A
/// definition (`fn commit_event(`) and a transition constructor
/// (`Transition::append_compaction(`) are not calls to the split write.
const SPLIT_WRITES: &[&str] = &[
    "commit_event",
    "commit_event_batch",
    "complete_task_with_result",
    "fail_task",
    "fail_task_with_reason",
    "pause_on_confirmation",
    "reject_confirmation",
    "repark_after_steering",
    "append_compaction",
    "persist_intent",
    "update_intent",
];
/// `(path relative to the crates directory, calls)`. Streaming
/// `TextDelta` / `ThinkingDelta` commits and other events that record no
/// state change are the legitimate bulk of these.
const ALLOWED: &[(&str, usize)] = &[
    ("agent-server/src/worker/activity.rs", 2),
    ("agent-server/src/worker/root_turn.rs", 7),
    ("agent-service-host/src/grpc.rs", 1),
    ("agent-service-host/src/host.rs", 4),
    ("agent-service-host/src/postgres/store.rs", 4),
    ("agent-service-host/src/sqlite/store.rs", 4),
];

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let path = entry?.path();
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if path.is_dir() {
            if name != "journal" && name != "tests" {
                rust_sources(&path, out)?;
            }
        } else if path.extension().is_some_and(|extension| extension == "rs")
            && !name.ends_with("_test.rs")
            && name != "conformance.rs"
        {
            out.push(path);
        }
    }
    Ok(())
}

/// The file up to its first inline `#[cfg(test)]` module.
fn production_part(source: &str) -> &str {
    let mut offset = 0;
    let mut lines = source.split_inclusive('\n').peekable();
    while let Some(line) = lines.next() {
        if line.trim() == "#[cfg(test)]"
            && lines.peek().is_some_and(|next| {
                let next = next.trim();
                let module = next
                    .strip_prefix("pub(crate) ")
                    .or_else(|| next.strip_prefix("pub "))
                    .unwrap_or(next);
                module.starts_with("mod ") && module.ends_with('{')
            })
        {
            return &source[..offset];
        }
        offset += line.len();
    }
    source
}

/// Calls of `method` in `source`: each `method(` not preceded by an
/// identifier character, `fn ` or `Transition::`.
fn call_count(source: &str, method: &str) -> usize {
    let call = format!("{method}(");
    source
        .match_indices(&call)
        .filter(|(at, _)| {
            let before = &source[..*at];
            !before.ends_with(|c: char| c.is_alphanumeric() || c == '_')
                && !before.ends_with("fn ")
                && !before.ends_with("Transition::")
        })
        .count()
}

fn split_write_counts() -> Result<BTreeMap<String, usize>> {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .context("agent-server has a parent directory")?;
    let mut files = Vec::new();
    for root in ["agent-server/src", "agent-service-host/src"] {
        rust_sources(&crates.join(root), &mut files)?;
    }
    let mut counts = BTreeMap::new();
    for file in files {
        let source =
            std::fs::read_to_string(&file).with_context(|| format!("read {}", file.display()))?;
        let production = production_part(&source);
        let calls: usize = SPLIT_WRITES
            .iter()
            .map(|method| call_count(production, method))
            .sum();
        if calls > 0 {
            let relative = file
                .strip_prefix(crates)
                .context("source under the crates directory")?
                .to_string_lossy()
                .replace('\\', "/");
            counts.insert(relative, calls);
        }
    }
    Ok(counts)
}

#[test]
fn split_writes_outside_the_journal_only_shrink() -> Result<()> {
    let counts = split_write_counts()?;
    let allowed: BTreeMap<String, usize> = ALLOWED
        .iter()
        .map(|(path, calls)| ((*path).to_owned(), *calls))
        .collect();
    let mut grown = Vec::new();
    let mut shrunk = Vec::new();
    for (path, calls) in &counts {
        match allowed.get(path) {
            Some(limit) if calls > limit => grown.push(format!("{path}: {calls} > {limit}")),
            Some(_) => {}
            None => grown.push(format!("{path}: {calls} (not allowlisted)")),
        }
    }
    for (path, limit) in &allowed {
        let calls = counts.get(path).copied().unwrap_or(0);
        if calls < *limit {
            shrunk.push(format!("(\"{path}\", {calls}), // was {limit}"));
        }
    }
    let current: Vec<String> = counts
        .iter()
        .map(|(path, calls)| format!("(\"{path}\", {calls}),"))
        .collect();
    ensure!(
        grown.is_empty(),
        "new split writes outside journal/ (commit them through a transition instead):\n{}\n\ncurrent counts:\n{}",
        grown.join("\n"),
        current.join("\n"),
    );
    ensure!(
        shrunk.is_empty(),
        "split writes were removed; shrink ALLOWED to:\n{}",
        shrunk.join("\n"),
    );
    Ok(())
}
