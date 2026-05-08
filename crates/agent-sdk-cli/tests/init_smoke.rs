//! Smoke test: run the binary against a tempdir and assert the
//! emitted files match the in-tree compose stack byte-for-byte.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use tempfile::TempDir;

fn workspace_root() -> Result<PathBuf> {
    // CARGO_MANIFEST_DIR points at crates/agent-sdk-cli; go up two
    // levels to the workspace root.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .context("CARGO_MANIFEST_DIR should have two ancestors")
}

fn read_canonical(name: &str) -> Result<String> {
    let path = workspace_root()?
        .join("dev/observability/langfuse")
        .join(name);
    fs::read_to_string(&path).with_context(|| format!("reading canonical {}", path.display()))
}

fn read_doc() -> Result<String> {
    let path = workspace_root()?.join("crates/agent-sdk/docs/observability/LANGFUSE.md");
    fs::read_to_string(&path).with_context(|| format!("reading canonical {}", path.display()))
}

fn run_cli(dest: &Path, force: bool) -> Result<std::process::Output> {
    let bin = env!("CARGO_BIN_EXE_agent-sdk");
    let mut cmd = Command::new(bin);
    cmd.args(["local-langfuse", "init", "--dest"]).arg(dest);
    if force {
        cmd.arg("--force");
    }
    cmd.output().context("spawning agent-sdk binary")
}

#[test]
fn init_writes_canonical_files() -> Result<()> {
    let tmp = TempDir::new()?;
    let dest = tmp.path().join("nested/dest");

    let output = run_cli(&dest, false)?;
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let compose = fs::read_to_string(dest.join("docker-compose.yml"))?;
    let collector = fs::read_to_string(dest.join("otel-collector.yaml"))?;
    let doc = fs::read_to_string(dest.join("LANGFUSE.md"))?;

    assert_eq!(compose, read_canonical("docker-compose.yml")?);
    assert_eq!(collector, read_canonical("otel-collector.yaml")?);
    assert_eq!(doc, read_doc()?);

    Ok(())
}

#[test]
fn init_refuses_to_overwrite_without_force() -> Result<()> {
    let tmp = TempDir::new()?;
    let dest = tmp.path().to_path_buf();

    let first = run_cli(&dest, false)?;
    assert!(first.status.success());

    let second = run_cli(&dest, false)?;
    assert!(!second.status.success(), "second init should fail");
    let stderr = String::from_utf8_lossy(&second.stderr);
    assert!(
        stderr.contains("refusing to overwrite"),
        "unexpected stderr: {stderr}"
    );

    let forced = run_cli(&dest, true)?;
    assert!(forced.status.success(), "force should succeed");
    Ok(())
}
