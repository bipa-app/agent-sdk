//! `agent-sdk doctor` — sanity checks for the local Langfuse stack.
//!
//! Read-only. Does not start, stop, or modify anything. Prints a
//! checklist and exits non-zero if any required item fails so the
//! command can drive a CI gate or a pre-flight check.

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{Result, bail};
use clap::Args as ClapArgs;

use crate::embed::DEFAULT_DEST_REL;

const REQUIRED_PORTS: &[u16] = &[4000, 4317, 4318];

#[derive(ClapArgs, Debug)]
pub struct DoctorArgs {
    /// Directory that `local-langfuse init` would write into.
    #[arg(long, default_value = DEFAULT_DEST_REL)]
    pub dest: PathBuf,
}

pub use DoctorArgs as Args;

pub fn run(args: DoctorArgs) -> Result<()> {
    let mut failures: Vec<String> = Vec::new();

    let docker_ok = check_command("docker", &["--version"]);
    report("docker", &docker_ok);
    if let Err(msg) = &docker_ok {
        failures.push(format!("docker missing: {msg}"));
    }

    let compose_ok = check_command("docker", &["compose", "version"]);
    report("docker compose", &compose_ok);
    if let Err(msg) = &compose_ok {
        failures.push(format!("docker compose missing: {msg}"));
    }

    for port in REQUIRED_PORTS {
        let port_ok = check_port_free(*port);
        report(&format!("port {port} free"), &port_ok);
        if let Err(msg) = &port_ok {
            failures.push(format!("port {port} not free: {msg}"));
        }
    }

    let dest = args.dest;
    let dest_ok = check_dest_writable(&dest);
    report(&format!("dest {} writable", dest.display()), &dest_ok);
    if let Err(msg) = &dest_ok {
        failures.push(format!("dest {} not writable: {msg}", dest.display()));
    }

    if failures.is_empty() {
        println!("\nall checks passed");
        Ok(())
    } else {
        bail!(
            "doctor: {} check(s) failed:\n  - {}",
            failures.len(),
            failures.join("\n  - ")
        );
    }
}

fn check_command(program: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|e| format!("failed to spawn `{program}`: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "`{program} {}` exited with {}",
            args.join(" "),
            output.status
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn check_port_free(port: u16) -> Result<String, String> {
    match TcpListener::bind(("127.0.0.1", port)) {
        Ok(_) => Ok(format!("127.0.0.1:{port}")),
        Err(e) => Err(e.to_string()),
    }
}

fn check_dest_writable(dest: &std::path::Path) -> Result<String, String> {
    // Prefer the existing parent if `dest` itself doesn't exist yet.
    let probe = if dest.exists() {
        dest.to_path_buf()
    } else {
        match dest.parent() {
            Some(p) if p.as_os_str().is_empty() => {
                std::env::current_dir().map_err(|e| format!("cannot read current dir: {e}"))?
            }
            Some(p) => p.to_path_buf(),
            None => std::env::current_dir().map_err(|e| format!("cannot read current dir: {e}"))?,
        }
    };

    let probe_existing = if probe.exists() {
        probe
    } else {
        std::env::current_dir().map_err(|e| format!("cannot read current dir: {e}"))?
    };

    let tmp = tempfile_in(&probe_existing)?;
    drop(tmp);
    Ok(format!("{}", probe_existing.display()))
}

fn tempfile_in(dir: &std::path::Path) -> Result<std::fs::File, String> {
    let mut path = dir.to_path_buf();
    path.push(".agent-sdk-doctor-write-probe");
    let file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&path)
        .map_err(|e| format!("cannot write to {}: {e}", dir.display()))?;
    // Best-effort cleanup; ignore failure (the worst case is a stale
    // empty file in the user's repo, which is harmless and visible).
    let _ = std::fs::remove_file(&path);
    Ok(file)
}

fn report(label: &str, result: &Result<String, String>) {
    match result {
        Ok(detail) if detail.is_empty() => println!("ok    {label}"),
        Ok(detail) => println!("ok    {label} ({detail})"),
        Err(msg) => println!("FAIL  {label}: {msg}"),
    }
}
