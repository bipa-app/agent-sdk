//! `agent-sdk local-langfuse` — manage the local Langfuse + `OTel`
//! collector dev stack.

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};

use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand};

use crate::embed::{
    COLLECTOR_FILENAME, COLLECTOR_YAML, COMPOSE_FILENAME, COMPOSE_YAML, DEFAULT_DEST_REL,
    DOC_FILENAME, LANGFUSE_DOC,
};

#[derive(Subcommand, Debug)]
pub enum Action {
    /// Materialize the compose stack into `<dest>` (creates the dir).
    Init(InitArgs),
    /// Run `<runtime> compose -f <dest>/docker-compose.yml up -d`.
    Up(DestArgs),
    /// Run `<runtime> compose -f <dest>/docker-compose.yml down`.
    Down(DownArgs),
    /// Run `<runtime> compose -f <dest>/docker-compose.yml ps`.
    Status(DestArgs),
}

#[derive(Args, Debug)]
pub struct InitArgs {
    /// Directory to write the compose files into. Matches the SDK's
    /// own `dev/observability/langfuse` layout by default.
    #[arg(long, default_value = DEFAULT_DEST_REL)]
    pub dest: PathBuf,

    /// Overwrite existing files instead of failing.
    #[arg(long)]
    pub force: bool,
}

#[derive(Args, Debug)]
pub struct DestArgs {
    /// Directory previously initialized via `init`.
    #[arg(long, default_value = DEFAULT_DEST_REL)]
    pub dest: PathBuf,
}

#[derive(Args, Debug)]
pub struct DownArgs {
    /// Directory previously initialized via `init`.
    #[arg(long, default_value = DEFAULT_DEST_REL)]
    pub dest: PathBuf,

    /// Pass `-v` to `compose down` so the named volumes are
    /// dropped (resets the local Langfuse database on next boot).
    #[arg(long)]
    pub volumes: bool,
}

pub fn run(action: Action) -> Result<()> {
    match action {
        Action::Init(args) => init(args),
        Action::Up(args) => up(&args),
        Action::Down(args) => down(&args),
        Action::Status(args) => status(&args),
    }
}

fn init(args: InitArgs) -> Result<()> {
    let dest = args.dest;
    fs::create_dir_all(&dest)
        .with_context(|| format!("creating destination directory {}", dest.display()))?;

    let entries: [(&str, &str); 3] = [
        (COMPOSE_FILENAME, COMPOSE_YAML),
        (COLLECTOR_FILENAME, COLLECTOR_YAML),
        (DOC_FILENAME, LANGFUSE_DOC),
    ];

    if !args.force {
        for (name, _) in entries {
            let path = dest.join(name);
            if path.exists() {
                bail!(
                    "refusing to overwrite existing file {} (rerun with --force to replace)",
                    path.display()
                );
            }
        }
    }

    for (name, contents) in entries {
        let path = dest.join(name);
        fs::write(&path, contents).with_context(|| format!("writing {}", path.display()))?;
        println!("wrote {}", path.display());
    }

    println!();
    println!(
        "next: `agent-sdk local-langfuse up --dest {}` to start the stack",
        dest.display()
    );
    Ok(())
}

fn up(args: &DestArgs) -> Result<()> {
    let compose_path = compose_path(&args.dest)?;
    run_compose(&compose_path, ["up", "-d"])
}

fn down(args: &DownArgs) -> Result<()> {
    let compose_path = compose_path(&args.dest)?;
    if args.volumes {
        run_compose(&compose_path, ["down", "-v"])
    } else {
        run_compose(&compose_path, ["down"])
    }
}

fn status(args: &DestArgs) -> Result<()> {
    let compose_path = compose_path(&args.dest)?;
    run_compose(&compose_path, ["ps"])
}

fn compose_path(dest: &Path) -> Result<PathBuf> {
    let path = dest.join(COMPOSE_FILENAME);
    if !path.exists() {
        bail!(
            "compose file not found at {}; run `agent-sdk local-langfuse init` first",
            path.display()
        );
    }
    Ok(path)
}

/// The compose invocation to use, as `(program, leading args)`.
///
/// Probes the known runtimes in order; `AGENT_SDK_CONTAINER_RUNTIME` selects
/// one explicitly. Standalone `podman-compose` / `docker-compose` are tried
/// last, since a runtime does not necessarily ship the `compose` subcommand.
pub fn compose_command() -> Result<(String, Vec<String>)> {
    let probe = |program: &str, args: &[&str]| -> bool {
        Command::new(program)
            .args(args)
            .arg("version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    };

    let runtimes: Vec<String> = match std::env::var("AGENT_SDK_CONTAINER_RUNTIME") {
        Ok(explicit) if !explicit.is_empty() => vec![explicit],
        _ => vec!["podman".to_owned(), "docker".to_owned()],
    };
    for runtime in &runtimes {
        if probe(runtime, &["compose"]) {
            return Ok((runtime.clone(), vec!["compose".to_owned()]));
        }
    }
    for standalone in ["podman-compose", "docker-compose"] {
        if probe(standalone, &[]) {
            return Ok((standalone.to_owned(), Vec::new()));
        }
    }
    bail!(
        "no working compose provider found (tried: podman compose, docker compose, \
         podman-compose, docker-compose). Install one, or set \
         AGENT_SDK_CONTAINER_RUNTIME."
    )
}

fn run_compose<I, S>(compose_path: &Path, args: I) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let (program, leading) = compose_command()?;
    let mut cmd = Command::new(&program);
    cmd.args(&leading).arg("-f").arg(compose_path);
    for a in args {
        cmd.arg(a);
    }
    let status: ExitStatus = cmd
        .status()
        .with_context(|| format!("spawning `{program}`"))?;
    if !status.success() {
        bail!("`{program} compose` exited with {status}");
    }
    Ok(())
}
