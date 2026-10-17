#![cfg(feature = "sqlite")]

//! Design §9, kill -9 "process start": a process effect whose record
//! committed before the process started survives the death of the process
//! that started it, so the next owner finds it and recovers it instead of
//! starting the process a second time.

use std::process::Command;
use std::time::Duration;

use agent_sdk_foundation::{ThreadId, ToolReplay};
use agent_server::journal::effect::{
    EffectIntent, EffectKey, EffectKind, EffectRun, EffectState, EffectSubject, EffectWrite,
    run_effect, started_effects_for_task,
};
use agent_server::journal::recovery::RecoveryAction;
use agent_server::journal::store::AgentTaskStore;
use agent_server::journal::task::{AgentTask, AgentTaskId, LeaseId, TaskStatus, WorkerId};
use agent_server::journal::thread_store::ThreadStore;
use agent_server::journal::{DurableStore, Guard, Rejected, Transition, Write};
use agent_service_host::sqlite::SqliteDurableStore;
use anyhow::{Context, Result, bail, ensure};
use time::OffsetDateTime;

const CHILD_ENV: &str = "AGENT_SDK_PROCESS_START_CRASH_CHILD";
const DB_ENV: &str = "AGENT_SDK_PROCESS_START_CRASH_DB";
const READY_ENV: &str = "AGENT_SDK_PROCESS_START_CRASH_READY";
const PROCESS: &str = "dev-server";

fn at(secs: i64) -> OffsetDateTime {
    OffsetDateTime::UNIX_EPOCH + time::Duration::seconds(secs)
}

fn subject() -> EffectSubject {
    EffectSubject::Process {
        project_hash: "project-1".into(),
        name: PROCESS.into(),
        argv_digest: "sha256:npm-run-dev".into(),
    }
}

fn leased(task: &AgentTaskId, worker: &str, lease: &str) -> Guard {
    Guard::TaskIs {
        task: task.clone(),
        status: TaskStatus::Running,
        lease: Some((WorkerId::from_string(worker), LeaseId::from_string(lease))),
    }
}

/// Child-process half: start the process effect, then wait to be killed
/// while the "process" runs.
#[tokio::test]
async fn process_start_crash_child() -> Result<()> {
    if std::env::var_os(CHILD_ENV).is_none() {
        return Ok(());
    }
    let url = std::env::var(DB_ENV).context("missing crash database URL")?;
    let ready = std::env::var(READY_ENV).context("missing crash ready path")?;
    let store = SqliteDurableStore::connect(&url).await?;

    let thread = ThreadId::from_string("process-start-crash");
    ThreadStore::get_or_create(&store, &thread, at(1)).await?;
    let task = AgentTask::new_root_turn(thread.clone(), at(1), 3);
    let task_id = task.id.clone();
    store.submit_root_turn(task).await?;
    store
        .try_acquire_task(
            &task_id,
            WorkerId::from_string("crashed-worker"),
            LeaseId::from_string("crashed-lease"),
            at(600),
            at(2),
        )
        .await?
        .context("the owner was not acquired")?;

    let intent = EffectIntent {
        key: EffectKey::process(&task_id, PROCESS),
        replay: ToolReplay::Unsafe,
        owner: task_id.clone(),
        thread,
        subject: subject(),
        attempt: 1,
    };
    let start = Transition {
        guards: vec![leased(&task_id, "crashed-worker", "crashed-lease")],
        writes: Vec::new(),
    };
    let run = run_effect(
        &store,
        intent,
        start,
        || async {
            std::fs::write(&ready, task_id.as_str())?;
            tokio::time::sleep(Duration::from_mins(1)).await;
            anyhow::Ok(())
        },
        |_| bail!("the child was not killed while the process ran"),
        at(3),
    )
    .await?;
    if let EffectRun::Refused(rejected) = run {
        bail!("the start was refused: {rejected}");
    }
    bail!("the child was not killed while the process ran")
}

/// Spawn the child, wait until its process effect is running, then
/// `kill -9` it.
async fn crash_while_the_process_runs(
    url: &str,
    ready_path: &std::path::Path,
) -> Result<AgentTaskId> {
    let mut child = Command::new(std::env::current_exe()?)
        .arg("process_start_crash_child")
        .arg("--exact")
        .arg("--nocapture")
        .env(CHILD_ENV, "1")
        .env(DB_ENV, url)
        .env(READY_ENV, ready_path)
        .spawn()
        .context("spawn the process-start child")?;
    for _ in 0..200 {
        if let Ok(task_id) = std::fs::read_to_string(ready_path)
            && !task_id.is_empty()
        {
            let killed = Command::new("kill")
                .arg("-9")
                .arg(child.id().to_string())
                .status()
                .context("kill -9 the process-start child")?;
            ensure!(killed.success(), "kill -9 failed with {killed}");
            let _ = child.wait()?;
            return Ok(AgentTaskId::from_string(task_id.trim()));
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let _ = child.kill();
    let output = child.wait_with_output()?;
    bail!(
        "the child never started the process: {}",
        String::from_utf8_lossy(&output.stderr)
    )
}

#[tokio::test]
async fn kill_nine_after_a_process_start_leaves_a_started_record_the_next_owner_recovers()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let url = format!(
        "sqlite://{}?mode=rwc",
        temp.path().join("process-start-crash.db").display()
    );
    let task_id = crash_while_the_process_runs(&url, &temp.path().join("started")).await?;

    let store = SqliteDurableStore::connect(&url).await?;
    let key = EffectKey::process(&task_id, PROCESS);
    let started = store.started_effects(EffectKind::ProcessStart).await?;
    let [record] = started.as_slice() else {
        bail!("the boot re-drive must find the one started process, got {started:?}");
    };
    ensure!(record.key == key && record.owner == task_id, "{record:?}");
    ensure!(record.subject == subject(), "{:?}", record.subject);
    ensure!(record.state == EffectState::Started, "{:?}", record.state);

    let records = store.release_expired_leases(at(601)).await?;
    ensure!(
        records
            .iter()
            .any(|record| record.id == task_id && record.action == RecoveryAction::Requeue),
        "the lease sweep did not requeue the dead owner: {records:?}"
    );
    store
        .try_acquire_task(
            &task_id,
            WorkerId::from_string("boot-worker"),
            LeaseId::from_string("boot-lease"),
            at(1200),
            at(602),
        )
        .await?
        .context("boot did not reacquire the owner")?;
    let owned = started_effects_for_task(&store, &task_id).await?;
    ensure!(
        owned.len() == 1 && owned[0].key == key,
        "the new owner must see the started process: {owned:?}"
    );

    let interrupt = |worker: &str, lease: &str| Transition {
        guards: vec![leased(&task_id, worker, lease)],
        writes: vec![Write::Effect(EffectWrite::Finish {
            key: key.clone(),
            owner: task_id.clone(),
            state: EffectState::Interrupted {
                captured_output: None,
            },
        })],
    };
    let stale = store
        .commit(interrupt("crashed-worker", "crashed-lease"), at(603))
        .await;
    ensure!(
        matches!(stale, Err(Rejected::Guard { .. })),
        "the dead owner's lease cannot end the effect: {stale:?}"
    );
    store
        .commit(interrupt("boot-worker", "boot-lease"), at(604))
        .await?;
    ensure!(
        store
            .started_effects(EffectKind::ProcessStart)
            .await?
            .is_empty(),
        "a recovered process is not re-driven again"
    );
    let ended = store.effect(&key).await?.context("the record is gone")?;
    ensure!(
        matches!(ended.state, EffectState::Interrupted { .. }),
        "{:?}",
        ended.state
    );
    Ok(())
}
