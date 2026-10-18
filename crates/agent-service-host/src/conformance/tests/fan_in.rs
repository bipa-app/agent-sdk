//! Seeded, serial schedules of store operations. Check each committed state,
//! before any host repair can hide a missed fan-in. These are not lock-race tests.

use anyhow::{Context, Result, ensure};
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{Config, RngSeed, TestRunner};

use agent_sdk_foundation::TerminalReason;
use agent_server::journal::recovery::RecoveryAction;
use agent_server::journal::store::{AgentTaskStore, SubagentInvocationSpawn};
use agent_server::journal::task::{
    AgentTask, AgentTaskId, ChildSpawnSpec, LeaseId, TaskKind, TaskStatus, WorkerId,
};
use agent_server::journal::thread_store::ThreadStore;
use agent_server::journal::{LeaseRecoveryScope, Settle, SpawnChildrenExt, Transition};

use super::{sample_subagent_input, sample_subagent_spec, suspension_payload, t_plus, thread_id};

#[derive(Clone, Copy, Debug)]
enum Operation {
    Tools,
    Subagent,
    Complete,
    Fail,
    Cancel,
    Expire,
}

fn schedule() -> impl Strategy<Value = Vec<(Operation, usize, usize)>> {
    let operation = prop_oneof![
        2 => Just(Operation::Tools),
        2 => Just(Operation::Subagent),
        2 => Just(Operation::Complete),
        2 => Just(Operation::Fail),
        1 => Just(Operation::Cancel),
        2 => Just(Operation::Expire),
    ];
    proptest::collection::vec((operation, 0usize..1024, 1usize..4), 48)
}

struct Tree<'a> {
    tasks: &'a dyn AgentTaskStore,
    threads: &'a dyn ThreadStore,
    ids: Vec<AgentTaskId>,
    seed: u64,
    at: i64,
}

impl Tree<'_> {
    async fn rows(&self) -> Result<Vec<AgentTask>> {
        let mut rows = Vec::new();
        for id in &self.ids {
            rows.push(
                self.tasks
                    .get(id)
                    .await?
                    .context("generated task disappeared")?,
            );
        }
        Ok(rows)
    }

    async fn check(&self) -> Result<()> {
        let rows = self.rows().await?;
        for parent in &rows {
            if parent.status != TaskStatus::WaitingOnChildren {
                continue;
            }
            // Subagent child roots are parentless. Read the durable forward
            // link, not the reverse lookup (which only returns parked links).
            let linked = parent
                .state
                .subagent_invocation()
                .map(|state| &state.child_root_task_id);
            let live = rows
                .iter()
                .filter(|child| {
                    !child.status.is_terminal()
                        && (child.parent_id.as_ref() == Some(&parent.id)
                            || linked == Some(&child.id))
                })
                .count();
            ensure!(
                live > 0,
                "WaitingOnChildren task {} ({:?}) has zero non-terminal children; stored count={}; rows={rows:#?}",
                parent.id,
                parent.kind,
                parent.pending_child_count
            );
            ensure!(
                usize::try_from(parent.pending_child_count)? == live,
                "task {} has stored count {}, but {live} live children",
                parent.id,
                parent.pending_child_count
            );
        }
        Ok(())
    }

    async fn root(&mut self) -> Result<()> {
        let name = format!("fan-in-{}-{}", self.seed, self.ids.len());
        let root = AgentTask::new_root_turn(thread_id(&name), t_plus(self.at), 3);
        let admitted = self.tasks.submit_root_turn(root).await?;
        ensure!(
            admitted.status == TaskStatus::Pending,
            "fresh root must be pending"
        );
        self.ids.push(admitted.id);
        self.check().await
    }

    async fn apply(&mut self, operation: Operation, selector: usize, width: usize) -> Result<()> {
        self.at += 100;
        let rows = self.rows().await?;
        let candidates: Vec<_> = rows
            .iter()
            .filter(|row| match operation {
                Operation::Tools | Operation::Subagent => {
                    row.kind == TaskKind::RootTurn
                        && row.status == TaskStatus::Pending
                        && self.ids.len() < 48
                }
                Operation::Cancel => !row.status.is_terminal(),
                Operation::Complete | Operation::Fail | Operation::Expire => {
                    row.status == TaskStatus::Pending
                }
            })
            .collect();
        if candidates.is_empty() {
            return self.check().await;
        }
        let task = candidates[selector % candidates.len()];
        if matches!(operation, Operation::Cancel) {
            self.tasks.cancel_tree(&task.id, t_plus(self.at)).await?;
            return self.check().await;
        }

        let worker = WorkerId::new();
        let lease = LeaseId::new();
        let acquired = self
            .tasks
            .try_acquire_task(
                &task.id,
                worker.clone(),
                lease.clone(),
                t_plus(self.at + 1),
                t_plus(self.at),
            )
            .await?;
        self.check().await?;
        // Repeated spawn/resume cycles can exhaust a root's retry budget.
        // Acquisition then fails it closed; check that transition too.
        let Some(acquired) = acquired else {
            ensure!(
                self.tasks
                    .get(&task.id)
                    .await?
                    .context("acquired task")?
                    .status
                    .is_terminal(),
                "a pending task must acquire or fail closed"
            );
            return Ok(());
        };
        self.apply_owned(&acquired, operation, width).await?;
        self.check().await
    }

    async fn apply_owned(
        &mut self,
        task: &AgentTask,
        operation: Operation,
        width: usize,
    ) -> Result<()> {
        let worker = task.worker_id.clone().context("acquired worker")?;
        let lease = task.lease_id.clone().context("acquired lease")?;
        match operation {
            Operation::Tools => {
                let (_, children) = self
                    .tasks
                    .spawn_tool_children(
                        &task.id,
                        &worker,
                        &lease,
                        vec![ChildSpawnSpec::new(1); width],
                        suspension_payload(&task.thread_id.to_string()),
                        None,
                        Vec::new(),
                        t_plus(self.at),
                    )
                    .await?;
                ensure!(children.len() == width, "tool batch must be created");
                self.ids.extend(children.into_iter().map(|child| child.id));
            }
            Operation::Subagent => {
                let child_thread =
                    thread_id(&format!("fan-in-{}-child-{}", self.seed, self.ids.len()));
                self.threads
                    .get_or_create(&child_thread, t_plus(self.at))
                    .await?;
                let (_, invocation, child) = self
                    .tasks
                    .spawn_subagent_invocation(
                        &task.id,
                        &worker,
                        &lease,
                        SubagentInvocationSpawn {
                            child_thread_id: child_thread,
                            spec: sample_subagent_spec(),
                            child_root_input: sample_subagent_input(),
                            spawn_index: 0,
                            child_caller_metadata: None,
                            payload: suspension_payload(&task.thread_id.to_string()),
                        },
                        Vec::new(),
                        t_plus(self.at),
                    )
                    .await?;
                self.ids.extend([invocation.id, child.id]);
            }
            Operation::Complete | Operation::Fail => {
                let settle = if matches!(operation, Operation::Complete) {
                    Settle::CompleteWithoutResult
                } else {
                    Settle::FailWithReason {
                        error: "generated failure".into(),
                        reason: TerminalReason::WatchdogStall,
                        result: None,
                    }
                };
                self.tasks
                    .commit(
                        Transition::settle_only(task.id.clone(), worker, lease, settle),
                        t_plus(self.at),
                    )
                    .await?;
                ensure!(
                    self.tasks
                        .get(&task.id)
                        .await?
                        .context("settled task")?
                        .status
                        .is_terminal(),
                    "settle must reach its terminal write"
                );
            }
            Operation::Expire => self.expire(task).await?,
            Operation::Cancel => anyhow::bail!("cancel does not require ownership"),
        }
        Ok(())
    }

    async fn expire(&mut self, acquired: &AgentTask) -> Result<()> {
        // Finish the remaining budget through real leases and sweeps,
        // never by patching attempt/status fields in the store.
        for attempt in acquired.attempt..=acquired.max_attempts {
            self.at += 2;
            let records = self
                .tasks
                .release_expired_leases(t_plus(self.at), LeaseRecoveryScope::Expired)
                .await?;
            self.check().await?;
            let record = records
                .iter()
                .find(|record| record.id == acquired.id)
                .context("expired lease must be recovered")?;
            if attempt == acquired.max_attempts {
                ensure!(
                    matches!(record.action, RecoveryAction::FailClosed(_)),
                    "last lease must fail closed"
                );
            } else {
                self.tasks
                    .try_acquire_task(
                        &acquired.id,
                        WorkerId::new(),
                        LeaseId::new(),
                        t_plus(self.at + 1),
                        t_plus(self.at),
                    )
                    .await?
                    .context("retry must acquire")?;
                self.check().await?;
            }
        }
        Ok(())
    }
}

async fn run_case(tasks: &dyn AgentTaskStore, threads: &dyn ThreadStore, seed: u64) -> Result<()> {
    let mut runner = TestRunner::new(Config {
        rng_seed: RngSeed::Fixed(seed),
        failure_persistence: None,
        ..Config::default()
    });
    let operations = schedule()
        .new_tree(&mut runner)
        .map_err(|error| anyhow::anyhow!("generate fan-in schedule: {error}"))?
        .current();
    let mut tree = Tree {
        tasks,
        threads,
        ids: Vec::new(),
        seed,
        at: 0,
    };
    // Start with both edge kinds, and put tool grandchildren below the
    // linked root. Later operations can spawn further waves and subagents.
    tree.root().await?;
    tree.apply(Operation::Tools, 0, operations[0].2).await?;
    tree.root().await?;
    tree.apply(Operation::Subagent, 0, 1).await?;
    tree.apply(Operation::Tools, 0, operations[1].2).await?;
    for (step, &(operation, selector, width)) in operations.iter().enumerate() {
        tree.apply(operation, selector, width)
            .await
            .with_context(|| {
                format!("step {step}: {operation:?}, selector={selector}, width={width}")
            })?;
        if tree
            .rows()
            .await?
            .iter()
            .all(|row| row.status.is_terminal())
        {
            tree.root().await?;
        }
    }
    // Drain without spawning or cancelling an ancestor: each last child
    // must wake its parent before the next operation can settle that parent.
    for step in 0..tree.ids.len() {
        if tree
            .rows()
            .await?
            .iter()
            .all(|row| row.status.is_terminal())
        {
            return Ok(());
        }
        let (_, selector, _) = operations[step % operations.len()];
        let operation = [Operation::Complete, Operation::Fail, Operation::Expire][selector % 3];
        tree.apply(operation, selector, 1)
            .await
            .with_context(|| format!("drain step {step}: {operation:?}"))?;
    }
    ensure!(
        tree.rows()
            .await?
            .iter()
            .all(|row| row.status.is_terminal()),
        "drain left live tasks"
    );
    Ok(())
}

async fn fan_in_property(tasks: &dyn AgentTaskStore, threads: &dyn ThreadStore) -> Result<()> {
    let seed = match std::env::var("FAN_IN_SEED") {
        Ok(value) => value.parse::<u64>().context("FAN_IN_SEED must be a u64")?,
        Err(std::env::VarError::NotPresent) => uuid::Uuid::new_v4().as_u64_pair().0,
        Err(error) => return Err(error.into()),
    };
    let cases = match std::env::var("FAN_IN_CASES") {
        Ok(value) => value.parse::<u32>().context("FAN_IN_CASES must be a u32")?,
        Err(std::env::VarError::NotPresent) => 16,
        Err(error) => return Err(error.into()),
    };
    ensure!(cases > 0, "FAN_IN_CASES must be positive");
    for case in 0..cases {
        let seed = seed.wrapping_add(u64::from(case));
        eprintln!("fan-in case {case}: FAN_IN_SEED={seed} FAN_IN_CASES=1");
        run_case(tasks, threads, seed)
            .await
            .with_context(|| format!("replay with FAN_IN_SEED={seed} FAN_IN_CASES=1"))?;
    }
    Ok(())
}

#[tokio::test]
async fn conformance_in_memory_fan_in_property() -> Result<()> {
    let stores = super::fresh_in_memory_stores();
    fan_in_property(stores.task.as_ref(), stores.thread.as_ref()).await
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn conformance_sqlite_fan_in_property() -> Result<()> {
    let store = crate::sqlite::SqliteDurableStore::connect("sqlite::memory:").await?;
    fan_in_property(&store, &store).await
}

#[cfg(feature = "postgres")]
#[tokio::test]
async fn conformance_postgres_fan_in_property() -> Result<()> {
    let Some((store, _guard)) = super::pg_test_store().await? else {
        return Ok(());
    };
    fan_in_property(&store, &store).await
}
