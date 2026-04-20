# Plan: Integrate `agent-service-host` into Bipa's main server

> Source PRD: Linear project [Integrate into Bipa server](https://linear.app/bipa/project/integrate-into-bipa-server-6bcf8073b320/overview)

## Architectural decisions

Durable decisions that apply across all phases:

- **Public API boundary**: Bipa keeps the existing mobile-facing gRPC/API contract. Mobile clients do not talk directly to the host in this project.
- **Execution boundary**: `agent-service-host` runs inside Bipa's main server behind a real internal gRPC loopback boundary.
- **Routing model**: Thread backend is explicit, durable, and sticky. A thread is either `legacy_sdk` or `embedded_host`, and that choice is made on first accepted submission.
- **Rollout model**: The Bipa agent is already employee-only, but host-backed routing still gets its own separate gate. Only new allowlisted employee threads are eligible for the embedded-host path in this project. Existing legacy threads do not migrate in place.
- **Persistence model**: The embedded host owns authoritative execution state for migrated traffic once work is admitted. If Bipa projection fails after host admission, the thread is marked degraded and reconciled asynchronously rather than rolling host work back.
- **Schema shape**: Bipa stores a thread binding model keyed by the product thread id, with exactly one stable host thread id for the lifetime of a host-backed thread, plus projection checkpoint, loop/session metadata, and health state. Bipa also stores confirmation payload state that is not part of the host proto.
- **Read model**: Bipa continues to serve `AgentStream`, thread history, and thread list from Bipa-owned projection tables. Bipa projects the full client-relevant event history needed for replay/history rather than relying on lossy reconstruction.
- **Ordering and replay**: Host event sequence is the canonical ordering source for host-backed threads. Bipa preserves that order in projection while keeping current `last_event_id` reconnect semantics externally. Replay gaps are a degraded state, not a silent fast-forward.
- **Bipa-owned semantics**: `RateLimited`, `UiBlock`, deeplink/navigation semantics, `ask_user`, and payloads such as `pin_token`, `wallet_signature`, and `user_input_value` stay Bipa-owned. `ask_user` remains a Bipa-owned tool, not a host-native primitive.
- **Session model**: `agent_loops` stay as Bipa product-session metadata only. `agent_turns` remain legacy-only and are not written for host-backed threads.
- **Parity bar**: Rollout requires 100% behavioral parity for employee-facing tools, confirmations, event output, analytics semantics, and user-visible failures. Unsupported or drifting behavior is a blocker, not a soft warning.
- **Failure policy**: Degraded host-backed threads stop accepting new submissions and fail visibly. They do not silently fall back to `legacy_sdk`.
- **Config and secrets**: Host-backed execution reuses Bipa's existing secret/config and authorization decisions. The embedded host uses the same Postgres cluster with a separate schema and separate pool lifecycle.
- **Auth and trust boundary**: Mobile authentication and authorization remain in Bipa. The host is reached through an internal trusted client only.
- **Service boundaries**: Primary implementation repo is `~/work/bipa/master`. The supporting reference repo is `~/work/agent-sdk`.

## Derived user stories

- **US1**: As a Bipa client, I can keep using the current public agent API while the backend execution engine changes.
- **US2**: As a platform engineer, I can route only new allowlisted threads to the embedded host while keeping legacy traffic stable.
- **US3**: As the embedded host, I can resolve Bipa-specific models, providers, tools, and confirmations without forking Bipa business logic.
- **US4**: As Bipa, I can project host execution back into the current read model so existing streaming/history APIs continue to work.
- **US5**: As a Bipa user, confirmation flows and product-only UX semantics behave the same on host-backed threads.
- **US6**: As an operator, I can recover from restarts, observe rollout health, and stop rollout safely if the host-backed path regresses.

## Implementation workflow

- Create a dedicated worktree in the Bipa repo named after the issue id:
  - `cd ~/work/bipa/master`
  - `git fetch origin`
  - `git worktree add ~/worktrees/bipa/ENG-XXXX -b ENG-XXXX origin/master`
- Use `~/work/bipa/master` for implementation and `~/work/agent-sdk` as the reference checkout for host behavior and API contracts.
- Before coding, recover context from:
  - the project overview
  - the parent tracker issue
  - direct blocker issues
  - the current Bipa implementation in `~/work/bipa/master`
  - the host/proto behavior in `~/work/agent-sdk`
- Before closing a card, leave behind context in Linear:
  - summarize the decisions made
  - list any migrations, flags, or rollout constraints added
  - record any follow-up issues created or newly discovered
  - update the issue description if the durable scope changed

---

## Phase 0: Establish the embedded host baseline in Bipa

**User stories**: US1, US2

### What to build

Create the dependency, process, and storage baseline that lets Bipa run the embedded host without exposing it publicly or breaking the existing client contract.

### Acceptance criteria

- [ ] Bipa and the embedded host build against the same runtime family
- [ ] The embedded host can boot inside Bipa and be reached through an internal client
- [ ] Host-specific config and Postgres lifecycle are isolated enough to operate independently

---

## Phase 1: Add thread ownership and routing for host-backed threads

**User stories**: US1, US2

### What to build

Make backend ownership explicit so Bipa can decide per thread whether work stays on the legacy runtime or goes to the embedded host, while keeping the rollout limited to new allowlisted threads.

### Acceptance criteria

- [ ] Backend choice is durable and queryable for every thread
- [ ] New allowlisted threads can route to the embedded host
- [ ] Existing legacy threads remain on the legacy path

---

## Phase 2: Implement Bipa runtime adapters for host execution

**User stories**: US3, US5

### What to build

Provide the runtime adapters that let the host resolve Bipa models, providers, tools, and confirmation rules from Bipa-owned state instead of introducing a second source of truth for business logic.

### Acceptance criteria

- [ ] Host-backed execution can resolve the same model/provider choices as Bipa today
- [ ] Host-backed tool calls reach the current Bipa tool implementations
- [ ] Confirmation classification remains Bipa-owned and consistent with the legacy path

---

## Phase 3: Project host execution into Bipa's existing client contract

**User stories**: US1, US4, US5

### What to build

Project committed host execution into Bipa's current read model so stream, history, and product-only UX semantics remain stable for current clients.

### Acceptance criteria

- [ ] Host-backed threads can replay/follow events into Bipa's read model
- [ ] Existing Bipa stream and history APIs work for host-backed threads
- [ ] Bipa-only events and interaction semantics remain available

---

## Phase 4: Bridge confirmation payloads and harden rollout

**User stories**: US5, US6

### What to build

Persist the Bipa-only confirmation payloads that are outside the host proto, recover projection followers safely after restarts, and add the observability and rollout guardrails needed for an internal launch.

### Acceptance criteria

- [ ] Confirmation payloads survive approval/resume lifecycles
- [ ] Host-backed threads recover cleanly after process restarts
- [ ] Rollout is measurable, bounded, and reversible for an internal cohort
