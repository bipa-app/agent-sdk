# Event Contract

Authoritative specification for the agent-server event system.
All implementation and client integration MUST conform to this contract.

---

## 1. Event Lifecycle

Every agent-server turn produces a stream of **lifecycle events** that
describe what happened during execution. Events are server-authoritative:
workers submit raw `AgentEvent` payloads; the server assigns all metadata.

### 1.1 Server-Owned Metadata

Each committed event carries:

| Field | Type | Authority |
|-------|------|-----------|
| `event_id` | UUID v7 | Server-allocated. Globally unique, time-ordered. |
| `thread_id` | `ThreadId` | Thread the event belongs to. |
| `sequence` | `u64` | Monotonic per-thread. Contiguous within batches. Starts at 0. |
| `timestamp` | `OffsetDateTime` | Server commit-time (UTC). |
| `event` | `AgentEvent` | The event payload. |

The **`(thread_id, sequence)` pair is the durable unique key**. Clients use
`sequence` as their resume token.

### 1.2 Event Types

| Event | Description | Emitted By |
|-------|-------------|------------|
| `Start` | Root turn begins execution. | Root worker |
| `Thinking` | LLM thinking/reasoning content. | Root worker |
| `Text` | LLM text response content. | Root worker |
| `ToolCallStart` | Tool invocation initiated. One per tool call. | Root worker (suspension) |
| `ToolProgress` | Mid-execution progress from a tool. | Tool worker |
| `ToolCallEnd` | Tool execution completed (success or failure). | Tool worker |
| `TurnComplete` | One LLM round-trip finished. | Root worker (commit) |
| `Done` | Agent loop completed successfully. | Root worker (commit) |
| `Error` | Execution failure. | Root worker (fail path) |
| `Refusal` | LLM refused the request. | Root worker (commit) |

### 1.3 Canonical Event Sequences

**Text-only turn:**
```
Start → [Thinking →] Text → TurnComplete → Done
```

**Refusal turn:**
```
Start → Text → Refusal → TurnComplete → Done
```

**Tool-call turn (suspend → tool → resume):**
```
Start → [Thinking →] ToolCallStart(1) [→ ToolCallStart(2) …]
  → [ToolProgress(1a) → ToolProgress(1b) →] ToolCallEnd(1)
  → [ToolProgress(2a) →] ToolCallEnd(2)
  → [Thinking →] Text → TurnComplete → Done
```

**Failed turn:**
```
Error
```

---

## 2. Durability and Commit Rules

### 2.1 Atomic Commit Path

All projections — thread aggregate, message history, checkpoint, turn
attempt audit, and lifecycle events — are committed atomically through
`commit_completed_turn`. Either all five steps succeed or none do.

Events are committed **after** state projections succeed (step 5 of 5).
This guarantees that if events exist, the underlying state transitions
are durable.

### 2.2 Single Allocation Authority

The `EventRepository` is the **only** place that assigns `event_id`,
`sequence`, and `timestamp` for committed events. Sequence allocation
and persistence happen under the same write lock (or serializable
transaction in a database backend). No two callers can observe the same
sequence for a thread.

### 2.3 Thread-Scoped Sequencing

Each thread maintains an independent monotonic counter:

- Starts at 0 for the first event on a new thread.
- Increments by 1 for each committed event.
- Sequences are contiguous within batches and across single/batch
  operations on the same thread.
- Different threads have fully independent sequence spaces.

### 2.4 Append-Only Semantics

The event repository is append-only. Committed events are never modified
or deleted through the normal operation path. The `(thread_id, sequence)`
uniqueness constraint is enforced by construction.

---

## 3. Live Streaming

### 3.1 Committed-Envelope-Only Delivery

Live streaming surfaces **only committed events**. Events that have not
been durably committed to the `EventRepository` are never delivered to
subscribers. The notification path (`EventNotifier::notify` or
`LiveTailHub::publish`) is called exclusively **after** durable commit.

This means:

- In-memory events from in-flight turns are invisible.
- Events from cancelled or failed turns that never call
  `commit_completed_turn` do not appear (except explicit `Error` events
  committed through `fail_root_turn`).
- If a commit succeeds but notification fails (process crash), the
  events are still durable and recoverable via replay.

### 3.2 Non-Blocking Producers

Workers are never stalled by subscriber backpressure:

- `EventNotifier`: uses `broadcast::send` which drops messages for
  lagged subscribers.
- `LiveTailHub`: uses `try_send` and never awaits. A slow subscriber
  transitions to lag state, not a blocked producer.

### 3.3 Per-Subscriber Isolation

Each subscriber in the `LiveTailHub` receives its own bounded `mpsc`
channel. A slow subscriber never affects other subscribers on the same
thread. Cross-thread subscribers are completely independent.

---

## 4. Replay

### 4.1 Broad Replay Scope

Replay covers the **entire committed event history** for a thread. All
event types — `Start`, `Thinking`, `Text`, `ToolCallStart`,
`ToolProgress`, `ToolCallEnd`, `TurnComplete`, `Done`, `Error`,
`Refusal` — are replay-eligible. There is no event type that is
live-only or ephemeral.

The replay surface is `stream_events(thread_id, after_sequence)`. The
client provides its last-seen sequence as `after_sequence` (or `None`
for full replay). The stream returns every committed event with
`sequence > after_sequence` in order.

### 4.2 Race-Free Replay-to-Live Handoff

The handoff protocol is a three-step process that closes every race
window:

1. **Subscribe** — acquire a live tail receiver *before* reading durable
   state. This ensures any event committed after this point arrives
   through the live channel.

2. **Capture watermark** — read the current committed high-water mark
   from `EventRepository::next_sequence`. The highest committed
   sequence is `watermark - 1`.

3. **Replay + switch** — yield durable events with
   `sequence > after_sequence && sequence <= watermark`, then switch to
   the live tail for `sequence > watermark`.

**Guarantees:**

- **No gaps**: subscribe-before-read closes the race window.
- **No duplicates**: `last_yielded` tracking skips overlap between
  replay and live tail during handoff.
- **No unpublished events**: live tail is fed only by post-commit
  notification.
- **Thread-scoped, stateless**: the client's `after_sequence` is the
  only resume token. No hidden server-side cursor state.

### 4.3 Canonical Append-Only Replay vs. Future Cleanup

The current replay implementation reads directly from the append-only
event repository. Every committed event — across all turns, tool
executions, and failures — is preserved indefinitely.

**Future cleanup/compaction** (out of scope):

If a future phase introduces derived compacted replay artifacts (e.g.,
merging fine-grained events into summary events, or truncating old
history), those artifacts:

- MUST be a **separate derived layer**, not modifications to the
  canonical append-only event stream.
- MUST NOT violate the `(thread_id, sequence)` uniqueness contract.
- MUST NOT break existing `after_sequence` resume tokens. A client
  reconnecting with a valid `after_sequence` from before compaction
  must either receive the expected suffix or a clear signal to
  re-establish from a new baseline.
- SHOULD be marked as derived/compacted in metadata so clients can
  distinguish original events from compacted summaries.

Until a compaction layer ships, the replay surface is identical to the
canonical append-only repository.

---

## 5. Lag Handling

### 5.1 Bounded-Wait Lag Detection

When a subscriber's buffer fills up, the `LiveTailHub` transitions the
subscriber from `Healthy` to `Lagging`. The lag state machine:

```
Healthy → (buffer full) → Lagging → (grace expired) → Disconnected
```

During the **Lagging** state:

- The hub stops delivering events to the subscriber.
- Events are still durable and recoverable via replay.
- The subscriber's existing buffer can be drained.

### 5.2 Grace Period

After lag is detected, the subscriber enters a **bounded-wait grace
period**. During this window:

- No new events are delivered (they're durable).
- The subscriber is NOT disconnected.
- The subscriber can drain its existing buffer.

After the grace period expires, the **next `publish` call** triggers
disconnect with `ReplayRequired` semantics.

### 5.3 Replay-Required Disconnect

On disconnect, the hub:

1. Records the last successfully delivered `sequence` in shared
   out-of-band state.
2. Sets the `replay_required` flag.
3. Drops the sender half of the mpsc channel.

The receiver:

1. Drains any remaining buffered events.
2. Returns `LiveTailEvent::ReplayRequired { last_delivered_sequence }`
   exactly once.
3. Returns `None` on subsequent calls (stream closed).

### 5.4 Reconnect Semantics

After receiving `ReplayRequired`, the client reconnects via
`stream_events` with `after_sequence` set to `last_delivered_sequence`.
This picks up exactly the missed events through durable replay and
seamlessly transitions to live delivery.

---

## 6. Operational Defaults

### 6.1 Subscriber Buffer Size

| Parameter | Default | Description |
|-----------|---------|-------------|
| `buffer_capacity` | **256** | Maximum events buffered per subscriber in `LiveTailHub` before lag detection triggers. |
| `EventNotifier` channel capacity | **256** | Broadcast channel capacity per thread. |

The `LiveTailHub` clamps `buffer_capacity` to a minimum of **1** (Tokio
`mpsc::channel` requires non-zero capacity).

### 6.2 Lag Grace Period

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lag_grace_period` | **5 seconds** | Time between lag detection and subscriber disconnect. |

During the grace period, the subscriber can drain its existing buffer.
Events committed during this window are durable and recoverable via
replay after reconnect.

### 6.3 Event Sequencing

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sequence start | **0** | First event on each thread starts at sequence 0. |
| Sequence stride | **1** | Each event increments by exactly 1. |
| Scope | **per-thread** | Each thread has an independent sequence counter. |
| UUID version | **v7** | Time-ordered, globally unique `event_id`. |

### 6.4 Retry and Recovery Guardrails

| Scenario | Behavior |
|----------|----------|
| Worker crash mid-turn | No partial events committed. The turn's events are committed atomically at completion. |
| Stale worker after lease loss | CAS guard prevents duplicate event emission. `Start` event is committed inside the idempotency guard. |
| Tool task cancellation | `Cancelled` outcome emits no events. |
| Tool task failure | `ToolCallEnd` with `success: false` is committed. |
| Duplicate `Start` prevention | `Start` is committed inside each response branch, after the idempotency guard fires. |

---

## 7. Wire Format

### 7.1 CommittedEvent (Storage Format)

```json
{
  "event_id": "01905a8c-...",
  "thread_id": "thread_abc",
  "sequence": 42,
  "timestamp": "2024-01-15T10:30:00Z",
  "event": {
    "type": "text",
    "message_id": "msg_123",
    "text": "Hello"
  }
}
```

The `event` payload is **nested** (not flattened) in the storage format.

### 7.2 AgentEventEnvelope (Client Wire Format)

```json
{
  "event_id": "01905a8c-...",
  "sequence": 42,
  "timestamp": "2024-01-15T10:30:00Z",
  "type": "text",
  "message_id": "msg_123",
  "text": "Hello"
}
```

The envelope is **flattened**: the `type` field and event-specific fields
appear at the top level. `thread_id` is dropped because the delivery
channel implies it.

Conversion: `CommittedEvent::into_envelope()` /
`CommittedEvent::to_envelope()`.

---

## 8. Decision Record

The event model is **decision-complete** for implementation. The
following decisions are final:

1. **Server-authoritative metadata**: the server owns `event_id`,
   `sequence`, and `timestamp`. Workers submit raw `AgentEvent` payloads.

2. **Append-only canonical stream**: events are never modified or
   deleted. Future compaction is a separate derived layer.

3. **Committed-envelope-only live streaming**: only durably committed
   events are delivered to subscribers.

4. **Non-blocking producers**: workers are never stalled by subscriber
   backpressure.

5. **Bounded-wait lag handling**: lagging subscribers are disconnected
   after a configurable grace period with replay-required semantics.

6. **Race-free handoff**: subscribe-before-read protocol with
   `last_yielded` deduplication.

7. **Thread-scoped independence**: each thread has fully independent
   sequence counters, subscribers, and replay state.

8. **Atomic event commit**: events are committed as the final step of
   the atomic commit path, after all state projections succeed.

9. **Fail-closed on failure**: error events are committed for failed
   turns. Cancelled tasks emit no events.

10. **Idempotent event emission**: `Start` events are committed inside
    the response branch, preventing duplicates from stale workers.

---

## Out of Scope

The following are explicitly out of scope:

- Cross-instance live fanout (distributed pub/sub).
- AMQP or outbox integration.
- Cleanup or compaction implementation.
- Delta/compacted replay artifacts.
- Event schema versioning or migration.
