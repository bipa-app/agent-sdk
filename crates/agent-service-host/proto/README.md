# gRPC Contract

This directory is the reviewable transport contract for the service-host path.

- [`agent/service/v1/control.proto`](agent/service/v1/control.proto) defines unary thread, transcript, task, submission, and confirmation RPCs.
- [`agent/service/v1/events.proto`](agent/service/v1/events.proto) defines the replay-first event stream.
- [`agent/service/v1/common.proto`](agent/service/v1/common.proto) defines committed thread, task, message, and confirmation types.
- [`agent/service/v1/errors.proto`](agent/service/v1/errors.proto) defines rich error-detail payloads for unary RPC failures.

## Service boundaries

- `AgentControlService` owns committed snapshots and durable mutations.
- `AgentEventService` owns replay and live-follow of committed thread events.

Task inspection is intentionally unary-only in this slice. The journal has a durable task row, but it does not yet have a durable task-change log with replay cursors. The contract therefore keeps replay semantics on the committed event stream alone and uses `GetTask` / `ListThreadTasks` for current task state.

## Replay semantics

`StreamThreadEvents` is the only replayable stream.

Rules:

1. `after_sequence` is exclusive. Reconnect with the last committed event sequence the client has durably processed.
2. Every `event` frame carries a server-assigned `sequence` and `commit_time`.
3. Only `event` frames participate in ordering. Control frames never consume a sequence number.
4. A successful stream starts with `replay_opened`, emits zero or more replay `event` frames, then emits `replay_catchup_complete`.
5. `FOLLOW_MODE_REPLAY_ONLY` ends with `closed(reason = STREAM_CLOSE_REASON_REPLAY_EXHAUSTED)` and EOF.
6. `FOLLOW_MODE_REPLAY_AND_FOLLOW` continues with live committed `event` frames after `replay_catchup_complete`.

Terminal control frames:

- `retention_gap`: the requested cursor is older than the retained event window. The server closes normally after emitting the frame.
- `replay_required`: the subscriber fell behind live delivery and must reconnect using `last_delivered_sequence`.
- `closed`: the server intentionally ended the stream because replay was exhausted, the thread completed, or the host shut down.

## Retention-gap recovery

When a client receives `retention_gap`, the recovery path is explicit:

1. Call `GetThread` to fetch the current thread snapshot and the current event-window bounds.
2. Call `GetThreadMessages` to rebuild the committed transcript snapshot.
3. Call `ListThreadTasks` or `GetTask` for any task-level UI that needs to be refreshed.
4. Open a new `StreamThreadEvents` call from a fresh cursor derived from the latest committed event sequence.

This keeps recovery on committed server state instead of hidden in-memory replay buffers.

## Confirmation semantics

Pending confirmations are part of the committed task snapshot through `TaskSnapshot.awaiting_confirmation`.

`DecideConfirmation` is durable and idempotent:

- approval moves the child back to `pending` for authoritative policy recheck and execution
- rejection fails the child with a canonical reason
- timeout fails the child as an explicit terminal decision

The request carries both `thread_id` and `task_id` so the server can reject cross-thread mismatches instead of relying on client-local routing assumptions.

## Unary error model

Recommended gRPC status mapping:

- `INVALID_ARGUMENT`: malformed IDs, empty input, missing idempotency key
- `NOT_FOUND`: unknown thread or task
- `FAILED_PRECONDITION`: thread is completed, task is not awaiting confirmation, or the requested state transition is not valid for the current committed row
- `ALREADY_EXISTS`: reused idempotency key with a different logical request body
- `UNAVAILABLE`: transient storage or shutdown conditions

Recommended rich detail payloads are defined in `errors.proto`.
