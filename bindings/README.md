# Bindings spike — Python / TypeScript over the durable core

This directory holds the **Phase 13·D binding spike** and its feasibility
recommendation. It is a proof-of-concept, not a shipping SDK: full bindings are
a follow-up.

## TL;DR recommendation

**Build the language bindings as thin gRPC clients against the
`agent-service-host` durable serving API — not as PyO3/napi-rs wrappers over the
in-process Rust SDK.** A working Python PoC lives in [`python/`](./python).

This is the same architecture the Claude Agent SDK uses (a thin binding over a
long-lived engine), and it is the option that actually plays to this codebase's
unique strength: the durable, crash-safe, resumable serving core.

## The two options we evaluated

### Option A — gRPC client over `agent-service-host` (RECOMMENDED, PoC built)

The host already exposes a clean, versioned gRPC contract
(`crates/agent-service-host/proto/agent/service/v1/`):

- `AgentControlService`: `CreateThread`, `SubmitThreadWork`, `GetThread`,
  `GetThreadMessages`, `ListThreadTasks`, `GetTask`, `ForkThread`,
  `DecideConfirmation`.
- `AgentEventService.StreamThreadEvents`: server-streaming replay + live follow
  of the durable event log.

A binding is then *just* generated stubs plus a thin ergonomic facade. The PoC
in `python/` does exactly this:

```python
from agent_sdk_client import AgentClient

with AgentClient("localhost:50051") as client:
    thread_id = client.create_thread()
    work = client.submit_text(thread_id, "Summarize the latest release notes.")
    for event in client.stream_events(thread_id, follow=True):
        print(event)
```

**Why this is the right call:**

| Dimension | gRPC client |
|-----------|-------------|
| Effort | Low — `protoc`/`grpcio-tools` generate the wire types; the facade is ~150 LOC |
| Maintenance | Contract is the `.proto`; bindings can't drift from the server semantics |
| Durability story | **Inherited for free** — the engine owns crash-safe threads, resumable turns, durable HITL confirmations. This is the "Temporal-grade backend for Python/TS agent loops" the gap analysis calls out |
| Concurrency / GIL | None of our problem — the agent loop runs in the Rust host, not the Python/Node process |
| Multi-language | One contract serves Python, TypeScript, Go, anything with a gRPC codegen |
| Deployment | Engine runs as a service (local or remote); bindings are pure clients |
| ABI / build | No native extension to compile, ship per-platform, or match to a Python/Node ABI |

**Costs / caveats:**

- Requires running the host process (a service, not an `import`). For a
  library-style "just call a function" use case this is heavier than Option B.
- Tool *execution* in the durable/external runtime is host-side. A binding that
  wants to run tools written *in Python* would need either the in-process model
  (Option B) or a tool-runtime callback channel over gRPC (a larger follow-up).

### Option B — PyO3 / napi-rs over the in-process SDK (NOT recommended for v1)

Wrap the `agent-sdk` crate directly with PyO3 (Python) or napi-rs (Node).

**Why we did not pick this for the spike:**

- The in-process SDK is deeply async (`tokio`) and generic over a user context
  type `<C>`, a provider trait, and a tool registry of Rust trait objects.
  Exposing that across an FFI boundary means erasing the generics, bridging two
  async runtimes, and marshalling closures/tools — a large surface with sharp
  edges (GIL, `Send`/`Sync`, panic-safety across the boundary).
- It throws away the durability/serving story, which is the differentiator. A
  PyO3 binding would be "another in-process agent loop, but slower to call,"
  competing with mature Python-native frameworks on their turf.
- It forces per-platform native wheels / prebuilt binaries and an ABI-matching
  build matrix (cp39/cp310/…, manylinux, macOS arm64/x86, Windows).

Option B remains the right tool *if and when* a pure-embedded, no-service,
Python-tools-in-process use case becomes a priority. It is complementary, not a
prerequisite.

## What the PoC demonstrates

- `python/` — a runnable thin client (`agent_sdk_client/`), a demo driver
  (`examples/durable_agent.py`), stub-generation from the canonical protos
  (`generate_stubs.sh`), and `requirements.txt`.
- The client covers the full happy path **and** the durable HITL flow
  (`approve` / `reject` against `DecideConfirmation`).
- Validated in this environment: the four `.proto` files compile cleanly with
  `protoc`; the package imports and degrades gracefully before stubs are
  generated. (Codegen + a live round-trip need `pip install grpcio-tools` and a
  running host — see below.)

## Running the PoC end-to-end

```bash
cd bindings/python
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./generate_stubs.sh                       # protoc -> agent_sdk_client/_generated

# In another terminal, from the repo root, start the durable host:
#   cargo run -p agent-service-host        # listens on its configured gRPC addr

AGENT_HOST=localhost:50051 python examples/durable_agent.py "Hello!"
```

## Recommended follow-up (full binding, out of scope here)

1. Package the gRPC client as `agent-sdk` on PyPI and `@bipa/agent-sdk` on npm,
   each generating stubs at build time from the versioned protos.
2. Add an ergonomic high-level API (async iterators for the event stream, typed
   event dataclasses/interfaces decoded from `EventEnvelope`).
3. Optionally add a gRPC **tool-runtime callback** channel so bindings can
   register tools that execute in the client process while the loop stays in the
   durable engine — the best of both options.
4. Publish a versioning policy that ties the binding version to the proto
   `agent.service.v1` contract revision.
