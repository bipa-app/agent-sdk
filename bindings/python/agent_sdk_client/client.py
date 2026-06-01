"""Thin Python client over the agent-service-host durable gRPC API.

This is the binding-spike facade: it wraps the generated gRPC stubs in a small,
ergonomic surface that mirrors the in-process Rust SDK's happy path
(`create thread -> submit work -> stream events`) plus the durable HITL
confirmation flow. The durable engine (the Rust `agent-service-host`) does all
the work; this client only marshals requests and yields decoded events.

Design: a "thin binding over the engine" — the same model the Claude Agent SDK
uses. Nothing about the agent loop, providers, tools, or persistence lives here;
those stay in the Rust core behind a stable, versioned gRPC contract.

Run `generate_stubs.sh` first to produce `agent_sdk_client/_generated`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Iterator, Optional

# The generated stubs use absolute `agent.service.v1...` imports, so the
# _generated dir must be importable as a roots dir.
_GEN_DIR = os.path.join(os.path.dirname(__file__), "_generated")
if _GEN_DIR not in sys.path:
    sys.path.insert(0, _GEN_DIR)

try:
    import grpc  # type: ignore

    from agent.service.v1 import common_pb2  # type: ignore
    from agent.service.v1 import control_pb2  # type: ignore
    from agent.service.v1 import control_pb2_grpc  # type: ignore
    from agent.service.v1 import events_pb2  # type: ignore
    from agent.service.v1 import events_pb2_grpc  # type: ignore

    _STUBS_AVAILABLE = True
except ImportError:  # pragma: no cover - stubs not generated yet
    _STUBS_AVAILABLE = False


@dataclass
class SubmittedWork:
    """Result of submitting work to a thread."""

    thread_id: str
    task_id: str
    queue_depth: int


class AgentClient:
    """Synchronous client for the durable agent serving host.

    Example:
        client = AgentClient("localhost:50051")
        thread_id = client.create_thread()
        work = client.submit_text(thread_id, "Summarize the latest release notes.")
        for event in client.stream_events(thread_id, follow=True):
            print(event)
    """

    def __init__(self, target: str = "localhost:50051", *, secure: bool = False):
        if not _STUBS_AVAILABLE:
            raise RuntimeError(
                "gRPC stubs not generated. Run bindings/python/generate_stubs.sh "
                "and `pip install -r requirements.txt` first."
            )
        if secure:
            self._channel = grpc.secure_channel(target, grpc.ssl_channel_credentials())
        else:
            self._channel = grpc.insecure_channel(target)
        self._control = control_pb2_grpc.AgentControlServiceStub(self._channel)
        self._events = events_pb2_grpc.AgentEventServiceStub(self._channel)

    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> "AgentClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ── Control plane ────────────────────────────────────────────────
    def create_thread(self, request_id: str = "") -> str:
        """Create a durable thread and return its id."""
        resp = self._control.CreateThread(
            control_pb2.CreateThreadRequest(request_id=request_id)
        )
        return resp.thread.thread_id

    def submit_text(
        self, thread_id: str, text: str, *, request_id: str = ""
    ) -> SubmittedWork:
        """Submit a single text input as the next root turn on a thread."""
        resp = self._control.SubmitThreadWork(
            control_pb2.SubmitThreadWorkRequest(
                request_id=request_id,
                thread_id=thread_id,
                input=[common_pb2.UserInputItem(text=text)],
            )
        )
        return SubmittedWork(
            thread_id=resp.thread.thread_id,
            task_id=resp.task.task_id,
            queue_depth=resp.current_queue_depth,
        )

    def approve(self, thread_id: str, task_id: str, *, request_id: str = "") -> None:
        """Approve a tool call awaiting a durable confirmation (HITL)."""
        self._control.DecideConfirmation(
            control_pb2.DecideConfirmationRequest(
                request_id=request_id,
                thread_id=thread_id,
                task_id=task_id,
                decision=common_pb2.ConfirmationDecision(
                    approved=common_pb2.ApprovedConfirmation()
                ),
            )
        )

    def reject(self, thread_id: str, task_id: str, *, request_id: str = "") -> None:
        """Reject a tool call awaiting a durable confirmation (HITL)."""
        self._control.DecideConfirmation(
            control_pb2.DecideConfirmationRequest(
                request_id=request_id,
                thread_id=thread_id,
                task_id=task_id,
                decision=common_pb2.ConfirmationDecision(
                    rejected=common_pb2.RejectedConfirmation()
                ),
            )
        )

    # ── Event plane ──────────────────────────────────────────────────
    def stream_events(
        self,
        thread_id: str,
        *,
        after_sequence: Optional[int] = None,
        follow: bool = True,
    ) -> Iterator[object]:
        """Replay (and optionally follow) the durable event log for a thread.

        Yields the decoded `oneof` payload of each `StreamThreadEventsResponse`
        frame (an `EventEnvelope` for committed events, or a control frame).
        The stream ends on `closed`, `retention_gap`, or `replay_required`.
        """
        follow_mode = (
            events_pb2.FOLLOW_MODE_REPLAY_AND_FOLLOW
            if follow
            else events_pb2.FOLLOW_MODE_REPLAY_ONLY
        )
        req = events_pb2.StreamThreadEventsRequest(
            thread_id=thread_id, follow_mode=follow_mode
        )
        if after_sequence is not None:
            req.after_sequence = after_sequence

        for frame in self._events.StreamThreadEvents(req):
            which = frame.WhichOneof("item")
            yield getattr(frame, which)
