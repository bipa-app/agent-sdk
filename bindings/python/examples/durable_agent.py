"""End-to-end demo: drive the durable agent engine from Python.

Prerequisites:
  1. pip install -r ../requirements.txt
  2. ../generate_stubs.sh
  3. Run the host (from the repo root):
       cargo run -p agent-service-host
     and point AGENT_HOST at its gRPC listen address (default localhost:50051).

Usage:
  AGENT_HOST=localhost:50051 python examples/durable_agent.py "Your prompt"
"""

import os
import sys

# Make the sibling package importable when run directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_sdk_client import AgentClient  # noqa: E402


def main() -> int:
    target = os.environ.get("AGENT_HOST", "localhost:50051")
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Say hello in one sentence."

    with AgentClient(target) as client:
        thread_id = client.create_thread()
        print(f"thread: {thread_id}")

        work = client.submit_text(thread_id, prompt)
        print(f"submitted task {work.task_id} (queue depth {work.queue_depth})")

        # Replay + follow the durable event log until the turn closes.
        for frame in client.stream_events(thread_id, follow=True):
            kind = type(frame).__name__
            print(f"  <{kind}> {frame}")
            if kind in ("EventStreamClosed", "RetentionGap", "ReplayRequired"):
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
