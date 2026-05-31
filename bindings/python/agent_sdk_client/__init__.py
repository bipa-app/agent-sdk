"""Python binding spike for the Agent SDK durable serving host.

A thin gRPC client over `agent-service-host`. See the package README for the
feasibility recommendation and architecture.
"""

from .client import AgentClient, SubmittedWork

__all__ = ["AgentClient", "SubmittedWork"]
