#!/usr/bin/env bash
# Regenerate the Python gRPC stubs from the agent-service-host .proto contract.
#
# The proto files are the single source of truth (crates/agent-service-host/
# proto). This script does NOT vendor a copy — it generates straight from the
# repo's protos so the binding can never drift from the server contract.
#
# Usage:
#   bindings/python/generate_stubs.sh
#
# Requires (see requirements.txt):
#   pip install grpcio-tools
set -euo pipefail

# Resolve repo root from this script's location so it runs from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PROTO_ROOT="$REPO_ROOT/crates/agent-service-host/proto"
OUT_DIR="$SCRIPT_DIR/agent_sdk_client/_generated"

mkdir -p "$OUT_DIR"
touch "$OUT_DIR/__init__.py"

python3 -m grpc_tools.protoc \
  -I "$PROTO_ROOT" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  --pyi_out="$OUT_DIR" \
  "$PROTO_ROOT"/agent/service/v1/common.proto \
  "$PROTO_ROOT"/agent/service/v1/control.proto \
  "$PROTO_ROOT"/agent/service/v1/events.proto \
  "$PROTO_ROOT"/agent/service/v1/errors.proto

echo "Generated stubs into $OUT_DIR"
echo "Note: grpc_python_out emits 'import agent.service.v1...' absolute imports."
echo "Add $OUT_DIR to PYTHONPATH, or run via the package which does so."
