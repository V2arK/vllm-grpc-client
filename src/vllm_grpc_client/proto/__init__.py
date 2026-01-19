"""
Proto buffer definitions for vLLM gRPC client.

This module contains auto-generated Python code from vllm_engine.proto.
The proto file is copied from the vLLM project and should be kept in sync
when the vLLM gRPC interface changes.

To regenerate the proto files:
    python -m grpc_tools.protoc -I=src/vllm_grpc_client/proto \
        --python_out=src/vllm_grpc_client/proto \
        --grpc_python_out=src/vllm_grpc_client/proto \
        --pyi_out=src/vllm_grpc_client/proto \
        src/vllm_grpc_client/proto/vllm_engine.proto
"""

# These imports will be available after proto compilation
from vllm_grpc_client.proto import vllm_engine_pb2, vllm_engine_pb2_grpc

__all__ = [
    "vllm_engine_pb2",
    "vllm_engine_pb2_grpc",
]
