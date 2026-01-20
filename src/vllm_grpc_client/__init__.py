"""
vLLM gRPC Client - Python client for vLLM gRPC server with OpenAI-style interface.

This package provides a high-level Python client for interacting with vLLM's
gRPC server. The interface is designed to be similar to the OpenAI Python
client for familiarity and ease of use.

Usage:
    from vllm_grpc_client import VLLMGrpcClient, AsyncVLLMGrpcClient

    # Sync client
    client = VLLMGrpcClient(host="localhost", port=9000)
    response = client.completions.create(prompt="Hello, world!")

    # Async client
    async with AsyncVLLMGrpcClient(host="localhost", port=9000) as client:
        response = await client.completions.create(prompt="Hello, world!")
"""

from vllm_grpc_client._client import AsyncVLLMGrpcClient, VLLMGrpcClient
from vllm_grpc_client._exceptions import (
    VLLMGrpcAbortedError,
    VLLMGrpcConnectionError,
    VLLMGrpcError,
    VLLMGrpcInvalidArgumentError,
    VLLMGrpcTimeoutError,
    VLLMGrpcUnavailableError,
    VLLMGrpcUnimplementedError,
)
from vllm_grpc_client._streaming import AsyncGenerateStream, GenerateStream
from vllm_grpc_client._types import (
    ChoiceConstraint,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CompletionChunkChoice,
    CompletionUsage,
    Embedding,
    EmbeddingResponse,
    EmbeddingUsage,
    HealthCheckResponse,
    ModelInfo,
    SamplingParams,
    ServerInfo,
    StructuredOutputs,
    TokenizedInput,
)
from vllm_grpc_client.utils import TokenDecoder

__version__ = "0.1.0"

__all__ = [
    # Client classes
    "VLLMGrpcClient",
    "AsyncVLLMGrpcClient",
    # Streaming
    "GenerateStream",
    "AsyncGenerateStream",
    # Types
    "SamplingParams",
    "StructuredOutputs",
    "ChoiceConstraint",
    "Completion",
    "CompletionChoice",
    "CompletionUsage",
    "CompletionChunk",
    "CompletionChunkChoice",
    "Embedding",
    "EmbeddingResponse",
    "EmbeddingUsage",
    "HealthCheckResponse",
    "ModelInfo",
    "ServerInfo",
    "TokenizedInput",
    # Exceptions
    "VLLMGrpcError",
    "VLLMGrpcConnectionError",
    "VLLMGrpcTimeoutError",
    "VLLMGrpcAbortedError",
    "VLLMGrpcInvalidArgumentError",
    "VLLMGrpcUnavailableError",
    "VLLMGrpcUnimplementedError",
    # Utilities
    "TokenDecoder",
]
