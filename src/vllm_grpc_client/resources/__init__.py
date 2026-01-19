"""
Resource modules for vLLM gRPC client.

Each resource module provides a class that handles a specific set of
gRPC RPCs, similar to the OpenAI Python client's resource structure.
"""

from vllm_grpc_client.resources.completions import AsyncCompletions, Completions
from vllm_grpc_client.resources.embeddings import AsyncEmbeddings, Embeddings
from vllm_grpc_client.resources.health import AsyncHealth, Health
from vllm_grpc_client.resources.models import AsyncModels, Models

__all__ = [
    "Completions",
    "AsyncCompletions",
    "Embeddings",
    "AsyncEmbeddings",
    "Models",
    "AsyncModels",
    "Health",
    "AsyncHealth",
]
