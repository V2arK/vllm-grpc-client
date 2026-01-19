"""
Embeddings resource for vLLM gRPC client.

Handles the Embed RPC for generating embeddings.

NOTE: The Embed RPC is not yet implemented in vLLM gRPC server.
This resource is provided for API completeness and future compatibility.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, List, Optional, Union

from vllm_grpc_client._exceptions import _exception_from_grpc_error
from vllm_grpc_client._types import (
    Embedding,
    EmbeddingResponse,
    EmbeddingUsage,
    TokenizedInput,
)
from vllm_grpc_client.proto import vllm_engine_pb2

if TYPE_CHECKING:
    from vllm_grpc_client._client import AsyncVLLMGrpcClient, VLLMGrpcClient


# Type alias for embedding input
EmbeddingInput = Union[str, List[int], TokenizedInput]


class Embeddings:
    """
    Synchronous embeddings resource.

    Provides methods for generating embeddings using the vLLM gRPC server's
    Embed RPC.

    NOTE: The Embed RPC is not yet implemented in vLLM gRPC server.
    Calling this method will raise VLLMGrpcUnimplementedError.
    """

    def __init__(self, client: "VLLMGrpcClient"):
        """
        Initialize the embeddings resource.

        Args:
            client: The parent VLLMGrpcClient instance.
        """
        self._client = client

    def create(
        self,
        *,
        input: EmbeddingInput,
        request_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> EmbeddingResponse:
        """
        Create embeddings for the given input.

        NOTE: This RPC is not yet implemented in vLLM gRPC server.
        Calling this method will raise VLLMGrpcUnimplementedError.

        Args:
            input: The input to embed. Can be a string, list of token IDs,
                or a TokenizedInput object.
            request_id: Optional unique request ID. Auto-generated if not provided.
            timeout: Request timeout in seconds.

        Returns:
            EmbeddingResponse containing the embedding vectors.

        Raises:
            VLLMGrpcUnimplementedError: This RPC is not yet implemented.
        """
        if request_id is None:
            request_id = f"emb-{uuid.uuid4().hex[:24]}"

        # Build the gRPC request
        grpc_request = self._build_embed_request(input=input, request_id=request_id)

        try:
            response = self._client._stub.Embed(
                grpc_request,
                timeout=timeout or self._client._timeout,
            )

            return EmbeddingResponse(
                model=self._client._model_name,
                data=[
                    Embedding(
                        embedding=list(response.embedding),
                        index=0,
                    )
                ],
                usage=EmbeddingUsage(
                    prompt_tokens=response.prompt_tokens,
                    total_tokens=response.prompt_tokens,
                ),
            )

        except Exception as e:
            import grpc

            if isinstance(e, grpc.RpcError):
                raise _exception_from_grpc_error(e) from e
            raise

    def _build_embed_request(
        self,
        input: EmbeddingInput,
        request_id: str,
    ) -> vllm_engine_pb2.EmbedRequest:
        """Build a gRPC EmbedRequest from parameters."""
        # Determine input type and convert to TokenizedInput
        if isinstance(input, str):
            # String input - need tokenization (not directly supported by gRPC)
            # In practice, the server should handle this, but current proto
            # only accepts TokenizedInput
            tokenized = vllm_engine_pb2.TokenizedInput(
                original_text=input,
                input_ids=[],  # Server may need to tokenize
            )
        elif isinstance(input, TokenizedInput):
            tokenized = vllm_engine_pb2.TokenizedInput(
                original_text=input.original_text,
                input_ids=input.input_ids,
            )
        elif isinstance(input, list):
            # List of token IDs
            tokenized = vllm_engine_pb2.TokenizedInput(
                input_ids=input,
            )
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

        return vllm_engine_pb2.EmbedRequest(
            request_id=request_id,
            tokenized=tokenized,
        )


class AsyncEmbeddings:
    """
    Asynchronous embeddings resource.

    Provides async methods for generating embeddings using the vLLM gRPC
    server's Embed RPC.

    NOTE: The Embed RPC is not yet implemented in vLLM gRPC server.
    Calling this method will raise VLLMGrpcUnimplementedError.
    """

    def __init__(self, client: "AsyncVLLMGrpcClient"):
        """
        Initialize the async embeddings resource.

        Args:
            client: The parent AsyncVLLMGrpcClient instance.
        """
        self._client = client

    async def create(
        self,
        *,
        input: EmbeddingInput,
        request_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> EmbeddingResponse:
        """
        Create embeddings for the given input asynchronously.

        NOTE: This RPC is not yet implemented in vLLM gRPC server.
        Calling this method will raise VLLMGrpcUnimplementedError.

        See Embeddings.create for detailed parameter documentation.
        """
        if request_id is None:
            request_id = f"emb-{uuid.uuid4().hex[:24]}"

        # Build the gRPC request
        grpc_request = _build_embed_request(input=input, request_id=request_id)

        try:
            response = await self._client._stub.Embed(
                grpc_request,
                timeout=timeout or self._client._timeout,
            )

            return EmbeddingResponse(
                model=self._client._model_name,
                data=[
                    Embedding(
                        embedding=list(response.embedding),
                        index=0,
                    )
                ],
                usage=EmbeddingUsage(
                    prompt_tokens=response.prompt_tokens,
                    total_tokens=response.prompt_tokens,
                ),
            )

        except Exception as e:
            import grpc

            if isinstance(e, grpc.RpcError):
                raise _exception_from_grpc_error(e) from e
            raise


# Helper function (shared by sync and async)
def _build_embed_request(
    input: EmbeddingInput,
    request_id: str,
) -> vllm_engine_pb2.EmbedRequest:
    """Build a gRPC EmbedRequest from parameters."""
    if isinstance(input, str):
        tokenized = vllm_engine_pb2.TokenizedInput(
            original_text=input,
            input_ids=[],
        )
    elif isinstance(input, TokenizedInput):
        tokenized = vllm_engine_pb2.TokenizedInput(
            original_text=input.original_text,
            input_ids=input.input_ids,
        )
    elif isinstance(input, list):
        tokenized = vllm_engine_pb2.TokenizedInput(
            input_ids=input,
        )
    else:
        raise ValueError(f"Unsupported input type: {type(input)}")

    return vllm_engine_pb2.EmbedRequest(
        request_id=request_id,
        tokenized=tokenized,
    )
