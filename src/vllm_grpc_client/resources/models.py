"""
Models resource for vLLM gRPC client.

Handles the GetModelInfo RPC for retrieving model information.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from vllm_grpc_client._exceptions import _exception_from_grpc_error
from vllm_grpc_client._types import ModelInfo
from vllm_grpc_client.proto import vllm_engine_pb2

if TYPE_CHECKING:
    from vllm_grpc_client._client import AsyncVLLMGrpcClient, VLLMGrpcClient


class Models:
    """
    Synchronous models resource.

    Provides methods for retrieving model information from the vLLM gRPC server.
    """

    def __init__(self, client: "VLLMGrpcClient"):
        """
        Initialize the models resource.

        Args:
            client: The parent VLLMGrpcClient instance.
        """
        self._client = client

    def retrieve(self, timeout: Optional[float] = None) -> ModelInfo:
        """
        Retrieve information about the loaded model.

        Args:
            timeout: Request timeout in seconds.

        Returns:
            ModelInfo containing model metadata.
        """
        try:
            response = self._client._stub.GetModelInfo(
                vllm_engine_pb2.GetModelInfoRequest(),
                timeout=timeout or self._client._timeout,
            )

            return ModelInfo(
                model_path=response.model_path,
                is_generation=response.is_generation,
                max_context_length=response.max_context_length,
                vocab_size=response.vocab_size,
                supports_vision=response.supports_vision,
            )

        except Exception as e:
            import grpc

            if isinstance(e, grpc.RpcError):
                raise _exception_from_grpc_error(e) from e
            raise

    def list(self, timeout: Optional[float] = None) -> ModelInfo:
        """
        List model information (alias for retrieve).

        vLLM gRPC server only supports a single loaded model, so this
        method is equivalent to retrieve().

        Args:
            timeout: Request timeout in seconds.

        Returns:
            ModelInfo containing model metadata.
        """
        return self.retrieve(timeout=timeout)


class AsyncModels:
    """
    Asynchronous models resource.

    Provides async methods for retrieving model information from the
    vLLM gRPC server.
    """

    def __init__(self, client: "AsyncVLLMGrpcClient"):
        """
        Initialize the async models resource.

        Args:
            client: The parent AsyncVLLMGrpcClient instance.
        """
        self._client = client

    async def retrieve(self, timeout: Optional[float] = None) -> ModelInfo:
        """
        Retrieve information about the loaded model asynchronously.

        Args:
            timeout: Request timeout in seconds.

        Returns:
            ModelInfo containing model metadata.
        """
        try:
            response = await self._client._stub.GetModelInfo(
                vllm_engine_pb2.GetModelInfoRequest(),
                timeout=timeout or self._client._timeout,
            )

            return ModelInfo(
                model_path=response.model_path,
                is_generation=response.is_generation,
                max_context_length=response.max_context_length,
                vocab_size=response.vocab_size,
                supports_vision=response.supports_vision,
            )

        except Exception as e:
            import grpc

            if isinstance(e, grpc.RpcError):
                raise _exception_from_grpc_error(e) from e
            raise

    async def list(self, timeout: Optional[float] = None) -> ModelInfo:
        """
        List model information asynchronously (alias for retrieve).

        Args:
            timeout: Request timeout in seconds.

        Returns:
            ModelInfo containing model metadata.
        """
        return await self.retrieve(timeout=timeout)
