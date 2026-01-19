"""
Main client classes for vLLM gRPC client.

Provides both synchronous (VLLMGrpcClient) and asynchronous (AsyncVLLMGrpcClient)
client classes with an interface similar to the OpenAI Python client.
"""

from __future__ import annotations

import os
from typing import Optional

import grpc

from vllm_grpc_client._exceptions import VLLMGrpcConnectionError, _exception_from_grpc_error
from vllm_grpc_client.proto import vllm_engine_pb2_grpc


# Default timeout for gRPC calls (in seconds)
DEFAULT_TIMEOUT = 60.0

# Default max message sizes (unlimited like vLLM server)
DEFAULT_MAX_MESSAGE_LENGTH = -1


class VLLMGrpcClient:
    """
    Synchronous vLLM gRPC client.

    Provides a high-level interface for interacting with vLLM's gRPC server,
    with an API similar to the OpenAI Python client.

    Usage:
        client = VLLMGrpcClient(host="localhost", port=9000)
        completion = client.completions.create(
            prompt="Hello, world!",
            max_tokens=100,
        )

        # Or using context manager
        with VLLMGrpcClient(host="localhost", port=9000) as client:
            completion = client.completions.create(prompt="Hello")

    Environment Variables:
        VLLM_GRPC_HOST: Default host if not specified
        VLLM_GRPC_PORT: Default port if not specified
        VLLM_GRPC_TIMEOUT: Default timeout if not specified
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        *,
        timeout: Optional[float] = None,
        max_send_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
        max_receive_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
    ):
        """
        Initialize the vLLM gRPC client.

        Args:
            host: The gRPC server host. Defaults to VLLM_GRPC_HOST env var or "localhost".
            port: The gRPC server port. Defaults to VLLM_GRPC_PORT env var or 9000.
            timeout: Default timeout for RPC calls in seconds.
                Defaults to VLLM_GRPC_TIMEOUT env var or 60.0.
            max_send_message_length: Maximum send message size (-1 for unlimited).
            max_receive_message_length: Maximum receive message size (-1 for unlimited).
        """
        # Resolve configuration from environment or defaults
        self._host = host or os.environ.get("VLLM_GRPC_HOST", "localhost")
        self._port = port or int(os.environ.get("VLLM_GRPC_PORT", "9000"))
        self._timeout = timeout or float(os.environ.get("VLLM_GRPC_TIMEOUT", str(DEFAULT_TIMEOUT)))

        # Build server address
        self._address = f"{self._host}:{self._port}"

        # Create gRPC channel options
        options = [
            ("grpc.max_send_message_length", max_send_message_length),
            ("grpc.max_receive_message_length", max_receive_message_length),
        ]

        # Create the gRPC channel
        self._channel: grpc.Channel = grpc.insecure_channel(self._address, options=options)
        self._stub = vllm_engine_pb2_grpc.VllmEngineStub(self._channel)

        # Model name (populated by retrieve on first access if needed)
        self._model_name: str = ""

        # Initialize resources lazily
        self._completions: Optional["Completions"] = None
        self._embeddings: Optional["Embeddings"] = None
        self._models: Optional["Models"] = None
        self._health: Optional["Health"] = None

    @property
    def completions(self) -> "Completions":
        """Completions resource for text generation."""
        if self._completions is None:
            from vllm_grpc_client.resources.completions import Completions

            self._completions = Completions(self)
        return self._completions

    @property
    def embeddings(self) -> "Embeddings":
        """Embeddings resource (not yet implemented in vLLM gRPC server)."""
        if self._embeddings is None:
            from vllm_grpc_client.resources.embeddings import Embeddings

            self._embeddings = Embeddings(self)
        return self._embeddings

    @property
    def models(self) -> "Models":
        """Models resource for model information."""
        if self._models is None:
            from vllm_grpc_client.resources.models import Models

            self._models = Models(self)
        return self._models

    @property
    def health(self) -> "Health":
        """Health resource for health checks and server info."""
        if self._health is None:
            from vllm_grpc_client.resources.health import Health

            self._health = Health(self)
        return self._health

    def close(self) -> None:
        """Close the gRPC channel."""
        if self._channel is not None:
            self._channel.close()

    def __enter__(self) -> "VLLMGrpcClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def is_healthy(self, timeout: Optional[float] = None) -> bool:
        """
        Check if the server is healthy.

        Args:
            timeout: Request timeout in seconds.

        Returns:
            True if the server is healthy, False otherwise.
        """
        try:
            response = self.health.check(timeout=timeout)
            return response.healthy
        except Exception:
            return False

    def wait_for_ready(
        self,
        timeout: float = 60.0,
        poll_interval: float = 1.0,
    ) -> bool:
        """
        Wait for the server to become ready.

        Args:
            timeout: Maximum time to wait in seconds.
            poll_interval: Time between health check attempts in seconds.

        Returns:
            True if the server became ready, False if timeout was reached.
        """
        import time

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_healthy(timeout=poll_interval):
                # Update model name
                try:
                    model_info = self.models.retrieve(timeout=poll_interval)
                    self._model_name = model_info.model_path
                except Exception:
                    pass
                return True
            time.sleep(poll_interval)
        return False


class AsyncVLLMGrpcClient:
    """
    Asynchronous vLLM gRPC client.

    Provides a high-level async interface for interacting with vLLM's gRPC server,
    with an API similar to the OpenAI Python client.

    Usage:
        async with AsyncVLLMGrpcClient(host="localhost", port=9000) as client:
            completion = await client.completions.create(
                prompt="Hello, world!",
                max_tokens=100,
            )

    Environment Variables:
        VLLM_GRPC_HOST: Default host if not specified
        VLLM_GRPC_PORT: Default port if not specified
        VLLM_GRPC_TIMEOUT: Default timeout if not specified
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        *,
        timeout: Optional[float] = None,
        max_send_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
        max_receive_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
    ):
        """
        Initialize the async vLLM gRPC client.

        Args:
            host: The gRPC server host. Defaults to VLLM_GRPC_HOST env var or "localhost".
            port: The gRPC server port. Defaults to VLLM_GRPC_PORT env var or 9000.
            timeout: Default timeout for RPC calls in seconds.
                Defaults to VLLM_GRPC_TIMEOUT env var or 60.0.
            max_send_message_length: Maximum send message size (-1 for unlimited).
            max_receive_message_length: Maximum receive message size (-1 for unlimited).
        """
        # Resolve configuration from environment or defaults
        self._host = host or os.environ.get("VLLM_GRPC_HOST", "localhost")
        self._port = port or int(os.environ.get("VLLM_GRPC_PORT", "9000"))
        self._timeout = timeout or float(os.environ.get("VLLM_GRPC_TIMEOUT", str(DEFAULT_TIMEOUT)))

        # Build server address
        self._address = f"{self._host}:{self._port}"

        # Create gRPC channel options
        options = [
            ("grpc.max_send_message_length", max_send_message_length),
            ("grpc.max_receive_message_length", max_receive_message_length),
        ]

        # Create the async gRPC channel
        self._channel: grpc.aio.Channel = grpc.aio.insecure_channel(self._address, options=options)
        self._stub = vllm_engine_pb2_grpc.VllmEngineStub(self._channel)

        # Model name (populated by retrieve on first access if needed)
        self._model_name: str = ""

        # Initialize resources lazily
        self._completions: Optional["AsyncCompletions"] = None
        self._embeddings: Optional["AsyncEmbeddings"] = None
        self._models: Optional["AsyncModels"] = None
        self._health: Optional["AsyncHealth"] = None

    @property
    def completions(self) -> "AsyncCompletions":
        """Completions resource for text generation."""
        if self._completions is None:
            from vllm_grpc_client.resources.completions import AsyncCompletions

            self._completions = AsyncCompletions(self)
        return self._completions

    @property
    def embeddings(self) -> "AsyncEmbeddings":
        """Embeddings resource (not yet implemented in vLLM gRPC server)."""
        if self._embeddings is None:
            from vllm_grpc_client.resources.embeddings import AsyncEmbeddings

            self._embeddings = AsyncEmbeddings(self)
        return self._embeddings

    @property
    def models(self) -> "AsyncModels":
        """Models resource for model information."""
        if self._models is None:
            from vllm_grpc_client.resources.models import AsyncModels

            self._models = AsyncModels(self)
        return self._models

    @property
    def health(self) -> "AsyncHealth":
        """Health resource for health checks and server info."""
        if self._health is None:
            from vllm_grpc_client.resources.health import AsyncHealth

            self._health = AsyncHealth(self)
        return self._health

    async def close(self) -> None:
        """Close the gRPC channel."""
        if self._channel is not None:
            await self._channel.close()

    async def __aenter__(self) -> "AsyncVLLMGrpcClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def is_healthy(self, timeout: Optional[float] = None) -> bool:
        """
        Check if the server is healthy.

        Args:
            timeout: Request timeout in seconds.

        Returns:
            True if the server is healthy, False otherwise.
        """
        try:
            response = await self.health.check(timeout=timeout)
            return response.healthy
        except Exception:
            return False

    async def wait_for_ready(
        self,
        timeout: float = 60.0,
        poll_interval: float = 1.0,
    ) -> bool:
        """
        Wait for the server to become ready.

        Args:
            timeout: Maximum time to wait in seconds.
            poll_interval: Time between health check attempts in seconds.

        Returns:
            True if the server became ready, False if timeout was reached.
        """
        import asyncio
        import time

        start_time = time.time()
        while time.time() - start_time < timeout:
            if await self.is_healthy(timeout=poll_interval):
                # Update model name
                try:
                    model_info = await self.models.retrieve(timeout=poll_interval)
                    self._model_name = model_info.model_path
                except Exception:
                    pass
                return True
            await asyncio.sleep(poll_interval)
        return False


# Type hints for lazy imports
if False:  # TYPE_CHECKING equivalent that doesn't cause import issues
    from vllm_grpc_client.resources.completions import AsyncCompletions, Completions
    from vllm_grpc_client.resources.embeddings import AsyncEmbeddings, Embeddings
    from vllm_grpc_client.resources.health import AsyncHealth, Health
    from vllm_grpc_client.resources.models import AsyncModels, Models
