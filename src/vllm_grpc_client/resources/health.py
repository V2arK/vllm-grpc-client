"""
Health resource for vLLM gRPC client.

Handles HealthCheck, GetServerInfo, and Abort RPCs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from vllm_grpc_client._exceptions import _exception_from_grpc_error
from vllm_grpc_client._types import HealthCheckResponse, ServerInfo
from vllm_grpc_client.proto import vllm_engine_pb2

if TYPE_CHECKING:
    from vllm_grpc_client._client import AsyncVLLMGrpcClient, VLLMGrpcClient


class Health:
    """
    Synchronous health resource.

    Provides methods for health checking, server info, and request abortion.
    """

    def __init__(self, client: "VLLMGrpcClient"):
        """
        Initialize the health resource.

        Args:
            client: The parent VLLMGrpcClient instance.
        """
        self._client = client

    def check(self, timeout: Optional[float] = None) -> HealthCheckResponse:
        """
        Perform a health check on the server.

        Args:
            timeout: Request timeout in seconds.

        Returns:
            HealthCheckResponse indicating server health status.
        """
        try:
            response = self._client._stub.HealthCheck(
                vllm_engine_pb2.HealthCheckRequest(),
                timeout=timeout or self._client._timeout,
            )

            return HealthCheckResponse(
                healthy=response.healthy,
                message=response.message,
            )

        except Exception as e:
            import grpc

            if isinstance(e, grpc.RpcError):
                raise _exception_from_grpc_error(e) from e
            raise

    def server_info(self, timeout: Optional[float] = None) -> ServerInfo:
        """
        Get server information and status.

        Args:
            timeout: Request timeout in seconds.

        Returns:
            ServerInfo containing server status and statistics.
        """
        try:
            response = self._client._stub.GetServerInfo(
                vllm_engine_pb2.GetServerInfoRequest(),
                timeout=timeout or self._client._timeout,
            )

            return ServerInfo(
                active_requests=response.active_requests,
                is_paused=response.is_paused,
                last_receive_timestamp=response.last_receive_timestamp,
                uptime_seconds=response.uptime_seconds,
                server_type=response.server_type,
            )

        except Exception as e:
            import grpc

            if isinstance(e, grpc.RpcError):
                raise _exception_from_grpc_error(e) from e
            raise

    def abort(
        self,
        request_ids: List[str],
        timeout: Optional[float] = None,
    ) -> None:
        """
        Abort one or more running requests.

        This is an out-of-band cancellation mechanism. The requests
        will receive a completion with finish_reason="abort".

        Args:
            request_ids: List of request IDs to abort.
            timeout: Request timeout in seconds.
        """
        try:
            self._client._stub.Abort(
                vllm_engine_pb2.AbortRequest(request_ids=request_ids),
                timeout=timeout or self._client._timeout,
            )

        except Exception as e:
            import grpc

            if isinstance(e, grpc.RpcError):
                raise _exception_from_grpc_error(e) from e
            raise


class AsyncHealth:
    """
    Asynchronous health resource.

    Provides async methods for health checking, server info, and request abortion.
    """

    def __init__(self, client: "AsyncVLLMGrpcClient"):
        """
        Initialize the async health resource.

        Args:
            client: The parent AsyncVLLMGrpcClient instance.
        """
        self._client = client

    async def check(self, timeout: Optional[float] = None) -> HealthCheckResponse:
        """
        Perform a health check on the server asynchronously.

        Args:
            timeout: Request timeout in seconds.

        Returns:
            HealthCheckResponse indicating server health status.
        """
        try:
            response = await self._client._stub.HealthCheck(
                vllm_engine_pb2.HealthCheckRequest(),
                timeout=timeout or self._client._timeout,
            )

            return HealthCheckResponse(
                healthy=response.healthy,
                message=response.message,
            )

        except Exception as e:
            import grpc

            if isinstance(e, grpc.RpcError):
                raise _exception_from_grpc_error(e) from e
            raise

    async def server_info(self, timeout: Optional[float] = None) -> ServerInfo:
        """
        Get server information and status asynchronously.

        Args:
            timeout: Request timeout in seconds.

        Returns:
            ServerInfo containing server status and statistics.
        """
        try:
            response = await self._client._stub.GetServerInfo(
                vllm_engine_pb2.GetServerInfoRequest(),
                timeout=timeout or self._client._timeout,
            )

            return ServerInfo(
                active_requests=response.active_requests,
                is_paused=response.is_paused,
                last_receive_timestamp=response.last_receive_timestamp,
                uptime_seconds=response.uptime_seconds,
                server_type=response.server_type,
            )

        except Exception as e:
            import grpc

            if isinstance(e, grpc.RpcError):
                raise _exception_from_grpc_error(e) from e
            raise

    async def abort(
        self,
        request_ids: List[str],
        timeout: Optional[float] = None,
    ) -> None:
        """
        Abort one or more running requests asynchronously.

        Args:
            request_ids: List of request IDs to abort.
            timeout: Request timeout in seconds.
        """
        try:
            await self._client._stub.Abort(
                vllm_engine_pb2.AbortRequest(request_ids=request_ids),
                timeout=timeout or self._client._timeout,
            )

        except Exception as e:
            import grpc

            if isinstance(e, grpc.RpcError):
                raise _exception_from_grpc_error(e) from e
            raise
