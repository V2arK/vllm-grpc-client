"""
Custom exceptions for vLLM gRPC client.

These exceptions are mapped from gRPC status codes to provide more
user-friendly error handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import grpc


class VLLMGrpcError(Exception):
    """Base exception for all vLLM gRPC client errors."""

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        details: Optional[str] = None,
    ):
        self.message = message
        self.code = code
        self.details = details
        super().__init__(message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.code:
            parts.append(f"code={self.code}")
        if self.details:
            parts.append(f"details={self.details}")
        return " | ".join(parts)


class VLLMGrpcConnectionError(VLLMGrpcError):
    """Raised when the client cannot connect to the gRPC server."""

    pass


class VLLMGrpcTimeoutError(VLLMGrpcError):
    """Raised when a gRPC call times out."""

    pass


class VLLMGrpcAbortedError(VLLMGrpcError):
    """Raised when a request is aborted by the server or client."""

    pass


class VLLMGrpcInvalidArgumentError(VLLMGrpcError):
    """Raised when the request contains invalid arguments."""

    pass


class VLLMGrpcUnavailableError(VLLMGrpcError):
    """Raised when the server is unavailable."""

    pass


class VLLMGrpcUnimplementedError(VLLMGrpcError):
    """Raised when the requested RPC is not implemented on the server."""

    pass


class VLLMGrpcInternalError(VLLMGrpcError):
    """Raised when an internal server error occurs."""

    pass


class VLLMGrpcCancelledError(VLLMGrpcError):
    """Raised when a request is cancelled."""

    pass


def _exception_from_grpc_error(error: grpc.RpcError) -> VLLMGrpcError:
    """
    Convert a gRPC RpcError to a VLLMGrpcError subclass.

    Maps gRPC status codes to appropriate exception types for better
    error handling.

    Args:
        error: The gRPC RpcError to convert.

    Returns:
        An appropriate VLLMGrpcError subclass instance.
    """
    import grpc

    code = error.code()
    details = error.details()
    code_name = code.name if hasattr(code, "name") else str(code)

    if code == grpc.StatusCode.UNAVAILABLE:
        return VLLMGrpcUnavailableError(
            f"Server unavailable: {details}",
            code=code_name,
            details=details,
        )
    elif code == grpc.StatusCode.DEADLINE_EXCEEDED:
        return VLLMGrpcTimeoutError(
            f"Request timed out: {details}",
            code=code_name,
            details=details,
        )
    elif code == grpc.StatusCode.INVALID_ARGUMENT:
        return VLLMGrpcInvalidArgumentError(
            f"Invalid argument: {details}",
            code=code_name,
            details=details,
        )
    elif code == grpc.StatusCode.ABORTED:
        return VLLMGrpcAbortedError(
            f"Request aborted: {details}",
            code=code_name,
            details=details,
        )
    elif code == grpc.StatusCode.UNIMPLEMENTED:
        return VLLMGrpcUnimplementedError(
            f"RPC not implemented: {details}",
            code=code_name,
            details=details,
        )
    elif code == grpc.StatusCode.INTERNAL:
        return VLLMGrpcInternalError(
            f"Internal server error: {details}",
            code=code_name,
            details=details,
        )
    elif code == grpc.StatusCode.CANCELLED:
        return VLLMGrpcCancelledError(
            f"Request cancelled: {details}",
            code=code_name,
            details=details,
        )
    else:
        return VLLMGrpcError(
            f"gRPC error: {details}",
            code=code_name,
            details=details,
        )
