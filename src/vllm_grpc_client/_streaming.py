"""
Streaming response handlers for vLLM gRPC client.

Provides both sync and async iterators for streaming generation responses,
similar to OpenAI's Stream and AsyncStream classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator, Iterator, List, Optional, Union

from vllm_grpc_client._exceptions import _exception_from_grpc_error
from vllm_grpc_client._types import Completion, CompletionChunk, CompletionChunkChoice, CompletionUsage

if TYPE_CHECKING:
    import grpc

    from vllm_grpc_client.proto import vllm_engine_pb2


class GenerateStream:
    """
    Synchronous streaming iterator for generation responses.

    Yields CompletionChunk objects for each streaming response, and provides
    access to the final Completion when streaming is complete.

    Usage:
        stream = client.completions.create(prompt="Hello", stream=True)
        for chunk in stream:
            print(chunk.choices[0].delta_token_ids)
        # Access final completion
        final = stream.get_final_completion()
    """

    def __init__(
        self,
        response_iterator: Iterator["vllm_engine_pb2.GenerateResponse"],
        request_id: str,
        model: str = "",
    ):
        """
        Initialize the streaming iterator.

        Args:
            response_iterator: The gRPC response iterator.
            request_id: The request ID for this generation.
            model: The model name (for response metadata).
        """
        self._response_iterator = response_iterator
        self._request_id = request_id
        self._model = model
        self._final_completion: Optional[Completion] = None
        self._accumulated_token_ids: List[int] = []
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._cached_tokens: int = 0

    def __iter__(self) -> Iterator[CompletionChunk]:
        return self

    def __next__(self) -> CompletionChunk:
        try:
            response = next(self._response_iterator)
            return self._process_response(response)
        except StopIteration:
            raise
        except Exception as e:
            import grpc

            if isinstance(e, grpc.RpcError):
                raise _exception_from_grpc_error(e) from e
            raise

    def _process_response(
        self, response: "vllm_engine_pb2.GenerateResponse"
    ) -> CompletionChunk:
        """Process a single gRPC response message."""
        if response.HasField("chunk"):
            chunk = response.chunk
            token_ids = list(chunk.token_ids)
            self._accumulated_token_ids.extend(token_ids)
            self._total_prompt_tokens = chunk.prompt_tokens
            self._total_completion_tokens += len(token_ids)
            self._cached_tokens = chunk.cached_tokens

            return CompletionChunk(
                id=self._request_id,
                model=self._model,
                choices=[
                    CompletionChunkChoice(
                        index=0,
                        delta_token_ids=token_ids,
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=chunk.prompt_tokens,
                    completion_tokens=len(token_ids),
                    total_tokens=chunk.prompt_tokens + len(token_ids),
                    cached_tokens=chunk.cached_tokens,
                ),
            )
        elif response.HasField("complete"):
            complete = response.complete
            # Store final completion
            from vllm_grpc_client._types import CompletionChoice

            self._final_completion = Completion(
                id=self._request_id,
                model=self._model,
                choices=[
                    CompletionChoice(
                        index=0,
                        token_ids=self._accumulated_token_ids + list(complete.output_ids),
                        finish_reason=complete.finish_reason or "stop",
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=complete.prompt_tokens or self._total_prompt_tokens,
                    completion_tokens=self._total_completion_tokens + len(complete.output_ids),
                    total_tokens=(
                        (complete.prompt_tokens or self._total_prompt_tokens)
                        + self._total_completion_tokens
                        + len(complete.output_ids)
                    ),
                    cached_tokens=complete.cached_tokens or self._cached_tokens,
                ),
            )

            # Return final chunk with finish_reason
            return CompletionChunk(
                id=self._request_id,
                model=self._model,
                choices=[
                    CompletionChunkChoice(
                        index=0,
                        delta_token_ids=list(complete.output_ids),
                        finish_reason=complete.finish_reason or "stop",
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=complete.prompt_tokens,
                    completion_tokens=len(complete.output_ids),
                    total_tokens=complete.prompt_tokens + len(complete.output_ids),
                    cached_tokens=complete.cached_tokens,
                ),
            )
        else:
            # Unknown response type, return empty chunk
            return CompletionChunk(
                id=self._request_id,
                model=self._model,
            )

    def get_final_completion(self) -> Optional[Completion]:
        """
        Get the final completion after streaming is complete.

        Returns None if streaming has not finished yet or if no complete
        response was received.
        """
        return self._final_completion


class AsyncGenerateStream:
    """
    Asynchronous streaming iterator for generation responses.

    Yields CompletionChunk objects for each streaming response, and provides
    access to the final Completion when streaming is complete.

    Usage:
        stream = await client.completions.create(prompt="Hello", stream=True)
        async for chunk in stream:
            print(chunk.choices[0].delta_token_ids)
        # Access final completion
        final = stream.get_final_completion()
    """

    def __init__(
        self,
        response_iterator: AsyncIterator["vllm_engine_pb2.GenerateResponse"],
        request_id: str,
        model: str = "",
    ):
        """
        Initialize the async streaming iterator.

        Args:
            response_iterator: The gRPC async response iterator.
            request_id: The request ID for this generation.
            model: The model name (for response metadata).
        """
        self._response_iterator = response_iterator
        # Get the actual async iterator from the gRPC call object
        self._aiter: Optional[AsyncIterator] = None
        self._request_id = request_id
        self._model = model
        self._final_completion: Optional[Completion] = None
        self._accumulated_token_ids: List[int] = []
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._cached_tokens: int = 0

    def __aiter__(self) -> AsyncIterator[CompletionChunk]:
        return self

    async def __anext__(self) -> CompletionChunk:
        try:
            # Initialize the async iterator if not already done
            if self._aiter is None:
                self._aiter = self._response_iterator.__aiter__()
            response = await self._aiter.__anext__()
            return self._process_response(response)
        except StopAsyncIteration:
            raise
        except Exception as e:
            import grpc

            if isinstance(e, grpc.RpcError):
                raise _exception_from_grpc_error(e) from e
            raise

    def _process_response(
        self, response: "vllm_engine_pb2.GenerateResponse"
    ) -> CompletionChunk:
        """Process a single gRPC response message."""
        if response.HasField("chunk"):
            chunk = response.chunk
            token_ids = list(chunk.token_ids)
            self._accumulated_token_ids.extend(token_ids)
            self._total_prompt_tokens = chunk.prompt_tokens
            self._total_completion_tokens += len(token_ids)
            self._cached_tokens = chunk.cached_tokens

            return CompletionChunk(
                id=self._request_id,
                model=self._model,
                choices=[
                    CompletionChunkChoice(
                        index=0,
                        delta_token_ids=token_ids,
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=chunk.prompt_tokens,
                    completion_tokens=len(token_ids),
                    total_tokens=chunk.prompt_tokens + len(token_ids),
                    cached_tokens=chunk.cached_tokens,
                ),
            )
        elif response.HasField("complete"):
            complete = response.complete
            # Store final completion
            from vllm_grpc_client._types import CompletionChoice

            self._final_completion = Completion(
                id=self._request_id,
                model=self._model,
                choices=[
                    CompletionChoice(
                        index=0,
                        token_ids=self._accumulated_token_ids + list(complete.output_ids),
                        finish_reason=complete.finish_reason or "stop",
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=complete.prompt_tokens or self._total_prompt_tokens,
                    completion_tokens=self._total_completion_tokens + len(complete.output_ids),
                    total_tokens=(
                        (complete.prompt_tokens or self._total_prompt_tokens)
                        + self._total_completion_tokens
                        + len(complete.output_ids)
                    ),
                    cached_tokens=complete.cached_tokens or self._cached_tokens,
                ),
            )

            # Return final chunk with finish_reason
            return CompletionChunk(
                id=self._request_id,
                model=self._model,
                choices=[
                    CompletionChunkChoice(
                        index=0,
                        delta_token_ids=list(complete.output_ids),
                        finish_reason=complete.finish_reason or "stop",
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=complete.prompt_tokens,
                    completion_tokens=len(complete.output_ids),
                    total_tokens=complete.prompt_tokens + len(complete.output_ids),
                    cached_tokens=complete.cached_tokens,
                ),
            )
        else:
            # Unknown response type, return empty chunk
            return CompletionChunk(
                id=self._request_id,
                model=self._model,
            )

    def get_final_completion(self) -> Optional[Completion]:
        """
        Get the final completion after streaming is complete.

        Returns None if streaming has not finished yet or if no complete
        response was received.
        """
        return self._final_completion


# Type alias for stream return types
StreamType = Union[GenerateStream, Completion]
AsyncStreamType = Union[AsyncGenerateStream, Completion]
