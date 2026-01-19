"""
Completions resource for vLLM gRPC client.

Handles the Generate RPC for text generation.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Dict, List, Optional, Union, overload

from typing_extensions import Literal

from vllm_grpc_client._exceptions import _exception_from_grpc_error
from vllm_grpc_client._streaming import AsyncGenerateStream, GenerateStream
from vllm_grpc_client._types import (
    Completion,
    CompletionChoice,
    CompletionUsage,
    PromptInput,
    SamplingParams,
    StructuredOutputs,
    TokenizedInput,
)
from vllm_grpc_client.proto import vllm_engine_pb2

if TYPE_CHECKING:
    import grpc

    from vllm_grpc_client._client import AsyncVLLMGrpcClient, VLLMGrpcClient


class Completions:
    """
    Synchronous completions resource.

    Provides methods for text generation using the vLLM gRPC server's
    Generate RPC.

    Usage:
        completion = client.completions.create(
            prompt="Hello, world!",
            max_tokens=100,
            temperature=0.7,
        )
    """

    def __init__(self, client: "VLLMGrpcClient"):
        """
        Initialize the completions resource.

        Args:
            client: The parent VLLMGrpcClient instance.
        """
        self._client = client

    @overload
    def create(
        self,
        *,
        prompt: PromptInput,
        stream: Literal[False] = False,
        request_id: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        max_tokens: Optional[int] = None,
        min_tokens: int = 0,
        stop: Optional[List[str]] = None,
        stop_token_ids: Optional[List[int]] = None,
        skip_special_tokens: bool = True,
        ignore_eos: bool = False,
        n: int = 1,
        logprobs: Optional[int] = None,
        prompt_logprobs: Optional[int] = None,
        seed: Optional[int] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        structured_outputs: Optional[StructuredOutputs] = None,
        timeout: Optional[float] = None,
    ) -> Completion:
        """Non-streaming completion."""
        ...

    @overload
    def create(
        self,
        *,
        prompt: PromptInput,
        stream: Literal[True],
        request_id: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        max_tokens: Optional[int] = None,
        min_tokens: int = 0,
        stop: Optional[List[str]] = None,
        stop_token_ids: Optional[List[int]] = None,
        skip_special_tokens: bool = True,
        ignore_eos: bool = False,
        n: int = 1,
        logprobs: Optional[int] = None,
        prompt_logprobs: Optional[int] = None,
        seed: Optional[int] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        structured_outputs: Optional[StructuredOutputs] = None,
        timeout: Optional[float] = None,
    ) -> GenerateStream:
        """Streaming completion."""
        ...

    def create(
        self,
        *,
        prompt: PromptInput,
        stream: bool = False,
        request_id: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        max_tokens: Optional[int] = None,
        min_tokens: int = 0,
        stop: Optional[List[str]] = None,
        stop_token_ids: Optional[List[int]] = None,
        skip_special_tokens: bool = True,
        ignore_eos: bool = False,
        n: int = 1,
        logprobs: Optional[int] = None,
        prompt_logprobs: Optional[int] = None,
        seed: Optional[int] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        structured_outputs: Optional[StructuredOutputs] = None,
        timeout: Optional[float] = None,
    ) -> Union[Completion, GenerateStream]:
        """
        Create a completion for the given prompt.

        Args:
            prompt: The prompt to generate completion for. Can be a string,
                list of token IDs, or a TokenizedInput object.
            stream: Whether to stream the response.
            request_id: Optional unique request ID. Auto-generated if not provided.
            temperature: Sampling temperature (0.0-2.0). None uses server default.
            top_p: Top-p nucleus sampling (0.0-1.0).
            top_k: Top-k sampling (0 to disable).
            min_p: Minimum probability threshold.
            frequency_penalty: Frequency penalty (-2.0 to 2.0).
            presence_penalty: Presence penalty (-2.0 to 2.0).
            repetition_penalty: Repetition penalty (1.0 = no penalty).
            max_tokens: Maximum tokens to generate. None for server default.
            min_tokens: Minimum tokens to generate.
            stop: List of stop sequences.
            stop_token_ids: List of stop token IDs.
            skip_special_tokens: Whether to skip special tokens in output.
            ignore_eos: Whether to ignore EOS token.
            n: Number of completions to generate.
            logprobs: Number of logprobs to return (not yet implemented in vLLM gRPC).
            prompt_logprobs: Number of prompt logprobs (not yet implemented).
            seed: Random seed for reproducibility.
            logit_bias: Token ID to bias mapping.
            structured_outputs: Structured output constraints.
            timeout: Request timeout in seconds.

        Returns:
            A Completion object if stream=False, otherwise a GenerateStream iterator.
        """
        if request_id is None:
            request_id = f"cmpl-{uuid.uuid4().hex[:24]}"

        # Build the gRPC request
        grpc_request = self._build_generate_request(
            prompt=prompt,
            request_id=request_id,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            stop=stop,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=skip_special_tokens,
            ignore_eos=ignore_eos,
            n=n,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            seed=seed,
            logit_bias=logit_bias,
            structured_outputs=structured_outputs,
        )

        try:
            response_iterator = self._client._stub.Generate(
                grpc_request,
                timeout=timeout or self._client._timeout,
            )

            if stream:
                return GenerateStream(
                    response_iterator=response_iterator,
                    request_id=request_id,
                    model=self._client._model_name,
                )
            else:
                # Collect all responses and return final completion
                final_response = None
                for response in response_iterator:
                    if response.HasField("complete"):
                        final_response = response.complete

                if final_response is None:
                    # No complete response received
                    return Completion(id=request_id, model=self._client._model_name)

                return Completion(
                    id=request_id,
                    model=self._client._model_name,
                    choices=[
                        CompletionChoice(
                            index=0,
                            token_ids=list(final_response.output_ids),
                            finish_reason=final_response.finish_reason or "stop",
                        )
                    ],
                    usage=CompletionUsage(
                        prompt_tokens=final_response.prompt_tokens,
                        completion_tokens=final_response.completion_tokens,
                        total_tokens=(
                            final_response.prompt_tokens + final_response.completion_tokens
                        ),
                        cached_tokens=final_response.cached_tokens,
                    ),
                )

        except Exception as e:
            import grpc

            if isinstance(e, grpc.RpcError):
                raise _exception_from_grpc_error(e) from e
            raise

    def _build_generate_request(
        self,
        prompt: PromptInput,
        request_id: str,
        stream: bool,
        temperature: Optional[float],
        top_p: float,
        top_k: int,
        min_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        repetition_penalty: float,
        max_tokens: Optional[int],
        min_tokens: int,
        stop: Optional[List[str]],
        stop_token_ids: Optional[List[int]],
        skip_special_tokens: bool,
        ignore_eos: bool,
        n: int,
        logprobs: Optional[int],
        prompt_logprobs: Optional[int],
        seed: Optional[int],
        logit_bias: Optional[Dict[int, float]],
        structured_outputs: Optional[StructuredOutputs],
    ) -> vllm_engine_pb2.GenerateRequest:
        """Build a gRPC GenerateRequest from parameters."""
        # Build sampling params
        sampling_params = vllm_engine_pb2.SamplingParams(
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            min_tokens=min_tokens,
            skip_special_tokens=skip_special_tokens,
            ignore_eos=ignore_eos,
            n=n,
        )

        if temperature is not None:
            sampling_params.temperature = temperature

        if max_tokens is not None:
            sampling_params.max_tokens = max_tokens

        if stop:
            sampling_params.stop.extend(stop)

        if stop_token_ids:
            sampling_params.stop_token_ids.extend(stop_token_ids)

        if logprobs is not None:
            sampling_params.logprobs = logprobs

        if prompt_logprobs is not None:
            sampling_params.prompt_logprobs = prompt_logprobs

        if seed is not None:
            sampling_params.seed = seed

        if logit_bias:
            for token_id, bias in logit_bias.items():
                sampling_params.logit_bias[token_id] = bias

        # Handle structured outputs
        if structured_outputs:
            if structured_outputs.json_schema:
                sampling_params.json_schema = structured_outputs.json_schema
            elif structured_outputs.regex:
                sampling_params.regex = structured_outputs.regex
            elif structured_outputs.grammar:
                sampling_params.grammar = structured_outputs.grammar
            elif structured_outputs.structural_tag:
                sampling_params.structural_tag = structured_outputs.structural_tag
            elif structured_outputs.json_object:
                sampling_params.json_object = structured_outputs.json_object
            elif structured_outputs.choice:
                sampling_params.choice.CopyFrom(
                    vllm_engine_pb2.ChoiceConstraint(
                        choices=structured_outputs.choice.choices
                    )
                )

        # Build the request
        request = vllm_engine_pb2.GenerateRequest(
            request_id=request_id,
            sampling_params=sampling_params,
            stream=stream,
        )

        # Set prompt input
        if isinstance(prompt, str):
            request.text = prompt
        elif isinstance(prompt, TokenizedInput):
            request.tokenized.CopyFrom(
                vllm_engine_pb2.TokenizedInput(
                    original_text=prompt.original_text,
                    input_ids=prompt.input_ids,
                )
            )
        elif isinstance(prompt, list):
            # List of token IDs
            request.tokenized.CopyFrom(
                vllm_engine_pb2.TokenizedInput(
                    input_ids=prompt,
                )
            )

        return request


class AsyncCompletions:
    """
    Asynchronous completions resource.

    Provides async methods for text generation using the vLLM gRPC server's
    Generate RPC.

    Usage:
        completion = await client.completions.create(
            prompt="Hello, world!",
            max_tokens=100,
            temperature=0.7,
        )
    """

    def __init__(self, client: "AsyncVLLMGrpcClient"):
        """
        Initialize the async completions resource.

        Args:
            client: The parent AsyncVLLMGrpcClient instance.
        """
        self._client = client

    @overload
    async def create(
        self,
        *,
        prompt: PromptInput,
        stream: Literal[False] = False,
        request_id: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        max_tokens: Optional[int] = None,
        min_tokens: int = 0,
        stop: Optional[List[str]] = None,
        stop_token_ids: Optional[List[int]] = None,
        skip_special_tokens: bool = True,
        ignore_eos: bool = False,
        n: int = 1,
        logprobs: Optional[int] = None,
        prompt_logprobs: Optional[int] = None,
        seed: Optional[int] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        structured_outputs: Optional[StructuredOutputs] = None,
        timeout: Optional[float] = None,
    ) -> Completion:
        """Non-streaming async completion."""
        ...

    @overload
    async def create(
        self,
        *,
        prompt: PromptInput,
        stream: Literal[True],
        request_id: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        max_tokens: Optional[int] = None,
        min_tokens: int = 0,
        stop: Optional[List[str]] = None,
        stop_token_ids: Optional[List[int]] = None,
        skip_special_tokens: bool = True,
        ignore_eos: bool = False,
        n: int = 1,
        logprobs: Optional[int] = None,
        prompt_logprobs: Optional[int] = None,
        seed: Optional[int] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        structured_outputs: Optional[StructuredOutputs] = None,
        timeout: Optional[float] = None,
    ) -> AsyncGenerateStream:
        """Streaming async completion."""
        ...

    async def create(
        self,
        *,
        prompt: PromptInput,
        stream: bool = False,
        request_id: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        max_tokens: Optional[int] = None,
        min_tokens: int = 0,
        stop: Optional[List[str]] = None,
        stop_token_ids: Optional[List[int]] = None,
        skip_special_tokens: bool = True,
        ignore_eos: bool = False,
        n: int = 1,
        logprobs: Optional[int] = None,
        prompt_logprobs: Optional[int] = None,
        seed: Optional[int] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        structured_outputs: Optional[StructuredOutputs] = None,
        timeout: Optional[float] = None,
    ) -> Union[Completion, AsyncGenerateStream]:
        """
        Create a completion for the given prompt asynchronously.

        See Completions.create for detailed parameter documentation.
        """
        if request_id is None:
            request_id = f"cmpl-{uuid.uuid4().hex[:24]}"

        # Build the gRPC request (reuse the sync method's helper)
        grpc_request = _build_generate_request(
            prompt=prompt,
            request_id=request_id,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            stop=stop,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=skip_special_tokens,
            ignore_eos=ignore_eos,
            n=n,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            seed=seed,
            logit_bias=logit_bias,
            structured_outputs=structured_outputs,
        )

        try:
            response_iterator = self._client._stub.Generate(
                grpc_request,
                timeout=timeout or self._client._timeout,
            )

            if stream:
                return AsyncGenerateStream(
                    response_iterator=response_iterator,
                    request_id=request_id,
                    model=self._client._model_name,
                )
            else:
                # Collect all responses and return final completion
                final_response = None
                async for response in response_iterator:
                    if response.HasField("complete"):
                        final_response = response.complete

                if final_response is None:
                    # No complete response received
                    return Completion(id=request_id, model=self._client._model_name)

                return Completion(
                    id=request_id,
                    model=self._client._model_name,
                    choices=[
                        CompletionChoice(
                            index=0,
                            token_ids=list(final_response.output_ids),
                            finish_reason=final_response.finish_reason or "stop",
                        )
                    ],
                    usage=CompletionUsage(
                        prompt_tokens=final_response.prompt_tokens,
                        completion_tokens=final_response.completion_tokens,
                        total_tokens=(
                            final_response.prompt_tokens + final_response.completion_tokens
                        ),
                        cached_tokens=final_response.cached_tokens,
                    ),
                )

        except Exception as e:
            import grpc

            if isinstance(e, grpc.RpcError):
                raise _exception_from_grpc_error(e) from e
            raise


# Helper function to build generate request (shared by sync and async)
def _build_generate_request(
    prompt: PromptInput,
    request_id: str,
    stream: bool,
    temperature: Optional[float],
    top_p: float,
    top_k: int,
    min_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    repetition_penalty: float,
    max_tokens: Optional[int],
    min_tokens: int,
    stop: Optional[List[str]],
    stop_token_ids: Optional[List[int]],
    skip_special_tokens: bool,
    ignore_eos: bool,
    n: int,
    logprobs: Optional[int],
    prompt_logprobs: Optional[int],
    seed: Optional[int],
    logit_bias: Optional[Dict[int, float]],
    structured_outputs: Optional[StructuredOutputs],
) -> vllm_engine_pb2.GenerateRequest:
    """Build a gRPC GenerateRequest from parameters."""
    # Build sampling params
    sampling_params = vllm_engine_pb2.SamplingParams(
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        min_tokens=min_tokens,
        skip_special_tokens=skip_special_tokens,
        ignore_eos=ignore_eos,
        n=n,
    )

    if temperature is not None:
        sampling_params.temperature = temperature

    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens

    if stop:
        sampling_params.stop.extend(stop)

    if stop_token_ids:
        sampling_params.stop_token_ids.extend(stop_token_ids)

    if logprobs is not None:
        sampling_params.logprobs = logprobs

    if prompt_logprobs is not None:
        sampling_params.prompt_logprobs = prompt_logprobs

    if seed is not None:
        sampling_params.seed = seed

    if logit_bias:
        for token_id, bias in logit_bias.items():
            sampling_params.logit_bias[token_id] = bias

    # Handle structured outputs
    if structured_outputs:
        if structured_outputs.json_schema:
            sampling_params.json_schema = structured_outputs.json_schema
        elif structured_outputs.regex:
            sampling_params.regex = structured_outputs.regex
        elif structured_outputs.grammar:
            sampling_params.grammar = structured_outputs.grammar
        elif structured_outputs.structural_tag:
            sampling_params.structural_tag = structured_outputs.structural_tag
        elif structured_outputs.json_object:
            sampling_params.json_object = structured_outputs.json_object
        elif structured_outputs.choice:
            sampling_params.choice.CopyFrom(
                vllm_engine_pb2.ChoiceConstraint(choices=structured_outputs.choice.choices)
            )

    # Build the request
    request = vllm_engine_pb2.GenerateRequest(
        request_id=request_id,
        sampling_params=sampling_params,
        stream=stream,
    )

    # Set prompt input
    if isinstance(prompt, str):
        request.text = prompt
    elif isinstance(prompt, TokenizedInput):
        request.tokenized.CopyFrom(
            vllm_engine_pb2.TokenizedInput(
                original_text=prompt.original_text,
                input_ids=prompt.input_ids,
            )
        )
    elif isinstance(prompt, list):
        # List of token IDs
        request.tokenized.CopyFrom(
            vllm_engine_pb2.TokenizedInput(
                input_ids=prompt,
            )
        )

    return request
