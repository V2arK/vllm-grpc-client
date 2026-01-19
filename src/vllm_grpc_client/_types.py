"""
Type definitions for vLLM gRPC client.

These Pydantic models mirror the protobuf message types but provide
a more Pythonic interface with validation and serialization.
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


# =====================
# Sampling Parameters
# =====================


class ChoiceConstraint(BaseModel):
    """List of allowed choices for structured output."""

    choices: List[str] = Field(default_factory=list)


class StructuredOutputs(BaseModel):
    """
    Structured output constraints for generation.

    Only one of the fields should be set. These map to vLLM's
    StructuredOutputsParams options.
    """

    json_schema: Optional[str] = Field(default=None, description="JSON schema string")
    regex: Optional[str] = Field(default=None, description="Regex pattern")
    grammar: Optional[str] = Field(default=None, description="Grammar/EBNF string")
    structural_tag: Optional[str] = Field(default=None, description="Structural tag")
    json_object: Optional[bool] = Field(default=None, description="Force JSON object output")
    choice: Optional[ChoiceConstraint] = Field(default=None, description="List of allowed choices")


class SamplingParams(BaseModel):
    """
    Sampling parameters for text generation.

    These parameters control how the model generates text, including
    temperature, top-p, top-k sampling, and various penalties.
    """

    temperature: Optional[float] = Field(default=None, ge=0.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, gt=0.0, le=1.0, description="Top-p nucleus sampling")
    top_k: int = Field(default=0, ge=0, description="Top-k sampling (0 to disable)")
    min_p: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum probability threshold")

    frequency_penalty: float = Field(
        default=0.0, ge=-2.0, le=2.0, description="Frequency penalty"
    )
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    repetition_penalty: float = Field(
        default=1.0, ge=0.0, description="Repetition penalty (1.0 = no penalty)"
    )

    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum tokens to generate")
    min_tokens: int = Field(default=0, ge=0, description="Minimum tokens to generate")

    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    stop_token_ids: Optional[List[int]] = Field(default=None, description="Stop token IDs")

    skip_special_tokens: bool = Field(default=True, description="Skip special tokens in output")
    spaces_between_special_tokens: bool = Field(
        default=True, description="Add spaces between special tokens"
    )
    ignore_eos: bool = Field(default=False, description="Ignore EOS token")

    n: int = Field(default=1, ge=1, description="Number of parallel samples")

    logprobs: Optional[int] = Field(
        default=None, ge=-1, description="Number of log probabilities to return"
    )
    prompt_logprobs: Optional[int] = Field(
        default=None, ge=-1, description="Number of prompt log probabilities"
    )

    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    include_stop_str_in_output: bool = Field(
        default=False, description="Include stop strings in output"
    )
    logit_bias: Optional[Dict[int, float]] = Field(
        default=None, description="Token ID to bias mapping"
    )
    truncate_prompt_tokens: Optional[int] = Field(
        default=None, description="Truncate prompt to N tokens"
    )

    structured_outputs: Optional[StructuredOutputs] = Field(
        default=None, description="Structured output constraints"
    )


# =====================
# Completion Types
# =====================


class CompletionUsage(BaseModel):
    """Token usage statistics for a completion."""

    prompt_tokens: int = Field(description="Number of tokens in the prompt")
    completion_tokens: int = Field(description="Number of tokens in the completion")
    total_tokens: int = Field(description="Total tokens (prompt + completion)")
    cached_tokens: int = Field(default=0, description="Number of cached tokens")


class CompletionChoice(BaseModel):
    """A single completion choice."""

    index: int = Field(default=0, description="Index of the choice")
    text: str = Field(default="", description="Generated text")
    token_ids: List[int] = Field(default_factory=list, description="Output token IDs")
    finish_reason: Optional[str] = Field(
        default=None, description="Reason for completion (stop, length, abort)"
    )
    # logprobs will be added when vLLM gRPC implements it


class Completion(BaseModel):
    """
    A complete generation response.

    This is returned for non-streaming requests or as the final
    response in a streaming sequence.
    """

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:24]}")
    object: str = Field(default="text_completion")
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = Field(default="")
    choices: List[CompletionChoice] = Field(default_factory=list)
    usage: Optional[CompletionUsage] = Field(default=None)

    @classmethod
    def from_grpc_complete(
        cls,
        complete: "vllm_engine_pb2.GenerateComplete",  # noqa: F821
        request_id: str,
        model: str = "",
    ) -> "Completion":
        """Create a Completion from a gRPC GenerateComplete message."""
        return cls(
            id=request_id,
            model=model,
            choices=[
                CompletionChoice(
                    index=0,
                    token_ids=list(complete.output_ids),
                    finish_reason=complete.finish_reason or "stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=complete.prompt_tokens,
                completion_tokens=complete.completion_tokens,
                total_tokens=complete.prompt_tokens + complete.completion_tokens,
                cached_tokens=complete.cached_tokens,
            ),
        )


# =====================
# Streaming Types
# =====================


class CompletionChunkChoice(BaseModel):
    """A single choice in a streaming chunk."""

    index: int = Field(default=0)
    delta_token_ids: List[int] = Field(default_factory=list, description="Delta token IDs")
    delta_text: str = Field(default="", description="Delta text (if available)")
    finish_reason: Optional[str] = Field(default=None)


class CompletionChunk(BaseModel):
    """
    A streaming chunk response.

    These are yielded during streaming generation.
    """

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:24]}")
    object: str = Field(default="text_completion.chunk")
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = Field(default="")
    choices: List[CompletionChunkChoice] = Field(default_factory=list)
    usage: Optional[CompletionUsage] = Field(default=None)

    @classmethod
    def from_grpc_chunk(
        cls,
        chunk: "vllm_engine_pb2.GenerateStreamChunk",  # noqa: F821
        request_id: str,
        model: str = "",
    ) -> "CompletionChunk":
        """Create a CompletionChunk from a gRPC GenerateStreamChunk message."""
        return cls(
            id=request_id,
            model=model,
            choices=[
                CompletionChunkChoice(
                    index=0,
                    delta_token_ids=list(chunk.token_ids),
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=chunk.prompt_tokens,
                completion_tokens=chunk.completion_tokens,
                total_tokens=chunk.prompt_tokens + chunk.completion_tokens,
                cached_tokens=chunk.cached_tokens,
            ),
        )


# =====================
# Embedding Types
# =====================


class Embedding(BaseModel):
    """A single embedding result."""

    object: str = Field(default="embedding")
    embedding: List[float] = Field(description="The embedding vector")
    index: int = Field(default=0)


class EmbeddingUsage(BaseModel):
    """Token usage for embedding request."""

    prompt_tokens: int = Field(description="Number of tokens in the input")
    total_tokens: int = Field(description="Total tokens processed")


class EmbeddingResponse(BaseModel):
    """Response from an embedding request."""

    object: str = Field(default="list")
    data: List[Embedding] = Field(default_factory=list)
    model: str = Field(default="")
    usage: Optional[EmbeddingUsage] = Field(default=None)

    @classmethod
    def from_grpc_response(
        cls,
        response: "vllm_engine_pb2.EmbedResponse",  # noqa: F821
        model: str = "",
    ) -> "EmbeddingResponse":
        """Create an EmbeddingResponse from a gRPC EmbedResponse message."""
        return cls(
            model=model,
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


# =====================
# Info Types
# =====================


class HealthCheckResponse(BaseModel):
    """Response from a health check."""

    healthy: bool = Field(description="Whether the server is healthy")
    message: str = Field(default="", description="Health status message")


class ModelInfo(BaseModel):
    """Model information from the server."""

    model_config = {"protected_namespaces": ()}

    model_path: str = Field(description="Path or name of the loaded model")
    is_generation: bool = Field(description="Whether this is a generation model")
    max_context_length: int = Field(description="Maximum context length")
    vocab_size: int = Field(description="Vocabulary size")
    supports_vision: bool = Field(description="Whether model supports vision inputs")


class ServerInfo(BaseModel):
    """Server information and status."""

    active_requests: int = Field(description="Number of active requests")
    is_paused: bool = Field(description="Whether the server is paused")
    last_receive_timestamp: float = Field(description="Timestamp of last received request")
    uptime_seconds: float = Field(description="Server uptime in seconds")
    server_type: str = Field(description="Server type identifier")


# =====================
# Request Types
# =====================


class TokenizedInput(BaseModel):
    """Pre-tokenized input for generation."""

    original_text: str = Field(default="", description="Original text (for debugging)")
    input_ids: List[int] = Field(description="Token IDs to process")


# Type alias for prompt input
PromptInput = Union[str, TokenizedInput, List[int]]
