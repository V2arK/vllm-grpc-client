"""
Tests for vLLM gRPC client against a live server.

Run with:
    pytest tests/test_client.py -v

Or to run against a specific server:
    VLLM_GRPC_HOST=your-server-ip VLLM_GRPC_PORT=9000 pytest tests/test_client.py -v
"""

import asyncio
import os

import pytest

from vllm_grpc_client import (
    AsyncVLLMGrpcClient,
    VLLMGrpcClient,
    VLLMGrpcError,
)

# Server configuration
GRPC_HOST = os.environ.get("VLLM_GRPC_HOST", "localhost")
GRPC_PORT = int(os.environ.get("VLLM_GRPC_PORT", "9000"))


class TestSyncClient:
    """Tests for synchronous VLLMGrpcClient."""

    @pytest.fixture
    def client(self):
        """Create a sync client for testing."""
        client = VLLMGrpcClient(host=GRPC_HOST, port=GRPC_PORT)
        yield client
        client.close()

    def test_health_check(self, client):
        """Test basic health check."""
        response = client.health.check()
        assert response.healthy is True
        assert response.message == "Health"
        print(f"Health check passed: {response}")

    def test_get_model_info(self, client):
        """Test retrieving model information."""
        model_info = client.models.retrieve()
        assert model_info.model_path != ""
        assert model_info.max_context_length > 0
        assert model_info.vocab_size > 0
        print(f"Model info: {model_info}")

    def test_get_server_info(self, client):
        """Test retrieving server information."""
        server_info = client.health.server_info()
        assert server_info.server_type == "vllm-grpc"
        assert server_info.uptime_seconds >= 0
        print(f"Server info: {server_info}")

    def test_non_streaming_completion(self, client):
        """Test non-streaming text completion."""
        completion = client.completions.create(
            prompt="The capital of France is",
            max_tokens=10,
            temperature=0.0,
        )
        assert completion.id is not None
        assert len(completion.choices) > 0
        assert len(completion.choices[0].token_ids) > 0
        assert completion.usage is not None
        assert completion.usage.prompt_tokens > 0
        assert completion.usage.completion_tokens > 0
        print(f"Completion: {completion}")

    def test_streaming_completion(self, client):
        """Test streaming text completion."""
        stream = client.completions.create(
            prompt="Count from 1 to 5:",
            max_tokens=30,
            temperature=0.0,
            stream=True,
        )

        chunks = []
        for chunk in stream:
            chunks.append(chunk)
            print(f"Chunk: {chunk.choices[0].delta_token_ids}")

        assert len(chunks) > 0
        # Last chunk should have finish_reason
        assert chunks[-1].choices[0].finish_reason in ["stop", "length"]

        # Check final completion
        final = stream.get_final_completion()
        assert final is not None
        print(f"Final completion: {final}")

    def test_completion_with_stop_strings(self, client):
        """Test completion with stop strings."""
        completion = client.completions.create(
            prompt="Say hello, then stop.",
            max_tokens=50,
            temperature=0.0,
            stop=["\n", ".", "!"],
        )
        assert completion.choices[0].finish_reason in ["stop", "length"]
        print(f"Completion with stop: {completion}")

    def test_completion_with_seed(self, client):
        """Test deterministic completion with seed."""
        params = dict(
            prompt="The meaning of life is",
            max_tokens=20,
            temperature=1.0,
            seed=42,
        )

        completion1 = client.completions.create(**params)
        completion2 = client.completions.create(**params)

        # Same seed should produce same output
        assert completion1.choices[0].token_ids == completion2.choices[0].token_ids
        print(f"Deterministic completion: {completion1.choices[0].token_ids}")


class TestAsyncClient:
    """Tests for asynchronous AsyncVLLMGrpcClient."""

    @pytest.fixture
    async def client(self):
        """Create an async client for testing."""
        client = AsyncVLLMGrpcClient(host=GRPC_HOST, port=GRPC_PORT)
        yield client
        await client.close()

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test async health check."""
        response = await client.health.check()
        assert response.healthy is True
        print(f"Async health check passed: {response}")

    @pytest.mark.asyncio
    async def test_get_model_info(self, client):
        """Test async model info retrieval."""
        model_info = await client.models.retrieve()
        assert model_info.model_path != ""
        print(f"Async model info: {model_info}")

    @pytest.mark.asyncio
    async def test_non_streaming_completion(self, client):
        """Test async non-streaming completion."""
        completion = await client.completions.create(
            prompt="Hello, how are you?",
            max_tokens=20,
            temperature=0.7,
        )
        assert len(completion.choices) > 0
        print(f"Async completion: {completion}")

    @pytest.mark.asyncio
    async def test_streaming_completion(self, client):
        """Test async streaming completion."""
        stream = await client.completions.create(
            prompt="List three colors:",
            max_tokens=30,
            temperature=0.0,
            stream=True,
        )

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
            print(f"Async chunk: {chunk.choices[0].delta_token_ids}")

        assert len(chunks) > 0
        print(f"Async streaming complete, {len(chunks)} chunks received")

    @pytest.mark.asyncio
    async def test_concurrent_completions(self, client):
        """Test concurrent async completions."""
        prompts = [
            "The sun is",
            "Water is",
            "Fire is",
        ]

        async def create_completion(prompt):
            return await client.completions.create(
                prompt=prompt,
                max_tokens=10,
                temperature=0.0,
            )

        # Run completions concurrently
        results = await asyncio.gather(*[create_completion(p) for p in prompts])

        assert len(results) == 3
        for i, result in enumerate(results):
            assert len(result.choices) > 0
            print(f"Concurrent result {i}: {result.choices[0].token_ids}")


class TestContextManagers:
    """Test context manager functionality."""

    def test_sync_context_manager(self):
        """Test sync client context manager."""
        with VLLMGrpcClient(host=GRPC_HOST, port=GRPC_PORT) as client:
            response = client.health.check()
            assert response.healthy is True

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async client context manager."""
        async with AsyncVLLMGrpcClient(host=GRPC_HOST, port=GRPC_PORT) as client:
            response = await client.health.check()
            assert response.healthy is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
