#!/usr/bin/env python3
"""
Basic usage examples for vLLM gRPC client.

This example demonstrates how to use the vLLM gRPC client with both
sync and async interfaces.

Usage:
    # Set server address via environment variables
    export VLLM_GRPC_HOST=10.28.115.40
    export VLLM_GRPC_PORT=9000

    # Or run with specific server
    python examples/basic_usage.py --host 10.28.115.40 --port 9000
"""

import argparse
import asyncio

from vllm_grpc_client import AsyncVLLMGrpcClient, VLLMGrpcClient


def sync_example(host: str, port: int):
    """Synchronous client example."""
    print("=" * 60)
    print("Synchronous Client Example")
    print("=" * 60)

    # Create client (can also use context manager)
    client = VLLMGrpcClient(host=host, port=port)

    try:
        # Health check
        print("\n1. Health Check:")
        health = client.health.check()
        print(f"   Healthy: {health.healthy}")
        print(f"   Message: {health.message}")

        # Get model info
        print("\n2. Model Info:")
        model_info = client.models.retrieve()
        print(f"   Model: {model_info.model_path}")
        print(f"   Max context: {model_info.max_context_length}")
        print(f"   Vocab size: {model_info.vocab_size}")

        # Non-streaming completion
        print("\n3. Non-Streaming Completion:")
        completion = client.completions.create(
            prompt="The quick brown fox",
            max_tokens=20,
            temperature=0.7,
        )
        print(f"   Request ID: {completion.id}")
        print(f"   Tokens: {completion.choices[0].token_ids}")
        print(f"   Finish reason: {completion.choices[0].finish_reason}")
        print(f"   Usage: {completion.usage}")

        # Streaming completion
        print("\n4. Streaming Completion:")
        stream = client.completions.create(
            prompt="Count from 1 to 3:",
            max_tokens=30,
            temperature=0.0,
            stream=True,
        )

        all_tokens = []
        for chunk in stream:
            tokens = chunk.choices[0].delta_token_ids
            all_tokens.extend(tokens)
            print(f"   Chunk: {tokens}")

        final = stream.get_final_completion()
        print(f"   Final completion: {final.choices[0].finish_reason}")

        # Completion with structured output (JSON)
        print("\n5. Structured Output (JSON Schema):")
        from vllm_grpc_client import StructuredOutputs

        completion = client.completions.create(
            prompt="Generate a JSON object with name and age fields:",
            max_tokens=50,
            temperature=0.0,
            structured_outputs=StructuredOutputs(
                json_schema='{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}'
            ),
        )
        print(f"   Tokens: {completion.choices[0].token_ids}")

    finally:
        client.close()


async def async_example(host: str, port: int):
    """Asynchronous client example."""
    print("\n" + "=" * 60)
    print("Asynchronous Client Example")
    print("=" * 60)

    async with AsyncVLLMGrpcClient(host=host, port=port) as client:
        # Health check
        print("\n1. Async Health Check:")
        health = await client.health.check()
        print(f"   Healthy: {health.healthy}")

        # Server info
        print("\n2. Server Info:")
        server_info = await client.health.server_info()
        print(f"   Server type: {server_info.server_type}")
        print(f"   Active requests: {server_info.active_requests}")
        print(f"   Uptime: {server_info.uptime_seconds:.2f}s")

        # Async streaming completion
        print("\n3. Async Streaming Completion:")
        stream = await client.completions.create(
            prompt="List three fruits:",
            max_tokens=30,
            temperature=0.0,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta_token_ids:
                print(f"   Chunk: {chunk.choices[0].delta_token_ids}")

        # Concurrent completions
        print("\n4. Concurrent Completions:")
        prompts = [
            "The sun is",
            "Water is made of",
            "The Earth orbits",
        ]

        async def create_completion(prompt):
            return await client.completions.create(
                prompt=prompt,
                max_tokens=10,
                temperature=0.0,
            )

        results = await asyncio.gather(*[create_completion(p) for p in prompts])

        for prompt, result in zip(prompts, results):
            print(f"   '{prompt}' -> {len(result.choices[0].token_ids)} tokens")


def main():
    parser = argparse.ArgumentParser(description="vLLM gRPC client examples")
    parser.add_argument("--host", default="10.28.115.40", help="gRPC server host")
    parser.add_argument("--port", type=int, default=9000, help="gRPC server port")
    args = parser.parse_args()

    # Run sync example
    sync_example(args.host, args.port)

    # Run async example
    asyncio.run(async_example(args.host, args.port))

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
