#!/usr/bin/env python3
"""
Example 07: Asynchronous Client

This example demonstrates the async client for high-performance
concurrent operations.

Features covered:
- Async client creation and usage
- Async streaming completions
- Concurrent completions with asyncio.gather
- Async context manager
- Async health checks and model info

Usage:
    python examples/07_async_client.py
"""

import asyncio
import time

from vllm_grpc_client import AsyncVLLMGrpcClient, TokenDecoder

# Server configuration
GRPC_HOST = "localhost"
GRPC_PORT = 9000


async def main():
    # =========================================================================
    # 1. Create Async Client
    # =========================================================================
    print("=" * 60)
    print("ASYNC CLIENT CREATION")
    print("=" * 60)

    # Create the async client
    client = AsyncVLLMGrpcClient(host=GRPC_HOST, port=GRPC_PORT)
    print(f"Connected to {GRPC_HOST}:{GRPC_PORT}")

    # Create decoder (sync operation, but can be done async with afrom_client)
    decoder = await TokenDecoder.afrom_client(client)
    print("Tokenizer loaded!\n")

    # =========================================================================
    # 2. Basic Async Completion
    # =========================================================================
    print("=" * 60)
    print("BASIC ASYNC COMPLETION")
    print("=" * 60)

    completion = await client.completions.create(
        prompt="Hello, how are you?",
        max_tokens=20,
        temperature=0.7,
    )

    text = decoder.decode_completion(completion)
    print(f"Prompt: 'Hello, how are you?'")
    print(f"Response: {text}")
    print()

    # =========================================================================
    # 3. Async Streaming
    # =========================================================================
    print("=" * 60)
    print("ASYNC STREAMING")
    print("=" * 60)

    stream = await client.completions.create(
        prompt="Count from 1 to 5:",
        max_tokens=30,
        temperature=0.0,
        stream=True,
    )

    print("Streaming output: ", end="", flush=True)
    async for chunk in stream:
        delta_text = decoder.decode_chunk(chunk)
        print(delta_text, end="", flush=True)
    print("\n")

    # =========================================================================
    # 4. Concurrent Completions
    # =========================================================================
    print("=" * 60)
    print("CONCURRENT COMPLETIONS")
    print("=" * 60)

    prompts = [
        "The capital of France is",
        "The largest planet is",
        "Water boils at",
        "The speed of light is",
        "Python was created by",
    ]

    async def create_completion(prompt: str) -> tuple[str, str]:
        """Create a completion and return prompt + result."""
        completion = await client.completions.create(
            prompt=prompt,
            max_tokens=15,
            temperature=0.0,
        )
        text = decoder.decode_completion(completion)
        return prompt, text

    print(f"Sending {len(prompts)} concurrent requests...")
    start_time = time.time()

    # Run all completions concurrently
    results = await asyncio.gather(*[create_completion(p) for p in prompts])

    elapsed = time.time() - start_time
    print(f"All {len(prompts)} completions finished in {elapsed:.2f}s\n")

    for prompt, text in results:
        print(f"  '{prompt}' → {text}")
    print()

    # =========================================================================
    # 5. Async Health Check and Server Info
    # =========================================================================
    print("=" * 60)
    print("ASYNC HEALTH AND SERVER INFO")
    print("=" * 60)

    # Health check
    health = await client.health.check()
    print(f"Health: {health.healthy} - {health.message}")

    # Server info
    server_info = await client.health.server_info()
    print(f"Server type: {server_info.server_type}")
    print(f"Uptime: {server_info.uptime_seconds:.2f}s")

    # Model info
    model_info = await client.models.retrieve()
    print(f"Model: {model_info.model_path}")
    print()

    # =========================================================================
    # 6. Async Wait for Ready
    # =========================================================================
    print("=" * 60)
    print("ASYNC WAIT FOR READY")
    print("=" * 60)

    is_ready = await client.wait_for_ready(timeout=5.0, poll_interval=0.5)
    print(f"Server ready: {is_ready}")
    print()

    # Close the client
    await client.close()

    # =========================================================================
    # 7. Async Context Manager
    # =========================================================================
    print("=" * 60)
    print("ASYNC CONTEXT MANAGER")
    print("=" * 60)

    async with AsyncVLLMGrpcClient(host=GRPC_HOST, port=GRPC_PORT) as ctx_client:
        health = await ctx_client.health.check()
        print(f"Context manager health check: {health.healthy}")

    print("Client automatically closed\n")

    # =========================================================================
    # 8. Concurrent Streaming (Advanced)
    # =========================================================================
    print("=" * 60)
    print("CONCURRENT STREAMING (ADVANCED)")
    print("=" * 60)

    async with AsyncVLLMGrpcClient(host=GRPC_HOST, port=GRPC_PORT) as ctx_client:
        decoder = await TokenDecoder.afrom_client(ctx_client)

        async def stream_completion(name: str, prompt: str):
            """Stream a completion and collect results."""
            stream = await ctx_client.completions.create(
                prompt=prompt,
                max_tokens=20,
                temperature=0.7,
                stream=True,
            )

            tokens = []
            async for chunk in stream:
                tokens.extend(chunk.choices[0].delta_token_ids)

            text = decoder.decode(tokens)
            return name, text

        # Run multiple streams concurrently
        stream_tasks = [
            stream_completion("Stream A", "The sun is"),
            stream_completion("Stream B", "The moon is"),
            stream_completion("Stream C", "Stars are"),
        ]

        print("Running 3 concurrent streams...")
        start_time = time.time()
        results = await asyncio.gather(*stream_tasks)
        elapsed = time.time() - start_time

        print(f"All streams completed in {elapsed:.2f}s")
        for name, text in results:
            print(f"  {name}: {text}")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
