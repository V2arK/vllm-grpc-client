#!/usr/bin/env python3
"""
Example 01: Basic Text Completion

This example demonstrates the fundamental usage of the vLLM gRPC client
for text generation, including both non-streaming and streaming modes.

Features covered:
- Creating a client connection
- Non-streaming text completion
- Streaming text completion
- Accessing token IDs and usage statistics

Usage:
    python examples/01_basic_completion.py
"""

from vllm_grpc_client import VLLMGrpcClient

# Server configuration - modify these to match your server
GRPC_HOST = "10.28.115.40"
GRPC_PORT = 9000


def main():
    # =========================================================================
    # 1. Create the gRPC client
    # =========================================================================
    # The client establishes a connection to the vLLM gRPC server.
    # You can also use environment variables:
    #   - VLLM_GRPC_HOST: default host
    #   - VLLM_GRPC_PORT: default port
    #   - VLLM_GRPC_TIMEOUT: default timeout in seconds

    print("Connecting to vLLM gRPC server...")
    client = VLLMGrpcClient(host=GRPC_HOST, port=GRPC_PORT)
    print(f"Connected to {GRPC_HOST}:{GRPC_PORT}\n")

    # =========================================================================
    # 2. Non-Streaming Completion
    # =========================================================================
    # Non-streaming mode waits for the entire response before returning.
    # This is simpler but has higher latency for long responses.

    print("=" * 60)
    print("NON-STREAMING COMPLETION")
    print("=" * 60)

    completion = client.completions.create(
        prompt="The quick brown fox",
        max_tokens=30,
        temperature=0.7,
    )

    # The response contains:
    # - id: Unique request identifier
    # - choices: List of completion choices (usually 1)
    # - usage: Token usage statistics
    print(f"Request ID: {completion.id}")
    print(f"Token IDs: {completion.choices[0].token_ids}")
    print(f"Finish reason: {completion.choices[0].finish_reason}")
    print(f"Prompt tokens: {completion.usage.prompt_tokens}")
    print(f"Completion tokens: {completion.usage.completion_tokens}")
    print(f"Total tokens: {completion.usage.total_tokens}")
    print()

    # =========================================================================
    # 3. Streaming Completion
    # =========================================================================
    # Streaming mode yields tokens as they are generated.
    # This provides lower time-to-first-token and real-time output.

    print("=" * 60)
    print("STREAMING COMPLETION")
    print("=" * 60)

    stream = client.completions.create(
        prompt="Once upon a time",
        max_tokens=50,
        temperature=0.8,
        stream=True,  # Enable streaming
    )

    # Iterate over the stream to receive chunks
    print("Streaming token IDs:")
    all_tokens = []
    for chunk in stream:
        # Each chunk contains:
        # - delta_token_ids: New tokens in this chunk
        # - finish_reason: Set only on the final chunk
        delta_tokens = chunk.choices[0].delta_token_ids
        all_tokens.extend(delta_tokens)
        print(f"  Chunk: {delta_tokens}")

        # Check if this is the final chunk
        if chunk.choices[0].finish_reason:
            print(f"  [Finish reason: {chunk.choices[0].finish_reason}]")

    print(f"\nTotal tokens received: {len(all_tokens)}")

    # You can access the final accumulated completion
    final_completion = stream.get_final_completion()
    if final_completion:
        print(f"Final completion ID: {final_completion.id}")
    print()

    # =========================================================================
    # 4. Using Context Manager
    # =========================================================================
    # The client can be used as a context manager for automatic cleanup.

    print("=" * 60)
    print("USING CONTEXT MANAGER")
    print("=" * 60)

    with VLLMGrpcClient(host=GRPC_HOST, port=GRPC_PORT) as ctx_client:
        result = ctx_client.completions.create(
            prompt="Hello, world!",
            max_tokens=10,
        )
        print(f"Completion using context manager: {result.choices[0].token_ids}")

    print("Client automatically closed after context manager exit\n")

    # =========================================================================
    # Cleanup
    # =========================================================================
    client.close()
    print("Done!")


if __name__ == "__main__":
    main()
