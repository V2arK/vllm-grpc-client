#!/usr/bin/env python3
"""
Example 06: Aborting Requests

This example demonstrates how to cancel running requests using the
out-of-band abort mechanism.

Features covered:
- Aborting a streaming request
- Aborting multiple requests
- Handling abort finish reason

Usage:
    python examples/06_abort_requests.py
"""

import threading
import time
import uuid

from vllm_grpc_client import VLLMGrpcClient, TokenDecoder

# Server configuration
GRPC_HOST = "localhost"
GRPC_PORT = 9000


def main():
    client = VLLMGrpcClient(host=GRPC_HOST, port=GRPC_PORT)
    decoder = TokenDecoder.from_client(client)
    print("Client ready!\n")

    # =========================================================================
    # 1. Basic Abort Example
    # =========================================================================
    print("=" * 60)
    print("BASIC ABORT EXAMPLE")
    print("=" * 60)

    # Generate a unique request ID
    request_id = f"abort-test-{uuid.uuid4().hex[:8]}"
    print(f"Request ID: {request_id}")

    # Start a long-running streaming request
    stream = client.completions.create(
        prompt="Write a very long story about a dragon:",
        max_tokens=500,  # Request many tokens
        min_tokens=100,  # Ensure it generates enough
        temperature=0.7,
        stream=True,
        request_id=request_id,
    )

    # Collect some tokens, then abort
    tokens_received = 0
    abort_after = 20  # Abort after receiving 20 token chunks

    print(f"Streaming (will abort after {abort_after} chunks)...")

    for chunk in stream:
        tokens = chunk.choices[0].delta_token_ids
        tokens_received += len(tokens)

        if tokens_received >= abort_after and not chunk.choices[0].finish_reason:
            print(f"\n[Received {tokens_received} tokens, sending abort...]")
            # Abort the request out-of-band
            client.health.abort([request_id])
            # Note: The stream will continue until the server processes the abort

        # Check for abort completion
        if chunk.choices[0].finish_reason:
            print(f"[Finish reason: {chunk.choices[0].finish_reason}]")
            if chunk.choices[0].finish_reason == "abort":
                print("[Request was successfully aborted!]")
            break
        else:
            # Print a dot for each chunk to show progress
            print(".", end="", flush=True)

    print(f"\nTotal tokens received: {tokens_received}")
    print()

    # =========================================================================
    # 2. Abort with Threading
    # =========================================================================
    print("=" * 60)
    print("ABORT WITH THREADING")
    print("=" * 60)

    request_id = f"thread-abort-{uuid.uuid4().hex[:8]}"
    print(f"Request ID: {request_id}")

    # Track state
    generation_done = threading.Event()
    tokens_generated = []

    def generate_text():
        """Run generation in a background thread."""
        stream = client.completions.create(
            prompt="Count from 1 to 100:",
            max_tokens=300,
            temperature=0.0,
            stream=True,
            request_id=request_id,
        )

        for chunk in stream:
            tokens = chunk.choices[0].delta_token_ids
            tokens_generated.extend(tokens)

            if chunk.choices[0].finish_reason:
                print(f"\n  [Generation finished: {chunk.choices[0].finish_reason}]")
                break
            else:
                print(".", end="", flush=True)

        generation_done.set()

    def abort_after_delay(delay: float):
        """Abort after a delay."""
        time.sleep(delay)
        print(f"\n  [Sending abort after {delay}s delay...]")
        client.health.abort([request_id])

    # Start generation and abort threads
    gen_thread = threading.Thread(target=generate_text)
    abort_thread = threading.Thread(target=abort_after_delay, args=(0.5,))

    print("Starting generation with abort after 0.5 seconds...")
    gen_thread.start()
    abort_thread.start()

    # Wait for both threads
    gen_thread.join()
    abort_thread.join()

    print(f"Tokens generated before abort: {len(tokens_generated)}")
    print()

    # =========================================================================
    # 3. Abort Multiple Requests
    # =========================================================================
    print("=" * 60)
    print("ABORT MULTIPLE REQUESTS")
    print("=" * 60)

    # Create multiple request IDs
    request_ids = [f"multi-{i}-{uuid.uuid4().hex[:8]}" for i in range(3)]
    print(f"Request IDs: {request_ids}")

    # Note: In a real scenario, you would have multiple concurrent requests.
    # Here we demonstrate the API for aborting multiple requests at once.

    print("Aborting multiple requests (no active requests with these IDs)...")
    client.health.abort(request_ids)  # This is a no-op if IDs don't exist
    print("Abort request sent for all IDs")
    print()

    # =========================================================================
    # 4. Abort with Custom Timeout
    # =========================================================================
    print("=" * 60)
    print("ABORT WITH CUSTOM TIMEOUT")
    print("=" * 60)

    request_id = f"timeout-test-{uuid.uuid4().hex[:8]}"
    print(f"Request ID: {request_id}")

    # Start generation
    stream = client.completions.create(
        prompt="Explain quantum physics:",
        max_tokens=200,
        temperature=0.7,
        stream=True,
        request_id=request_id,
    )

    # Collect a few tokens then abort with timeout
    for i, chunk in enumerate(stream):
        if i >= 10:
            print(f"\n[Aborting with 5 second timeout...]")
            client.health.abort([request_id], timeout=5.0)

        if chunk.choices[0].finish_reason:
            print(f"[Finish reason: {chunk.choices[0].finish_reason}]")
            break
        else:
            print(".", end="", flush=True)

    print("\nAbort with timeout completed")
    print()

    # =========================================================================
    # Cleanup
    # =========================================================================
    client.close()
    print("Done!")


if __name__ == "__main__":
    main()
