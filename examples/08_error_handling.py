#!/usr/bin/env python3
"""
Example 08: Error Handling

This example demonstrates how to handle various errors that can occur
when using the vLLM gRPC client.

Features covered:
- Connection errors
- Timeout errors
- Invalid argument errors
- Unimplemented errors (e.g., embeddings)
- Aborted request errors
- Generic error handling

Usage:
    python examples/08_error_handling.py
"""

from vllm_grpc_client import (
    VLLMGrpcClient,
    VLLMGrpcError,
    VLLMGrpcConnectionError,
    VLLMGrpcTimeoutError,
    VLLMGrpcInvalidArgumentError,
    VLLMGrpcUnimplementedError,
    VLLMGrpcUnavailableError,
    VLLMGrpcAbortedError,
)

# Server configuration
GRPC_HOST = "10.28.115.40"
GRPC_PORT = 9000


def main():
    # =========================================================================
    # 1. Connection Errors
    # =========================================================================
    print("=" * 60)
    print("CONNECTION ERRORS")
    print("=" * 60)

    # Try connecting to a non-existent server
    print("Attempting to connect to non-existent server...")
    try:
        bad_client = VLLMGrpcClient(host="non-existent-host", port=9999)
        # The connection is lazy, so we need to make a request to trigger the error
        bad_client.health.check(timeout=2.0)
    except VLLMGrpcUnavailableError as e:
        print(f"Caught VLLMGrpcUnavailableError: {e.message}")
        print(f"  Code: {e.code}")
    except VLLMGrpcError as e:
        print(f"Caught VLLMGrpcError: {e.message}")
    except Exception as e:
        print(f"Caught unexpected error: {type(e).__name__}: {e}")
    print()

    # =========================================================================
    # 2. Timeout Errors
    # =========================================================================
    print("=" * 60)
    print("TIMEOUT ERRORS")
    print("=" * 60)

    client = VLLMGrpcClient(host=GRPC_HOST, port=GRPC_PORT)

    # Request with very short timeout (may or may not timeout depending on network)
    print("Attempting request with very short timeout (0.001s)...")
    try:
        client.completions.create(
            prompt="Hello",
            max_tokens=100,
            timeout=0.001,  # 1 millisecond - likely to timeout
        )
        print("Request completed (fast network!)")
    except VLLMGrpcTimeoutError as e:
        print(f"Caught VLLMGrpcTimeoutError: {e.message}")
        print(f"  Code: {e.code}")
    except VLLMGrpcError as e:
        # May get other errors depending on timing
        print(f"Caught VLLMGrpcError: {e.message}")
    print()

    # =========================================================================
    # 3. Invalid Argument Errors
    # =========================================================================
    print("=" * 60)
    print("INVALID ARGUMENT ERRORS")
    print("=" * 60)

    # Invalid top_p value
    print("Attempting request with invalid top_p (-1.0)...")
    try:
        client.completions.create(
            prompt="Hello",
            max_tokens=10,
            top_p=-1.0,  # Invalid: must be between 0 and 1
        )
    except VLLMGrpcInvalidArgumentError as e:
        print(f"Caught VLLMGrpcInvalidArgumentError: {e.message}")
        print(f"  Details: {e.details}")
    except VLLMGrpcError as e:
        print(f"Caught VLLMGrpcError: {e.message}")
    print()

    # Invalid temperature value
    print("Attempting request with invalid temperature (-5.0)...")
    try:
        client.completions.create(
            prompt="Hello",
            max_tokens=10,
            temperature=-5.0,  # Invalid: must be >= 0
        )
    except VLLMGrpcInvalidArgumentError as e:
        print(f"Caught VLLMGrpcInvalidArgumentError: {e.message}")
    except VLLMGrpcError as e:
        print(f"Caught VLLMGrpcError: {e.message}")
    print()

    # =========================================================================
    # 4. Unimplemented Errors
    # =========================================================================
    print("=" * 60)
    print("UNIMPLEMENTED ERRORS")
    print("=" * 60)

    # Embeddings are not yet implemented in vLLM gRPC
    print("Attempting to use embeddings (not implemented)...")
    try:
        client.embeddings.create(
            input=[1, 2, 3, 4, 5],  # Token IDs
        )
    except VLLMGrpcUnimplementedError as e:
        print(f"Caught VLLMGrpcUnimplementedError: {e.message}")
        print(f"  Details: {e.details}")
    except VLLMGrpcError as e:
        print(f"Caught VLLMGrpcError: {e.message}")
    print()

    # =========================================================================
    # 5. Generic Error Handling Pattern
    # =========================================================================
    print("=" * 60)
    print("GENERIC ERROR HANDLING PATTERN")
    print("=" * 60)

    def safe_completion(prompt: str, max_tokens: int = 10):
        """
        Example of safe completion with comprehensive error handling.
        """
        try:
            completion = client.completions.create(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return completion.choices[0].token_ids

        except VLLMGrpcInvalidArgumentError as e:
            print(f"[ERROR] Invalid argument: {e.details}")
            return None

        except VLLMGrpcTimeoutError as e:
            print(f"[ERROR] Request timed out: {e.message}")
            return None

        except VLLMGrpcUnavailableError as e:
            print(f"[ERROR] Server unavailable: {e.message}")
            return None

        except VLLMGrpcUnimplementedError as e:
            print(f"[ERROR] Feature not implemented: {e.message}")
            return None

        except VLLMGrpcAbortedError as e:
            print(f"[ERROR] Request aborted: {e.message}")
            return None

        except VLLMGrpcError as e:
            # Catch-all for other gRPC errors
            print(f"[ERROR] gRPC error ({e.code}): {e.message}")
            return None

        except Exception as e:
            # Non-gRPC errors (e.g., network issues)
            print(f"[ERROR] Unexpected error: {type(e).__name__}: {e}")
            return None

    # Test the safe completion function
    print("Testing safe_completion function...")
    result = safe_completion("Hello, world!")
    if result:
        print(f"Success: {result}")
    print()

    # =========================================================================
    # 6. Retry Pattern
    # =========================================================================
    print("=" * 60)
    print("RETRY PATTERN")
    print("=" * 60)

    import time

    def completion_with_retry(
        prompt: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Completion with automatic retry on transient errors.
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                completion = client.completions.create(
                    prompt=prompt,
                    max_tokens=10,
                    temperature=0.7,
                )
                return completion.choices[0].token_ids

            except (VLLMGrpcTimeoutError, VLLMGrpcUnavailableError) as e:
                # Transient errors - retry
                last_error = e
                print(f"  Attempt {attempt + 1} failed: {e.message}")
                if attempt < max_retries - 1:
                    print(f"  Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)

            except VLLMGrpcInvalidArgumentError:
                # Non-transient error - don't retry
                raise

        raise last_error

    print("Testing retry pattern...")
    try:
        result = completion_with_retry("Test prompt")
        print(f"Success: {result}")
    except VLLMGrpcError as e:
        print(f"All retries failed: {e.message}")
    print()

    # =========================================================================
    # 7. Error Information
    # =========================================================================
    print("=" * 60)
    print("ERROR INFORMATION")
    print("=" * 60)

    print("All exception types in vllm_grpc_client:")
    print("  - VLLMGrpcError (base class)")
    print("  - VLLMGrpcConnectionError")
    print("  - VLLMGrpcTimeoutError")
    print("  - VLLMGrpcInvalidArgumentError")
    print("  - VLLMGrpcUnavailableError")
    print("  - VLLMGrpcUnimplementedError")
    print("  - VLLMGrpcAbortedError")
    print("  - VLLMGrpcInternalError")
    print("  - VLLMGrpcCancelledError")
    print()

    print("Error attributes:")
    print("  - message: Human-readable error message")
    print("  - code: gRPC status code name")
    print("  - details: Additional error details from server")
    print()

    # =========================================================================
    # Cleanup
    # =========================================================================
    client.close()
    print("Done!")


if __name__ == "__main__":
    main()
