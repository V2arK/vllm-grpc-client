#!/usr/bin/env python3
"""
Example 02: Decoding Token IDs to Text

The vLLM gRPC server returns token IDs (integers) rather than text.
This example shows how to use TokenDecoder to convert token IDs back to
human-readable text.

Features covered:
- Creating a TokenDecoder from the client
- Decoding non-streaming completions
- Decoding streaming completions in real-time
- Manual token encoding/decoding

Requirements:
    pip install transformers

Usage:
    python examples/02_text_decoding.py
"""

from vllm_grpc_client import VLLMGrpcClient, TokenDecoder

# Server configuration
GRPC_HOST = "10.28.115.40"
GRPC_PORT = 9000


def main():
    # =========================================================================
    # 1. Create Client and TokenDecoder
    # =========================================================================
    print("Connecting to server...")
    client = VLLMGrpcClient(host=GRPC_HOST, port=GRPC_PORT)

    # TokenDecoder automatically:
    # 1. Queries the model name from the server
    # 2. Downloads the tokenizer from HuggingFace (cached after first use)
    # 3. Wraps it for convenient decoding
    print("Loading tokenizer (first run may download from HuggingFace)...")
    decoder = TokenDecoder.from_client(client)
    print("Tokenizer loaded!\n")

    # =========================================================================
    # 2. Decode Non-Streaming Completion
    # =========================================================================
    print("=" * 60)
    print("DECODING NON-STREAMING COMPLETION")
    print("=" * 60)

    completion = client.completions.create(
        prompt="The capital of France is",
        max_tokens=30,
        temperature=0.0,  # Deterministic output
    )

    # Method 1: Use decode_completion() helper
    text = decoder.decode_completion(completion)
    print(f"Prompt: 'The capital of France is'")
    print(f"Generated text: {text}")
    print(f"Token IDs: {completion.choices[0].token_ids}")
    print()

    # =========================================================================
    # 3. Decode Streaming Completion (Real-time)
    # =========================================================================
    print("=" * 60)
    print("DECODING STREAMING COMPLETION (REAL-TIME)")
    print("=" * 60)

    stream = client.completions.create(
        prompt="Write a haiku about programming:",
        max_tokens=50,
        temperature=0.7,
        stream=True,
    )

    print("Prompt: 'Write a haiku about programming:'")
    print("Generated text: ", end="", flush=True)

    # Decode and print each chunk as it arrives
    for chunk in stream:
        # Method 2: Use decode_chunk() helper for streaming
        delta_text = decoder.decode_chunk(chunk)
        print(delta_text, end="", flush=True)

    print("\n")

    # =========================================================================
    # 4. Manual Token Encoding and Decoding
    # =========================================================================
    print("=" * 60)
    print("MANUAL ENCODING AND DECODING")
    print("=" * 60)

    # Encode text to token IDs
    text = "Hello, how are you?"
    token_ids = decoder.encode(text)
    print(f"Original text: '{text}'")
    print(f"Encoded token IDs: {token_ids}")

    # Decode token IDs back to text
    decoded_text = decoder.decode(token_ids)
    print(f"Decoded text: '{decoded_text}'")
    print()

    # Decode arbitrary token IDs
    some_tokens = [791, 6520, 374, 6366, 0]
    decoded = decoder.decode(some_tokens)
    print(f"Token IDs {some_tokens} decode to: '{decoded}'")
    print()

    # =========================================================================
    # 5. Decode with Special Tokens
    # =========================================================================
    print("=" * 60)
    print("HANDLING SPECIAL TOKENS")
    print("=" * 60)

    completion = client.completions.create(
        prompt="Say hello",
        max_tokens=20,
        temperature=0.0,
    )

    # By default, special tokens are skipped
    text_without_special = decoder.decode(
        completion.choices[0].token_ids,
        skip_special_tokens=True,  # Default
    )
    print(f"Without special tokens: '{text_without_special}'")

    # Include special tokens (e.g., <|endoftext|>)
    text_with_special = decoder.decode(
        completion.choices[0].token_ids,
        skip_special_tokens=False,
    )
    print(f"With special tokens: '{text_with_special}'")
    print()

    # =========================================================================
    # Cleanup
    # =========================================================================
    client.close()
    print("Done!")


if __name__ == "__main__":
    main()
