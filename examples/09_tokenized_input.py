#!/usr/bin/env python3
"""
Example 09: Using Tokenized Input

This example demonstrates how to send pre-tokenized input to the server,
which can be useful when you want to:
- Avoid double tokenization (if you've already tokenized on client side)
- Send specific token sequences
- Integrate with systems that work with token IDs

Features covered:
- Sending token IDs directly
- Using TokenizedInput object
- Comparing text vs tokenized input

Usage:
    python examples/09_tokenized_input.py
"""

from vllm_grpc_client import VLLMGrpcClient, TokenDecoder, TokenizedInput

# Server configuration
GRPC_HOST = "10.28.115.40"
GRPC_PORT = 9000


def main():
    client = VLLMGrpcClient(host=GRPC_HOST, port=GRPC_PORT)
    decoder = TokenDecoder.from_client(client)
    print("Client and decoder ready!\n")

    # =========================================================================
    # 1. Text Input (Standard)
    # =========================================================================
    print("=" * 60)
    print("TEXT INPUT (STANDARD)")
    print("=" * 60)

    prompt_text = "The capital of France is"

    completion = client.completions.create(
        prompt=prompt_text,  # String input
        max_tokens=10,
        temperature=0.0,
    )

    text_result = decoder.decode_completion(completion)
    print(f"Prompt (text): '{prompt_text}'")
    print(f"Result: {text_result}")
    print(f"Prompt tokens: {completion.usage.prompt_tokens}")
    print()

    # =========================================================================
    # 2. Token IDs Input (List)
    # =========================================================================
    print("=" * 60)
    print("TOKEN IDS INPUT (LIST)")
    print("=" * 60)

    # First, encode the prompt to get token IDs
    prompt_tokens = decoder.encode(prompt_text)
    print(f"Encoded '{prompt_text}' to tokens: {prompt_tokens}")

    # Send token IDs directly
    completion = client.completions.create(
        prompt=prompt_tokens,  # List of integers
        max_tokens=10,
        temperature=0.0,
    )

    token_result = decoder.decode_completion(completion)
    print(f"Prompt (tokens): {prompt_tokens}")
    print(f"Result: {token_result}")
    print(f"Prompt tokens: {completion.usage.prompt_tokens}")
    print()

    # =========================================================================
    # 3. TokenizedInput Object
    # =========================================================================
    print("=" * 60)
    print("TOKENIZED INPUT OBJECT")
    print("=" * 60)

    # Use TokenizedInput for more control
    tokenized_input = TokenizedInput(
        original_text=prompt_text,  # For debugging/logging
        input_ids=prompt_tokens,
    )

    completion = client.completions.create(
        prompt=tokenized_input,  # TokenizedInput object
        max_tokens=10,
        temperature=0.0,
    )

    tokenized_result = decoder.decode_completion(completion)
    print(f"TokenizedInput.original_text: '{tokenized_input.original_text}'")
    print(f"TokenizedInput.input_ids: {tokenized_input.input_ids}")
    print(f"Result: {tokenized_result}")
    print()

    # =========================================================================
    # 4. Custom Token Sequences
    # =========================================================================
    print("=" * 60)
    print("CUSTOM TOKEN SEQUENCES")
    print("=" * 60)

    # You can create custom token sequences for special purposes
    # For example, continuing from specific tokens

    # Get some example tokens
    hello_tokens = decoder.encode("Hello")
    world_tokens = decoder.encode(" world")

    print(f"'Hello' tokens: {hello_tokens}")
    print(f"' world' tokens: {world_tokens}")

    # Combine tokens
    combined_tokens = hello_tokens + world_tokens
    combined_text = decoder.decode(combined_tokens)
    print(f"Combined: {combined_tokens} -> '{combined_text}'")

    # Generate continuation
    completion = client.completions.create(
        prompt=combined_tokens,
        max_tokens=15,
        temperature=0.7,
    )

    continuation = decoder.decode_completion(completion)
    print(f"Continuation: {continuation}")
    print()

    # =========================================================================
    # 5. Comparing Results
    # =========================================================================
    print("=" * 60)
    print("COMPARING TEXT VS TOKENIZED INPUT")
    print("=" * 60)

    test_prompt = "Python is a"

    # With text input
    completion_text = client.completions.create(
        prompt=test_prompt,
        max_tokens=10,
        temperature=0.0,
        seed=42,
    )

    # With tokenized input
    test_tokens = decoder.encode(test_prompt)
    completion_tokens = client.completions.create(
        prompt=test_tokens,
        max_tokens=10,
        temperature=0.0,
        seed=42,
    )

    result_text = decoder.decode_completion(completion_text)
    result_tokens = decoder.decode_completion(completion_tokens)

    print(f"Prompt: '{test_prompt}'")
    print(f"Token IDs: {test_tokens}")
    print(f"Result (text input): {result_text}")
    print(f"Result (token input): {result_tokens}")
    print(f"Results match: {result_text == result_tokens}")
    print()

    # =========================================================================
    # 6. Special Tokens
    # =========================================================================
    print("=" * 60)
    print("WORKING WITH SPECIAL TOKENS")
    print("=" * 60)

    # Access special token IDs from the tokenizer
    tokenizer = decoder.tokenizer

    # Get some special token IDs (availability depends on the model)
    special_tokens = {}
    for attr in ["bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"]:
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            special_tokens[attr] = token_id

    print("Special tokens for this model:")
    for name, token_id in special_tokens.items():
        token_str = tokenizer.decode([token_id])
        print(f"  {name}: {token_id} -> '{repr(token_str)}'")

    # You can include special tokens in your input if needed
    if "bos_token_id" in special_tokens:
        bos_id = special_tokens["bos_token_id"]
        prompt_with_bos = [bos_id] + decoder.encode("Hello")
        print(f"\nPrompt with BOS: {prompt_with_bos}")
        print(f"Decoded: '{decoder.decode(prompt_with_bos, skip_special_tokens=False)}'")
    print()

    # =========================================================================
    # Cleanup
    # =========================================================================
    client.close()
    print("Done!")


if __name__ == "__main__":
    main()
