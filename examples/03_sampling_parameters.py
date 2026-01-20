#!/usr/bin/env python3
"""
Example 03: Sampling Parameters

This example demonstrates all available sampling parameters for controlling
text generation behavior.

Features covered:
- Temperature sampling
- Top-p (nucleus) sampling
- Top-k sampling
- Min-p sampling
- Frequency and presence penalties
- Repetition penalty
- Stop sequences and stop tokens
- Seed for reproducibility
- Logit bias
- Other parameters (min_tokens, ignore_eos, etc.)

Usage:
    python examples/03_sampling_parameters.py
"""

from vllm_grpc_client import VLLMGrpcClient, TokenDecoder

# Server configuration
GRPC_HOST = "localhost"
GRPC_PORT = 9000


def main():
    client = VLLMGrpcClient(host=GRPC_HOST, port=GRPC_PORT)
    decoder = TokenDecoder.from_client(client)
    print("Client and decoder ready!\n")

    # =========================================================================
    # 1. Temperature
    # =========================================================================
    # Controls randomness: 0.0 = deterministic, higher = more random
    print("=" * 60)
    print("TEMPERATURE SAMPLING")
    print("=" * 60)

    for temp in [0.0, 0.5, 1.0, 1.5]:
        completion = client.completions.create(
            prompt="The meaning of life is",
            max_tokens=20,
            temperature=temp,
        )
        text = decoder.decode_completion(completion)
        print(f"Temperature {temp}: {text}")
    print()

    # =========================================================================
    # 2. Top-p (Nucleus Sampling)
    # =========================================================================
    # Only consider tokens whose cumulative probability exceeds top_p
    print("=" * 60)
    print("TOP-P (NUCLEUS) SAMPLING")
    print("=" * 60)

    for top_p in [0.1, 0.5, 0.9, 1.0]:
        completion = client.completions.create(
            prompt="Colors of the rainbow:",
            max_tokens=20,
            temperature=1.0,
            top_p=top_p,
        )
        text = decoder.decode_completion(completion)
        print(f"Top-p {top_p}: {text}")
    print()

    # =========================================================================
    # 3. Top-k Sampling
    # =========================================================================
    # Only consider the top k most likely tokens
    print("=" * 60)
    print("TOP-K SAMPLING")
    print("=" * 60)

    for top_k in [1, 10, 50, 100]:
        completion = client.completions.create(
            prompt="My favorite food is",
            max_tokens=15,
            temperature=1.0,
            top_k=top_k,
        )
        text = decoder.decode_completion(completion)
        print(f"Top-k {top_k}: {text}")
    print()

    # =========================================================================
    # 4. Min-p Sampling
    # =========================================================================
    # Filter tokens with probability less than min_p * max_probability
    print("=" * 60)
    print("MIN-P SAMPLING")
    print("=" * 60)

    for min_p in [0.0, 0.05, 0.1]:
        completion = client.completions.create(
            prompt="The best programming language is",
            max_tokens=15,
            temperature=1.0,
            min_p=min_p,
        )
        text = decoder.decode_completion(completion)
        print(f"Min-p {min_p}: {text}")
    print()

    # =========================================================================
    # 5. Frequency and Presence Penalties
    # =========================================================================
    # frequency_penalty: Penalize tokens based on their frequency in the text
    # presence_penalty: Penalize tokens that have appeared at all
    print("=" * 60)
    print("FREQUENCY AND PRESENCE PENALTIES")
    print("=" * 60)

    # Without penalties (may repeat)
    completion = client.completions.create(
        prompt="The word 'happy' is",
        max_tokens=30,
        temperature=0.7,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    text = decoder.decode_completion(completion)
    print(f"No penalties: {text}")

    # With frequency penalty (discourages repetition)
    completion = client.completions.create(
        prompt="The word 'happy' is",
        max_tokens=30,
        temperature=0.7,
        frequency_penalty=1.5,
    )
    text = decoder.decode_completion(completion)
    print(f"Frequency penalty 1.5: {text}")

    # With presence penalty (encourages new topics)
    completion = client.completions.create(
        prompt="The word 'happy' is",
        max_tokens=30,
        temperature=0.7,
        presence_penalty=1.5,
    )
    text = decoder.decode_completion(completion)
    print(f"Presence penalty 1.5: {text}")
    print()

    # =========================================================================
    # 6. Repetition Penalty
    # =========================================================================
    # Alternative to frequency/presence penalties (1.0 = no penalty)
    print("=" * 60)
    print("REPETITION PENALTY")
    print("=" * 60)

    for rep_penalty in [1.0, 1.2, 1.5]:
        completion = client.completions.create(
            prompt="Repeat after me: hello hello hello",
            max_tokens=20,
            temperature=0.7,
            repetition_penalty=rep_penalty,
        )
        text = decoder.decode_completion(completion)
        print(f"Repetition penalty {rep_penalty}: {text}")
    print()

    # =========================================================================
    # 7. Stop Sequences
    # =========================================================================
    # Stop generation when these strings are encountered
    print("=" * 60)
    print("STOP SEQUENCES")
    print("=" * 60)

    completion = client.completions.create(
        prompt="List three fruits:\n1.",
        max_tokens=50,
        temperature=0.0,
        stop=["\n\n", "4."],  # Stop at double newline or "4."
    )
    text = decoder.decode_completion(completion)
    print(f"Stop at newline or '4.': {text}")
    print(f"Finish reason: {completion.choices[0].finish_reason}")
    print()

    # =========================================================================
    # 8. Seed for Reproducibility
    # =========================================================================
    # Same seed + same parameters = same output
    print("=" * 60)
    print("SEED FOR REPRODUCIBILITY")
    print("=" * 60)

    params = dict(
        prompt="Generate a random number:",
        max_tokens=10,
        temperature=1.0,
        seed=42,
    )

    completion1 = client.completions.create(**params)
    completion2 = client.completions.create(**params)

    text1 = decoder.decode_completion(completion1)
    text2 = decoder.decode_completion(completion2)

    print(f"Seed 42 (run 1): {text1}")
    print(f"Seed 42 (run 2): {text2}")
    print(f"Same output: {text1 == text2}")
    print()

    # =========================================================================
    # 9. Logit Bias
    # =========================================================================
    # Bias specific tokens to be more or less likely
    # Token ID -> bias value (-100 to 100)
    print("=" * 60)
    print("LOGIT BIAS")
    print("=" * 60)

    # First, encode some tokens to get their IDs
    yes_tokens = decoder.encode(" yes")
    no_tokens = decoder.encode(" no")
    print(f"Token IDs for ' yes': {yes_tokens}")
    print(f"Token IDs for ' no': {no_tokens}")

    # Bias towards "yes"
    if yes_tokens and no_tokens:
        # Use the first token ID from each
        bias_yes = {yes_tokens[0]: 100.0, no_tokens[0]: -100.0}
        completion = client.completions.create(
            prompt="Should I learn Python? Answer with yes or no:",
            max_tokens=5,
            temperature=0.0,
            logit_bias=bias_yes,
        )
        text = decoder.decode_completion(completion)
        print(f"Biased towards 'yes': {text}")
    print()

    # =========================================================================
    # 10. Other Parameters
    # =========================================================================
    print("=" * 60)
    print("OTHER PARAMETERS")
    print("=" * 60)

    # min_tokens: Minimum tokens to generate
    completion = client.completions.create(
        prompt="Say hi",
        max_tokens=50,
        min_tokens=20,  # Force at least 20 tokens
        temperature=0.7,
    )
    text = decoder.decode_completion(completion)
    print(f"min_tokens=20: '{text}' ({completion.usage.completion_tokens} tokens)")

    # skip_special_tokens: Whether to skip special tokens (default: True)
    completion = client.completions.create(
        prompt="Hello",
        max_tokens=10,
        skip_special_tokens=False,
    )
    # Note: This affects server-side detokenization if stop strings are used
    print(f"skip_special_tokens=False: {completion.choices[0].token_ids}")

    # include_stop_str_in_output: Include stop string in output
    completion = client.completions.create(
        prompt="Say hello.",
        max_tokens=20,
        stop=["."],
        include_stop_str_in_output=True,
    )
    text = decoder.decode_completion(completion)
    print(f"include_stop_str_in_output=True: '{text}'")
    print()

    # =========================================================================
    # Cleanup
    # =========================================================================
    client.close()
    print("Done!")


if __name__ == "__main__":
    main()
