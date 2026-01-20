#!/usr/bin/env python3
"""
Example 04: Structured Outputs

This example demonstrates how to use structured output constraints to
guide the model to generate outputs in specific formats.

Features covered:
- JSON Schema constraint
- JSON Object constraint
- Regex pattern constraint
- Grammar (EBNF) constraint
- Choice constraint

Note: Not all models support all constraint types. The server may return
an error if a constraint is not supported.

Usage:
    python examples/04_structured_outputs.py
"""

from vllm_grpc_client import VLLMGrpcClient, TokenDecoder, StructuredOutputs, ChoiceConstraint

# Server configuration
GRPC_HOST = "10.28.115.40"
GRPC_PORT = 9000


def main():
    client = VLLMGrpcClient(host=GRPC_HOST, port=GRPC_PORT)
    decoder = TokenDecoder.from_client(client)
    print("Client and decoder ready!\n")

    # =========================================================================
    # 1. JSON Schema Constraint
    # =========================================================================
    # Force output to match a JSON schema
    print("=" * 60)
    print("JSON SCHEMA CONSTRAINT")
    print("=" * 60)

    json_schema = '''{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "city": {"type": "string"}
        },
        "required": ["name", "age", "city"]
    }'''

    try:
        completion = client.completions.create(
            prompt="Generate a person's info as JSON:",
            max_tokens=100,
            temperature=0.0,
            structured_outputs=StructuredOutputs(json_schema=json_schema),
        )
        text = decoder.decode_completion(completion)
        print(f"JSON Schema result: {text}")
    except Exception as e:
        print(f"JSON Schema not supported or error: {e}")
    print()

    # =========================================================================
    # 2. JSON Object Constraint
    # =========================================================================
    # Force output to be any valid JSON object
    print("=" * 60)
    print("JSON OBJECT CONSTRAINT")
    print("=" * 60)

    try:
        completion = client.completions.create(
            prompt="Create a JSON object describing a book:",
            max_tokens=100,
            temperature=0.7,
            structured_outputs=StructuredOutputs(json_object=True),
        )
        text = decoder.decode_completion(completion)
        print(f"JSON Object result: {text}")
    except Exception as e:
        print(f"JSON Object not supported or error: {e}")
    print()

    # =========================================================================
    # 3. Regex Pattern Constraint
    # =========================================================================
    # Force output to match a regex pattern
    print("=" * 60)
    print("REGEX CONSTRAINT")
    print("=" * 60)

    # Match a phone number pattern: (XXX) XXX-XXXX
    phone_regex = r"\(\d{3}\) \d{3}-\d{4}"

    try:
        completion = client.completions.create(
            prompt="Generate a US phone number:",
            max_tokens=20,
            temperature=0.0,
            structured_outputs=StructuredOutputs(regex=phone_regex),
        )
        text = decoder.decode_completion(completion)
        print(f"Regex pattern: {phone_regex}")
        print(f"Regex result: {text}")
    except Exception as e:
        print(f"Regex not supported or error: {e}")
    print()

    # Match an email pattern
    email_regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

    try:
        completion = client.completions.create(
            prompt="Generate an email address:",
            max_tokens=30,
            temperature=0.0,
            structured_outputs=StructuredOutputs(regex=email_regex),
        )
        text = decoder.decode_completion(completion)
        print(f"Email regex: {email_regex}")
        print(f"Email result: {text}")
    except Exception as e:
        print(f"Email regex not supported or error: {e}")
    print()

    # =========================================================================
    # 4. Choice Constraint
    # =========================================================================
    # Force output to be one of the specified choices
    print("=" * 60)
    print("CHOICE CONSTRAINT")
    print("=" * 60)

    choices = ["positive", "negative", "neutral"]

    try:
        completion = client.completions.create(
            prompt="Classify the sentiment of 'I love this product!':",
            max_tokens=10,
            temperature=0.0,
            structured_outputs=StructuredOutputs(
                choice=ChoiceConstraint(choices=choices)
            ),
        )
        text = decoder.decode_completion(completion)
        print(f"Choices: {choices}")
        print(f"Choice result: {text}")
    except Exception as e:
        print(f"Choice constraint not supported or error: {e}")
    print()

    # =========================================================================
    # 5. Grammar (EBNF) Constraint
    # =========================================================================
    # Force output to match a grammar
    print("=" * 60)
    print("GRAMMAR (EBNF) CONSTRAINT")
    print("=" * 60)

    # Simple arithmetic expression grammar
    grammar = """
    root ::= expr
    expr ::= term (('+' | '-') term)*
    term ::= factor (('*' | '/') factor)*
    factor ::= number | '(' expr ')'
    number ::= [0-9]+
    """

    try:
        completion = client.completions.create(
            prompt="Generate a simple math expression:",
            max_tokens=20,
            temperature=0.0,
            structured_outputs=StructuredOutputs(grammar=grammar),
        )
        text = decoder.decode_completion(completion)
        print(f"Grammar result: {text}")
    except Exception as e:
        print(f"Grammar not supported or error: {e}")
    print()

    # =========================================================================
    # 6. Combining with Sampling Parameters
    # =========================================================================
    print("=" * 60)
    print("STRUCTURED OUTPUT WITH SAMPLING PARAMETERS")
    print("=" * 60)

    try:
        # Structured output with temperature for variety
        for i in range(3):
            completion = client.completions.create(
                prompt="Generate a color name (one word):",
                max_tokens=10,
                temperature=1.0,  # Add randomness
                structured_outputs=StructuredOutputs(
                    choice=ChoiceConstraint(
                        choices=["red", "blue", "green", "yellow", "purple", "orange"]
                    )
                ),
            )
            text = decoder.decode_completion(completion)
            print(f"  Run {i+1}: {text}")
    except Exception as e:
        print(f"Error: {e}")
    print()

    # =========================================================================
    # Cleanup
    # =========================================================================
    client.close()
    print("Done!")


if __name__ == "__main__":
    main()
