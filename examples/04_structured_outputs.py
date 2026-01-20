#!/usr/bin/env python3
"""
Example 04: Structured Outputs

This example demonstrates how to use structured output constraints to
guide the model to generate outputs in specific formats.

Features covered:
- JSON Schema constraint
- JSON Object constraint
- Regex pattern constraint
- Grammar (EBNF/Lark) constraint
- Choice constraint

Note: Not all models support all constraint types. The server may return
an error if a constraint is not supported.

IMPORTANT - Grammar Constraint Notes:
=====================================
vLLM supports two grammar formats depending on the backend:

1. EBNF (Extended Backus-Naur Form) - llama.cpp/xgrammar style:
   - Uses `::=` for rule definitions
   - Root rule should be named `root`
   - Example:
     ```
     root ::= "hello" | "world"
     ```
   - Reference: https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md

2. Lark Grammar Format:
   - Uses `:` for rule definitions
   - Root rule should be named `start`
   - Example:
     ```
     start: "hello" | "world"
     ```
   - Reference: https://lark-parser.readthedocs.io/en/latest/grammar.html

The vLLM server will attempt to auto-detect the format:
- If the grammar contains `::=`, it's treated as EBNF
- Otherwise, it's treated as Lark and converted to EBNF internally

Common Issues:
- Complex EBNF patterns like `('+' | '-')` may fail parsing
- Some character classes like `[0-9]+` may not be supported
- The server uses xgrammar which has specific syntax requirements

Usage:
    python examples/04_structured_outputs.py
"""

from vllm_grpc_client import (
    ChoiceConstraint,
    StructuredOutputs,
    TokenDecoder,
    VLLMGrpcClient,
)

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

    json_schema = """{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "city": {"type": "string"}
        },
        "required": ["name", "age", "city"]
    }"""

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
    # 5. Grammar Constraints
    # =========================================================================
    print("=" * 60)
    print("GRAMMAR CONSTRAINTS")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 5a. EBNF Grammar (llama.cpp / xgrammar style)
    # -------------------------------------------------------------------------
    # The vLLM xgrammar backend expects EBNF format with specific syntax.
    # Root rule should be named 'root'.
    #
    # NOTE: Complex patterns may fail. The xgrammar parser is strict about
    # certain syntax constructs. If you see errors like:
    #   "Failed to convert the grammar from GBNF to Lark"
    # it means the grammar syntax is not compatible with xgrammar.
    #
    # Working EBNF example (simple SQL-like grammar):
    print("\n--- EBNF Grammar (xgrammar compatible) ---")

    ebnf_grammar = """
root ::= select_statement
select_statement ::= "SELECT" column "from" table "where" condition
column ::= "col_1" | "col_2"
table ::= "table_1" | "table_2"
condition ::= column "=" number
number ::= "1" | "2"
"""

    try:
        completion = client.completions.create(
            prompt="Generate a SQL SELECT statement:",
            max_tokens=50,
            temperature=0.0,
            structured_outputs=StructuredOutputs(grammar=ebnf_grammar),
        )
        text = decoder.decode_completion(completion)
        print(f"EBNF Grammar result: {text}")
    except Exception as e:
        # Common error: xgrammar may reject certain EBNF syntax
        print(f"EBNF Grammar error: {e}")
    print()

    # -------------------------------------------------------------------------
    # 5b. Lark Grammar Format
    # -------------------------------------------------------------------------
    # vLLM can also accept Lark-style grammars, which use ':' instead of '::='
    # The server will auto-convert Lark to EBNF internally.
    # Root rule should be named 'start'.
    print("--- Lark Grammar (auto-converted to EBNF) ---")

    lark_grammar = """
start: select_statement
select_statement: "SELECT" column "from" table "where" condition
column: "col_1" | "col_2"
table: "table_1" | "table_2"
condition: column "=" number
number: "1" | "2"
"""

    try:
        completion = client.completions.create(
            prompt="Generate a SQL SELECT statement:",
            max_tokens=50,
            temperature=0.0,
            structured_outputs=StructuredOutputs(grammar=lark_grammar),
        )
        text = decoder.decode_completion(completion)
        print(f"Lark Grammar result: {text}")
    except Exception as e:
        print(f"Lark Grammar error: {e}")
    print()

    # -------------------------------------------------------------------------
    # 5c. Simple Yes/No Grammar
    # -------------------------------------------------------------------------
    # A very simple grammar that should work on most backends
    print("--- Simple Yes/No Grammar ---")

    simple_grammar = """
root ::= "yes" | "no"
"""

    try:
        completion = client.completions.create(
            prompt="Is the sky blue? Answer with yes or no:",
            max_tokens=10,
            temperature=0.0,
            structured_outputs=StructuredOutputs(grammar=simple_grammar),
        )
        text = decoder.decode_completion(completion)
        print(f"Simple Grammar result: {text}")
    except Exception as e:
        print(f"Simple Grammar error: {e}")
    print()

    # -------------------------------------------------------------------------
    # 5d. Known Issue: Complex Arithmetic Grammar
    # -------------------------------------------------------------------------
    # NOTE: The following grammar demonstrates a known compatibility issue
    # with the vLLM xgrammar backend. Grammars using certain constructs like
    # grouped alternatives `('+' | '-')` or character classes `[0-9]+` may
    # fail with parsing errors.
    #
    # Error example:
    #   "Failed to convert the grammar from GBNF to Lark: Expected ')' at line X"
    #
    # This is a SERVER-SIDE limitation, not a client issue. The client
    # correctly passes the grammar string to the server via gRPC.
    print("--- Complex Grammar (may fail on some backends) ---")

    # This grammar uses syntax that may not be fully supported
    complex_grammar = """
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
            structured_outputs=StructuredOutputs(grammar=complex_grammar),
        )
        text = decoder.decode_completion(completion)
        print(f"Complex Grammar result: {text}")
    except Exception as e:
        # Expected: This may fail due to xgrammar parsing limitations
        # The error originates from vLLM's backend_xgrammar.py which calls
        # xgr.Grammar.from_ebnf() - certain EBNF constructs are not supported
        print(f"Complex Grammar error (expected on xgrammar): {e}")
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
            print(f"  Run {i + 1}: {text}")
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
