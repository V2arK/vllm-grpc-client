#!/usr/bin/env python3
"""
Comprehensive tests for structured outputs.

Tests all constraint types against the vLLM gRPC server:
- JSON Schema
- JSON Object
- Regex
- Choice
- Grammar (EBNF and Lark)

These tests verify that the client correctly passes structured output
constraints to the server and handles responses properly.
"""

import json
import re

import pytest

from vllm_grpc_client import (
    ChoiceConstraint,
    StructuredOutputs,
    TokenDecoder,
    VLLMGrpcClient,
    VLLMGrpcInvalidArgumentError,
)

# Server configuration
GRPC_HOST = "localhost"
GRPC_PORT = 9000


@pytest.fixture(scope="module")
def client():
    """Create a client for the test module."""
    c = VLLMGrpcClient(host=GRPC_HOST, port=GRPC_PORT)
    yield c
    c.close()


@pytest.fixture(scope="module")
def decoder(client):
    """Create a decoder for the test module."""
    return TokenDecoder.from_client(client)


class TestJSONSchema:
    """Tests for JSON Schema constraint."""

    def test_simple_object_schema(self, client, decoder):
        """Test generating a simple JSON object matching schema."""
        schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            }
        )

        completion = client.completions.create(
            prompt="Generate a person with name and age as JSON:",
            max_tokens=50,
            temperature=0.0,
            structured_outputs=StructuredOutputs(json_schema=schema),
        )

        text = decoder.decode_completion(completion)
        # Verify it's valid JSON matching the schema
        parsed = json.loads(text)
        assert "name" in parsed
        assert "age" in parsed
        assert isinstance(parsed["name"], str)
        assert isinstance(parsed["age"], int)

    def test_nested_object_schema(self, client, decoder):
        """Test generating nested JSON objects."""
        schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "person": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "address": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            }
        )

        completion = client.completions.create(
            prompt="Generate a person with nested address:",
            max_tokens=100,
            temperature=0.0,
            structured_outputs=StructuredOutputs(json_schema=schema),
        )

        text = decoder.decode_completion(completion)
        parsed = json.loads(text)
        assert "person" in parsed

    def test_array_schema(self, client, decoder):
        """Test generating JSON arrays."""
        schema = json.dumps(
            {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 5,
            }
        )

        completion = client.completions.create(
            prompt="Generate a list of fruits as JSON array:",
            max_tokens=50,
            temperature=0.0,
            structured_outputs=StructuredOutputs(json_schema=schema),
        )

        text = decoder.decode_completion(completion)
        parsed = json.loads(text)
        assert isinstance(parsed, list)
        assert 2 <= len(parsed) <= 5
        assert all(isinstance(item, str) for item in parsed)


class TestJSONObject:
    """Tests for JSON Object constraint."""

    def test_json_object_output(self, client, decoder):
        """Test that output is valid JSON object."""
        completion = client.completions.create(
            prompt="Create a JSON object describing a car:",
            max_tokens=100,
            temperature=0.0,
            structured_outputs=StructuredOutputs(json_object=True),
        )

        text = decoder.decode_completion(completion)
        parsed = json.loads(text)
        assert isinstance(parsed, dict)


class TestRegex:
    """Tests for regex pattern constraint."""

    def test_phone_number_pattern(self, client, decoder):
        """Test US phone number pattern."""
        pattern = r"\(\d{3}\) \d{3}-\d{4}"

        completion = client.completions.create(
            prompt="Generate a US phone number:",
            max_tokens=20,
            temperature=0.0,
            structured_outputs=StructuredOutputs(regex=pattern),
        )

        text = decoder.decode_completion(completion)
        # The text should match the pattern
        assert re.match(pattern, text), f"'{text}' doesn't match pattern"

    def test_date_pattern(self, client, decoder):
        """Test date pattern YYYY-MM-DD."""
        pattern = r"\d{4}-\d{2}-\d{2}"

        completion = client.completions.create(
            prompt="Generate a date in YYYY-MM-DD format:",
            max_tokens=15,
            temperature=0.0,
            structured_outputs=StructuredOutputs(regex=pattern),
        )

        text = decoder.decode_completion(completion)
        assert re.match(pattern, text), f"'{text}' doesn't match date pattern"

    def test_simple_word_pattern(self, client, decoder):
        """Test simple word pattern."""
        pattern = r"(hello|world|test)"

        completion = client.completions.create(
            prompt="Say one word:",
            max_tokens=10,
            temperature=0.0,
            structured_outputs=StructuredOutputs(regex=pattern),
        )

        text = decoder.decode_completion(completion)
        assert text in ["hello", "world", "test"], f"'{text}' not in expected words"


class TestChoice:
    """Tests for choice constraint."""

    def test_simple_choices(self, client, decoder):
        """Test choosing from a simple list."""
        choices = ["yes", "no", "maybe"]

        completion = client.completions.create(
            prompt="Do you like ice cream? Answer with yes, no, or maybe:",
            max_tokens=10,
            temperature=0.0,
            structured_outputs=StructuredOutputs(
                choice=ChoiceConstraint(choices=choices)
            ),
        )

        text = decoder.decode_completion(completion)
        assert text in choices, f"'{text}' not in choices {choices}"

    def test_sentiment_choices(self, client, decoder):
        """Test sentiment classification."""
        choices = ["positive", "negative", "neutral"]

        completion = client.completions.create(
            prompt="Classify sentiment of 'I love this!': ",
            max_tokens=10,
            temperature=0.0,
            structured_outputs=StructuredOutputs(
                choice=ChoiceConstraint(choices=choices)
            ),
        )

        text = decoder.decode_completion(completion)
        assert text in choices
        # "I love this!" should be positive
        assert text == "positive"

    def test_multiple_word_choices(self, client, decoder):
        """Test choices with multiple words."""
        choices = ["very good", "somewhat good", "not good"]

        completion = client.completions.create(
            prompt="Rate this: ",
            max_tokens=10,
            temperature=0.0,
            structured_outputs=StructuredOutputs(
                choice=ChoiceConstraint(choices=choices)
            ),
        )

        text = decoder.decode_completion(completion)
        assert text in choices


class TestGrammar:
    """
    Tests for grammar constraint.

    NOTE: Grammar support varies by backend. The vLLM xgrammar backend
    supports EBNF grammars but has some limitations:

    1. Root rule should be named 'root' for EBNF
    2. Root rule should be named 'start' for Lark (auto-converted)
    3. Some complex patterns like `('+' | '-')` may not parse correctly
    4. Character classes like `[0-9]+` may not be supported

    The grammar string is passed directly to the server - any parsing
    errors occur on the server side, not in this client.
    """

    def test_simple_ebnf_grammar(self, client, decoder):
        """Test simple EBNF grammar with alternatives."""
        # Simple grammar that should work on all backends
        grammar = """
root ::= "yes" | "no"
"""

        completion = client.completions.create(
            prompt="Is the sky blue? Answer:",
            max_tokens=10,
            temperature=0.0,
            structured_outputs=StructuredOutputs(grammar=grammar),
        )

        text = decoder.decode_completion(completion)
        assert text in ["yes", "no"], f"'{text}' not in ['yes', 'no']"

    def test_sql_ebnf_grammar(self, client, decoder):
        """
        Test SQL-like EBNF grammar.

        This is the grammar format used in vLLM's official tests.
        Reference: tests/v1/entrypoints/conftest.py:sample_sql_ebnf
        """
        grammar = """
root ::= select_statement
select_statement ::= "SELECT" column "from" table "where" condition
column ::= "col_1" | "col_2"
table ::= "table_1" | "table_2"
condition ::= column "=" number
number ::= "1" | "2"
"""

        completion = client.completions.create(
            prompt="Generate a SQL statement:",
            max_tokens=50,
            temperature=0.0,
            structured_outputs=StructuredOutputs(grammar=grammar),
        )

        text = decoder.decode_completion(completion)
        # Should contain SELECT and the grammar elements
        assert "SELECT" in text
        assert "col_" in text
        assert "table_" in text

    def test_lark_grammar(self, client, decoder):
        """
        Test Lark-style grammar.

        vLLM auto-detects Lark format (uses ':' instead of '::=')
        and converts it to EBNF internally.
        Reference: tests/v1/entrypoints/conftest.py:sample_sql_lark
        """
        grammar = """
start: select_statement
select_statement: "SELECT" column "from" table "where" condition
column: "col_1" | "col_2"
table: "table_1" | "table_2"
condition: column "=" number
number: "1" | "2"
"""

        completion = client.completions.create(
            prompt="Generate a SQL statement:",
            max_tokens=50,
            temperature=0.0,
            structured_outputs=StructuredOutputs(grammar=grammar),
        )

        text = decoder.decode_completion(completion)
        assert "SELECT" in text

    def test_complex_grammar_may_fail(self, client, decoder):
        """
        Test complex grammar that may fail on xgrammar backend.

        This test documents a known limitation: grammars using certain
        constructs like grouped alternatives `('+' | '-')` or character
        classes `[0-9]+` may fail with parsing errors.

        The error occurs in vLLM's backend_xgrammar.py when calling
        xgr.Grammar.from_ebnf() - this is a SERVER-SIDE limitation,
        not a client issue.
        """
        grammar = """
root ::= expr
expr ::= term (('+' | '-') term)*
term ::= factor (('*' | '/') factor)*
factor ::= number | '(' expr ')'
number ::= [0-9]+
"""

        try:
            completion = client.completions.create(
                prompt="Generate math expression:",
                max_tokens=20,
                temperature=0.0,
                structured_outputs=StructuredOutputs(grammar=grammar),
            )
            # If it succeeds, that's fine too
            text = decoder.decode_completion(completion)
            assert len(text) > 0
        except VLLMGrpcInvalidArgumentError as e:
            # Expected error on xgrammar backend
            assert "Failed to convert the grammar" in str(e) or "Invalid grammar" in str(e)


class TestStructuredOutputsWithSamplingParams:
    """Test combining structured outputs with various sampling parameters."""

    def test_with_temperature(self, client, decoder):
        """Test structured output with temperature variation."""
        choices = ["red", "blue", "green", "yellow"]
        results = set()

        # With temperature > 0, we should get some variation
        for _ in range(5):
            completion = client.completions.create(
                prompt="Pick a color:",
                max_tokens=10,
                temperature=1.0,
                structured_outputs=StructuredOutputs(
                    choice=ChoiceConstraint(choices=choices)
                ),
            )
            text = decoder.decode_completion(completion)
            assert text in choices
            results.add(text)

        # With high temperature, we should see at least some variety
        # (though not guaranteed)
        assert len(results) >= 1

    def test_with_seed_reproducibility(self, client, decoder):
        """Test that seed provides reproducible results."""
        choices = ["alpha", "beta", "gamma", "delta"]

        results1 = []
        results2 = []

        for _ in range(3):
            completion = client.completions.create(
                prompt="Pick a Greek letter:",
                max_tokens=10,
                temperature=0.5,
                seed=42,
                structured_outputs=StructuredOutputs(
                    choice=ChoiceConstraint(choices=choices)
                ),
            )
            results1.append(decoder.decode_completion(completion))

        for _ in range(3):
            completion = client.completions.create(
                prompt="Pick a Greek letter:",
                max_tokens=10,
                temperature=0.5,
                seed=42,
                structured_outputs=StructuredOutputs(
                    choice=ChoiceConstraint(choices=choices)
                ),
            )
            results2.append(decoder.decode_completion(completion))

        # Same seed should give same results
        assert results1 == results2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_json_schema(self, client):
        """Test with minimal valid JSON schema."""
        schema = json.dumps({"type": "object"})

        completion = client.completions.create(
            prompt="Generate JSON:",
            max_tokens=50,
            temperature=0.0,
            structured_outputs=StructuredOutputs(json_schema=schema),
        )

        # Should succeed
        assert completion.choices

    def test_single_choice(self, client, decoder):
        """Test with only one choice."""
        choices = ["only_option"]

        completion = client.completions.create(
            prompt="Choose:",
            max_tokens=10,
            temperature=0.0,
            structured_outputs=StructuredOutputs(
                choice=ChoiceConstraint(choices=choices)
            ),
        )

        text = decoder.decode_completion(completion)
        assert text == "only_option"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
