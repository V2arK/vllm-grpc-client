# vLLM gRPC Client Examples

This directory contains comprehensive examples demonstrating all features of the vLLM gRPC client.

## Prerequisites

```bash
# Install the client
pip install -e ..

# For text decoding examples
pip install transformers
```

## Starting a vLLM gRPC Server

Before running the examples, you need a running vLLM gRPC server. Start one locally:

```bash
# Start vLLM with gRPC enabled
python -m vllm.entrypoints.grpc_server \
    --model <your-model-name> \
    --host 0.0.0.0 \
    --port 9000

# Example with a specific model:
python -m vllm.entrypoints.grpc_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 9000
```

Alternatively, if running vLLM via Docker or Kubernetes, ensure the gRPC port (default: 9000) is exposed.

## Server Configuration

All examples use these default settings:
- Host: `localhost`
- Port: `9000`

Modify the `GRPC_HOST` and `GRPC_PORT` variables in each example to connect to a remote server.

## Examples

### [01_basic_completion.py](01_basic_completion.py)
**Basic Text Completion**

Fundamental usage of the client including:
- Creating a client connection
- Non-streaming text completion
- Streaming text completion
- Using context managers

```bash
python examples/01_basic_completion.py
```

### [02_text_decoding.py](02_text_decoding.py)
**Decoding Token IDs to Text**

How to convert token IDs back to readable text:
- Creating a TokenDecoder
- Decoding non-streaming completions
- Real-time streaming text decoding
- Manual encoding/decoding

```bash
python examples/02_text_decoding.py
```

### [03_sampling_parameters.py](03_sampling_parameters.py)
**Sampling Parameters**

All available parameters for controlling generation:
- Temperature, top-p, top-k, min-p
- Frequency and presence penalties
- Repetition penalty
- Stop sequences
- Seed for reproducibility
- Logit bias

```bash
python examples/03_sampling_parameters.py
```

### [04_structured_outputs.py](04_structured_outputs.py)
**Structured Outputs**

Constraining output format:
- JSON Schema constraint
- JSON Object constraint
- Regex pattern constraint
- Choice constraint
- Grammar (EBNF) constraint

```bash
python examples/04_structured_outputs.py
```

### [05_health_and_server_info.py](05_health_and_server_info.py)
**Health Checks and Server Information**

Monitoring the server:
- Health checks
- Server information
- Model information
- Waiting for server readiness

```bash
python examples/05_health_and_server_info.py
```

### [06_abort_requests.py](06_abort_requests.py)
**Aborting Requests**

Canceling running requests:
- Basic abort
- Abort with threading
- Aborting multiple requests
- Handling abort completion

```bash
python examples/06_abort_requests.py
```

### [07_async_client.py](07_async_client.py)
**Asynchronous Client**

High-performance async operations:
- Async client usage
- Async streaming
- Concurrent completions with asyncio.gather
- Async context manager

```bash
python examples/07_async_client.py
```

### [08_error_handling.py](08_error_handling.py)
**Error Handling**

Comprehensive error handling:
- Connection errors
- Timeout errors
- Invalid argument errors
- Unimplemented errors
- Retry patterns

```bash
python examples/08_error_handling.py
```

### [09_tokenized_input.py](09_tokenized_input.py)
**Using Tokenized Input**

Pre-tokenized input handling:
- Sending token IDs directly
- Using TokenizedInput object
- Working with special tokens

```bash
python examples/09_tokenized_input.py
```

## Quick Start

```python
from vllm_grpc_client import VLLMGrpcClient, TokenDecoder

# Connect to server
client = VLLMGrpcClient(host="localhost", port=9000)

# Create decoder for text output
decoder = TokenDecoder.from_client(client)

# Generate text
completion = client.completions.create(
    prompt="Hello, world!",
    max_tokens=50,
    temperature=0.7,
)

# Get readable text
text = decoder.decode_completion(completion)
print(text)

# Cleanup
client.close()
```

## Running All Examples

```bash
# Run all examples in sequence
for f in examples/0*.py; do
    echo "Running $f..."
    python "$f"
    echo ""
done
```

## Notes

- The vLLM gRPC server returns **token IDs only**. Use `TokenDecoder` for text output.
- **Embeddings** are not yet implemented in vLLM gRPC server.
- **Logprobs** support is marked as TODO in vLLM gRPC.
- Structured outputs support depends on the model capabilities.
