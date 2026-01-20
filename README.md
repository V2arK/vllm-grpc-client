# vLLM gRPC Client

A Python gRPC client for vLLM with an OpenAI-style interface.

This package provides a high-level Python client for interacting with vLLM's gRPC server. The interface is designed to be similar to the OpenAI Python client for familiarity and ease of use.

## Features

- **OpenAI-style API**: Familiar interface for users of the OpenAI Python client
- **Sync and Async support**: Both synchronous and asynchronous clients
- **Streaming support**: Stream generation responses in real-time
- **Type-safe**: Full type hints with Pydantic models
- **Text decoding**: Built-in utilities to decode token IDs to text
- **Production-ready**: Supports all vLLM gRPC features including:
  - Text generation (streaming and non-streaming)
  - Health checks
  - Model and server info
  - Request abortion
  - Structured outputs (JSON schema, regex, grammar)
  - Sampling parameters (temperature, top-p, top-k, etc.)

## Installation

```bash
pip install vllm-grpc-client

# For text decoding support
pip install vllm-grpc-client transformers
```

Or install from source:

```bash
git clone https://github.com/your-repo/vllm-grpc-client.git
cd vllm-grpc-client
pip install -e .
```

## Quick Start

### Synchronous Client

```python
from vllm_grpc_client import VLLMGrpcClient

# Create client
client = VLLMGrpcClient(host="localhost", port=9000)

# Generate text
completion = client.completions.create(
    prompt="The capital of France is",
    max_tokens=50,
    temperature=0.7,
)

print(completion.choices[0].token_ids)
print(f"Usage: {completion.usage}")

# Streaming
stream = client.completions.create(
    prompt="Count from 1 to 5:",
    max_tokens=30,
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta_token_ids)

# Close client
client.close()
```

### Asynchronous Client

```python
import asyncio
from vllm_grpc_client import AsyncVLLMGrpcClient

async def main():
    async with AsyncVLLMGrpcClient(host="localhost", port=9000) as client:
        # Generate text
        completion = await client.completions.create(
            prompt="Hello, world!",
            max_tokens=20,
        )
        print(completion.choices[0].token_ids)

        # Streaming
        stream = await client.completions.create(
            prompt="List three colors:",
            max_tokens=30,
            stream=True,
        )

        async for chunk in stream:
            print(chunk.choices[0].delta_token_ids)

asyncio.run(main())
```

### Context Manager

```python
# Sync
with VLLMGrpcClient(host="localhost", port=9000) as client:
    result = client.completions.create(prompt="Hello", max_tokens=10)

# Async
async with AsyncVLLMGrpcClient(host="localhost", port=9000) as client:
    result = await client.completions.create(prompt="Hello", max_tokens=10)
```

### Text Decoding

The vLLM gRPC server returns **token IDs only**. To get plain text, use the `TokenDecoder`:

```python
from vllm_grpc_client import VLLMGrpcClient, TokenDecoder

client = VLLMGrpcClient(host="localhost", port=9000)

# Create decoder (automatically downloads the tokenizer)
decoder = TokenDecoder.from_client(client)

# Non-streaming with text
completion = client.completions.create(
    prompt="The capital of France is",
    max_tokens=50,
)
text = decoder.decode_completion(completion)
print(f"Generated text: {text}")

# Streaming with text
stream = client.completions.create(
    prompt="Count from 1 to 5:",
    max_tokens=30,
    stream=True,
)

for chunk in stream:
    delta_text = decoder.decode_chunk(chunk)
    print(delta_text, end="", flush=True)

client.close()
```

**Note**: Text decoding requires the `transformers` library:
```bash
pip install transformers
```

## API Reference

### Client Classes

- `VLLMGrpcClient`: Synchronous client
- `AsyncVLLMGrpcClient`: Asynchronous client

Both support the following resources:

- `client.completions`: Text generation
- `client.embeddings`: Embeddings (not yet implemented in vLLM gRPC)
- `client.models`: Model information
- `client.health`: Health checks, server info, and request abortion

### Completions

```python
completion = client.completions.create(
    prompt="Hello, world!",           # Text prompt or list of token IDs
    max_tokens=100,                   # Maximum tokens to generate
    temperature=0.7,                  # Sampling temperature
    top_p=0.95,                       # Top-p nucleus sampling
    top_k=50,                         # Top-k sampling
    stream=False,                     # Enable streaming
    stop=[".", "\n"],                 # Stop sequences
    seed=42,                          # Random seed for reproducibility
    structured_outputs=StructuredOutputs(
        json_schema='{"type": "object"}'
    ),
)
```

### Health & Server Info

```python
# Health check
health = client.health.check()
print(health.healthy)

# Server info
info = client.health.server_info()
print(f"Active requests: {info.active_requests}")
print(f"Uptime: {info.uptime_seconds}s")

# Abort requests
client.health.abort(["request-id-1", "request-id-2"])
```

### Model Info

```python
model = client.models.retrieve()
print(f"Model: {model.model_path}")
print(f"Max context: {model.max_context_length}")
print(f"Vocab size: {model.vocab_size}")
```

## Environment Variables

- `VLLM_GRPC_HOST`: Default gRPC server host (default: "localhost")
- `VLLM_GRPC_PORT`: Default gRPC server port (default: 9000)
- `VLLM_GRPC_TIMEOUT`: Default request timeout in seconds (default: 60.0)

## Architecture

The client is designed to be easy to maintain when vLLM's gRPC interface changes:

```
vllm_grpc_client/
├── proto/                    # Protocol buffer definitions
│   ├── vllm_engine.proto     # Copy from vLLM (keep in sync)
│   ├── vllm_engine_pb2.py    # Generated Python code
│   └── vllm_engine_pb2_grpc.py
├── resources/                # API resources (OpenAI-style)
│   ├── completions.py        # Generate RPC
│   ├── embeddings.py         # Embed RPC (placeholder)
│   ├── models.py             # GetModelInfo RPC
│   └── health.py             # HealthCheck, ServerInfo, Abort RPCs
├── _client.py                # Main client classes
├── _types.py                 # Pydantic models for responses
├── _streaming.py             # Streaming response handlers
└── _exceptions.py            # Custom exceptions
```

### Updating for New vLLM Versions

1. Copy the updated `vllm_engine.proto` from vLLM
2. Regenerate Python stubs:
   ```bash
   python -m grpc_tools.protoc -I=src/vllm_grpc_client/proto \
       --python_out=src/vllm_grpc_client/proto \
       --grpc_python_out=src/vllm_grpc_client/proto \
       --pyi_out=src/vllm_grpc_client/proto \
       src/vllm_grpc_client/proto/vllm_engine.proto
   ```
3. Fix the import in `vllm_engine_pb2_grpc.py`
4. Update resource classes if needed

## Limitations

- **Token IDs only**: The gRPC interface returns token IDs by default. Use `TokenDecoder` to convert to text (requires `transformers` library)
- **Embeddings**: The Embed RPC is not yet implemented in vLLM gRPC server
- **Logprobs**: Logprobs support is marked as TODO in vLLM gRPC

## License

Apache-2.0
