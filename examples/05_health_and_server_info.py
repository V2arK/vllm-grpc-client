#!/usr/bin/env python3
"""
Example 05: Health Checks and Server Information

This example demonstrates how to monitor the vLLM gRPC server's health
and retrieve server/model information.

Features covered:
- Health check
- Server information
- Model information
- Checking server readiness

Usage:
    python examples/05_health_and_server_info.py
"""

import time

from vllm_grpc_client import VLLMGrpcClient

# Server configuration
GRPC_HOST = "10.28.115.40"
GRPC_PORT = 9000


def main():
    client = VLLMGrpcClient(host=GRPC_HOST, port=GRPC_PORT)
    print(f"Connected to {GRPC_HOST}:{GRPC_PORT}\n")

    # =========================================================================
    # 1. Health Check
    # =========================================================================
    # Check if the server is healthy and ready to accept requests
    print("=" * 60)
    print("HEALTH CHECK")
    print("=" * 60)

    health = client.health.check()

    print(f"Healthy: {health.healthy}")
    print(f"Message: {health.message}")
    print()

    # You can also use the convenience method
    is_healthy = client.is_healthy()
    print(f"Quick health check: {is_healthy}")
    print()

    # =========================================================================
    # 2. Server Information
    # =========================================================================
    # Get detailed server status and statistics
    print("=" * 60)
    print("SERVER INFORMATION")
    print("=" * 60)

    server_info = client.health.server_info()

    print(f"Server type: {server_info.server_type}")
    print(f"Active requests: {server_info.active_requests}")
    print(f"Is paused: {server_info.is_paused}")
    print(f"Uptime: {server_info.uptime_seconds:.2f} seconds")
    print(f"  ({server_info.uptime_seconds / 3600:.2f} hours)")
    print(f"Last receive timestamp: {server_info.last_receive_timestamp}")
    print()

    # =========================================================================
    # 3. Model Information
    # =========================================================================
    # Get information about the loaded model
    print("=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)

    model_info = client.models.retrieve()

    print(f"Model path: {model_info.model_path}")
    print(f"Is generation model: {model_info.is_generation}")
    print(f"Max context length: {model_info.max_context_length:,} tokens")
    print(f"Vocabulary size: {model_info.vocab_size:,} tokens")
    print(f"Supports vision: {model_info.supports_vision}")
    print()

    # Alternative: use list() method (same as retrieve() since vLLM only has one model)
    model_info_list = client.models.list()
    print(f"Model via list(): {model_info_list.model_path}")
    print()

    # =========================================================================
    # 4. Wait for Server Ready
    # =========================================================================
    # Useful for startup scripts that need to wait for the server
    print("=" * 60)
    print("WAIT FOR SERVER READY")
    print("=" * 60)

    print("Simulating wait_for_ready() (server already running)...")
    start_time = time.time()
    is_ready = client.wait_for_ready(timeout=5.0, poll_interval=0.5)
    elapsed = time.time() - start_time

    print(f"Server ready: {is_ready}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print()

    # =========================================================================
    # 5. Continuous Health Monitoring
    # =========================================================================
    print("=" * 60)
    print("CONTINUOUS HEALTH MONITORING (5 checks)")
    print("=" * 60)

    for i in range(5):
        health = client.health.check()
        server_info = client.health.server_info()
        print(
            f"  Check {i+1}: healthy={health.healthy}, "
            f"active_requests={server_info.active_requests}"
        )
        time.sleep(0.5)
    print()

    # =========================================================================
    # 6. Error Handling for Unhealthy Server
    # =========================================================================
    print("=" * 60)
    print("ERROR HANDLING")
    print("=" * 60)

    # Example of handling connection to non-existent server
    try:
        bad_client = VLLMGrpcClient(host="non-existent-host", port=9999)
        # This will fail when we try to make a request
        bad_client.health.check(timeout=2.0)
    except Exception as e:
        print(f"Expected error connecting to bad host: {type(e).__name__}")

    # Check if our good client is still healthy
    print(f"Original client still healthy: {client.is_healthy()}")
    print()

    # =========================================================================
    # Cleanup
    # =========================================================================
    client.close()
    print("Done!")


if __name__ == "__main__":
    main()
