#!/usr/bin/env python3
"""
List available models from Draw Things gRPC server.

Usage:
    uv run python model_list.py [--server HOST] [--port PORT]
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../draw-things-comfyui/src"))

import grpc
import json
import base64
from google.protobuf.json_format import MessageToJson
from generated import imageService_pb2, imageService_pb2_grpc


def list_models(server: str = "localhost", port: int = 7859, use_tls: bool = False):
    """Get list of available models from Draw Things gRPC server."""

    options = [
        ["grpc.max_send_message_length", -1],
        ["grpc.max_receive_message_length", -1],
    ]

    if use_tls:
        channel = grpc.secure_channel(
            f"{server}:{port}", grpc.ssl_channel_credentials(), options=options
        )
    else:
        channel = grpc.insecure_channel(f"{server}:{port}", options=options)

    stub = imageService_pb2_grpc.ImageGenerationServiceStub(channel)

    print(f"Connecting to {server}:{port}...")

    request = imageService_pb2.EchoRequest(name="grpc-model-list")
    response = stub.Echo(request)

    response_json = json.loads(MessageToJson(response))

    print(f"\n=== Draw Things Server Info ===")
    print(f"Message: {response.message}")
    print(f"Server ID: {response.serverIdentifier}")
    print(f"Shared Secret Missing: {response.sharedSecretMissing}")

    files = list(response.files)
    print(f"\n=== Available Models ({len(files)}) ===")
    for f in sorted(files):
        print(f"  {f}")

    if response.override:
        override = dict(response.override)
        if override:
            print(f"\n=== Model Metadata ===")
            for key, b64_value in sorted(override.items()):
                try:
                    value = json.loads(str(base64.b64decode(b64_value), "utf8"))
                    print(f"  {key}: {value}")
                except Exception:
                    print(f"  {key}: <decode error>")

    if response.HasField("thresholds"):
        t = response.thresholds
        print(f"\n=== Compute Unit Thresholds ===")
        print(f"  Max Steps: {t.maxSteps}")
        print(f"  Max Resolution: {t.maxResolution}")

    channel.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="List models from Draw Things gRPC server")
    parser.add_argument("--server", default="localhost", help="gRPC server address")
    parser.add_argument("--port", type=int, default=7859, help="gRPC server port")
    parser.add_argument("--tls", action="store_true", help="Use TLS")

    args = parser.parse_args()

    try:
        list_models(server=args.server, port=args.port, use_tls=args.tls)
    except grpc.RpcError as e:
        print(f"[Error] gRPC error: {e.code()}: {e.details()}")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
