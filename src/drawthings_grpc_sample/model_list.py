#!/usr/bin/env python3
"""
List available models from Draw Things gRPC server.

Usage:
    uv run python model_list.py [--server HOST] [--port PORT]
"""

import sys

import grpc
import json
import base64
from google.protobuf.json_format import MessageToJson
from drawthings_grpc_sample.generated import imageService_pb2, imageService_pb2_grpc


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

    # Parse override for typed model lists
    all_items = []  # (type, name, info)

    if response.HasField("override") and response.override:
        override_json = json.loads(MessageToJson(response.override))
        for key, value in override_json.items():
            if isinstance(value, str) and value:
                try:
                    items = json.loads(str(base64.b64decode(value), "utf8"))
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                name = item.get("name", "")
                                file = item.get("file", "")
                                if name and file and name != file:
                                    display = f"{name} ({file})"
                                else:
                                    display = name or file or "unknown"
                                all_items.append((key, display, item))
                            else:
                                all_items.append((key, str(item), None))
                except Exception:
                    pass

    # Sort by type, then by name
    all_items.sort(key=lambda x: (x[0].lower(), x[1].lower()))

    print(f"\n=== Available Models ({len(all_items)}) ===")

    current_type = None
    for item_type, name, info in all_items:
        if item_type != current_type:
            print(f"\n--- {item_type.upper()} ---")
            current_type = item_type
        print(f"  {name}")

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
