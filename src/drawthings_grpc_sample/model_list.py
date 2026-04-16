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
from typing import Any
from google.protobuf.json_format import MessageToJson
from drawthings_grpc_sample.defaults import (
    DEFAULT_GRPC_COMPRESSION,
    DEFAULT_PORT,
    DEFAULT_SERVER,
    DEFAULT_TLS_CA_FILE,
    DEFAULT_USE_TLS,
)
from drawthings_grpc_sample.generated import imageService_pb2, imageService_pb2_grpc
from drawthings_grpc_sample.samplers import get_sampler_names
from drawthings_grpc_sample.tls import create_channel


def _decode_override_lists(response: imageService_pb2.EchoReply) -> dict[str, list[Any]]:
    """Decode base64 JSON arrays from Echo.override into Python lists by key."""

    parsed: dict[str, list[Any]] = {}
    if not response.HasField("override") or not response.override:
        return parsed

    override_json = json.loads(MessageToJson(response.override))
    for key, value in override_json.items():
        if not isinstance(value, str) or not value:
            continue
        try:
            items = json.loads(str(base64.b64decode(value), "utf8"))
        except Exception:
            continue
        if isinstance(items, list):
            parsed[key] = items

    return parsed


def get_available_model_files(
    server: str = DEFAULT_SERVER,
    port: int = DEFAULT_PORT,
    use_tls: bool = DEFAULT_USE_TLS,
    tls_ca_file: str | None = DEFAULT_TLS_CA_FILE,
    use_compression: bool = DEFAULT_GRPC_COMPRESSION,
) -> list[str]:
    """Return sorted model filenames from the Draw Things server's models list."""

    channel = create_channel(
        server,
        port,
        use_tls,
        ca_cert_file=tls_ca_file,
        use_compression=use_compression,
    )
    try:
        stub = imageService_pb2_grpc.ImageGenerationServiceStub(channel)
        request = imageService_pb2.EchoRequest(name="grpc-model-check")
        response = stub.Echo(request)

        decoded = _decode_override_lists(response)
        model_entries = decoded.get("models", [])

        files: set[str] = set()
        for item in model_entries:
            if isinstance(item, dict):
                model_file = item.get("file") or item.get("name")
                if model_file:
                    files.add(str(model_file))
            elif isinstance(item, str) and item:
                files.add(item)

        return sorted(files)
    finally:
        channel.close()


def list_models(
    server: str = DEFAULT_SERVER,
    port: int = DEFAULT_PORT,
    use_tls: bool = DEFAULT_USE_TLS,
    tls_ca_file: str | None = DEFAULT_TLS_CA_FILE,
    use_compression: bool = DEFAULT_GRPC_COMPRESSION,
):
    """Get list of available models from Draw Things gRPC server."""

    channel = create_channel(
        server,
        port,
        use_tls,
        ca_cert_file=tls_ca_file,
        use_compression=use_compression,
    )

    stub = imageService_pb2_grpc.ImageGenerationServiceStub(channel)

    print(f"Connecting to {server}:{port}...")

    request = imageService_pb2.EchoRequest(name="grpc-model-list")
    response = stub.Echo(request)

    print(f"\n=== Draw Things Server Info ===")
    print(f"Message: {response.message}")
    print(f"Server ID: {response.serverIdentifier}")
    print(f"Shared Secret Missing: {response.sharedSecretMissing}")

    # Parse override for typed model lists
    all_items = []  # (type, name, info)

    decoded_override = _decode_override_lists(response)
    for key, items in decoded_override.items():
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

    samplers = get_sampler_names()
    print(f"\n=== Available Samplers ({len(samplers)}) ===")
    for index, sampler_name in enumerate(samplers):
        print(f"  {index:>2}: {sampler_name}")

    channel.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="List models from Draw Things gRPC server")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="gRPC server address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="gRPC server port")
    parser.set_defaults(tls=DEFAULT_USE_TLS)
    parser.add_argument("--tls", dest="tls", action="store_true", help="Use TLS (default)")
    parser.add_argument("--no-tls", dest="tls", action="store_false", help="Disable TLS and use plaintext")
    parser.add_argument(
        "--tls-ca-file",
        default=DEFAULT_TLS_CA_FILE,
        help="Path to PEM CA certificate used to verify TLS server certificate",
    )

    args = parser.parse_args()

    try:
        list_models(
            server=args.server,
            port=args.port,
            use_tls=args.tls,
            tls_ca_file=args.tls_ca_file,
            use_compression=DEFAULT_GRPC_COMPRESSION,
        )
    except grpc.RpcError as e:
        print(f"[Error] gRPC error: {e.code()}: {e.details()}")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
