"""
Simple Draw Things gRPC Image Generation Example

This script demonstrates how to call the Draw Things gRPC API to generate images.
It connects to a running Draw Things gRPC server and sends a GenerateImage request.

Requirements:
    - Draw Things gRPC server running (e.g., gRPCServerCLI)
    - A model file available on the server
"""

import os
import sys
import time
from datetime import datetime

import flatbuffers
import grpc
import numpy as np
from PIL import Image

# Add path to draw-things-comfyui generated code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "third_party/draw-things-comfyui"))

from generated import config_generated, imageService_pb2, imageService_pb2_grpc


def clamp(value):
    return max(min(int(value if np.isfinite(value) else 0), 255), 0)


def build_generation_config(
    model: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    seed: int = -1,
    steps: int = 20,
    cfg: float = 7.0,
    strength: float = 1.0,
    sampler: int = 0,  # 0 = DPMPP2MKarras
) -> bytes:
    """Build a FlatBuffers GenerationConfiguration from parameters."""

    config = config_generated.GenerationConfigurationT()
    config.model = model
    config.startWidth = width // 64
    config.startHeight = height // 64
    config.seed = max(0, seed) if seed >= 0 else 0  # Ensure non-negative
    config.steps = steps
    config.guidanceScale = cfg
    config.strength = strength
    config.sampler = sampler

    builder = flatbuffers.Builder(0)
    builder.Finish(config.Pack(builder))
    return bytes(builder.Output())


def decode_response_image(response_image: bytes) -> Image.Image:
    """Decode a Draw Things image response to PIL Image."""

    if len(response_image) < 68:
        raise ValueError(f"Image data too short: {len(response_image)} bytes (expected >= 68)")

    int_buffer = np.frombuffer(response_image[:68], dtype=np.uint32, count=17)
    height, width, channels = int_buffer[6:9]
    is_compressed = int_buffer[0] == 1012247

    if is_compressed:
        import fpzip
        uncompressed = fpzip.decompress(response_image[68:], order="C")
        data = uncompressed.astype(np.float16).tobytes()
    else:
        data = response_image[68:]

    fp16_data = np.frombuffer(data, dtype=np.float16, count=width * height * channels)
    fp16_data = np.clip((fp16_data + 1) * 127, 0, 255).astype(np.uint8)

    if channels == 4:
        mode = "RGBA"
    elif channels == 3:
        mode = "RGB"
    else:
        mode = "L"

    img = Image.frombytes(mode, (width, height), fp16_data.tobytes())
    return img


def generate_image(
    server: str = "localhost",
    port: int = 7859,
    model: str = "realism_sdxl_by_stable_yogi_f16.ckpt",
    prompt: str = "a gorgeous blonde woman riding a unicorn in a yellow bikini",
    negative_prompt: str = "blurry, low quality",
    width: int = 512,
    height: int = 512,
    seed: int = -1,
    steps: int = 20,
    cfg: float = 7.0,
    use_tls: bool = True,
):
    """
    Generate an image via Draw Things gRPC API.

    Args:
        server: gRPC server address
        port: gRPC server port
        model: Model filename on server
        prompt: Text prompt
        negative_prompt: Negative prompt
        width: Image width (will be rounded to 64px)
        height: Image height (will be rounded to 64px)
        seed: Random seed (random if None)
        steps: Number of inference steps
        cfg: Guidance scale
        use_tls: Whether to use TLS
    """

    if seed is None or seed < 0:
        seed = int(time.time() * 1000) % 4294967295

    # Build FlatBuffers configuration
    config_fbs = build_generation_config(
        model=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        seed=seed,
        steps=steps,
        cfg=cfg,
    )

    # Create gRPC channel
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

    request = imageService_pb2.ImageGenerationRequest(
        prompt=prompt,
        negativePrompt=negative_prompt,
        configuration=config_fbs,
        user="grpc-example",
        device="LAPTOP",
        chunked=True,
    )

    print(f"[gRPC] Sending GenerateImage request...")
    print(f"[gRPC]   Model: {model}")
    print(f"[gRPC]   Prompt: {prompt}")
    print(f"[gRPC]   Seed: {seed}")
    print(f"[gRPC]   Steps: {steps}, CFG: {cfg}")
    print(f"[gRPC]   Size: {width}x{height}")

    # Stream response
    response_stream = stub.GenerateImage(request)

    response_images = []

    while True:
        try:
            response = response_stream.next()
        except StopIteration:
            break

        # Print progress
        signpost = response.currentSignpost
        if signpost:
            if signpost.HasField("sampling"):
                print(f"[gRPC] Progress: step {signpost.sampling.step}/{steps}")
            elif signpost.HasField("imageDecoded"):
                print(f"[gRPC] Status: image decoded")

        # Collect generated images
        if response.generatedImages:
            response_images.extend(response.generatedImages)

        # LAST_CHUNK = 1
        if response.chunkState == 1:
            break

    channel.close()

    if not response_images:
        raise Exception("No images returned from server")

    print(f"[gRPC] Received {len(response_images)} image(s)")

    # Decode and save images
    os.makedirs("img", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, img_data in enumerate(response_images):
        img = decode_response_image(img_data)
        filename = f"img/generated_{timestamp}_{i}.png"
        img.save(filename)
        print(f"[Saved] {filename} ({img.width}x{img.height})")

    return filename


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate image via Draw Things gRPC")
    parser.add_argument("--server", default="localhost", help="gRPC server address")
    parser.add_argument("--port", type=int, default=7859, help="gRPC server port")
    parser.add_argument("--model", default="realism_sdxl_by_stable_yogi_f16.ckpt", help="Model filename (.ckpt)")
    parser.add_argument("--prompt", default="a gorgeous blonde woman riding a unicorn in a yellow bikini", help="Text prompt")
    parser.add_argument("--negative", default="blurry, low quality", help="Negative prompt")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=7.0, help="CFG scale")
    parser.add_argument("--tls", action="store_true", help="Use TLS")

    args = parser.parse_args()

    try:
        generate_image(
            server=args.server,
            port=args.port,
            model=args.model,
            prompt=args.prompt,
            negative_prompt=args.negative,
            width=args.width,
            height=args.height,
            seed=args.seed,
            steps=args.steps,
            cfg=args.cfg,
            use_tls=args.tls,
        )
    except grpc.RpcError as e:
        print(f"[Error] gRPC error: {e.code()}: {e.details()}")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
