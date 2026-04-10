"""
Simple Draw Things gRPC Image Generation Example

This script demonstrates how to call the Draw Things gRPC API to generate images.
It connects to a running Draw Things gRPC server and sends a GenerateImage request.

Requirements:
    - Draw Things gRPC server running (e.g., gRPCServerCLI)
    - A model file available on the server
"""

import os
import struct
import sys
import time
from datetime import datetime

import flatbuffers
import grpc
import numpy as np
from PIL import Image, ImageOps

from drawthings_grpc_sample.generated import (
    config_generated,
    imageService_pb2,
    imageService_pb2_grpc,
)
from drawthings_grpc_sample.tls import create_channel


CCV_TENSOR_CPU_MEMORY = 0x1
CCV_TENSOR_FORMAT_NHWC = 0x02
CCV_16F = 0x20000

DEFAULT_T2I_PROMPT = "a gorgeous blonde woman riding a unicorn in a yellow bikini"
DEFAULT_I2I_PROMPT = "a 22 year old african woman a Dark cave wearing a light blue lycra bodysuit."
DEFAULT_I2I__WIDTH = 512
DEFAULT_I2I__HEIGHT = 512
DEFAULT_I2I__STRENGTH = 0.6

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


def encode_request_image(source_image: str, width: int, height: int) -> bytes:
    """Encode an input image to Draw Things tensor bytes for img2img requests."""

    if not os.path.exists(source_image):
        raise FileNotFoundError(f"Source image not found: {source_image}")

    resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS

    with Image.open(source_image) as img:
        rgb = ImageOps.fit(img.convert("RGB"), (width, height), method=resample)

    image_array = np.asarray(rgb, dtype=np.float32)
    image_array = image_array / 255.0 * 2.0 - 1.0
    fp16_bytes = image_array.astype(np.float16).tobytes(order="C")

    image_bytes = bytearray(68 + len(fp16_bytes))
    struct.pack_into(
        "<9I",
        image_bytes,
        0,
        0,
        CCV_TENSOR_CPU_MEMORY,
        CCV_TENSOR_FORMAT_NHWC,
        CCV_16F,
        0,
        1,
        height,
        width,
        3,
    )
    image_bytes[68:] = fp16_bytes
    return bytes(image_bytes)


def generate_image(
    server: str = "localhost",
    port: int = 7859,
    model: str = "realism_sdxl_by_stable_yogi_f16.ckpt",
    prompt: str = DEFAULT_T2I_PROMPT,
    negative_prompt: str = "blurry, low quality",
    width: int = 512,
    height: int = 512,
    seed: int = -1,
    steps: int = 20,
    cfg: float = 7.0,
    strength: float = 1.0,
    source_image: str | None = None,
    use_tls: bool = True,
    tls_ca_file: str | None = None,
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
        strength: Denoising strength (used by img2img)
        source_image: Input image path for img2img mode
        use_tls: Whether to use TLS
        tls_ca_file: Optional path to PEM CA certificate for TLS verification
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
        strength=strength,
    )

    channel = create_channel(server, port, use_tls, ca_cert_file=tls_ca_file)

    stub = imageService_pb2_grpc.ImageGenerationServiceStub(channel)

    request_kwargs = {
        "prompt": prompt,
        "negativePrompt": negative_prompt,
        "configuration": config_fbs,
        "user": "grpc-example",
        "device": "LAPTOP",
        "chunked": True,
    }

    if source_image:
        request_kwargs["image"] = encode_request_image(source_image, width, height)
        request_kwargs["scaleFactor"] = 1

    request = imageService_pb2.ImageGenerationRequest(**request_kwargs)

    print(f"[gRPC] Sending GenerateImage request...")
    print(f"[gRPC]   Mode: {'i2i' if source_image else 't2i'}")
    print(f"[gRPC]   Model: {model}")
    print(f"[gRPC]   Prompt: {prompt}")
    if source_image:
        print(f"[gRPC]   Source: {source_image}")
        print(f"[gRPC]   Strength: {strength}")
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


def image_to_image(
    server: str = "localhost",
    port: int = 7859,
    model: str = "realism_sdxl_by_stable_yogi_f16.ckpt",
    prompt: str = DEFAULT_I2I_PROMPT,
    negative_prompt: str = "blurry, low quality",
    source_image: str = "img_src/test_1.png",
    strength: float = 0.6,
    width: int = 512,
    height: int = 512,
    seed: int = -1,
    steps: int = 20,
    cfg: float = 7.0,
    use_tls: bool = True,
    tls_ca_file: str | None = None,
):
    """Generate an image from a source image (img2img)."""

    return generate_image(
        server=server,
        port=port,
        model=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        seed=seed,
        steps=steps,
        cfg=cfg,
        strength=strength,
        source_image=source_image,
        use_tls=use_tls,
        tls_ca_file=tls_ca_file,
    )


i2i = image_to_image


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate image via Draw Things gRPC")
    parser.add_argument("mode", nargs="?", choices=["t2i", "i2i"], default=None, help="Generation mode (positional: t2i/i2i)")
    parser.add_argument("--mode", choices=["t2i", "i2i"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--server", default="localhost", help="gRPC server address")
    parser.add_argument("--port", type=int, default=7859, help="gRPC server port")
    parser.add_argument("--model", default="realism_sdxl_by_stable_yogi_f16.ckpt", help="Model filename (.ckpt)")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Text prompt (mode-specific default used when omitted)",
    )
    parser.add_argument("--negative", default="blurry, low quality", help="Negative prompt")
    parser.add_argument("--source-image", default="img_src/test_1.png", help="Input source image for i2i mode")
    parser.add_argument("--strength", type=float, default=0.6, help="Denoising strength for i2i mode")
    parser.add_argument("--width", type=int, default=None, help="Image width")
    parser.add_argument("--height", type=int, default=None, help="Image height")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=7.0, help="CFG scale")
    parser.add_argument("--tls", action="store_true", help="Use TLS")
    parser.add_argument(
        "--tls-ca-file",
        default=None,
        help="Path to PEM CA certificate used to verify TLS server certificate",
    )

    args = parser.parse_args()
    # Determine mode: positional arg takes precedence, fallback to --mode, default t2i
    mode = args.mode or args.mode or "t2i"
    prompt = args.prompt or (DEFAULT_I2I_PROMPT if mode == "i2i" else DEFAULT_T2I_PROMPT)
    width = args.width if args.width is not None else (DEFAULT_I2I__WIDTH if mode == "i2i" else 512)
    height = args.height if args.height is not None else (DEFAULT_I2I__HEIGHT if mode == "i2i" else 512)
    strength = args.strength if hasattr(args, "strength") and args.strength is not None else (DEFAULT_I2I__STRENGTH if mode == "i2i" else 1.0)

    try:
        if mode == "i2i":
            image_to_image(
                server=args.server,
                port=args.port,
                model=args.model,
                prompt=prompt,
                negative_prompt=args.negative,
                source_image=args.source_image,
                strength=strength,
                width=width,
                height=height,
                seed=args.seed,
                steps=args.steps,
                cfg=args.cfg,
                use_tls=args.tls,
                tls_ca_file=args.tls_ca_file,
            )
        else:
            generate_image(
                server=args.server,
                port=args.port,
                model=args.model,
                prompt=prompt,
                negative_prompt=args.negative,
                width=width,
                height=height,
                seed=args.seed,
                steps=args.steps,
                cfg=args.cfg,
                use_tls=args.tls,
                tls_ca_file=args.tls_ca_file,
            )
    except grpc.RpcError as e:
        print(f"[Error] gRPC error: {e.code()}: {e.details()}")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
