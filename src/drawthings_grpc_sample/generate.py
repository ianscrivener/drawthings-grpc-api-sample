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
from io import BytesIO
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
from drawthings_grpc_sample.defaults import (
    DEFAULT_GRPC_COMPRESSION,
    DEFAULT_I2I_CFG,
    DEFAULT_I2I_FACE_RESTORE_MODEL,
    DEFAULT_I2I_HEIGHT,
    DEFAULT_I2I_MODEL,
    DEFAULT_I2I_NEGATIVE_PROMPT,
    DEFAULT_I2I_PROMPT,
    DEFAULT_I2I_SAMPLER,
    DEFAULT_I2I_SEED,
    DEFAULT_I2I_SOURCE_IMAGE,
    DEFAULT_I2I_STEPS,
    DEFAULT_I2I_STRENGTH,
    DEFAULT_I2I_UPSCALE,
    DEFAULT_I2I_UPSCALE_FACTOR,
    DEFAULT_I2I_UPSCALE_MODEL,
    DEFAULT_I2I_WIDTH,
    DEFAULT_PORT,
    DEFAULT_REQUEST_CHUNKED,
    DEFAULT_SERVER,
    DEFAULT_T2I_CFG,
    DEFAULT_T2I_FACE_RESTORE_MODEL,
    DEFAULT_T2I_HEIGHT,
    DEFAULT_T2I_MODEL,
    DEFAULT_T2I_NEGATIVE_PROMPT,
    DEFAULT_T2I_PROMPT,
    DEFAULT_T2I_SAMPLER,
    DEFAULT_T2I_SEED,
    DEFAULT_T2I_STEPS,
    DEFAULT_T2I_UPSCALE,
    DEFAULT_T2I_UPSCALE_FACTOR,
    DEFAULT_T2I_UPSCALE_MODEL,
    DEFAULT_T2I_WIDTH,
    DEFAULT_TLS_CA_FILE,
    DEFAULT_USE_TLS,
)
from drawthings_grpc_sample.model_list import get_available_model_files
from drawthings_grpc_sample.samplers import resolve_sampler
from drawthings_grpc_sample.tls import create_channel


CCV_TENSOR_CPU_MEMORY = 0x1
CCV_TENSOR_FORMAT_NHWC = 0x02
CCV_16F = 0x20000


def clamp(value):
    return max(min(int(value if np.isfinite(value) else 0), 255), 0)


def build_generation_config(
    model: str,
    prompt: str,
    negative_prompt: str = DEFAULT_T2I_NEGATIVE_PROMPT,
    width: int = DEFAULT_T2I_WIDTH,
    height: int = DEFAULT_T2I_HEIGHT,
    seed: int = DEFAULT_T2I_SEED,
    steps: int = DEFAULT_T2I_STEPS,
    cfg: float = DEFAULT_T2I_CFG,
    strength: float = 1.0,
    upscaler: str | None = None,
    upscaler_scale_factor: int = 0,
    face_restoration: str | None = None,
    sampler: int = 0,
) -> bytes:
    """Build a FlatBuffers GenerationConfiguration from parameters."""

    config = config_generated.GenerationConfigurationT()
    config.model = model
    config.startWidth = width // 64
    config.startHeight = height // 64
    config.seed = max(0, seed) if seed >= 0 else 0
    config.steps = steps
    config.guidanceScale = cfg
    config.strength = strength
    config.sampler = sampler
    config.batchCount = 1
    config.batchSize = 1
    config.originalImageWidth = width
    config.originalImageHeight = height
    config.targetImageWidth = width
    config.targetImageHeight = height
    config.negativeOriginalImageWidth = max(1, width // 2)
    config.negativeOriginalImageHeight = max(1, height // 2)

    if upscaler:
        config.upscaler = upscaler
    if upscaler_scale_factor > 0:
        config.upscalerScaleFactor = upscaler_scale_factor
    if face_restoration:
        config.faceRestoration = face_restoration

    builder = flatbuffers.Builder(0)
    builder.Finish(config.Pack(builder))
    return bytes(builder.Output())


def decode_response_image(response_image: bytes) -> Image.Image:
    """Decode a Draw Things image response to PIL Image."""

    # Some server paths (for example, upscaled outputs) can return encoded images directly.
    if response_image.startswith(b"\x89PNG\r\n\x1a\n") or response_image.startswith(b"\xff\xd8") or response_image.startswith(b"RIFF"):
        with Image.open(BytesIO(response_image)) as img:
            return img.copy()

    if len(response_image) < 68:
        raise ValueError(f"Image data too short: {len(response_image)} bytes (expected >= 68)")

    int_buffer = np.frombuffer(response_image[:68], dtype=np.uint32, count=17)
    height, width, channels = int_buffer[6:9]
    is_compressed = int_buffer[0] == 1012247

    if width <= 0 or height <= 0 or channels not in (1, 3, 4):
        with Image.open(BytesIO(response_image)) as img:
            return img.copy()

    # Some upscaled responses arrive as raw packed RGB/RGBA instead of fp16 tensor data.
    rgb_size = width * height * 3
    rgba_size = width * height * 4
    if len(response_image) == rgba_size:
        return Image.frombytes("RGBA", (width, height), response_image).convert("RGB")
    if len(response_image) == rgb_size:
        return Image.frombytes("RGB", (width, height), response_image)

    if is_compressed:
        import fpzip

        uncompressed = fpzip.decompress(response_image[68:], order="C")
        data = uncompressed.astype(np.float16).tobytes()
    else:
        data = response_image[68:]

    if len(data) == rgba_size:
        return Image.frombytes("RGBA", (width, height), data).convert("RGB")
    if len(data) == rgb_size:
        return Image.frombytes("RGB", (width, height), data)

    required_bytes = width * height * channels * np.dtype(np.float16).itemsize
    if len(data) < required_bytes:
        with Image.open(BytesIO(response_image)) as img:
            return img.copy()

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
    server: str = DEFAULT_SERVER,
    port: int = DEFAULT_PORT,
    model: str = DEFAULT_T2I_MODEL,
    prompt: str = DEFAULT_T2I_PROMPT,
    negative_prompt: str = DEFAULT_T2I_NEGATIVE_PROMPT,
    width: int = DEFAULT_T2I_WIDTH,
    height: int = DEFAULT_T2I_HEIGHT,
    seed: int = DEFAULT_T2I_SEED,
    steps: int = DEFAULT_T2I_STEPS,
    cfg: float = DEFAULT_T2I_CFG,
    strength: float = 1.0,
    source_image: str | None = None,
    upscale: bool = DEFAULT_T2I_UPSCALE,
    upscale_model: str = DEFAULT_T2I_UPSCALE_MODEL,
    upscale_factor: int = DEFAULT_T2I_UPSCALE_FACTOR,
    face_restore: str | None = None,
    use_tls: bool = DEFAULT_USE_TLS,
    tls_ca_file: str | None = DEFAULT_TLS_CA_FILE,
    sampler: str | int = DEFAULT_T2I_SAMPLER,
    chunked: bool = DEFAULT_REQUEST_CHUNKED,
    use_compression: bool = DEFAULT_GRPC_COMPRESSION,
):
    """Generate an image via Draw Things gRPC API."""

    if seed is None or seed < 0:
        seed = int(time.time() * 1000) % 4294967295

    available_models = get_available_model_files(
        server=server,
        port=port,
        use_tls=use_tls,
        tls_ca_file=tls_ca_file,
        use_compression=use_compression,
    )

    if not available_models:
        raise ValueError(
            "Could not validate the requested model because the server returned no installed models."
        )

    if model not in available_models:
        preview = ", ".join(available_models[:10])
        suffix = " ..." if len(available_models) > 10 else ""
        raise ValueError(
            f"Requested model '{model}' does not exist on {server}:{port}. "
            f"Available models ({len(available_models)}): {preview}{suffix}"
        )

    sampler_index, sampler_name = resolve_sampler(sampler)

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
        upscaler=upscale_model if upscale else None,
        upscaler_scale_factor=upscale_factor if upscale else 0,
        face_restoration=face_restore,
        sampler=sampler_index,
    )

    channel = create_channel(
        server,
        port,
        use_tls,
        ca_cert_file=tls_ca_file,
        use_compression=use_compression,
    )
    stub = imageService_pb2_grpc.ImageGenerationServiceStub(channel)

    request_kwargs = {
        "prompt": prompt,
        "negativePrompt": negative_prompt,
        "configuration": config_fbs,
        "user": "grpc-example",
        "device": "LAPTOP",
        "chunked": chunked,
    }

    if source_image:
        request_kwargs["image"] = encode_request_image(source_image, width, height)
        request_kwargs["scaleFactor"] = 1

    request = imageService_pb2.ImageGenerationRequest(**request_kwargs)

    print("[gRPC] Sending GenerateImage request...")
    print(f"[gRPC]   Mode: {'i2i' if source_image else 't2i'}")
    print(f"[gRPC]   Model: {model}")
    print(f"[gRPC]   Prompt: {prompt}")
    if source_image:
        print(f"[gRPC]   Source: {source_image}")
        print(f"[gRPC]   Strength: {strength}")
    print(f"[gRPC]   Seed: {seed}")
    print(f"[gRPC]   Steps: {steps}, CFG: {cfg}, Sampler: {sampler_name} ({sampler_index})")
    print(f"[gRPC]   Size: {width}x{height}")
    print(f"[gRPC]   Chunked response: {chunked}")
    if upscale:
        print(f"[gRPC]   Upscale: {upscale_model} x{upscale_factor}")
    if face_restore:
        print(f"[gRPC]   Face restore: {face_restore if face_restore else '<server-default>'}")

    response_stream = stub.GenerateImage(request)
    response_images = []
    pending_image_chunks = []

    while True:
        try:
            response = response_stream.next()
        except StopIteration:
            break

        signpost = response.currentSignpost
        if signpost:
            if signpost.HasField("sampling"):
                print(f"[gRPC] Progress: step {signpost.sampling.step}/{steps}")
            elif signpost.HasField("imageDecoded"):
                print("[gRPC] Status: image decoded")

        if response.generatedImages:
            # In chunked mode, large images can be split across multiple responses.
            # We accumulate chunks while MORE_CHUNKS and finalize when LAST_CHUNK arrives.
            if response.chunkState == 1:
                pending_image_chunks.extend(response.generatedImages)
            else:
                if pending_image_chunks:
                    pending_image_chunks.extend(response.generatedImages)
                    response_images.append(b"".join(pending_image_chunks))
                    pending_image_chunks = []
                else:
                    response_images.extend(response.generatedImages)

    # If server closed stream after MORE_CHUNKS chunks, flush what we have.
    if pending_image_chunks:
        response_images.append(b"".join(pending_image_chunks))

    channel.close()

    if not response_images:
        raise Exception("No images returned from server")

    print(f"[gRPC] Received {len(response_images)} image(s)")

    os.makedirs("img", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    last_file = None
    for i, img_data in enumerate(response_images):
        img = decode_response_image(img_data)
        filename = f"img/generated_{timestamp}_{i}.png"
        img.save(filename)
        print(f"[Saved] {filename} ({img.width}x{img.height})")
        last_file = filename

    return last_file


def image_to_image(
    server: str = DEFAULT_SERVER,
    port: int = DEFAULT_PORT,
    model: str = DEFAULT_I2I_MODEL,
    prompt: str = DEFAULT_I2I_PROMPT,
    negative_prompt: str = DEFAULT_I2I_NEGATIVE_PROMPT,
    source_image: str = DEFAULT_I2I_SOURCE_IMAGE,
    strength: float = DEFAULT_I2I_STRENGTH,
    width: int = DEFAULT_I2I_WIDTH,
    height: int = DEFAULT_I2I_HEIGHT,
    seed: int = DEFAULT_I2I_SEED,
    steps: int = DEFAULT_I2I_STEPS,
    cfg: float = DEFAULT_I2I_CFG,
    upscale: bool = DEFAULT_I2I_UPSCALE,
    upscale_model: str = DEFAULT_I2I_UPSCALE_MODEL,
    upscale_factor: int = DEFAULT_I2I_UPSCALE_FACTOR,
    face_restore: str | None = None,
    use_tls: bool = DEFAULT_USE_TLS,
    tls_ca_file: str | None = DEFAULT_TLS_CA_FILE,
    sampler: str | int = DEFAULT_I2I_SAMPLER,
    chunked: bool = DEFAULT_REQUEST_CHUNKED,
    use_compression: bool = DEFAULT_GRPC_COMPRESSION,
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
        upscale=upscale,
        upscale_model=upscale_model,
        upscale_factor=upscale_factor,
        face_restore=face_restore,
        use_tls=use_tls,
        tls_ca_file=tls_ca_file,
        sampler=sampler,
        chunked=chunked,
        use_compression=use_compression,
    )


i2i = image_to_image


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Generate image via Draw Things gRPC")
    parser.add_argument("mode", nargs="?", choices=["t2i", "i2i"], default=None, help="Generation mode (positional: t2i/i2i)")
    parser.add_argument("--mode", dest="mode_flag", choices=["t2i", "i2i"], default=None, help="Generation mode")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="gRPC server address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="gRPC server port")
    parser.add_argument("--model", default=None, help="Model filename (.ckpt)")
    parser.add_argument("--prompt", default=None, help="Text prompt (mode-specific default used when omitted)")
    parser.add_argument("--negative", default=None, help="Negative prompt")
    parser.add_argument("--source-image", default=None, help="Input source image for i2i mode")
    parser.add_argument("--strength", type=float, default=None, help="Denoising strength for i2i mode")
    parser.add_argument("--width", type=int, default=None, help="Image width")
    parser.add_argument("--height", type=int, default=None, help="Image height")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--steps", type=int, default=None, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=None, help="CFG scale")
    parser.add_argument("--sampler", default=None, help="Sampler name or index")
    parser.add_argument("--upscale", nargs="?", const=True, default=None, help="Enable upscaling (use --upscale or --upscale true)")
    parser.add_argument("--upscale-model", default=None, help="Upscale model filename (.ckpt)")
    parser.add_argument("--upscale-factor", type=int, default=None, help="Upscale factor (for example, 2)")
    parser.add_argument("--face-restore", action="store_true", help="Enable face restoration using mode-specific default model")
    parser.set_defaults(tls=DEFAULT_USE_TLS)
    parser.set_defaults(chunked=DEFAULT_REQUEST_CHUNKED)
    parser.add_argument("--tls", dest="tls", action="store_true", help="Use TLS (default)")
    parser.add_argument("--no-tls", dest="tls", action="store_false", help="Disable TLS and use plaintext")
    parser.add_argument("--chunked", dest="chunked", action="store_true", help="Request chunked image responses")
    parser.add_argument("--no-chunked", dest="chunked", action="store_false", help="Request non-chunked image responses")
    parser.add_argument("--tls-ca-file", default=DEFAULT_TLS_CA_FILE, help="Path to PEM CA certificate used to verify TLS server certificate")

    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)

    mode = args.mode_flag or args.mode or "t2i"

    if mode == "i2i":
        mode_model = DEFAULT_I2I_MODEL
        mode_prompt = DEFAULT_I2I_PROMPT
        mode_negative = DEFAULT_I2I_NEGATIVE_PROMPT
        mode_width = DEFAULT_I2I_WIDTH
        mode_height = DEFAULT_I2I_HEIGHT
        mode_seed = DEFAULT_I2I_SEED
        mode_steps = DEFAULT_I2I_STEPS
        mode_cfg = DEFAULT_I2I_CFG
        mode_sampler = DEFAULT_I2I_SAMPLER
        mode_upscale = DEFAULT_I2I_UPSCALE
        mode_upscale_model = DEFAULT_I2I_UPSCALE_MODEL
        mode_upscale_factor = DEFAULT_I2I_UPSCALE_FACTOR
        mode_face_restore = DEFAULT_I2I_FACE_RESTORE_MODEL
        mode_source_image = DEFAULT_I2I_SOURCE_IMAGE
        mode_strength = DEFAULT_I2I_STRENGTH
    else:
        mode_model = DEFAULT_T2I_MODEL
        mode_prompt = DEFAULT_T2I_PROMPT
        mode_negative = DEFAULT_T2I_NEGATIVE_PROMPT
        mode_width = DEFAULT_T2I_WIDTH
        mode_height = DEFAULT_T2I_HEIGHT
        mode_seed = DEFAULT_T2I_SEED
        mode_steps = DEFAULT_T2I_STEPS
        mode_cfg = DEFAULT_T2I_CFG
        mode_sampler = DEFAULT_T2I_SAMPLER
        mode_upscale = DEFAULT_T2I_UPSCALE
        mode_upscale_model = DEFAULT_T2I_UPSCALE_MODEL
        mode_upscale_factor = DEFAULT_T2I_UPSCALE_FACTOR
        mode_face_restore = DEFAULT_T2I_FACE_RESTORE_MODEL
        mode_source_image = DEFAULT_I2I_SOURCE_IMAGE
        mode_strength = 1.0

    prompt = args.prompt if args.prompt is not None else mode_prompt
    model = args.model if args.model is not None else mode_model
    negative_prompt = args.negative if args.negative is not None else mode_negative
    width = args.width if args.width is not None else mode_width
    height = args.height if args.height is not None else mode_height
    seed = args.seed if args.seed is not None else mode_seed
    steps = args.steps if args.steps is not None else mode_steps
    cfg = args.cfg if args.cfg is not None else mode_cfg
    sampler = args.sampler if args.sampler is not None else mode_sampler
    source_image = args.source_image if args.source_image is not None else mode_source_image
    strength = args.strength if args.strength is not None else mode_strength
    upscale_input = args.upscale if args.upscale is not None else mode_upscale
    upscale = upscale_input if isinstance(upscale_input, bool) else str(upscale_input).lower() in {"1", "true", "yes", "on"}
    upscale_model = args.upscale_model if args.upscale_model is not None else mode_upscale_model
    upscale_factor = args.upscale_factor if args.upscale_factor is not None else mode_upscale_factor
    face_restore_model = mode_face_restore if args.face_restore else None

    try:
        if mode == "i2i":
            image_to_image(
                server=args.server,
                port=args.port,
                model=model,
                prompt=prompt,
                negative_prompt=negative_prompt,
                source_image=source_image,
                strength=strength,
                width=width,
                height=height,
                seed=seed,
                steps=steps,
                cfg=cfg,
                upscale=upscale,
                upscale_model=upscale_model,
                upscale_factor=upscale_factor,
                face_restore=face_restore_model,
                use_tls=args.tls,
                tls_ca_file=args.tls_ca_file,
                sampler=sampler,
                chunked=args.chunked,
                use_compression=DEFAULT_GRPC_COMPRESSION,
            )
        else:
            generate_image(
                server=args.server,
                port=args.port,
                model=model,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                seed=seed,
                steps=steps,
                cfg=cfg,
                upscale=upscale,
                upscale_model=upscale_model,
                upscale_factor=upscale_factor,
                face_restore=face_restore_model,
                use_tls=args.tls,
                tls_ca_file=args.tls_ca_file,
                sampler=sampler,
                chunked=args.chunked,
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
