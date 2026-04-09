# DrawThings gRPC Image Generation Example

Simple Python example demonstrating how to call the DrawThings gRPC API.

See Also: [**DrawThings gRPC Protocol**](DrawThings_gRPC_Protocol.md)

## Setup

```bash
# Install uv if you don't have it already 
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up the UV Python environment 
uv sync
```

Dependencies are already configured in `pyproject.toml`.

## Run

Make sure the DrawThings gRPC server is running first:


1. ✅ Server Online
1. ✅ gRPC (not HTTP)
1. ✅ Transport Layer Security
1. ✅ Enable Model Browsing


Then generate an image:

```bash
uv run python generate.py

# or vary the prompts
uv run python generate.py --prompt "a cat" --steps 20 --seed 42
```

## Options

```
--server localhost   gRPC server address
--port 7859          gRPC server port
--model <file>       Model filename on server
--prompt <text>      Text prompt
--negative <text>    Negative prompt
--width 512          Image width (rounded to 64px)
--height 512         Image height (rounded to 64px)
--seed <int>         Random seed
--steps 20           Inference steps
--cfg 7.0            CFG scale
--tls                Use TLS (requires server to have TLS enabled)
```

## Output

Images are saved to the `img/` directory with timestamp filenames:
- `img/generated_20260409_143052_0.png`


---
## View Available Models

```
uv run model_list.py
```


---
## Sources
- [**draw-things-comfyui**](https://github.com/drawthingsai/draw-things-comfyui) - Official Draw Things extension for ComfyUI (TypeScript frontend, Python backend)
- [**draw-things-community**](https://github.com/drawthingsai/draw-things-community) - Community repository with Swift diffusion model implementations and self-hosted gRPC server