# Draw Things gRPC Image Generation Example

Simple Python example demonstrating how to call the Draw Things gRPC API.

## Setup

```bash
uv sync
```

Dependencies are already configured in `pyproject.toml`.

## Run

Make sure the Draw Things gRPC server is running first:

```bash
# In draw-things-community/
bazel run Apps:gRPCServerCLI-macOS -- /path/to/models
# Or with Swift:
swift run -c release gRPCServerCLI /path/to/models
```

Then generate an image:

```bash
uv run python generate.py --prompt "a cat" --steps 20 --seed 42
```

## Options

```
--server localhost    gRPC server address
--port 7859          gRPC server port
--model <file>       Model filename on server
--prompt <text>      Text prompt
--negative <text>    Negative prompt
--width 512          Image width (rounded to 64px)
--height 512         Image height (rounded to 64px)
--seed <int>        Random seed
--steps 20           Inference steps
--cfg 7.0            CFG scale
--tls                Use TLS (requires server to have TLS enabled)
```

## Output

Images are saved to the `img/` directory with timestamp filenames:
- `img/generated_20260409_143052_0.png`
