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
1. ✅ Enable Model Browsing
1. ✅ Know whether server TLS is ON or OFF


Then generate an image (server TLS OFF):

```bash
uv run drawthings-grpc-generate

# or vary the prompts
uv run drawthings-grpc-generate --prompt "a cat" --steps 20 --seed 42

# compatibility wrapper still works
uv run python generate.py --prompt "a cat" --steps 20 --seed 42
```

If server TLS is ON, add `--tls`:

```bash
uv run drawthings-grpc-generate --tls

# compatibility wrapper still works
uv run python generate.py --tls
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
--tls-ca-file <path> Path to PEM CA certificate used to verify TLS server certificate
```

## TLS And Certificate Verification

- The client now includes the Draw Things root CA used by the server, so standard Draw Things TLS works without extra setup.
- If server TLS is ON, use `--tls`.
- If server TLS is OFF, do not use `--tls`.
- If your server uses a different/private CA, pass `--tls-ca-file /path/to/ca.pem`.

If you see `CERTIFICATE_VERIFY_FAILED`, the client does not trust the presented certificate chain. Use `--tls-ca-file` with the correct CA PEM.

### TLS/Plaintext Test Matrix

To verify all four commands pass, run them in two phases:

1. With server TLS ON:
	- `python model_list.py --tls`
	- `python generate.py --tls`
2. With server TLS OFF:
	- `python model_list.py`
	- `python generate.py`

## Output

Images are saved to the `img/` directory with timestamp filenames:
- `img/generated_20260409_143052_0.png`


---
## View Available Models

```
uv run drawthings-grpc-model-list

# compatibility wrapper still works
uv run python model_list.py

# TLS mode
uv run drawthings-grpc-model-list --tls
uv run python model_list.py --tls
```


---
## Sources
- [**draw-things-comfyui**](https://github.com/drawthingsai/draw-things-comfyui) - Official Draw Things extension for ComfyUI (TypeScript frontend, Python backend)
- [**draw-things-community**](https://github.com/drawthingsai/draw-things-community) - Community repository with Swift diffusion model implementations and self-hosted gRPC server