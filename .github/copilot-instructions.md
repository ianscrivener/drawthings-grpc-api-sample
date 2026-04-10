# Project Guidelines

## Build And Run
- Use UV for Python environment and dependency management.
- Install dependencies with `uv sync`.
- Run the main CLIs with project entry points:
  - `uv run drawthings-grpc-generate [args]`
  - `uv run drawthings-grpc-model-list [args]`
- Compatibility wrappers are still valid:
  - `uv run python generate.py [args]`
  - `uv run python model_list.py [args]`
- Python version requirement is `>=3.11`.
- There is no configured test suite yet. Do not invent test commands unless asked to add tests.

## Architecture
- Main implementation lives in `src/drawthings_grpc_sample/`.
- `src/drawthings_grpc_sample/generate.py` contains the image-generation flow:
  - FlatBuffers config build
  - gRPC streaming request/response
  - response image decode and save to `img/`
- `src/drawthings_grpc_sample/model_list.py` calls `Echo` and parses the model metadata response.
- Root scripts `generate.py` and `model_list.py` are compatibility wrappers to package entry points.
- Generated code boundaries:
  - `src/drawthings_grpc_sample/generated/` is generated protobuf/FlatBuffers code
  - `third_party/draw-things-comfyui/generated/` is generated reference code
  - Do not hand-edit generated files.
- `third_party/draw-things-comfyui/` is reference integration code. Keep sample CLI behavior isolated unless a task explicitly requires integration changes.

## Conventions And Pitfalls
- Preserve gRPC max message options in clients unless a task explicitly changes transport behavior.
- Width/height in generation config are quantized by 64 (`width // 64`, `height // 64`). Keep this behavior unless asked to change it.
- TLS behavior can be confusing:
  - CLI usage in both scripts is plaintext by default unless `--tls` is provided.
  - `generate_image()` has `use_tls=True` as a function default, but CLI args override it.
- Keep output behavior stable: generated images are written to `img/` with timestamped names.
- When extending API usage (ControlNet, upload, file checks), use existing generated types and protocol docs rather than ad hoc payloads.

## Docs And References
- Setup, usage, and options: `README.md`
- RPC and message details: `DrawThings_gRPC_Protocol.md`
- Planned feature direction: `ToDo.md`

Link to these docs instead of duplicating their content in future instruction files.
