from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_FALLBACK_DEFAULTS: dict[str, Any] = {
    "grpc": {
        "server": "localhost",
        "port": 7859,
        "compression": False,
        "request_chunked": True,
        "tls": True,
        "tls_ca_file": None,
    },
    "generation": {
        "t2i": {
            "model": "z_image_turbo_1.0_i8x.ckpt",
            "prompt": "a gorgeous blonde woman riding a unicorn in a yellow bikini",
            "negative_prompt": "blurry, low quality",
            "width": 512,
            "height": 512,
            "seed": -1,
            "steps": 20,
            "cfg": 7.0,
            "sampler": "DPM++ 2M Karras",
            "upscale": False,
            "upscale_model": "4x_ultrasharp_f16.ckpt",
            "upscale_factor": 2,
            "face_restore_model": "RestoreFormer.pth",
        },
        "i2i": {
            "model": "z_image_turbo_1.0_i8x.ckpt",
            "prompt": "a 22 year old african woman a Dark cave wearing a light blue lycra bodysuit.",
            "negative_prompt": "blurry, low quality",
            "source_image": "img_src/test_1.png",
            "width": 512,
            "height": 512,
            "strength": 0.6,
            "seed": -1,
            "steps": 20,
            "cfg": 7.0,
            "sampler": "DPM++ 2M Karras",
            "upscale": False,
            "upscale_model": "4x_ultrasharp_f16.ckpt",
            "upscale_factor": 2,
            "face_restore_model": "RestoreFormer.pth",
        },
    },
}

_CACHED_DEFAULTS: dict[str, Any] | None = None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _candidate_defaults_files() -> list[Path]:
    candidates: list[Path] = []

    env_path = os.getenv("DRAWTHINGS_DEFAULTS_FILE")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.append(Path.cwd() / "defaults.yaml")

    module_file = Path(__file__).resolve()
    repo_root = module_file.parents[2]
    candidates.append(repo_root / "defaults.yaml")

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)

    return unique


def _load_file_defaults(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if loaded is None:
        return {}

    if not isinstance(loaded, dict):
        raise ValueError(f"{file_path} must contain a top-level mapping")

    return loaded


def load_defaults(force_reload: bool = False) -> dict[str, Any]:
    global _CACHED_DEFAULTS

    if _CACHED_DEFAULTS is not None and not force_reload:
        return _CACHED_DEFAULTS

    merged = _deep_merge({}, _FALLBACK_DEFAULTS)

    for file_path in _candidate_defaults_files():
        if not file_path.exists():
            continue
        merged = _deep_merge(merged, _load_file_defaults(file_path))
        break

    _CACHED_DEFAULTS = merged
    return merged


def _as_str(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _as_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _as_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if value is None:
        return fallback
    return bool(value)


_DEFAULTS = load_defaults()
_GRPC_DEFAULTS = _DEFAULTS.get("grpc", {})
_GENERATION_DEFAULTS = _DEFAULTS.get("generation", {})
_T2I_DEFAULTS = _GENERATION_DEFAULTS.get("t2i", {})
_I2I_DEFAULTS = _GENERATION_DEFAULTS.get("i2i", {})

DEFAULT_SERVER = _as_str(_GRPC_DEFAULTS.get("server"), "localhost")
DEFAULT_PORT = _as_int(_GRPC_DEFAULTS.get("port"), 7859)
DEFAULT_GRPC_COMPRESSION = _as_bool(_GRPC_DEFAULTS.get("compression"), False)
DEFAULT_REQUEST_CHUNKED = _as_bool(_GRPC_DEFAULTS.get("request_chunked"), True)
DEFAULT_USE_TLS = _as_bool(_GRPC_DEFAULTS.get("tls"), True)
DEFAULT_TLS_CA_FILE = _as_optional_str(_GRPC_DEFAULTS.get("tls_ca_file"))

DEFAULT_T2I_MODEL = _as_str(_T2I_DEFAULTS.get("model"), "z_image_turbo_1.0_i8x.ckpt")
DEFAULT_T2I_PROMPT = _as_str(
    _T2I_DEFAULTS.get("prompt"),
    "a gorgeous blonde woman riding a unicorn in a yellow bikini",
)
DEFAULT_T2I_NEGATIVE_PROMPT = _as_str(_T2I_DEFAULTS.get("negative_prompt"), "blurry, low quality")
DEFAULT_T2I_WIDTH = _as_int(_T2I_DEFAULTS.get("width"), 512)
DEFAULT_T2I_HEIGHT = _as_int(_T2I_DEFAULTS.get("height"), 512)
DEFAULT_T2I_SEED = _as_int(_T2I_DEFAULTS.get("seed"), -1)
DEFAULT_T2I_STEPS = _as_int(_T2I_DEFAULTS.get("steps"), 20)
DEFAULT_T2I_CFG = _as_float(_T2I_DEFAULTS.get("cfg"), 7.0)
DEFAULT_T2I_SAMPLER = _as_str(_T2I_DEFAULTS.get("sampler"), "DPM++ 2M Karras")
DEFAULT_T2I_UPSCALE = _as_bool(_T2I_DEFAULTS.get("upscale"), False)
DEFAULT_T2I_UPSCALE_MODEL = _as_str(_T2I_DEFAULTS.get("upscale_model"), "4x_ultrasharp_f16.ckpt")
DEFAULT_T2I_UPSCALE_FACTOR = _as_int(_T2I_DEFAULTS.get("upscale_factor"), 2)
DEFAULT_T2I_FACE_RESTORE_MODEL = _as_str(_T2I_DEFAULTS.get("face_restore_model"), "RestoreFormer.pth")

DEFAULT_I2I_MODEL = _as_str(_I2I_DEFAULTS.get("model"), "z_image_turbo_1.0_i8x.ckpt")
DEFAULT_I2I_PROMPT = _as_str(
    _I2I_DEFAULTS.get("prompt"),
    "a 22 year old african woman a Dark cave wearing a light blue lycra bodysuit.",
)
DEFAULT_I2I_NEGATIVE_PROMPT = _as_str(_I2I_DEFAULTS.get("negative_prompt"), "blurry, low quality")
DEFAULT_I2I_SOURCE_IMAGE = _as_str(_I2I_DEFAULTS.get("source_image"), "img_src/test_1.png")
DEFAULT_I2I_WIDTH = _as_int(_I2I_DEFAULTS.get("width"), 512)
DEFAULT_I2I_HEIGHT = _as_int(_I2I_DEFAULTS.get("height"), 512)
DEFAULT_I2I_STRENGTH = _as_float(_I2I_DEFAULTS.get("strength"), 0.6)
DEFAULT_I2I_SEED = _as_int(_I2I_DEFAULTS.get("seed"), -1)
DEFAULT_I2I_STEPS = _as_int(_I2I_DEFAULTS.get("steps"), 20)
DEFAULT_I2I_CFG = _as_float(_I2I_DEFAULTS.get("cfg"), 7.0)
DEFAULT_I2I_SAMPLER = _as_str(_I2I_DEFAULTS.get("sampler"), "DPM++ 2M Karras")
DEFAULT_I2I_UPSCALE = _as_bool(_I2I_DEFAULTS.get("upscale"), False)
DEFAULT_I2I_UPSCALE_MODEL = _as_str(_I2I_DEFAULTS.get("upscale_model"), "4x_ultrasharp_f16.ckpt")
DEFAULT_I2I_UPSCALE_FACTOR = _as_int(_I2I_DEFAULTS.get("upscale_factor"), 2)
DEFAULT_I2I_FACE_RESTORE_MODEL = _as_str(_I2I_DEFAULTS.get("face_restore_model"), "RestoreFormer.pth")

# Compatibility aliases for existing imports and behavior.
DEFAULT_MODEL = DEFAULT_T2I_MODEL
DEFAULT_NEGATIVE_PROMPT = DEFAULT_T2I_NEGATIVE_PROMPT
DEFAULT_SOURCE_IMAGE = DEFAULT_I2I_SOURCE_IMAGE
DEFAULT_WIDTH = DEFAULT_T2I_WIDTH
DEFAULT_HEIGHT = DEFAULT_T2I_HEIGHT
DEFAULT_SEED = DEFAULT_T2I_SEED
DEFAULT_STEPS = DEFAULT_T2I_STEPS
DEFAULT_CFG = DEFAULT_T2I_CFG
DEFAULT_SAMPLER = DEFAULT_T2I_SAMPLER
DEFAULT_CHUNKED = DEFAULT_REQUEST_CHUNKED
DEFAULT_UPSCALE = DEFAULT_T2I_UPSCALE
DEFAULT_UPSCALE_MODEL = DEFAULT_T2I_UPSCALE_MODEL
DEFAULT_UPSCALE_FACTOR = DEFAULT_T2I_UPSCALE_FACTOR
DEFAULT_FACE_RESTORE_MODEL = DEFAULT_T2I_FACE_RESTORE_MODEL
