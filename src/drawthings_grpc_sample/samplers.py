from __future__ import annotations

import re
from typing import Any

from drawthings_grpc_sample.generated import config_generated


SAMPLER_NAMES: list[str] = [
    "DPM++ 2M Karras",
    "Euler A",
    "DDIM",
    "PLMS",
    "DPM++ SDE Karras",
    "UniPC",
    "LCM",
    "Euler A Substep",
    "DPM++ SDE Substep",
    "TCD",
    "Euler A Trailing",
    "DPM++ SDE Trailing",
    "DPM++ 2M AYS",
    "Euler A AYS",
    "DPM++ SDE AYS",
    "DPM++ 2M Trailing",
    "DDIM Trailing",
    "UniPC Trailing",
    "UniPC AYS",
    "TCD Trailing",
]


def _normalize_sampler_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def _build_enum_sampler_aliases() -> dict[str, int]:
    aliases: dict[str, int] = {}
    for name, value in config_generated.SamplerType.__dict__.items():
        if name.startswith("_"):
            continue
        if isinstance(value, int):
            aliases[_normalize_sampler_text(name)] = value
    return aliases


_NAME_ALIASES: dict[str, int] = {
    _normalize_sampler_text(name): index for index, name in enumerate(SAMPLER_NAMES)
}
_ENUM_ALIASES: dict[str, int] = _build_enum_sampler_aliases()


def get_sampler_names() -> list[str]:
    return list(SAMPLER_NAMES)


def resolve_sampler(value: str | int | None) -> tuple[int, str]:
    if value is None:
        return 0, SAMPLER_NAMES[0]

    if isinstance(value, int):
        if 0 <= value < len(SAMPLER_NAMES):
            return value, SAMPLER_NAMES[value]
        raise ValueError(
            f"Sampler index {value} is out of range. Valid range: 0-{len(SAMPLER_NAMES) - 1}."
        )

    text = str(value).strip()
    if not text:
        raise ValueError("Sampler cannot be empty.")

    if text.isdigit():
        index = int(text)
        if 0 <= index < len(SAMPLER_NAMES):
            return index, SAMPLER_NAMES[index]
        raise ValueError(
            f"Sampler index {index} is out of range. Valid range: 0-{len(SAMPLER_NAMES) - 1}."
        )

    normalized = _normalize_sampler_text(text)

    if normalized in _NAME_ALIASES:
        index = _NAME_ALIASES[normalized]
        return index, SAMPLER_NAMES[index]

    if normalized in _ENUM_ALIASES:
        index = _ENUM_ALIASES[normalized]
        if 0 <= index < len(SAMPLER_NAMES):
            return index, SAMPLER_NAMES[index]
        raise ValueError(
            f"Sampler enum resolves to unsupported index {index}. Valid range: 0-{len(SAMPLER_NAMES) - 1}."
        )

    sample = ", ".join(SAMPLER_NAMES[:6]) + " ..."
    raise ValueError(
        f"Unknown sampler '{value}'. Use a sampler name (for example: {sample}) "
        f"or an index in the range 0-{len(SAMPLER_NAMES) - 1}."
    )
