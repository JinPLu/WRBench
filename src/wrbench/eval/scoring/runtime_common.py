"""Shared runtime helpers for local and DashScope evaluation runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from wrbench.eval._io import read_json, resolve_video_path


def load_model_explicit_cuda(
    model_cls: Any,
    *,
    model_path: str,
    dtype: Any,
    attn_implementation: str,
    local_rank: int,
) -> Any:
    """Load on CPU, then place one complete model on the selected CUDA device.

    Avoiding ``device_map`` keeps the single-GPU path independent of Accelerate.
    The dtype/torch_dtype retry preserves compatibility across Transformers
    model classes.
    """
    common_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
        "attn_implementation": attn_implementation,
    }
    try:
        model = model_cls.from_pretrained(model_path, dtype=dtype, **common_kwargs)
    except TypeError:
        model = model_cls.from_pretrained(model_path, torch_dtype=dtype, **common_kwargs)
    return model.to(f"cuda:{int(local_rank)}")


def explicit_cuda_placement_metadata(local_rank: int) -> dict[str, Any]:
    return {
        "device_map": None,
        "device_placement": "explicit_model_to",
        "model_device": f"cuda:{int(local_rank)}",
    }


def require_section(mapping: dict[str, Any], key: str, scope: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Missing required mapping: {scope}.{key}")
    return value


def require_value(mapping: dict[str, Any], key: str, scope: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required value: {scope}.{key}")
    value = mapping[key]
    if value is None:
        raise ValueError(f"Required value is null: {scope}.{key}")
    return value


def require_text(mapping: dict[str, Any], key: str, scope: str) -> str:
    value = str(require_value(mapping, key, scope)).strip()
    if not value:
        raise ValueError(f"Required text is empty: {scope}.{key}")
    return value


def scoring_video_path(row: dict[str, Any], *, surface: str | None = None) -> str:
    return resolve_video_path(row, surface=surface)


def index_by_video_id(
    rows: list[dict[str, Any]],
    *,
    label: str = "rows",
    require_unique: bool = False,
) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for index, row in enumerate(rows):
        raw_id = row.get("video_id")
        if raw_id is None or str(raw_id).strip() == "":
            if require_unique:
                raise ValueError(f"{label} row {index} is missing video_id")
            continue
        video_id = str(raw_id).strip() if require_unique else str(raw_id)
        if video_id in mapped:
            duplicates.append(video_id)
        mapped[video_id] = row
    if require_unique and duplicates:
        raise ValueError(f"{label} contains duplicate video_id values: {sorted(set(duplicates))[:10]}")
    return mapped


def version_or_unavailable(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except Exception:
        return "unavailable"
    return str(getattr(module, "__version__", "unknown"))


def cuda_build_or_unavailable() -> str:
    try:
        import torch  # type: ignore
    except Exception:
        return "unavailable"
    return str(getattr(torch.version, "cuda", None) or "unavailable")


def model_config_model_type(model_path: Path) -> str | None:
    config_path = model_path / "config.json"
    if not config_path.exists():
        return None
    try:
        config = read_json(config_path)
    except Exception:
        return None
    value = config.get("model_type") if isinstance(config, dict) else None
    return str(value) if value is not None else None


def derive_frames_used_from_processor_inputs(inputs: Any, processor: Any) -> int:
    try:
        grid = inputs.get("video_grid_thw") if hasattr(inputs, "get") else inputs["video_grid_thw"]
    except Exception:
        return 0
    if grid is None:
        return 0
    try:
        temporal_grid = grid[0, 0]
    except Exception:
        try:
            temporal_grid = grid[0][0]
        except Exception:
            return 0
    try:
        temporal_grid_value = int(temporal_grid.item())
    except Exception:
        try:
            temporal_grid_value = int(temporal_grid)
        except Exception:
            return 0
    temporal_patch_size = getattr(getattr(processor, "video_processor", None), "temporal_patch_size", 1)
    try:
        patch_size = int(temporal_patch_size)
    except Exception:
        patch_size = 1
    return max(0, temporal_grid_value * max(1, patch_size))


META_KEYS: Final[list[str]] = [
    "video_id",
    "prompt_id",
    "variant_id",
    "family_id",
    "reasoning_tier",
    "oov_gap",
    "event_delta",
    "divergence_id",
    "scenario_type",
    "visibility_gap_level",
    "protocol",
    "predictability_level",
    "event_count",
    "expected_state",
    "expected_visibility",
    "model",
    "camera_type",
    "world_state_prompt",
    "expected_behavior",
]
