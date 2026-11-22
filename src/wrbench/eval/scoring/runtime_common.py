"""Shared runtime helpers for local and DashScope evaluation runners."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final, Mapping

from wrbench.eval._io import read_json, resolve_video_path

VISION_BOUNDARY_MODES: Final[tuple[str, ...]] = ("warn", "strip_outer_markers", "preserve")
DEFAULT_VISION_BOUNDARY_MODE: Final[str] = "warn"
HISTORICAL_TRANSFORMERS_PREFIX: Final[str] = "5.9."
DEFAULT_VISION_START_TOKEN_ID: Final[int] = 248053
DEFAULT_VISION_END_TOKEN_ID: Final[int] = 248054
DEFAULT_VIDEO_PAD_TOKEN_ID: Final[int] = 248056


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


def capture_scorer_version_metadata() -> dict[str, str]:
    """Capture runtime library versions relevant to Qwen D3-D6 replay."""
    return {
        "transformers_version": version_or_unavailable("transformers"),
        "torch_version": version_or_unavailable("torch"),
        "tokenizers_version": version_or_unavailable("tokenizers"),
        "decord_version": version_or_unavailable("decord"),
        "cuda_build": cuda_build_or_unavailable(),
    }


def resolve_vision_boundary_mode(mode: str | None = None) -> str:
    """Resolve and validate ``QWEN35_VISION_BOUNDARY_MODE``."""
    raw = mode if mode is not None else os.environ.get("QWEN35_VISION_BOUNDARY_MODE", DEFAULT_VISION_BOUNDARY_MODE)
    resolved = str(raw or DEFAULT_VISION_BOUNDARY_MODE).strip()
    if resolved not in VISION_BOUNDARY_MODES:
        raise ValueError(
            f"QWEN35_VISION_BOUNDARY_MODE must be one of {', '.join(VISION_BOUNDARY_MODES)}; got {resolved!r}"
        )
    return resolved


def _resolve_special_token_id(
    processor: Any,
    token_text: str,
    *,
    explicit: int | None,
    attr_names: tuple[str, ...],
    default_id: int,
) -> int:
    if explicit is not None:
        return int(explicit)
    tokenizer = getattr(processor, "tokenizer", processor) if processor is not None else None
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if callable(convert):
        try:
            tid = convert(token_text)
            if isinstance(tid, int) and tid >= 0 and tid != unk_id:
                return tid
        except Exception:
            pass
    for source in (processor, tokenizer):
        if source is None:
            continue
        for attr in attr_names:
            val = getattr(source, attr, None)
            if isinstance(val, int) and val >= 0:
                return val
    return default_id


def _sequence_to_int_list(input_ids: Any) -> tuple[list[int] | None, int]:
    if input_ids is None:
        return None, 0
    raw = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
    if not isinstance(raw, (list, tuple)) or not raw:
        return None, 0
    if isinstance(raw[0], (list, tuple)):
        if len(raw) != 1:
            return None, 0
        seq = [int(x) for x in raw[0]]
        return seq, len(seq)
    seq = [int(x) for x in raw]
    return seq, len(seq)


def _strip_sequence_indices(value: Any, remove: set[int], seq_len: int) -> Any:
    keep = [i for i in range(seq_len) if i not in remove]
    if isinstance(value, list):
        if value and isinstance(value[0], list) and len(value[0]) == seq_len:
            return [[row[i] for i in keep] for row in value]
        if len(value) == seq_len:
            return [value[i] for i in keep]
        return value
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 1 and shape[-1] == seq_len:
        return value[..., keep]
    return value


def _find_extra_vision_boundary_indices(
    seq: list[int],
    *,
    vision_start_id: int,
    vision_end_id: int,
    video_pad_id: int,
) -> list[int]:
    vs_positions = [i for i, tid in enumerate(seq) if tid == vision_start_id]
    ve_positions = [i for i, tid in enumerate(seq) if tid == vision_end_id]
    pad_positions = [i for i, tid in enumerate(seq) if tid == video_pad_id]

    if pad_positions and len(vs_positions) >= 2 and len(ve_positions) >= 2:
        first_pad = pad_positions[0]
        last_pad = pad_positions[-1]
        first_vs = vs_positions[0]
        last_ve = ve_positions[-1]
        has_outer_start = (
            first_vs < first_pad
            and first_vs + 1 < len(seq)
            and seq[first_vs + 1] != video_pad_id
            and any(first_vs < pos < first_pad for pos in vs_positions)
        )
        has_outer_end = (
            last_ve > last_pad
            and last_ve - 1 >= 0
            and seq[last_ve - 1] != video_pad_id
            and any(last_pad < pos < last_ve for pos in ve_positions)
        )
        if has_outer_start and has_outer_end:
            return [first_vs, last_ve]

    if not pad_positions and len(vs_positions) >= 2 and len(ve_positions) >= 2:
        if ve_positions[-1] == ve_positions[-2] + 1 and vs_positions[0] < vs_positions[1]:
            return [vs_positions[0], ve_positions[-1]]

    return []


def detect_or_align_vision_boundary_tokens(
    inputs: Mapping[str, Any],
    *,
    processor: Any = None,
    mode: str | None = None,
    vision_start_token_id: int | None = None,
    vision_end_token_id: int | None = None,
    video_pad_token_id: int | None = None,
    transformers_version: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Detect or align redundant outer ``<|vision_start|>/<|vision_end|>`` tokens (Issue #11).

    In Transformers >=5.22, the Qwen3.5/Qwen3-VL processor retains outer
    ``<|vision_start|>`` and ``<|vision_end|>`` markers around expanded video
    tokens compared to historical ``transformers==5.9.0``, shifting multimodal
    RoPE positions by 2 and lowering D3/D4 scores.
    """
    resolved_mode = resolve_vision_boundary_mode(mode)
    vs_id = _resolve_special_token_id(
        processor,
        "<|vision_start|>",
        explicit=vision_start_token_id,
        attr_names=("vision_start_token_id",),
        default_id=DEFAULT_VISION_START_TOKEN_ID,
    )
    ve_id = _resolve_special_token_id(
        processor,
        "<|vision_end|>",
        explicit=vision_end_token_id,
        attr_names=("vision_end_token_id",),
        default_id=DEFAULT_VISION_END_TOKEN_ID,
    )
    pad_id = _resolve_special_token_id(
        processor,
        "<|video_pad|>",
        explicit=video_pad_token_id,
        attr_names=("video_token_id", "video_pad_token_id"),
        default_id=DEFAULT_VIDEO_PAD_TOKEN_ID,
    )

    out_inputs: dict[str, Any] = dict(inputs)
    seq, seq_len = _sequence_to_int_list(out_inputs.get("input_ids"))
    extra_indices = (
        _find_extra_vision_boundary_indices(
            seq,
            vision_start_id=vs_id,
            vision_end_id=ve_id,
            video_pad_id=pad_id,
        )
        if seq is not None
        else []
    )

    tf_ver = (
        transformers_version
        if transformers_version is not None
        else version_or_unavailable("transformers")
    )
    tf_differs = tf_ver not in ("unavailable", "unknown", "") and not tf_ver.startswith(
        HISTORICAL_TRANSFORMERS_PREFIX
    )

    info: dict[str, Any] = {
        "mode": resolved_mode,
        "transformers_version": tf_ver,
        "transformers_version_differs_from_5_9": tf_differs,
        "extra_boundary_tokens_detected": bool(extra_indices),
        "extra_boundary_indices": list(extra_indices),
        "vision_boundary_aligned": False,
        "warning": None,
    }

    if extra_indices and resolved_mode == "strip_outer_markers":
        remove = set(extra_indices)
        for field_name in ("input_ids", "attention_mask", "mm_token_type_ids"):
            if field_name in out_inputs and out_inputs[field_name] is not None:
                out_inputs[field_name] = _strip_sequence_indices(out_inputs[field_name], remove, seq_len)
        info["vision_boundary_aligned"] = True
        info["original_seq_len"] = seq_len
        info["aligned_seq_len"] = seq_len - len(remove)
    elif resolved_mode == "warn":
        if extra_indices:
            info["warning"] = (
                f"Detected extra outer <|vision_start|>/<|vision_end|> tokens at indices {extra_indices} "
                f"(transformers=={tf_ver}). Newer Transformers processors retain redundant vision boundary "
                "markers around expanded video tokens, shifting multimodal RoPE by 2 and lowering D3/D4 scores "
                "(Issue #11; see docs/eval/hunyuan-d3-replay.md). Set "
                "QWEN35_VISION_BOUNDARY_MODE=strip_outer_markers to align tokens or install transformers==5.9.0 "
                "for historical replay."
            )
        elif tf_differs:
            info["warning"] = (
                f"Running Qwen3.5 scorer under transformers=={tf_ver} (historical replay baseline is 5.9.0). "
                "If your Transformers processor retains extra outer <|vision_start|>/<|vision_end|> tokens, "
                "D3/D4 scores can shift (Issue #11; see docs/eval/hunyuan-d3-replay.md). Set "
                "QWEN35_VISION_BOUNDARY_MODE=strip_outer_markers or install transformers==5.9.0."
            )

    return out_inputs, info


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
