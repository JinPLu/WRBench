#!/usr/bin/env python3
"""Extract full-FoV DINOv2 v2 visual-integrity features (D2).

Full-FoV preprocessing avoids Hugging Face AutoImageProcessor's implicit center
crop: it preserves aspect ratio by scaling the shorter edge (``resize_short_edge``),
applies rescale and ImageNet normalization before bottom/right zero-padding in
normalized tensor space to a multiple of ``patch_size``, and masks out
padding-only patch tokens during local best-match scoring.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from wrbench.eval._io import read_records, resolve_video_path, sha256_file, to_jsonable, write_json, write_jsonl

PREFIX = "d2_lg_v2_"
METHOD_NAME = "dinov2_fullfov_local_global_patch_masked_bestmatch_v2"
FORMULA_FAMILY = "fixed_nonlearned_fullfov_scalar_variants_v2"
LEARNING_STATUS = (
    "The candidate uses a pretrained DINOv2 backbone but no WRBench/D2-trained head, "
    "no fitted feature weights, and no normalization from the evaluated benchmark pool."
)

LOCAL_LOW_PERCENTILE = 20.0
FORBIDDEN_MAINLINE_FIELD = "d2_dinov2_temporal_consistency"
SELECTED_VISUAL_INTEGRITY_FIELD = "d2_selected_visual_integrity_score"
SIZE_FIELDS = ("height", "width", "longest_edge", "shortest_edge", "max_height", "max_width")
PAPER_SAMPLING = {"sample_policy": "time_fps", "sample_fps": 3.0, "min_frames": 4, "max_frames": 24, "batch_size": 4}


def _clip01(value: float) -> float:
    return 0.0 if not math.isfinite(float(value)) else float(min(1.0, max(0.0, float(value))))


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return _clip01(float(value))


def _l2_normalize(values: np.ndarray, axis: int = -1) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    return arr / np.where(norm <= 1e-12, 1.0, norm)


def _unique_sorted(indices: Iterable[int], source_frame_count: int) -> list[int]:
    return sorted({max(0, min(source_frame_count - 1, int(idx))) for idx in indices})


def _fixed_count_indices(source_frame_count: int, max_frames: int) -> list[int]:
    if source_frame_count <= 0 or max_frames <= 0:
        return []
    count = min(int(max_frames), int(source_frame_count))
    return [int(i) for i in np.linspace(0, source_frame_count - 1, count, dtype=np.int64)]


def _frame_indices_for_sampling(
    *,
    source_frame_count: int,
    source_fps: float | None,
    sample_policy: str,
    sample_fps: float,
    min_frames: int,
    max_frames: int,
) -> list[int]:
    if source_frame_count <= 0:
        return []
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0")
    min_frames = max(1, min(int(min_frames), int(max_frames), int(source_frame_count)))
    max_frames = min(int(max_frames), int(source_frame_count))
    if sample_policy == "fixed_count":
        return _fixed_count_indices(source_frame_count, max_frames)
    if sample_policy != "time_fps":
        raise ValueError(f"unsupported sample_policy: {sample_policy}")
    if source_fps is None or source_fps <= 0 or sample_fps <= 0:
        return _fixed_count_indices(source_frame_count, max_frames)

    step = max(float(source_fps) / float(sample_fps), 1.0)
    indices = _unique_sorted(
        [0, source_frame_count - 1, *np.arange(0, source_frame_count, step).round().astype(int).tolist()],
        source_frame_count,
    )
    if len(indices) < min_frames:
        indices = _unique_sorted(
            [*indices, *_fixed_count_indices(source_frame_count, min_frames)],
            source_frame_count,
        )
    if len(indices) > max_frames:
        positions = np.linspace(0, len(indices) - 1, max_frames, dtype=np.int64)
        indices = _unique_sorted([indices[int(pos)] for pos in positions], source_frame_count)
        if indices[0] != 0:
            indices = _unique_sorted([0, *indices], source_frame_count)
        if indices[-1] != source_frame_count - 1:
            indices = _unique_sorted([*indices, source_frame_count - 1], source_frame_count)
        if len(indices) > max_frames:
            middle = indices[1:-1]
            keep_middle = max_frames - 2
            if keep_middle > 0:
                pos = np.linspace(0, len(middle) - 1, keep_middle, dtype=np.int64)
                indices = [0, *[middle[int(i)] for i in pos], source_frame_count - 1]
            else:
                indices = [0, source_frame_count - 1][:max_frames]
    return indices


def _sample_frames(
    video_path: str,
    *,
    sample_policy: str,
    sample_fps: float,
    min_frames: int,
    max_frames: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    try:
        import decord

        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)
        total = len(vr)
        if total <= 0:
            return None, {"decode_backend": "decord", "decode_error": "empty_video"}
        source_fps = float(vr.get_avg_fps() or 0.0)
        idx = _frame_indices_for_sampling(
            source_frame_count=total,
            source_fps=source_fps,
            sample_policy=sample_policy,
            sample_fps=sample_fps,
            min_frames=min_frames,
            max_frames=max_frames,
        )
        return vr.get_batch(idx).asnumpy(), {
            "decode_backend": "decord",
            "decode_error": "",
            "source_fps": source_fps if source_fps > 0 else None,
            "source_frame_count": int(total),
            "sampled_frame_indices": idx,
            "sampled_frame_count": int(len(idx)),
            "sample_policy": sample_policy,
        }
    except Exception as exc:
        import cv2

        cap = cv2.VideoCapture(video_path)
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        reported_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            return None, {"decode_backend": "opencv", "decode_error": f"empty_video_after_decord:{exc}"}
        idx = _frame_indices_for_sampling(
            source_frame_count=len(frames),
            source_fps=source_fps if source_fps > 0 else None,
            sample_policy=sample_policy,
            sample_fps=sample_fps,
            min_frames=min_frames,
            max_frames=max_frames,
        )
        return np.stack([frames[int(i)] for i in idx], axis=0), {
            "decode_backend": "opencv",
            "decode_error": "",
            "source_fps": source_fps if source_fps > 0 else None,
            "source_frame_count": int(reported_total or len(frames)),
            "decoded_frame_count": int(len(frames)),
            "sampled_frame_indices": idx,
            "sampled_frame_count": int(len(idx)),
            "sample_policy": sample_policy,
        }


def split_dinov2_tokens(last_hidden_state: np.ndarray, num_register_tokens: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split HF DINOv2 hidden states into L2-normalized CLS and patch-token arrays."""
    hidden = np.asarray(last_hidden_state, dtype=np.float32)
    if hidden.ndim != 3:
        raise ValueError(f"expected hidden state shape [batch, tokens, dim], got {hidden.shape}")
    patch_start = 1 + max(0, int(num_register_tokens))
    if hidden.shape[1] <= patch_start:
        raise ValueError(f"DINOv2 output does not contain patch tokens: shape={hidden.shape}")
    return _l2_normalize(hidden[:, 0, :]), _l2_normalize(hidden[:, patch_start:, :])


def _size_value(size: Any, key: str) -> Any:
    if size is None:
        return None
    if isinstance(size, Mapping):
        return size.get(key)
    getter = getattr(size, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except TypeError:
            pass
    return getattr(size, key, None)


def _resolve_resize_short_edge(processor: Any) -> int:
    size = getattr(processor, "size", None)
    for key in ("shortest_edge", "short_edge"):
        value = _size_value(size, key)
        if isinstance(value, (int, float, np.generic)) and value > 0:
            return int(value)
    height, width = _size_value(size, "height"), _size_value(size, "width")
    if isinstance(height, (int, float, np.generic)) and isinstance(width, (int, float, np.generic)) and height > 0 and width > 0:
        return int(min(height, width))
    if isinstance(size, (int, float)) and size > 0:
        return int(size)
    crop_size = getattr(processor, "crop_size", None)
    numeric = [int(v) for v in (_size_value(crop_size, "height"), _size_value(crop_size, "width")) if isinstance(v, (int, float, np.generic)) and v > 0]
    return min(numeric) if numeric else 224


def _serialize_size_attr(value: Any) -> Any:
    if any(hasattr(value, key) for key in SIZE_FIELDS):
        return {key: to_jsonable(getattr(value, key)) for key in SIZE_FIELDS if getattr(value, key, None) is not None}
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        if converted is not value:
            return to_jsonable(converted)
    return to_jsonable(dict(value.items()) if hasattr(value, "items") else value)


def _processor_norm_config(processor: Any) -> dict[str, Any]:
    mean = getattr(processor, "image_mean", None)
    std = getattr(processor, "image_std", None)
    return {
        "image_mean": [float(v) for v in (mean if mean is not None else [0.485, 0.456, 0.406])],
        "image_std": [float(v) for v in (std if std is not None else [0.229, 0.224, 0.225])],
        "do_rescale": bool(getattr(processor, "do_rescale", True)),
        "rescale_factor": float(getattr(processor, "rescale_factor", 1.0 / 255.0)),
        "resize_short_edge": _resolve_resize_short_edge(processor),
        "processor_size": _serialize_size_attr(getattr(processor, "size", {}) or {}),
        "processor_crop_size": _serialize_size_attr(getattr(processor, "crop_size", {}) or {}),
        "processor_do_center_crop": bool(getattr(processor, "do_center_crop", False)),
    }


def resize_preserve_aspect_shape(height: int, width: int, resize_short_edge: int) -> tuple[int, int]:
    """Return an aspect-preserving resize shape as ``(height, width)``."""
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid source shape: {(height, width)}")
    scale = max(1, int(resize_short_edge)) / float(min(height, width))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def pad_shape_to_patch_multiple(height: int, width: int, patch_size: int) -> tuple[int, int]:
    """Return the next ``patch_size``-multiple shape as ``(height, width)``."""
    patch = max(1, int(patch_size))
    return int(math.ceil(height / patch) * patch), int(math.ceil(width / patch) * patch)


def valid_patch_mask_for_shape(
    resized_shape: tuple[int, int],
    padded_shape: tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    """Build a flat patch mask where patches intersecting only padding are false."""
    resized_h, resized_w = resized_shape
    padded_h, padded_w = padded_shape
    patch = max(1, int(patch_size))
    mask = np.zeros((padded_h // patch, padded_w // patch), dtype=bool)
    mask[: int(math.ceil(max(0, resized_h) / patch)), : int(math.ceil(max(0, resized_w) / patch))] = True
    return mask.reshape(-1)


def _preprocess_frame(
    frame: np.ndarray,
    *,
    norm_config: Mapping[str, Any],
    patch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from PIL import Image

    orig_h, orig_w = frame.shape[:2]
    res_h, res_w = resize_preserve_aspect_shape(orig_h, orig_w, int(norm_config["resize_short_edge"]))
    bicubic = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    arr = np.asarray(Image.fromarray(frame).resize((res_w, res_h), resample=bicubic), dtype=np.float32)
    if bool(norm_config["do_rescale"]):
        arr = arr * float(norm_config["rescale_factor"])
    mean = np.asarray(norm_config["image_mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(norm_config["image_std"], dtype=np.float32).reshape(1, 1, 3)
    normalized = (arr - mean) / std
    pad_h, pad_w = pad_shape_to_patch_multiple(res_h, res_w, patch_size)
    # Zero-pad in normalized space (bottom/right) rather than black RGB before normalization.
    pixel = np.zeros((3, pad_h, pad_w), dtype=np.float32)
    pixel[:, :res_h, :res_w] = normalized.transpose(2, 0, 1)
    return pixel, {
        PREFIX + "original_shape": [int(orig_h), int(orig_w), int(frame.shape[2] if frame.ndim == 3 else 1)],
        PREFIX + "resized_shape": [int(res_h), int(res_w)],
        PREFIX + "padded_shape": [int(pad_h), int(pad_w)],
        PREFIX + "patch_grid": [int(pad_h // patch_size), int(pad_w // patch_size)],
    }


def preprocess_fullfov_batch(
    frames: list[np.ndarray],
    *,
    norm_config: Mapping[str, Any],
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Preprocess frames to a batch-common padded size and valid-patch masks."""
    items = [_preprocess_frame(frame, norm_config=norm_config, patch_size=patch_size) for frame in frames]
    common_h, common_w = pad_shape_to_patch_multiple(
        max(p.shape[1] for p, _ in items),
        max(p.shape[2] for p, _ in items),
        patch_size,
    )
    batch = np.zeros((len(items), 3, common_h, common_w), dtype=np.float32)
    masks, metas = [], []
    for idx, (pixel, meta) in enumerate(items):
        _, h, w = pixel.shape
        batch[idx, :, :h, :w] = pixel
        mask = valid_patch_mask_for_shape(tuple(meta[PREFIX + "resized_shape"]), (common_h, common_w), patch_size)
        masks.append(mask)
        metas.append({
            **meta,
            PREFIX + "padded_shape": [int(common_h), int(common_w)],
            PREFIX + "patch_grid": [int(common_h // patch_size), int(common_w // patch_size)],
            PREFIX + "valid_patch_count": int(np.count_nonzero(mask)),
            PREFIX + "pad_patch_count": int(mask.size - np.count_nonzero(mask)),
            PREFIX + "valid_patch_mask": mask.astype(np.uint8).tolist(),
        })
    return batch, np.stack(masks, axis=0), metas, {
        PREFIX + "batch_common_padded_shape": [int(common_h), int(common_w)],
        PREFIX + "batch_common_patch_grid": [int(common_h // patch_size), int(common_w // patch_size)],
    }


def _local_bestmatch_stats(
    patches_a: np.ndarray,
    mask_a: np.ndarray,
    patches_b: np.ndarray,
    mask_b: np.ndarray,
) -> tuple[float, float]:
    valid_a = patches_a[np.asarray(mask_a, dtype=bool)]
    valid_b = patches_b[np.asarray(mask_b, dtype=bool)]
    if valid_a.size == 0 or valid_b.size == 0:
        return 0.0, 0.0
    sims = np.clip(np.matmul(valid_a.astype(np.float32), valid_b.astype(np.float32).T), 0.0, 1.0)
    matches = np.concatenate([np.max(sims, axis=1), np.max(sims, axis=0)], axis=0)
    return _clip01(float(np.mean(matches))), _clip01(float(np.percentile(matches.astype(np.float64), LOCAL_LOW_PERCENTILE)))


def compute_fullfov_candidate(
    global_features: np.ndarray,
    patch_features: np.ndarray,
    patch_masks: np.ndarray,
) -> dict[str, Any]:
    """Compute fixed-formula D2 full-FoV visual integrity score (``candidate_e``)."""
    globals_ = _l2_normalize(global_features)
    patches = _l2_normalize(patch_features)
    masks = np.asarray(patch_masks, dtype=bool)
    if globals_.ndim != 2 or patches.ndim != 3 or masks.ndim != 2:
        raise ValueError("invalid global/patch/mask array dimensions")
    if len(globals_) != len(patches) or len(globals_) != len(masks) or patches.shape[1] != masks.shape[1]:
        raise ValueError("global, patch, and mask shapes do not align")

    frame_count = int(len(globals_))
    if frame_count < 2:
        return {
            PREFIX + "candidate_e": None,
            PREFIX + "g_first_last": None,
            PREFIX + "local_mean_median": None,
            PREFIX + "local_low_p20": None,
            PREFIX + "local_low_min": None,
            PREFIX + "adjacent_global_cosines": [],
            PREFIX + "adjacent_local_means": [],
            PREFIX + "adjacent_local_lows": [],
            PREFIX + "sampled_frame_count": frame_count,
            PREFIX + "method": METHOD_NAME,
            PREFIX + "formula_family": FORMULA_FAMILY,
            PREFIX + "learned_head": False,
        }

    global_adjacent, local_means, local_lows = [], [], []
    for idx in range(frame_count - 1):
        global_adjacent.append(_clip01(float(np.sum(globals_[idx] * globals_[idx + 1]))))
        local_mean, local_low = _local_bestmatch_stats(patches[idx], masks[idx], patches[idx + 1], masks[idx + 1])
        local_means.append(local_mean)
        local_lows.append(local_low)

    l_p20 = _clip01(float(np.percentile(np.asarray(local_lows, dtype=np.float64), LOCAL_LOW_PERCENTILE)))
    g_first_last = _clip01(float(np.sum(globals_[0] * globals_[-1])))
    candidate_e = min(g_first_last, l_p20)

    return {
        PREFIX + "candidate_e": _finite_or_none(candidate_e),
        PREFIX + "g_first_last": g_first_last,
        PREFIX + "local_mean_median": _clip01(float(np.median(np.asarray(local_means, dtype=np.float64)))),
        PREFIX + "local_low_p20": l_p20,
        PREFIX + "local_low_min": _clip01(float(np.min(np.asarray(local_lows, dtype=np.float64)))),
        PREFIX + "adjacent_global_cosines": [float(v) for v in global_adjacent],
        PREFIX + "adjacent_local_means": [float(v) for v in local_means],
        PREFIX + "adjacent_local_lows": [float(v) for v in local_lows],
        PREFIX + "sampled_frame_count": frame_count,
        PREFIX + "method": METHOD_NAME,
        PREFIX + "formula_family": FORMULA_FAMILY,
        PREFIX + "learned_head": False,
    }


def _supports_interpolate_pos_encoding(model: Any) -> bool:
    try:
        return "interpolate_pos_encoding" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def _model_forward(model: Any, pixel_values: Any, *, interpolate_pos_encoding: bool) -> Any:
    if interpolate_pos_encoding and _supports_interpolate_pos_encoding(model):
        try:
            return model(pixel_values=pixel_values, interpolate_pos_encoding=True)
        except TypeError:
            pass
    return model(pixel_values=pixel_values)


def _extract_feature_tensors(
    frames: list[np.ndarray],
    processor: Any,
    model: Any,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    global_chunks, patch_chunks, mask_chunks, frame_metas, batch_metas = [], [], [], [], []
    patch_size = int(getattr(model.config, "patch_size", 14))
    num_register_tokens = int(getattr(model.config, "num_register_tokens", 0) or 0)
    norm_config = _processor_norm_config(processor)
    token_shape: list[int] | None = None

    for start in range(0, len(frames), batch_size):
        pixel_np, patch_masks, metas, batch_meta = preprocess_fullfov_batch(
            frames[start : start + batch_size],
            norm_config=norm_config,
            patch_size=patch_size,
        )
        with torch.inference_mode():
            out = _model_forward(model, torch.from_numpy(pixel_np).to(device), interpolate_pos_encoding=True)
        hidden = out.last_hidden_state.detach().float().cpu().numpy()
        global_feats, patch_feats = split_dinov2_tokens(hidden, num_register_tokens=num_register_tokens)
        if patch_feats.shape[1] != int(patch_masks.shape[1]):
            raise ValueError(
                f"DINOv2 patch-token count {patch_feats.shape[1]} != expected {int(patch_masks.shape[1])}"
            )
        token_shape = [int(v) for v in hidden.shape[1:]]
        global_chunks.append(global_feats)
        patch_chunks.append(patch_feats)
        mask_chunks.append(patch_masks)
        frame_metas.extend(metas)
        batch_metas.append(batch_meta)

    return (
        np.concatenate(global_chunks, axis=0),
        np.concatenate(patch_chunks, axis=0),
        np.concatenate(mask_chunks, axis=0),
        {
            PREFIX + "preprocess_policy": "full_fov_resize_short_edge_preserve_aspect_pad_patch_multiple_no_center_crop",
            PREFIX + "processor_contract": "manual preprocessing from processor mean/std/rescale/size; HF processor call bypassed to avoid silent center crop",
            PREFIX + "processor_config": norm_config,
            PREFIX + "patch_size": patch_size,
            PREFIX + "num_register_tokens": num_register_tokens,
            PREFIX + "hidden_token_shape": token_shape,
            PREFIX + "interpolate_pos_encoding_requested": True,
            PREFIX + "interpolate_pos_encoding_supported": _supports_interpolate_pos_encoding(model),
            PREFIX + "frame_preprocess": frame_metas,
            PREFIX + "batch_preprocess": batch_metas,
            PREFIX + "valid_patch_count_min": int(min(item[PREFIX + "valid_patch_count"] for item in frame_metas)),
            PREFIX + "valid_patch_count_max": int(max(item[PREFIX + "valid_patch_count"] for item in frame_metas)),
            PREFIX + "pad_patch_count_max": int(max(item[PREFIX + "pad_patch_count"] for item in frame_metas)),
        },
    )


def merge_candidate_fields(
    row: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    select_visual_integrity: bool = True,
) -> dict[str, Any]:
    out = dict(row)
    out.pop(FORBIDDEN_MAINLINE_FIELD, None)
    attempted = sorted({FORBIDDEN_MAINLINE_FIELD, SELECTED_VISUAL_INTEGRITY_FIELD}.intersection(candidate))
    if attempted:
        raise ValueError(f"candidate attempted to overwrite protected field(s): {attempted}")
    for key in candidate:
        if key.startswith("d2_") and not key.startswith(PREFIX):
            raise ValueError(f"candidate field lacks required {PREFIX} prefix: {key}")
    out.update(candidate)
    if select_visual_integrity:
        score = _finite_or_none(candidate.get(PREFIX + "candidate_e"))
        out[SELECTED_VISUAL_INTEGRITY_FIELD] = score
        if out.get("feature_status") == "ok" and score is not None:
            out["d2_status"] = "ok"
        elif out.get("feature_status") == "ok":
            out["d2_status"] = "insufficient_frames"
        else:
            out["d2_status"] = str(out.get("feature_status") or "error")
    return out


def _video_rows(path: Path, *, video_root: Path | None = None) -> list[dict[str, Any]]:
    out = []
    for row in read_records(path):
        vid = row.get("video_id")
        surface = "full_continuation" if row.get("evaluated_segment") == "generated_only" and row.get("original_concat_path") else None
        video_path = resolve_video_path(row, video_root=video_root, surface=surface)
        if isinstance(vid, str) and vid and video_path:
            out.append({**row, "video_path": video_path})
    return out


def _verified_video_sha256(row: Mapping[str, Any]) -> str | None:
    expected = row.get("sha256")
    if not expected:
        return None
    actual = sha256_file(str(row["video_path"]))
    if actual != str(expected).strip().lower():
        raise ValueError(f"Video SHA-256 mismatch for {row['video_id']}: expected {expected}, got {actual}")
    return actual


def _extract_one(
    row: Mapping[str, Any],
    processor: Any,
    model: Any,
    device: str,
    sample_policy: str,
    sample_fps: float,
    min_frames: int,
    max_frames: int,
    batch_size: int,
) -> dict[str, Any]:
    video_path = str(row["video_path"])
    actual_sha256 = _verified_video_sha256(row)
    sampling = {
        "sample_policy": sample_policy, "sample_fps": sample_fps,
        "min_frames": min_frames, "max_frames": max_frames, "batch_size": batch_size,
    }
    frames, sample_meta = _sample_frames(
        video_path,
        sample_policy=sample_policy,
        sample_fps=sample_fps,
        min_frames=min_frames,
        max_frames=max_frames,
    )
    base_row: dict[str, Any] = {
        **dict(row),
        "video_id": str(row["video_id"]),
        "video_path": video_path,
        "path": video_path,
        "model": row.get("model", ""),
        "family_id": row.get("family_id", ""),
        "variant_id": row.get("variant_id", ""),
        "camera_type": row.get("camera_type", ""),
        "event_tier": row.get("event_tier") or row.get("event") or "",
        "feature_status": "ok" if frames is not None else "decode_failed",
        PREFIX + "feature_status": "ok" if frames is not None else "decode_failed",
        PREFIX + "learning_status": LEARNING_STATUS,
        PREFIX + "formula_family": FORMULA_FAMILY,
        PREFIX + "sample_metadata": dict(sample_meta),
        PREFIX + "sampling_config": sampling,
        PREFIX + "paper_sampling": sampling == PAPER_SAMPLING,
        **sample_meta,
    }
    if actual_sha256:
        base_row[PREFIX + "video_sha256"] = actual_sha256
    if frames is None:
        empty = compute_fullfov_candidate(np.empty((0, 1)), np.empty((0, 1, 1)), np.empty((0, 1), dtype=bool))
        return merge_candidate_fields(base_row, empty)

    base_row[PREFIX + "input_frame_array_shape"] = [int(v) for v in frames.shape]
    g_feats, p_feats, p_masks, feat_meta = _extract_feature_tensors(list(frames), processor, model, device, batch_size)
    candidate = compute_fullfov_candidate(g_feats, p_feats, p_masks)
    candidate.update(feat_meta)
    return merge_candidate_fields(base_row, candidate)


def qa_rows(rows: list[Mapping[str, Any]], requested: int) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("feature_status") == "ok"]
    score_keys = {PREFIX + "candidate_e", PREFIX + "g_first_last", PREFIX + "local_low_p20", SELECTED_VISUAL_INTEGRITY_FIELD}
    missing_fields, out_of_bounds, wrong_prefix, forbidden_rows = [], [], [], []
    finite_candidate_count = selected_match_count = 0
    for row in rows:
        if row.get("feature_status") == "ok" and (PREFIX + "candidate_e") not in row:
            missing_fields.append({"video_id": row.get("video_id"), "field": PREFIX + "candidate_e"})
        cval = row.get(PREFIX + "candidate_e")
        sval = row.get(SELECTED_VISUAL_INTEGRITY_FIELD)
        if row.get("feature_status") == "ok" and isinstance(cval, (int, float)) and math.isfinite(float(cval)):
            finite_candidate_count += 1
            if row.get("d2_status") == "ok" and isinstance(sval, (int, float)) and math.isclose(float(sval), float(cval), rel_tol=0.0, abs_tol=1e-12):
                selected_match_count += 1
        if FORBIDDEN_MAINLINE_FIELD in row:
            forbidden_rows.append(row.get("video_id"))
        for key, value in row.items():
            if key.startswith("d2_lg_") and not key.startswith(PREFIX):
                wrong_prefix.append({"video_id": row.get("video_id"), "field": key})
            if key in score_keys and isinstance(value, (int, float)) and math.isfinite(float(value)) and not (0.0 <= float(value) <= 1.0):
                out_of_bounds.append({"video_id": row.get("video_id"), "field": key, "value": value})

    def _counts(key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            val = str(row.get(key) or "")
            counts[val] = counts.get(val, 0) + 1
        return dict(sorted(counts.items()))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requested": int(requested),
        "rows_written": len(rows),
        "rows_requested_equals_written": int(requested) == len(rows),
        "feature_status_counts": _counts("feature_status"),
        "ok_rows": len(ok_rows),
        "non_ok_rows": len(rows) - len(ok_rows),
        "selected_visual_integrity_field": SELECTED_VISUAL_INTEGRITY_FIELD,
        "all_ok_selected_scores_match_candidate_e": bool(ok_rows) and selected_match_count == len(ok_rows),
        "forbidden_mainline_field": FORBIDDEN_MAINLINE_FIELD,
        "forbidden_mainline_present": bool(forbidden_rows),
        "all_ok_candidate_e_scores_finite": bool(ok_rows) and finite_candidate_count == len(ok_rows),
        "all_d2_lg_v2_scores_in_0_1": not out_of_bounds,
        "out_of_bounds": out_of_bounds[:20],
        "missing_audit_fields": missing_fields[:50],
        "wrong_prefix_fields": wrong_prefix[:50],
        "camera_type_counts": _counts("camera_type"),
        "model_family_counts": {k: v for k, v in _counts("model").items()},
        "method": METHOD_NAME,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True, help="Video manifest JSON/JSONL.")
    parser.add_argument("--video-root", type=Path, default=None, help="Optional base directory for relative video paths.")
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, help="Local DINOv2 model directory.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-policy", choices=("fixed_count", "time_fps"), default=PAPER_SAMPLING["sample_policy"])
    parser.add_argument("--sample-fps", type=float, default=PAPER_SAMPLING["sample_fps"])
    parser.add_argument("--min-frames", type=int, default=PAPER_SAMPLING["min_frames"])
    parser.add_argument("--max-frames", type=int, default=PAPER_SAMPLING["max_frames"])
    parser.add_argument("--batch-size", type=int, default=PAPER_SAMPLING["batch_size"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--qa-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.qa_only:
        if args.model_dir is None:
            parser.error("missing required argument: --model-dir")
        if args.sample_fps <= 0 or args.min_frames <= 0 or args.max_frames <= 0 or args.batch_size <= 0:
            parser.error("--sample-fps, --min-frames, --max-frames, and --batch-size must be > 0")
        if args.min_frames > args.max_frames:
            parser.error("--min-frames must be <= --max-frames")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    rows = _video_rows(args.videos, video_root=args.video_root)
    if args.limit:
        rows = rows[: args.limit]
    if args.qa_only:
        existing_rows = read_records(args.out_jsonl, allow_missing=True)
        summary = qa_rows(existing_rows, requested=len(rows) or len(existing_rows))
        write_json(args.out_jsonl.with_suffix(".qa.json"), summary)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        ok = summary["all_d2_lg_v2_scores_in_0_1"] and not summary["missing_audit_fields"] and not summary["wrong_prefix_fields"] and not summary["forbidden_mainline_present"] and summary["all_ok_selected_scores_match_candidate_e"]
        return 0 if ok else 2

    from transformers import AutoImageProcessor, AutoModel

    sampling = {key: getattr(args, key) for key in PAPER_SAMPLING}
    model_dir = str(args.model_dir.resolve())
    done: set[str] = set()
    if args.resume:
        by_id = {row["video_id"]: row for row in rows}
        kept = []
        for saved in read_records(args.out_jsonl, allow_missing=True):
            if saved.get("d2_status") != "ok":
                continue
            vid = saved.get("video_id")
            row = by_id.get(vid)
            if (
                row is None or vid in done or saved.get("video_path") != row["video_path"]
                or saved.get(PREFIX + "sampling_config") != sampling
                or saved.get(PREFIX + "model_dir") != model_dir
                or saved.get(PREFIX + "video_sha256") != _verified_video_sha256(row)
            ):
                raise ValueError("Cannot resume with changed inputs or D2 settings. Choose a new --out-jsonl.")
            kept.append(saved)
            done.add(vid)
        write_jsonl(kept, args.out_jsonl, atomic=True)
    processor = AutoImageProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True).to(args.device)
    model.eval()
    rows_to_run = [row for row in rows if row.get("video_id") not in done]
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    errored = 0
    with args.out_jsonl.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows_to_run, start=1):
            try:
                out = _extract_one(row, processor, model, args.device, args.sample_policy, args.sample_fps, args.min_frames, args.max_frames, args.batch_size)
                out[PREFIX + "model_dir"] = model_dir
            except Exception as exc:
                errored += 1
                out = {
                    "video_id": str(row.get("video_id", "")),
                    "video_path": str(row.get("video_path") or row.get("path") or ""),
                    "feature_status": "error",
                    "d2_status": "error",
                    SELECTED_VISUAL_INTEGRITY_FIELD: None,
                    PREFIX + "feature_status": "error",
                    PREFIX + "error": repr(exc),
                    PREFIX + "method": METHOD_NAME,
                    PREFIX + "learned_head": False,
                }
            handle.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            if idx % 10 == 0:
                print(f"processed {idx}/{len(rows_to_run)} this_run, total_done={len(done) + idx}/{len(rows)}", flush=True)

    summary = qa_rows(read_records(args.out_jsonl, allow_missing=True), requested=len(rows))
    summary.update({"videos": str(args.videos), "model_dir": str(args.model_dir), "out_jsonl": str(args.out_jsonl), "errored_this_time": errored, "device": args.device, "sample_policy": args.sample_policy, "sample_fps": args.sample_fps, "min_frames": args.min_frames, "max_frames": args.max_frames, "batch_size": args.batch_size})
    write_json(args.out_jsonl.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    ok = errored == 0 and summary["rows_requested_equals_written"] and summary["all_d2_lg_v2_scores_in_0_1"] and not summary["missing_audit_fields"] and not summary["wrong_prefix_fields"] and not summary["forbidden_mainline_present"] and summary["all_ok_selected_scores_match_candidate_e"]
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
