"""Nominal D1 camera target helpers for manifest/profile evaluation without sidecars."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

import numpy as np

from wrbench.eval.d1._math import identity_poses, rot_y


_CAMERA_SUFFIXES = ("yaw_LR", "yaw_RL", "pan_LR", "pan_RL", "static")
_PROFILE_ANGLE_RE = re.compile(r"^(?:TI2V|TV2V|T2V|YAW)[_-]?(\d+(?:\.\d+)?)$", re.IGNORECASE)
_VIDEO_PROFILE_RE = re.compile(r"(?:^|__)profile_([A-Za-z0-9_-]+)(?:__|$)")


def camera_type(row: Mapping[str, Any]) -> str:
    """Resolve canonical camera label from row metadata or ``video_id`` suffix."""
    for field in ("camera_type", "camera", "canonical_camera", "intent"):
        value = row.get(field)
        if value not in (None, ""):
            return str(value).strip()
    video_id = str(row.get("video_id") or "")
    for suffix in _CAMERA_SUFFIXES:
        if f"__{suffix}__" in video_id or video_id.endswith(f"__{suffix}"):
            return suffix
    return ""


def _finite_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _registry_record(model: str) -> Any | None:
    if not model:
        return None
    try:
        from wrbench.registry import model_record

        return model_record(model)
    except Exception:
        return None


def is_prompt_only_row(row: Mapping[str, Any]) -> bool:
    """Return True for API prompt-camera rows without requested trajectories."""
    vct = str(row.get("viewpoint_condition") or row.get("viewpoint_condition_type") or "").strip()
    if vct:
        return vct == "prompt-only"
    if str(row.get("paper_group") or "").strip() == "API prompt-camera":
        return True
    if str(row.get("source_group") or "").strip() == "api_prompt_camera":
        return True
    record = _registry_record(str(row.get("model") or row.get("model_id") or ""))
    return bool(record is not None and record.viewpoint_condition_type == "prompt-only")


def _parse_profile_yaw_deg(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = _PROFILE_ANGLE_RE.match(text)
    if match:
        parsed = float(match.group(1))
        return parsed if parsed > 0.0 else None
    return None


def resolve_nominal_yaw(
    row: Mapping[str, Any] | None,
    *,
    camera: str,
    allow_model_default: bool = False,
) -> tuple[float | None, float | None, str]:
    """Resolve ``(signed_yaw_deg, abs_yaw_deg, source)`` from row fields/profiles."""
    if camera == "static":
        return 0.0, 0.0, "static_identity"
    if camera not in {"yaw_LR", "yaw_RL"}:
        return None, None, "unsupported_camera"
    sign = -1.0 if camera == "yaw_LR" else 1.0
    if row is not None:
        for field in ("d1_target_signed_yaw_deg", "yaw_peak_target_deg", "target_signed_yaw_deg"):
            value = _finite_float(row.get(field))
            if value is not None and abs(value) > 0.0:
                return float(value), abs(float(value)), f"abs({field})"
        for field in ("d1_target_abs_yaw_deg", "d1_target_nominal_yaw_deg", "requested_yaw_deg"):
            value = _finite_float(row.get(field))
            if value is not None and value > 0.0:
                abs_yaw = float(value)
                return sign * abs_yaw, abs_yaw, field
        for field in ("d1_profile", "profile", "pressure_level"):
            parsed = _parse_profile_yaw_deg(row.get(field))
            if parsed is not None:
                return sign * parsed, parsed, field
        video_id = str(row.get("video_id") or "")
        match = _VIDEO_PROFILE_RE.search(video_id)
        if match:
            parsed = _parse_profile_yaw_deg(match.group(1))
            if parsed is not None:
                return sign * parsed, parsed, "video_id_profile"
    if not allow_model_default:
        return None, None, "missing_nominal"
    if row is not None and is_prompt_only_row(row):
        return None, None, "prompt_only_ineligible"
    model = str((row or {}).get("model") or (row or {}).get("model_id") or "")
    record = _registry_record(model)
    if record is None:
        return None, None, "unsupported_model_policy"
    if record.model_input == "TV2V" or record.source_video_usage == "first_frame_extraction":
        return sign * 30.0, 30.0, "requested_control_tv2v_30deg_go_return"
    if record.model_input == "TI2V":
        return sign * 60.0, 60.0, "requested_control_ti2v_60deg_go_return"
    return None, None, "unsupported_model_policy"


def split_counts(total: int, ratios: tuple[float, ...] = (0.45, 0.10, 0.45)) -> list[int]:
    vals = [max(float(v), 0.0) for v in ratios]
    raw = [v / sum(vals) * total for v in vals]
    counts = [int(math.floor(v)) for v in raw]
    remain = total - sum(counts)
    order = sorted(range(len(vals)), key=lambda idx: raw[idx] - counts[idx], reverse=True)
    for idx in range(remain):
        counts[order[idx % len(vals)]] += 1
    return counts


def go_return_values(frames: int, peak: float, ratios: tuple[float, ...] = (0.45, 0.10, 0.45)) -> np.ndarray:
    frames = max(int(frames), 1)
    depart, hold, ret = split_counts(frames, ratios)
    depart = max(depart, 1)
    ret = max(ret, 1)
    values: list[float] = []
    values.extend(np.linspace(0.0, peak, depart, endpoint=True).tolist())
    values.extend([float(peak)] * hold)
    values.extend(np.linspace(peak, 0.0, ret, endpoint=True).tolist())
    if len(values) < frames:
        values.extend([0.0] * (frames - len(values)))
    return np.asarray(values[:frames], dtype=np.float64)


def yaw_target_trajectory(frames: int, peak_deg: float) -> np.ndarray:
    """Build an OpenCV C2W go-return yaw pose stack for ``frames`` frames."""
    poses = identity_poses(frames)
    for idx, value in enumerate(go_return_values(frames, peak_deg)):
        poses[idx, :3, :3] = rot_y(float(value))
    return poses


def build_script_nominal_target(row: Mapping[str, Any], frames: int) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Construct nominal OpenCV C2W requested-control trajectory from manifest/profile metadata."""
    camera = camera_type(row)
    if camera == "static":
        if is_prompt_only_row(row):
            return None, {
                "target_source": "excluded_prompt_only",
                "target_certification_status": "excluded",
                "d1_target_rule": "prompt_only_ineligible",
                "status": "excluded_prompt_only",
            }
        return identity_poses(frames), {
            "metric_tier": "opencv_c2w",
            "target_source": "nominal_profile_fallback",
            "target_certification_status": "nominal_requested_control",
            "target_coordinate_convention": "opencv_c2w",
            "evidence_level": "nominal_requested_control",
            "control_family": "static",
            "control_direction": "static",
            "control_profile": row.get("d1_profile") or row.get("profile") or "canonical_static",
            "d1_target_policy": "nominal_script_profile",
            "d1_target_rule": "static_identity",
            "d1_target_abs_yaw_deg": 0.0,
            "d1_target_signed_yaw_deg": 0.0,
            "yaw_peak_target_deg": 0.0,
            "status": "ok",
        }
    if camera not in {"yaw_LR", "yaw_RL"}:
        return None, {
            "target_source": "unsupported_camera_type",
            "target_certification_status": "missing",
            "d1_target_rule": "unsupported_camera_type",
            "status": "unsupported_camera_type",
        }
    signed_yaw, abs_yaw, source = resolve_nominal_yaw(row, camera=camera, allow_model_default=True)
    if signed_yaw is None or abs_yaw is None:
        status = "excluded_prompt_only" if source == "prompt_only_ineligible" else "unsupported_model_policy"
        return None, {
            "target_source": source,
            "target_certification_status": "excluded" if status == "excluded_prompt_only" else "missing",
            "d1_target_rule": source,
            "status": status,
        }
    return yaw_target_trajectory(frames, signed_yaw), {
        "metric_tier": "opencv_c2w",
        "target_source": "nominal_profile_fallback",
        "target_certification_status": "nominal_requested_control",
        "target_coordinate_convention": "opencv_c2w",
        "evidence_level": "nominal_requested_control",
        "control_family": "rotation",
        "control_direction": camera,
        "control_profile": row.get("d1_profile") or row.get("profile") or f"nominal_{int(round(abs_yaw))}deg",
        "d1_target_policy": "yaw_matrix_profile_explicit" if source not in {"requested_control_tv2v_30deg_go_return", "requested_control_ti2v_60deg_go_return"} else "requested_control_tv30_ti60",
        "d1_target_rule": source,
        "d1_target_abs_yaw_deg": float(abs_yaw),
        "d1_target_signed_yaw_deg": float(signed_yaw),
        "yaw_peak_target_deg": float(signed_yaw),
        "status": "ok",
    }
