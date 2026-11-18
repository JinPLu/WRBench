"""D1 prompt-camera alignment (CamAlign v1 and v2-lite) scoring from recovered poses."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from wrbench.eval.d1._math import (
    peak_signed as _peak_signed,
    relative_poses as _math_relative_poses,
    rotation_angle_deg as _rotation_angle_deg,
)
from wrbench.eval.d1.nominal_target import camera_type, resolve_nominal_yaw

SCORER_VERSION = "d1_camera_intent_v1"
THRESHOLD_VERSION = "d1_camera_intent_thresholds_v1"
FORMULA_VERSION = "direction_first_magnitude_norm_v1"
V2_LITE_SCORER_VERSION = "d1_camalign_v2_lite"
V2_LITE_THRESHOLD_VERSION = "d1_camalign_v2_lite_thresholds_v1"
V2_LITE_FORMULA_VERSION = "direction_response_v2_lite"
POSE_CONVENTION = "opencv_c2w"
SUPPORTED_INTENTS = frozenset({"yaw_LR", "yaw_RL", "static"})
V2_LITE_KNOWN_COMPONENTS = frozenset({"yaw_LR", "yaw_RL", "pan_LR", "pan_RL"})
V2_LITE_ENABLED_COMPONENTS = frozenset({"yaw_LR", "yaw_RL"})

WEAK_YAW_DEG = 2.0
NOMINAL_YAW_FLOOR_DEG = 10.0
DEFAULT_NOMINAL_YAW_DEG = 30.0
WEAK_PAN = 0.02
NOMINAL_PAN_FLOOR = 0.05
DEFAULT_NOMINAL_PAN = 0.30
STATIC_ROT_OBVIOUS_DEG = 5.0
STATIC_TRANS_OBVIOUS = 0.10


def _base_result(
    *,
    status: str,
    score: float | None,
    direction: str | None,
    magnitude: float | None,
    metric_scope: str,
    aggregation_scope: str,
) -> dict[str, Any]:
    return {
        "d1_camalign_score": score,
        "d1_camalign_status": status,
        "d1_camalign_direction": direction,
        "d1_camalign_magnitude": magnitude,
        "d1_camalign_metric_scope": metric_scope,
        "d1_camalign_aggregation_scope": aggregation_scope,
        "d1_camalign_scorer_version": SCORER_VERSION,
        "d1_camalign_threshold_version": THRESHOLD_VERSION,
        "d1_camalign_formula_version": FORMULA_VERSION,
        "d1_camalign_pose_convention": POSE_CONVENTION,
    }


def _empty_v2_lite_diagnostics() -> dict[str, Any]:
    return {
        "d1_camalign_diag_final_yaw_deg": None,
        "d1_camalign_diag_net_yaw_deg": None,
        "d1_camalign_diag_final_translation": None,
        "d1_camalign_diag_net_translation": None,
        "d1_camalign_diag_backtrack_ratio": None,
        "d1_camalign_diag_total_variation_deg": None,
    }


def _base_v2_lite_result(
    *,
    status: str,
    score: float | None,
    direction: str | None,
    magnitude: float | None,
    metric_scope: str,
    aggregation_scope: str,
    components: list[str],
    component_scores: dict[str, float | None] | None = None,
    component_statuses: dict[str, str] | None = None,
    observed_abs_peak: dict[str, float | None] | None = None,
    aligned_peak: dict[str, float | None] | None = None,
    opposite_peak: dict[str, float | None] | None = None,
    denominator: dict[str, float | None] | None = None,
    nominal_source: dict[str, str | None] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "d1_camalign_score": score,
        "d1_camalign_status": status,
        "d1_camalign_direction": direction,
        "d1_camalign_magnitude": magnitude,
        "d1_camalign_metric_scope": metric_scope,
        "d1_camalign_aggregation_scope": aggregation_scope,
        "d1_camalign_scorer_version": V2_LITE_SCORER_VERSION,
        "d1_camalign_threshold_version": V2_LITE_THRESHOLD_VERSION,
        "d1_camalign_formula_version": V2_LITE_FORMULA_VERSION,
        "d1_camalign_pose_convention": POSE_CONVENTION,
        "d1_camalign_v2_score": score,
        "d1_camalign_v2_status": status,
        "d1_camalign_v2_direction": direction,
        "d1_camalign_v2_components": components,
        "d1_camalign_v2_component_scores": component_scores or {},
        "d1_camalign_v2_component_statuses": component_statuses or {},
        "d1_camalign_v2_observed_abs_peak": observed_abs_peak or {},
        "d1_camalign_v2_aligned_peak": aligned_peak or {},
        "d1_camalign_v2_opposite_peak": opposite_peak or {},
        "d1_camalign_v2_denominator": denominator or {},
        "d1_camalign_v2_nominal_source": nominal_source or {},
        "d1_camalign_v2_scorer_version": V2_LITE_SCORER_VERSION,
        "d1_camalign_v2_threshold_version": V2_LITE_THRESHOLD_VERSION,
        "d1_camalign_v2_formula_version": V2_LITE_FORMULA_VERSION,
        "d1_camalign_v2_pose_convention": POSE_CONVENTION,
    }
    result.update(diagnostics or _empty_v2_lite_diagnostics())
    return result


def _validate_poses(poses_c2w: Any) -> np.ndarray | None:
    try:
        poses = np.asarray(poses_c2w, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
        return None
    if not np.all(np.isfinite(poses)):
        return None
    expected_last_row = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if not np.allclose(poses[:, 3, :], expected_last_row[None], atol=1e-6):
        return None
    return poses


def _relative_poses(poses: np.ndarray) -> np.ndarray | None:
    try:
        return _math_relative_poses(poses)
    except np.linalg.LinAlgError:
        return None


def _yaw_degrees(relative: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(relative[:, 0, 2], relative[:, 0, 0]))


def _yaw_degrees_unwrapped(relative: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.arctan2(relative[:, 0, 2], relative[:, 0, 0])))


def _v2_lite_diagnostics(relative: np.ndarray) -> dict[str, Any]:
    yaw = _yaw_degrees_unwrapped(relative)
    trans = relative[:, :3, 3]
    if len(yaw) == 0 or len(trans) == 0:
        return _empty_v2_lite_diagnostics()
    total_variation_deg = float(np.sum(np.abs(np.diff(yaw)))) if len(yaw) > 1 else 0.0
    net_yaw_deg = float(yaw[-1] - yaw[0])
    if total_variation_deg <= 1e-12:
        backtrack_ratio = 0.0
    else:
        backtrack_ratio = float(np.clip((total_variation_deg - abs(net_yaw_deg)) / total_variation_deg, 0.0, 1.0))
    net_translation = trans[-1] - trans[0]
    return {
        "d1_camalign_diag_final_yaw_deg": float(yaw[-1]),
        "d1_camalign_diag_net_yaw_deg": net_yaw_deg,
        "d1_camalign_diag_final_translation": [float(value) for value in trans[-1]],
        "d1_camalign_diag_net_translation": [float(value) for value in net_translation],
        "d1_camalign_diag_backtrack_ratio": backtrack_ratio,
        "d1_camalign_diag_total_variation_deg": total_variation_deg,
    }


def _motion_strength(relative: np.ndarray) -> tuple[float, float]:
    rot_deg = 0.0 if len(relative) == 0 else max(_rotation_angle_deg(pose[:3, :3]) for pose in relative)
    trans = relative[:, :3, 3]
    trans_norm = math.sqrt(float(np.max(np.sum(trans * trans, axis=1)))) if len(trans) else 0.0
    return float(rot_deg), float(trans_norm)


def _observed_yaw_direction(yaw_peak_deg: float, weak_threshold_deg: float) -> str:
    if abs(yaw_peak_deg) < weak_threshold_deg:
        return "static"
    return "yaw_RL" if yaw_peak_deg > 0.0 else "yaw_LR"


def _scopes(intent: str) -> tuple[str, str]:
    if intent in {"yaw_LR", "yaw_RL"}:
        return "D1-CamAlign_common_yaw", "matched_yaw_common_denominator_candidate"
    if intent == "static":
        return "D1-CamAlign_static_hold", "static_hold_candidate"
    parts = [part.strip() for part in str(intent or "").split("+") if part.strip()]
    if parts and set(parts).issubset({"yaw_LR", "yaw_RL"}):
        return "D1-CamAlign_common_yaw", "matched_yaw_common_denominator_candidate"
    return "D1-CamAlign_unsupported", "unsupported"


def _parse_v2_lite_components(intent: str) -> list[str]:
    intent = str(intent or "").strip()
    if intent == "static":
        return ["static"]
    return [part.strip() for part in intent.split("+") if part.strip()]


def _float_from_row(row: Mapping[str, Any] | None, field: str) -> float | None:
    if row is None:
        return None
    value = row.get(field)
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _v2_lite_nominal(component: str, row: Mapping[str, Any] | None) -> tuple[float, str]:
    if component.startswith("yaw_"):
        _signed, abs_yaw, source = resolve_nominal_yaw(row, camera=component, allow_model_default=False)
        if abs_yaw is not None and abs_yaw > 0.0:
            return max(float(abs_yaw), NOMINAL_YAW_FLOOR_DEG), source
        return DEFAULT_NOMINAL_YAW_DEG, "default_nominal_yaw_deg"
    if component.startswith("pan_"):
        for field in ("d1_target_pan_norm", "d1_target_translation_norm"):
            value = _float_from_row(row, field)
            if value is not None and value > 0.0:
                return max(value, NOMINAL_PAN_FLOOR), field
        return DEFAULT_NOMINAL_PAN, "default_nominal_pan"
    return 1.0, "unsupported"


def _v2_lite_component_values(relative: np.ndarray, component: str) -> tuple[np.ndarray, float, float]:
    if component == "yaw_LR":
        return _yaw_degrees_unwrapped(relative), -1.0, WEAK_YAW_DEG
    if component == "yaw_RL":
        return _yaw_degrees_unwrapped(relative), 1.0, WEAK_YAW_DEG
    if component == "pan_LR":
        return relative[:, 0, 3], -1.0, WEAK_PAN
    if component == "pan_RL":
        return relative[:, 0, 3], 1.0, WEAK_PAN
    raise ValueError(f"unsupported component: {component}")


def _v2_lite_observed_direction(component: str, values: np.ndarray, weak: float) -> str:
    peak = _peak_signed(values)
    if abs(peak) < weak:
        return "static"
    if component.startswith("yaw_"):
        return "yaw_RL" if peak > 0.0 else "yaw_LR"
    if component.startswith("pan_"):
        return "pan_RL" if peak > 0.0 else "pan_LR"
    return "unknown"


def _v2_lite_component_score(
    relative: np.ndarray,
    *,
    component: str,
    row: Mapping[str, Any] | None,
) -> tuple[float, str, str, dict[str, float | str]]:
    values, sign, weak = _v2_lite_component_values(relative, component)
    aligned = sign * values
    aligned_peak = float(np.max(aligned)) if len(aligned) else 0.0
    opposite_peak = float(np.max(np.maximum(-aligned, 0.0))) if len(aligned) else 0.0
    observed_abs_peak = float(np.max(np.abs(values))) if len(values) else 0.0
    direction = _v2_lite_observed_direction(component, values, weak)
    denom, nominal_source = _v2_lite_nominal(component, row)

    if observed_abs_peak < weak:
        status = "weak_motion"
        response = 0.0
    elif opposite_peak > aligned_peak:
        status = "direction_mismatch"
        response = 0.0
    else:
        status = "ok"
        response = float(np.clip((aligned_peak - weak) / max(denom - weak, 1e-12), 0.0, 1.0))
        if math.isclose(response, 1.0, rel_tol=0.0, abs_tol=1e-12):
            response = 1.0

    return response, status, direction, {
        "observed_abs_peak": observed_abs_peak,
        "aligned_peak": aligned_peak,
        "opposite_peak": opposite_peak,
        "denominator": float(denom),
        "nominal_source": nominal_source,
    }


def score_camera_intent_v2_lite(
    poses_c2w: Any,
    *,
    intent: str,
    row: Mapping[str, Any] | None = None,
    static_rot_obvious_deg: float = STATIC_ROT_OBVIOUS_DEG,
    static_trans_obvious: float = STATIC_TRANS_OBVIOUS,
) -> dict[str, Any]:
    """Score requested camera-control components with the v2-lite formula."""
    intent_text = str(intent or "").strip()
    metric_scope, aggregation_scope = _scopes(intent_text)
    components = _parse_v2_lite_components(intent_text)
    if not components:
        return _base_v2_lite_result(
            status="unsupported_intent",
            score=None,
            direction=None,
            magnitude=None,
            metric_scope=metric_scope,
            aggregation_scope=aggregation_scope,
            components=[],
        )

    if components == ["static"]:
        poses = _validate_poses(poses_c2w)
        if poses is None:
            return _base_v2_lite_result(
                status="invalid_pose",
                score=None,
                direction=None,
                magnitude=None,
                metric_scope=metric_scope,
                aggregation_scope=aggregation_scope,
                components=components,
            )
        relative = _relative_poses(poses)
        if relative is None:
            return _base_v2_lite_result(
                status="invalid_pose",
                score=None,
                direction=None,
                magnitude=None,
                metric_scope=metric_scope,
                aggregation_scope=aggregation_scope,
                components=components,
            )
        rot_strength_deg, trans_strength = _motion_strength(relative)
        rot_motion = rot_strength_deg / max(float(static_rot_obvious_deg), 1e-6)
        trans_motion = trans_strength / max(float(static_trans_obvious), 1e-12)
        normalized_motion = max(rot_motion, trans_motion)
        static_score = float(np.clip(1.0 - normalized_motion, 0.0, 1.0))
        status = "ok" if static_score >= 1.0 else "motion_detected"
        return _base_v2_lite_result(
            status=status,
            score=static_score,
            direction="static",
            magnitude=float(normalized_motion),
            metric_scope=metric_scope,
            aggregation_scope=aggregation_scope,
            components=components,
            component_scores={"static": static_score},
            component_statuses={"static": status},
            observed_abs_peak={"static": float(normalized_motion)},
            aligned_peak={"static": float(1.0 - normalized_motion)},
            opposite_peak={"static": float(normalized_motion)},
            denominator={"static": 1.0},
            nominal_source={"static": "static_motion_thresholds"},
            diagnostics=_v2_lite_diagnostics(relative),
        )

    unknown = [component for component in components if component not in V2_LITE_KNOWN_COMPONENTS]
    disabled = [
        component
        for component in components
        if component in V2_LITE_KNOWN_COMPONENTS and component not in V2_LITE_ENABLED_COMPONENTS
    ]
    if unknown or disabled:
        status = "unsupported_component" if (disabled and not unknown) else "unsupported_intent"
        component_statuses = {component: status for component in components}
        return _base_v2_lite_result(
            status=status,
            score=None,
            direction=None,
            magnitude=None,
            metric_scope="D1-CamAlign_unsupported",
            aggregation_scope="unsupported",
            components=components,
            component_scores={component: None for component in components},
            component_statuses=component_statuses,
            observed_abs_peak={component: None for component in components},
            aligned_peak={component: None for component in components},
            opposite_peak={component: None for component in components},
            denominator={component: None for component in components},
            nominal_source={component: None for component in components},
        )

    poses = _validate_poses(poses_c2w)
    if poses is None:
        return _base_v2_lite_result(
            status="invalid_pose",
            score=None,
            direction=None,
            magnitude=None,
            metric_scope=metric_scope,
            aggregation_scope=aggregation_scope,
            components=components,
        )
    relative = _relative_poses(poses)
    if relative is None:
        return _base_v2_lite_result(
            status="invalid_pose",
            score=None,
            direction=None,
            magnitude=None,
            metric_scope=metric_scope,
            aggregation_scope=aggregation_scope,
            components=components,
        )

    component_scores: dict[str, float | None] = {}
    component_statuses: dict[str, str] = {}
    observed_abs_peak: dict[str, float | None] = {}
    aligned_peak: dict[str, float | None] = {}
    opposite_peak: dict[str, float | None] = {}
    denominator: dict[str, float | None] = {}
    nominal_source: dict[str, str | None] = {}
    directions: dict[str, str] = {}

    for component in components:
        response, status, direction, details = _v2_lite_component_score(relative, component=component, row=row)
        component_scores[component] = response
        component_statuses[component] = status
        observed_abs_peak[component] = float(details["observed_abs_peak"])
        aligned_peak[component] = float(details["aligned_peak"])
        opposite_peak[component] = float(details["opposite_peak"])
        denominator[component] = float(details["denominator"])
        nominal_source[component] = str(details["nominal_source"])
        directions[component] = direction

    numeric_scores = [score for score in component_scores.values() if score is not None]
    score = min(numeric_scores) if numeric_scores else None
    if all(status == "ok" for status in component_statuses.values()):
        status = "ok"
    elif any(status == "direction_mismatch" for status in component_statuses.values()):
        status = "direction_mismatch"
    elif any(status == "weak_motion" for status in component_statuses.values()):
        status = "weak_motion"
    else:
        status = "component_issue"
    direction = directions[components[0]] if len(components) == 1 else "component_mixed"
    peaks = [val for val in observed_abs_peak.values() if val is not None]
    magnitude = float(observed_abs_peak[components[0]]) if len(components) == 1 else (max(peaks) if peaks else None)

    return _base_v2_lite_result(
        status=status,
        score=None if score is None else float(score),
        direction=direction,
        magnitude=magnitude,
        metric_scope=metric_scope,
        aggregation_scope=aggregation_scope,
        components=components,
        component_scores=component_scores,
        component_statuses=component_statuses,
        observed_abs_peak=observed_abs_peak,
        aligned_peak=aligned_peak,
        opposite_peak=opposite_peak,
        denominator=denominator,
        nominal_source=nominal_source,
        diagnostics=_v2_lite_diagnostics(relative),
    )


def score_camera_intent(
    poses_c2w: Any,
    *,
    intent: str,
    yaw_weak_threshold_deg: float = WEAK_YAW_DEG,
    yaw_full_score_deg: float = DEFAULT_NOMINAL_YAW_DEG,
    static_rot_obvious_deg: float = STATIC_ROT_OBVIOUS_DEG,
    static_trans_obvious: float = STATIC_TRANS_OBVIOUS,
) -> dict[str, Any]:
    """Legacy v1 prompt-camera scorer retained for explicit ``--scorer-version v1`` comparisons."""
    intent = str(intent or "").strip()
    metric_scope, aggregation_scope = _scopes(intent)
    if intent not in SUPPORTED_INTENTS:
        return _base_result(
            status="unsupported_intent",
            score=None,
            direction=None,
            magnitude=None,
            metric_scope=metric_scope,
            aggregation_scope=aggregation_scope,
        )

    poses = _validate_poses(poses_c2w)
    if poses is None:
        return _base_result(
            status="invalid_pose",
            score=None,
            direction=None,
            magnitude=None,
            metric_scope=metric_scope,
            aggregation_scope=aggregation_scope,
        )

    relative = _relative_poses(poses)
    if relative is None:
        return _base_result(
            status="invalid_pose",
            score=None,
            direction=None,
            magnitude=None,
            metric_scope=metric_scope,
            aggregation_scope=aggregation_scope,
        )

    yaw_peak_deg = _peak_signed(_yaw_degrees(relative))
    yaw_magnitude_deg = abs(yaw_peak_deg)
    rot_strength_deg, trans_strength = _motion_strength(relative)

    if intent in {"yaw_LR", "yaw_RL"}:
        observed_direction = _observed_yaw_direction(yaw_peak_deg, yaw_weak_threshold_deg)
        if observed_direction == "static":
            return _base_result(
                status="weak_motion",
                score=0.0,
                direction=observed_direction,
                magnitude=float(yaw_magnitude_deg),
                metric_scope=metric_scope,
                aggregation_scope=aggregation_scope,
            )
        if observed_direction != intent:
            return _base_result(
                status="direction_mismatch",
                score=0.0,
                direction=observed_direction,
                magnitude=float(yaw_magnitude_deg),
                metric_scope=metric_scope,
                aggregation_scope=aggregation_scope,
            )
        denominator = max(float(yaw_full_score_deg) - float(yaw_weak_threshold_deg), 1e-6)
        score = (yaw_magnitude_deg - float(yaw_weak_threshold_deg)) / denominator
        return _base_result(
            status="ok",
            score=float(np.clip(score, 0.0, 1.0)),
            direction=observed_direction,
            magnitude=float(yaw_magnitude_deg),
            metric_scope=metric_scope,
            aggregation_scope=aggregation_scope,
        )

    rot_motion = rot_strength_deg / max(float(static_rot_obvious_deg), 1e-6)
    trans_motion = trans_strength / max(float(static_trans_obvious), 1e-12)
    normalized_motion = max(rot_motion, trans_motion)
    static_score = float(np.clip(1.0 - normalized_motion, 0.0, 1.0))
    status = "ok" if static_score >= 1.0 else "motion_detected"
    return _base_result(
        status=status,
        score=static_score,
        direction="static",
        magnitude=float(normalized_motion),
        metric_scope=metric_scope,
        aggregation_scope=aggregation_scope,
    )


def score_camera_intent_row(
    row: dict[str, Any],
    poses_c2w: Any,
    *,
    intent_field: str = "camera_type",
    scorer_version: str = "v2_lite",
) -> dict[str, Any]:
    """Score one manifest row (defaults to published ``v2_lite`` formula)."""
    intent = row.get(intent_field) or camera_type(row)
    scored = dict(row)
    version = str(scorer_version or "v2_lite").strip().lower()
    if version in {"v2_lite", "d1_camalign_v2_lite"}:
        scored.update(score_camera_intent_v2_lite(poses_c2w, intent=str(intent or ""), row=scored))
    elif version in {"v1", "d1_camera_intent_v1"}:
        scored.update(score_camera_intent(poses_c2w, intent=str(intent or "")))
    else:
        raise ValueError(f"unsupported CamAlign scorer_version: {scorer_version!r}")
    return scored
