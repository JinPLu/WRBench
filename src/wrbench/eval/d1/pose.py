"""Pose trajectory loading, normalization, and SE(3) comparison helpers for D1."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from wrbench.eval.d1._math import rotation_angle_deg

PRIMARY_TARGET_POSE_PATH_FIELDS = (
    "target_pose_path",
    "input_pose_path",
    "trajectory_c2w_path",
)
POSE_INLINE_FIELDS = ("target_poses_c2w", "input_poses_c2w")


def _expand_poses(poses: Any) -> np.ndarray:
    arr = np.asarray(poses, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.shape[-2:] == (3, 4):
        bottom = np.zeros((*arr.shape[:-2], 1, 4), dtype=arr.dtype)
        bottom[..., 0, 3] = 1.0
        arr = np.concatenate([arr, bottom], axis=-2)
    if arr.ndim != 3 or arr.shape[-2:] != (4, 4):
        raise ValueError(f"expected pose shape T x 4 x 4 or T x 3 x 4, got {arr.shape}")
    if len(arr) == 0:
        raise ValueError("pose stack is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("pose contains non-finite values")
    expected_bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=arr.dtype)
    if not np.allclose(arr[:, 3, :], expected_bottom, atol=1e-6):
        raise ValueError("pose homogeneous bottom row is invalid")
    det = np.linalg.det(arr[:, :3, :3])
    if not np.all(np.isfinite(det)) or np.any(np.abs(det) <= 1e-8):
        raise ValueError("pose rotation/extrinsic block is singular")
    return arr


def _camera_basis(convention: str) -> np.ndarray:
    normalized = str(convention).lower()
    if normalized == "opencv":
        return np.eye(4, dtype=np.float64)
    if normalized == "opengl":
        return np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
    raise ValueError(f"unknown camera convention: {convention}")


def _normalize_predicted_poses(
    predicted: Any,
    pose_type: str,
    source_camera_convention: str,
    target_camera_convention: str,
) -> np.ndarray:
    predicted_c2w = _expand_poses(predicted)
    normalized_pose_type = str(pose_type).lower()
    if normalized_pose_type == "c2w":
        pass
    elif normalized_pose_type == "w2c":
        predicted_c2w = np.linalg.inv(predicted_c2w)
    else:
        raise ValueError(f"unknown predicted pose type: {pose_type}")

    source_basis = _camera_basis(source_camera_convention)
    target_basis = _camera_basis(target_camera_convention)
    return predicted_c2w @ np.linalg.inv(source_basis) @ target_basis


def load_target_poses(row: dict[str, Any]) -> tuple[np.ndarray | None, str | None]:
    for field in PRIMARY_TARGET_POSE_PATH_FIELDS:
        value = row.get(field)
        if not value:
            continue
        path = Path(value)
        if not path.exists():
            return None, "missing_target_pose"
        try:
            return _expand_poses(np.load(path, allow_pickle=False)), None
        except ValueError:
            return None, "invalid_pose"
        except Exception:
            return None, "error"
    for field in POSE_INLINE_FIELDS:
        if row.get(field) is not None:
            try:
                return _expand_poses(row[field]), None
            except ValueError:
                return None, "invalid_pose"
    return None, "missing_target_pose"


def _translation_shape_error(target: np.ndarray, predicted: np.ndarray) -> float:
    target_rel = target[:, :3, 3] - target[0, :3, 3]
    pred_rel = predicted[:, :3, 3] - predicted[0, :3, 3]
    pred_norm_sq = float(np.sum(pred_rel * pred_rel))
    if pred_norm_sq > 1e-12:
        scale = max(float(np.sum(target_rel * pred_rel)) / pred_norm_sq, 0.0)
        pred_rel = pred_rel * scale
    target_scale = math.sqrt(float(np.mean(np.sum(target_rel * target_rel, axis=1)))) + 1e-8
    rmse = math.sqrt(float(np.mean(np.sum((target_rel - pred_rel) ** 2, axis=1))))
    return rmse / target_scale


def score_trajectory_pose(
    target_poses_c2w: Any,
    predicted_poses_c2w: Any,
    *,
    rot_scale_deg: float,
    trans_scale: float,
    predicted_pose_type: str,
    predicted_camera_convention: str,
    target_camera_convention: str,
) -> dict[str, Any]:
    target = _expand_poses(target_poses_c2w)
    predicted = _normalize_predicted_poses(
        predicted_poses_c2w,
        predicted_pose_type,
        predicted_camera_convention,
        target_camera_convention,
    )
    if len(target) != len(predicted):
        return {"pose_status": "length_mismatch"}
    if len(target) == 0:
        return {"pose_status": "invalid_pose"}
    try:
        alignment = target[0] @ np.linalg.inv(predicted[0])
    except np.linalg.LinAlgError:
        return {"pose_status": "invalid_pose"}
    predicted = alignment @ predicted
    rot_errors = [
        rotation_angle_deg(target[idx, :3, :3].T @ predicted[idx, :3, :3])
        for idx in range(len(target))
    ]
    rot_error = float(np.mean(rot_errors))
    trans_error = _translation_shape_error(target, predicted)
    reward = math.exp(-(rot_error / max(float(rot_scale_deg), 1e-8) + trans_error / max(float(trans_scale), 1e-8)))
    return {
        "pose_status": "ok",
        "pose_reward": max(0.0, min(1.0, reward)),
        "pose_rot_error_deg": rot_error,
        "pose_trans_error": trans_error,
    }
