"""Shared SE(3) / OpenCV C2W geometry math helpers for D1 trajectory & intent scoring."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def rot_y(deg: float) -> np.ndarray:
    """Return a 3x3 OpenCV rotation matrix around the +Y (down) axis."""
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def identity_poses(frames: int) -> np.ndarray:
    return np.repeat(np.eye(4, dtype=np.float64)[None], max(int(frames), 1), axis=0)


def expand_poses(poses: Any) -> np.ndarray:
    arr = np.asarray(poses, dtype=np.float64)
    if arr.ndim == 2 and arr.shape == (4, 4):
        return arr[None, :, :]
    if arr.ndim == 3 and arr.shape[1:] == (4, 4):
        return arr
    raise ValueError(f"Expected (N, 4, 4) or (4, 4) poses array, got shape {arr.shape}")


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cos_angle = (np.trace(rotation) - 1.0) * 0.5
    return math.degrees(math.acos(float(np.clip(cos_angle, -1.0, 1.0))))


def relative_poses(poses: np.ndarray) -> np.ndarray:
    return np.linalg.inv(poses[0]) @ poses


def yaw_degrees(poses: np.ndarray, *, unwrap: bool = False) -> np.ndarray:
    rel = relative_poses(poses)
    angles = np.arctan2(rel[:, 0, 2], rel[:, 0, 0])
    if unwrap:
        angles = np.unwrap(angles)
    return np.degrees(angles)


def translation_offset(poses: np.ndarray) -> np.ndarray:
    return poses[:, :3, 3] - poses[0, :3, 3]


def translation_relative(poses: np.ndarray) -> np.ndarray:
    rel = relative_poses(poses)
    return rel[:, :3, 3]


def peak_signed(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(values[int(np.argmax(np.abs(values)))])


def movement_strength_rot(poses: np.ndarray) -> float:
    rel = relative_poses(poses)
    if len(rel) == 0:
        return 0.0
    return float(max(rotation_angle_deg(pose[:3, :3]) for pose in rel))


def movement_strength_trans(poses: np.ndarray) -> float:
    trans = translation_offset(poses)
    if len(trans) == 0:
        return 0.0
    return float(math.sqrt(float(np.max(np.sum(trans * trans, axis=1)))))


def return_metrics(poses: np.ndarray) -> tuple[float, float]:
    rel = relative_poses(poses)
    if len(rel) == 0:
        return 0.0, 0.0
    last = rel[-1]
    return float(rotation_angle_deg(last[:3, :3])), float(np.linalg.norm(last[:3, 3]))
