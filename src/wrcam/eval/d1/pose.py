"""External MegaSAM/MagaSAM pose rewards for offline candidate scoring."""

from __future__ import annotations

import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import safe_video_id

PRIMARY_TARGET_POSE_PATH_FIELDS = (
    "target_pose_path",
    "input_pose_path",
    "trajectory_c2w_path",
)
POSE_INLINE_FIELDS = ("target_poses_c2w", "input_poses_c2w")


def _is_enabled(config: dict[str, Any] | None) -> bool:
    return bool((config or {}).get("enabled", True))


def _video_path(row: dict[str, Any]) -> str | None:
    value = row.get("path") or row.get("video_path")
    return str(value) if value else None


def _with_pose_status(row: dict[str, Any], status: str) -> dict[str, Any]:
    enriched = dict(row)
    enriched["pose_status"] = status
    if status != "ok":
        enriched.pop("pose_reward", None)
        enriched.pop("pose_rot_error_deg", None)
        enriched.pop("pose_trans_error", None)
        components = dict(enriched.get("reward_components") or {})
        components.pop("pose", None)
        if components:
            enriched["reward_components"] = components
        else:
            enriched.pop("reward_components", None)
    return enriched


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


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cos_angle = (np.trace(rotation) - 1.0) * 0.5
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


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
    rot_scale_deg: float = 45.0,
    trans_scale: float = 1.0,
    predicted_pose_type: str = "c2w",
    predicted_camera_convention: str = "opencv",
    target_camera_convention: str = "opencv",
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
        _rotation_angle_deg(target[idx, :3, :3].T @ predicted[idx, :3, :3])
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


def _pose_output_dir(row: dict[str, Any], config: dict[str, Any]) -> Path:
    cache_root = Path(config.get("cache_root") or ".cache/reward_scoring")
    return cache_root / "pose" / safe_video_id(row.get("video_id"))


def _resolve_megasam(config: dict[str, Any]) -> tuple[str | None, str]:
    repo = (
        config.get("megasam_repo")
        or config.get("magasam_repo")
        or os.environ.get("MEGASAM_REPO")
        or os.environ.get("MAGASAM_REPO")
    )
    python = (
        config.get("megasam_python")
        or config.get("magasam_python")
        or os.environ.get("MEGASAM_PYTHON")
        or os.environ.get("MAGASAM_PYTHON")
        or sys.executable
    )
    return str(repo) if repo else None, str(python)


def _stringify_stream(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _write_megasam_command_log(
    output_dir: Path,
    *,
    command: str | list[str],
    status: str,
    returncode: int | None = None,
    stdout: Any = None,
    stderr: Any = None,
    timeout: Any = None,
) -> None:
    command_text = command if isinstance(command, str) else shlex.join(map(str, command))
    lines = [
        f"status: {status}",
        f"command: {command_text}",
        f"returncode: {'' if returncode is None else returncode}",
    ]
    if timeout is not None:
        lines.append(f"timeout: {timeout}")
    lines.extend(
        [
            "stdout:",
            _stringify_stream(stdout),
            "stderr:",
            _stringify_stream(stderr),
        ]
    )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "megasam_command.log").write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass


def _run_megasam(row: dict[str, Any], output_dir: Path, config: dict[str, Any]) -> str | None:
    repo, python = _resolve_megasam(config)
    if not repo or not Path(repo).exists() or not Path(python).exists():
        return "megasam_unavailable"
    video_path = _video_path(row)
    if not video_path:
        return "skipped"
    output_dir.mkdir(parents=True, exist_ok=True)
    entrypoint = config.get("entrypoint", "demo.py")
    template = config.get("command_template")
    if template:
        command = template.format(
            python=shlex.quote(str(python)),
            repo=shlex.quote(str(repo)),
            entrypoint=shlex.quote(str(entrypoint)),
            video_path=shlex.quote(str(video_path)),
            video_id=shlex.quote(str(row.get("video_id"))),
            output_dir=shlex.quote(str(output_dir)),
            device=shlex.quote(str(config.get("device", "cpu"))),
        )
        shell = True
    else:
        command = [
            python,
            str(Path(repo) / entrypoint),
            "--video",
            video_path,
            "--out_dir",
            str(output_dir),
        ]
        shell = False
    try:
        result = subprocess.run(command, shell=shell, check=False, capture_output=True, text=True, timeout=config.get("timeout"))
    except FileNotFoundError as exc:
        _write_megasam_command_log(output_dir, command=command, status="file_not_found", stderr=exc)
        return "megasam_unavailable"
    except subprocess.TimeoutExpired as exc:
        _write_megasam_command_log(
            output_dir,
            command=command,
            status="timeout",
            stdout=exc.stdout,
            stderr=exc.stderr,
            timeout=exc.timeout,
        )
        return "megasam_failed"
    _write_megasam_command_log(
        output_dir,
        command=command,
        status="completed",
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
    if result.returncode != 0:
        return "megasam_failed"
    return None


def score_pose_row(row: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score one row against a target camera trajectory using cached/external MegaSAM poses."""
    cfg = dict(config or {})
    enriched = dict(row)
    if not _is_enabled(cfg):
        return _with_pose_status(row, "disabled")
    target, status = load_target_poses(row)
    if status:
        return _with_pose_status(row, status)
    output_dir = _pose_output_dir(row, cfg)
    pose_file = output_dir / cfg.get("poses_file", "poses.npy")
    had_cached_pose = pose_file.exists()
    if not had_cached_pose:
        status = _run_megasam(row, output_dir, cfg)
        if status:
            return _with_pose_status(row, status)
    if not pose_file.exists():
        return _with_pose_status(row, "missing_output")
    try:
        predicted = _expand_poses(np.load(pose_file, allow_pickle=False))
        score = score_trajectory_pose(
            target,
            predicted,
            rot_scale_deg=float(cfg.get("rot_scale_deg", 45.0)),
            trans_scale=float(cfg.get("trans_scale", 1.0)),
            predicted_pose_type=cfg.get("predicted_pose_type", "c2w"),
            predicted_camera_convention=cfg.get("predicted_camera_convention", "opencv"),
            target_camera_convention=cfg.get("target_camera_convention", "opencv"),
        )
    except ValueError:
        return _with_pose_status(row, "invalid_pose")
    except Exception:
        return _with_pose_status(row, "error")
    if score.get("pose_status") == "length_mismatch" and had_cached_pose:
        try:
            pose_file.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            return _with_pose_status(row, "error")
        status = _run_megasam(row, output_dir, cfg)
        if status:
            return _with_pose_status(row, status)
        if not pose_file.exists():
            return _with_pose_status(row, "missing_output")
        try:
            predicted = _expand_poses(np.load(pose_file, allow_pickle=False))
            score = score_trajectory_pose(
                target,
                predicted,
                rot_scale_deg=float(cfg.get("rot_scale_deg", 45.0)),
                trans_scale=float(cfg.get("trans_scale", 1.0)),
                predicted_pose_type=cfg.get("predicted_pose_type", "c2w"),
                predicted_camera_convention=cfg.get("predicted_camera_convention", "opencv"),
                target_camera_convention=cfg.get("target_camera_convention", "opencv"),
            )
        except ValueError:
            return _with_pose_status(row, "invalid_pose")
        except Exception:
            return _with_pose_status(row, "error")
    enriched.update(score)
    if score.get("pose_status") == "ok":
        components = dict(enriched.get("reward_components") or {})
        components["pose"] = score["pose_reward"]
        enriched["reward_components"] = components
    else:
        return _with_pose_status(enriched, str(score.get("pose_status", "error")))
    return enriched
