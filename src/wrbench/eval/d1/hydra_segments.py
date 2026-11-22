"""Select generated continuations for camera scoring of concatenated clips."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from wrbench.eval._io import safe_video_id, sha256_file, write_json
from wrbench.eval.plugins import ConcatSegmentPolicy, SegmentPolicyPlugin, registered_segment_policies


def _matching_segment_policy(row: dict[str, Any] | Any) -> SegmentPolicyPlugin | None:
    if not isinstance(row, dict):
        return None
    policies = list(registered_segment_policies().values())
    for policy in policies:
        if not isinstance(policy, ConcatSegmentPolicy) and policy.applies(row):
            return policy
    return _matching_concat_policy(row)


def _matching_concat_policy(row: dict[str, Any]) -> ConcatSegmentPolicy | None:
    if row.get("scoring_video_surface") == "source_generated_concat":
        condition = _positive_int(row.get("condition_frame_count"))
        generated = _positive_int(row.get("generated_frame_count"))
        if condition and generated:
            return ConcatSegmentPolicy("manifest_concat", frozenset(), condition, generated)
    for policy in registered_segment_policies().values():
        if isinstance(policy, ConcatSegmentPolicy) and policy.applies(row):
            return policy
    return None


def is_hydra_row(row: dict[str, Any]) -> bool:
    if str(row.get("model") or "").lower() == "hydra":
        return True
    return _matching_segment_policy(row) is not None


def load_camera_sidecar(video_path: Path) -> dict[str, Any] | None:
    sidecar_path = Path(str(video_path) + ".camera.json")
    if not sidecar_path.is_file():
        return None
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _positive_int(value: Any) -> int | None:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count > 0 else None


def sidecar_condition_frames(sidecar: dict[str, Any] | None) -> int | None:
    if not isinstance(sidecar, dict):
        return None
    for key in ("condition_frame_count", "source_frame_count", "model_frame_count", "num_frames"):
        count = _positive_int(sidecar.get(key))
        if count is not None:
            return count
    return None


def _sidecar_generated_frames(sidecar: dict[str, Any] | None) -> int | None:
    if not isinstance(sidecar, dict):
        return None
    return _positive_int(sidecar.get("generated_frame_count"))


def video_frame_count(video_path: Path) -> int:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if count > 0:
        cap.release()
        return count
    decoded = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        decoded += 1
    cap.release()
    return decoded


def _safe_stem(video_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", video_path.stem).strip("._")
    return stem or "video"


def materialize_generated_only_clip(
    video_path: Path,
    out_path: Path,
    start_frame: int,
    frame_count: int,
) -> Path:
    import cv2  # type: ignore

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 8.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise ValueError(f"could not decode first frame: {video_path}")
        height, width = frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    written = 0
    try:
        while written < int(frame_count):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        cap.release()
    if written != int(frame_count):
        raise ValueError(
            f"expected to write {frame_count} generated frames from {video_path}, wrote {written}"
        )
    return out_path


def _existing_eval_path(row: dict[str, Any]) -> Path | None:
    value = row.get("eval_video_path")
    if value in (None, ""):
        return None
    return Path(str(value))


def _metadata(
    *,
    eval_video_path: Path,
    original_concat_path: Path,
    status: str,
    condition_frames: int | None = None,
    generated_frames: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    source_video_path: Any = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "eval_video_path": str(eval_video_path),
        "original_concat_path": str(original_concat_path),
        "evaluated_segment": "generated_only" if status != "unresolved" else "unresolved",
        "hydra_segment_status": status,
    }
    if condition_frames is not None:
        out["condition_frame_count"] = int(condition_frames)
    if generated_frames is not None:
        out["generated_frame_count"] = int(generated_frames)
    if start_frame is not None:
        out["eval_video_frame_start"] = int(start_frame)
    if end_frame is not None:
        out["eval_video_frame_end_exclusive"] = int(end_frame)
    if source_video_path not in (None, ""):
        out["source_video_path"] = str(source_video_path)
    return out


def _pose_segment_path(pose_file: Path) -> Path:
    return pose_file.with_suffix(pose_file.suffix + ".segment.json")


def save_pose_segment(row: dict[str, Any], pose_file: Path, segment: dict[str, Any]) -> None:
    """Record the input segment used to produce this exact pose stack."""
    if segment.get("evaluated_segment") == "generated_only":
        write_json(_pose_segment_path(pose_file), {
            "video_id": row.get("video_id"),
            "pose_sha256": sha256_file(pose_file),
            "segment": segment,
        })


def clear_pose_segment(pose_file: Path) -> None:
    """Remove any stale generated-segment receipt when writing a full-concat pose stack."""
    receipt = _pose_segment_path(pose_file)
    if receipt.exists():
        receipt.unlink()


def read_pose_segment(row: dict[str, Any], pose_file: Path) -> dict[str, Any]:
    try:
        receipt = json.loads(_pose_segment_path(pose_file).read_text(encoding="utf-8"))
        if (
            receipt["video_id"] == row.get("video_id")
            and receipt["pose_sha256"]
            and receipt["pose_sha256"] == sha256_file(pose_file)
            and receipt["segment"]["evaluated_segment"] == "generated_only"
        ):
            return receipt["segment"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return {}


def load_pose_segment(row: dict[str, Any], pose_file: Path) -> dict[str, Any]:
    """Attach generated-clip provenance without changing the original manifest."""
    return {**row, **read_pose_segment(row, pose_file)}


HYDRA_CANONICAL_CONDITION_FRAMES = 77
HYDRA_CANONICAL_GENERATED_FRAMES = 77


def _resolve_concat_counts(
    row: dict[str, Any],
    original_path: Path,
    *,
    total_count: int,
    allow_canonical_default: bool,
    policy: ConcatSegmentPolicy | None = None,
) -> tuple[dict[str, Any] | None, int | None, int | None, str | None, Any]:
    sidecar = load_camera_sidecar(original_path) if str(original_path) else None
    condition_frames = sidecar_condition_frames(sidecar)
    if sidecar is None and condition_frames is None:
        condition_frames = _positive_int(row.get("condition_frame_count"))
    source_video_path = row.get("source_video_path")
    if isinstance(sidecar, dict) and sidecar.get("source_video_path") not in (None, ""):
        source_video_path = sidecar.get("source_video_path")
    generated_frames = _positive_int(row.get("generated_frame_count")) or _sidecar_generated_frames(sidecar)

    active_policy = policy or _matching_concat_policy(row)
    cond_default = active_policy.condition_frames if active_policy else HYDRA_CANONICAL_CONDITION_FRAMES
    gen_default = active_policy.generated_frames if active_policy else HYDRA_CANONICAL_GENERATED_FRAMES
    segment_policy: str | None = None
    if sidecar is None and condition_frames is None and allow_canonical_default and total_count == cond_default + gen_default:
        condition_frames = cond_default
        if generated_frames is None:
            generated_frames = gen_default
        if cond_default == HYDRA_CANONICAL_CONDITION_FRAMES and gen_default == HYDRA_CANONICAL_GENERATED_FRAMES:
            segment_policy = "canonical_77_plus_77_default"
        elif active_policy is not None:
            segment_policy = active_policy.name
    return sidecar, condition_frames, generated_frames, segment_policy, source_video_path


def resolve_eval_video_for_row(
    row: dict[str, Any],
    clip_root: Path,
    materialize: bool = True,
    *,
    video_root: Path | None = None,
    allow_canonical_default: bool = True,
) -> tuple[Path, dict[str, Any]]:
    input_value = row.get("path") or row.get("video_path")
    if not input_value:
        return Path(""), {"hydra_segment_status": "missing_video_path"}
    original_path = Path(str(input_value))
    if not original_path.is_absolute() and video_root is not None:
        original_path = Path(video_root) / original_path
    policy = _matching_segment_policy(row)
    if not isinstance(policy, ConcatSegmentPolicy):
        return original_path, {}

    early_sidecar = load_camera_sidecar(original_path)
    early_cond = sidecar_condition_frames(early_sidecar) or _positive_int(row.get("condition_frame_count"))
    early_gen = _positive_int(row.get("generated_frame_count")) or _sidecar_generated_frames(early_sidecar)
    early_src = (
        early_sidecar.get("source_video_path")
        if isinstance(early_sidecar, dict) and early_sidecar.get("source_video_path") not in (None, "")
        else row.get("source_video_path")
    )

    existing = _existing_eval_path(row)
    if existing is not None:
        if not existing.is_absolute() and video_root is not None:
            existing = Path(video_root) / existing
        expected = early_gen or early_cond or policy.generated_frames
        if existing.is_file() and video_frame_count(existing) == expected:
            return existing, _metadata(
                eval_video_path=existing,
                original_concat_path=Path(str(row.get("original_concat_path") or original_path)),
                status="generated_only_existing",
                condition_frames=early_cond,
                generated_frames=expected,
                start_frame=0,
                end_frame=expected,
                source_video_path=early_src,
            )

    if not original_path.exists():
        return original_path, _metadata(
            eval_video_path=original_path,
            original_concat_path=original_path,
            status="unresolved",
            condition_frames=early_cond,
            generated_frames=early_gen,
            source_video_path=early_src,
        )

    total_frames = video_frame_count(original_path)
    sidecar, condition_frames, generated_frames, segment_policy, source_video_path = _resolve_concat_counts(
        row,
        original_path,
        total_count=total_frames,
        allow_canonical_default=allow_canonical_default,
        policy=policy,
    )

    if condition_frames is None:
        return original_path, _metadata(
            eval_video_path=original_path,
            original_concat_path=original_path,
            status="unresolved",
            condition_frames=condition_frames,
            generated_frames=generated_frames,
            source_video_path=source_video_path,
        )

    if total_frames == (generated_frames or condition_frames) and (
        row.get("evaluated_segment") == "generated_only"
        or (
            isinstance(sidecar, dict)
            and (
                sidecar.get("evaluated_segment") == "generated_only"
                or sidecar.get("hydra_segment_status") in {"already_generated_only", "generated_only_materialized"}
            )
        )
    ):
        return original_path, _metadata(
            eval_video_path=original_path,
            original_concat_path=Path(str(row.get("original_concat_path") or original_path)),
            status="already_generated_only",
            condition_frames=condition_frames,
            generated_frames=total_frames,
            start_frame=0,
            end_frame=total_frames,
            source_video_path=source_video_path,
        )

    if generated_frames is None:
        generated_frames = condition_frames

    if generated_frames is None or total_frames != condition_frames + generated_frames:
        return original_path, _metadata(
            eval_video_path=original_path,
            original_concat_path=original_path,
            status="unresolved",
            condition_frames=condition_frames,
            generated_frames=generated_frames,
            source_video_path=source_video_path,
        )

    video_id = safe_video_id(row.get("video_id") or _safe_stem(original_path))
    source = original_path.stat()
    clip_id = hashlib.sha256(
        repr((str(original_path.resolve()), source.st_size, source.st_mtime_ns, condition_frames, generated_frames)).encode()
    ).hexdigest()[:16]
    out_path = clip_root / f"{video_id}.{clip_id}.generated_only.mp4"
    if materialize and (not out_path.exists() or video_frame_count(out_path) != generated_frames):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        materialize_generated_only_clip(
            original_path,
            out_path,
            start_frame=condition_frames,
            frame_count=generated_frames,
        )
        generated_sidecar = dict(sidecar) if isinstance(sidecar, dict) else {"model": str(row.get("model") or "hydra")}
        if segment_policy:
            generated_sidecar["hydra_segment_policy"] = segment_policy
        generated_sidecar.update(
            {
                "evaluated_segment": "generated_only",
                "hydra_segment_status": "generated_only_materialized",
                "original_concat_path": str(original_path),
                "eval_video_path": str(out_path),
                "condition_frame_count": int(condition_frames),
                "generated_frame_count": int(generated_frames),
                "eval_video_frame_start": int(condition_frames),
                "eval_video_frame_end_exclusive": int(condition_frames + generated_frames),
            }
        )
        write_json(out_path.with_suffix(out_path.suffix + ".camera.json"), generated_sidecar, sort_keys=True)

    meta = _metadata(
        eval_video_path=out_path,
        original_concat_path=original_path,
        status="generated_only_materialized",
        condition_frames=condition_frames,
        generated_frames=generated_frames,
        start_frame=condition_frames,
        end_frame=condition_frames + generated_frames,
        source_video_path=source_video_path,
    )
    if segment_policy:
        meta["hydra_segment_policy"] = segment_policy
    return out_path, meta


def resolve_full_concat_pose_input(
    row: dict[str, Any],
    clip_root: Path,
    materialize: bool = True,
    *,
    video_root: Path | None = None,
    allow_canonical_default: bool = True,
) -> tuple[Path, dict[str, Any], int | None, bool]:
    """Resolve pose-extraction input for concat-output models using full-concat poses.

    Returns ``(video_path, eval_metadata, expected_frames, is_concat_row)``:
    - Non-concat rows return ``(original_path, {}, None, False)`` so standard
      ``max_frames`` sampling applies.
    - Rows that are already ``generated_only`` (or point to an existing generated
      clip) delegate to ``resolve_eval_video_for_row`` with ``is_concat_row=True``
      so ``max_frames`` will not truncate their generated segment.
    - Full-concat rows (e.g. HyDRA 77+77 = 154 frames) return the original concat
      video with ``eval_metadata={}`` and ``expected_frames=total_frames``, so
      VGGT runs on the full concat sequence without truncation and D1/CamAlign
      slice ``[condition:condition+generated]`` during scoring.
    """
    input_value = row.get("path") or row.get("video_path")
    if not input_value:
        return Path(""), {"hydra_segment_status": "missing_video_path"}, None, False
    original_path = Path(str(input_value))
    if not original_path.is_absolute() and video_root is not None:
        original_path = Path(video_root) / original_path
    policy = _matching_segment_policy(row)
    if not isinstance(policy, ConcatSegmentPolicy):
        return original_path, {}, None, False

    if (
        row.get("evaluated_segment") == "generated_only"
        or _existing_eval_path(row) is not None
    ):
        eval_path, meta = resolve_eval_video_for_row(
            row,
            clip_root=clip_root,
            materialize=materialize,
            video_root=video_root,
            allow_canonical_default=allow_canonical_default,
        )
        return eval_path, meta, meta.get("generated_frame_count"), True

    if not original_path.exists():
        early_sidecar = load_camera_sidecar(original_path)
        early_cond = sidecar_condition_frames(early_sidecar) or _positive_int(row.get("condition_frame_count"))
        early_gen = _positive_int(row.get("generated_frame_count")) or _sidecar_generated_frames(early_sidecar)
        return (
            original_path,
            _metadata(
                eval_video_path=original_path,
                original_concat_path=original_path,
                status="unresolved",
                condition_frames=early_cond,
                generated_frames=early_gen,
                source_video_path=row.get("source_video_path"),
            ),
            None,
            True,
        )

    total_frames = video_frame_count(original_path)
    sidecar, condition_frames, generated_frames, _segment_policy, source_video_path = _resolve_concat_counts(
        row,
        original_path,
        total_count=total_frames,
        allow_canonical_default=allow_canonical_default,
        policy=policy,
    )
    if condition_frames is None:
        return (
            original_path,
            _metadata(
                eval_video_path=original_path,
                original_concat_path=original_path,
                status="unresolved",
                condition_frames=condition_frames,
                generated_frames=generated_frames,
                source_video_path=source_video_path,
            ),
            None,
            True,
        )

    if total_frames == (generated_frames or condition_frames) and (
        isinstance(sidecar, dict)
        and (
            sidecar.get("evaluated_segment") == "generated_only"
            or sidecar.get("hydra_segment_status") in {"already_generated_only", "generated_only_materialized"}
        )
    ):
        meta = _metadata(
            eval_video_path=original_path,
            original_concat_path=Path(str(row.get("original_concat_path") or original_path)),
            status="already_generated_only",
            condition_frames=condition_frames,
            generated_frames=total_frames,
            start_frame=0,
            end_frame=total_frames,
            source_video_path=source_video_path,
        )
        return original_path, meta, total_frames, True

    if generated_frames is None:
        generated_frames = condition_frames

    if generated_frames is None or total_frames != condition_frames + generated_frames:
        return (
            original_path,
            _metadata(
                eval_video_path=original_path,
                original_concat_path=original_path,
                status="unresolved",
                condition_frames=condition_frames,
                generated_frames=generated_frames,
                source_video_path=source_video_path,
            ),
            None,
            True,
        )

    return original_path, {}, int(total_frames), True


def slice_concat_poses(
    row: dict[str, Any] | Any,
    poses: Any,
    *,
    policy: ConcatSegmentPolicy | None = None,
    allow_canonical_default: bool = True,
) -> tuple[Any, dict[str, Any]]:
    row_dict = dict(row) if not isinstance(row, dict) else row
    if (
        row_dict.get("evaluated_segment") == "generated_only"
        or str(row_dict.get("hydra_segment_status") or "")
        in {"already_generated_only", "generated_only_existing", "generated_only_materialized"}
    ):
        total = len(poses)
        cond = _positive_int(row_dict.get("condition_frame_count"))
        gen = _positive_int(row_dict.get("generated_frame_count")) or total
        meta = _metadata(
            eval_video_path=Path(str(row_dict.get("eval_video_path") or row_dict.get("path") or row_dict.get("video_path") or "")),
            original_concat_path=Path(str(row_dict.get("original_concat_path") or row_dict.get("path") or row_dict.get("video_path") or "")),
            status=str(row_dict.get("hydra_segment_status") or "already_generated_only"),
            condition_frames=cond,
            generated_frames=gen,
            start_frame=_positive_int(row_dict.get("eval_video_frame_start")) or 0,
            end_frame=_positive_int(row_dict.get("eval_video_frame_end_exclusive")) or total,
            source_video_path=row_dict.get("source_video_path"),
        )
        if row_dict.get("hydra_segment_policy"):
            meta["hydra_segment_policy"] = row_dict["hydra_segment_policy"]
        return poses, meta

    source_raw = row_dict.get("original_concat_path") or row_dict.get("path") or row_dict.get("video_path") or ""
    original_path = Path(str(source_raw)) if source_raw else Path("")
    sidecar, condition_frames, generated_frames, segment_policy, _ = _resolve_concat_counts(
        row_dict,
        original_path,
        total_count=len(poses),
        allow_canonical_default=allow_canonical_default,
        policy=policy,
    )

    if condition_frames is None:
        return poses, _metadata(
            eval_video_path=original_path,
            original_concat_path=original_path,
            status="unresolved",
            condition_frames=condition_frames,
            generated_frames=generated_frames,
            source_video_path=row_dict.get("source_video_path"),
        )

    if generated_frames is None:
        generated_frames = condition_frames

    total = len(poses)
    if total == generated_frames and (
        isinstance(sidecar, dict)
        and (
            sidecar.get("evaluated_segment") == "generated_only"
            or sidecar.get("hydra_segment_status") in {"already_generated_only", "generated_only_materialized"}
        )
    ):
        return poses, _metadata(
            eval_video_path=original_path,
            original_concat_path=original_path,
            status="already_generated_only",
            condition_frames=condition_frames,
            generated_frames=generated_frames,
            start_frame=0,
            end_frame=total,
            source_video_path=row_dict.get("source_video_path"),
        )

    if total == condition_frames + generated_frames:
        meta = _metadata(
            eval_video_path=original_path,
            original_concat_path=original_path,
            status="generated_only_from_concat",
            condition_frames=condition_frames,
            generated_frames=generated_frames,
            start_frame=condition_frames,
            end_frame=condition_frames + generated_frames,
            source_video_path=row_dict.get("source_video_path"),
        )
        if segment_policy:
            meta["hydra_segment_policy"] = segment_policy
        return poses[condition_frames : condition_frames + generated_frames], meta

    return poses, _metadata(
        eval_video_path=original_path,
        original_concat_path=original_path,
        status="unresolved",
        condition_frames=condition_frames,
        generated_frames=generated_frames,
        source_video_path=row_dict.get("source_video_path"),
    )


def prediction_eval_poses(
    row: dict[str, Any],
    poses: Any,
    *,
    allow_canonical_default: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Return generated-only poses and segment metadata for concat-output models (e.g. HyDRA)."""
    policy = _matching_segment_policy(row)
    if policy is not None and not isinstance(policy, ConcatSegmentPolicy):
        return policy.slice_poses(row, poses)
    if not is_hydra_row(row):
        return poses, {}
    return slice_concat_poses(
        row,
        poses,
        policy=policy if isinstance(policy, ConcatSegmentPolicy) else None,
        allow_canonical_default=allow_canonical_default,
    )
