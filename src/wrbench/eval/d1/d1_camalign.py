#!/usr/bin/env python3
"""Score D1 prompt-camera alignment (CamAlign) from cached pose stacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from wrbench.eval._io import read_records, safe_video_id, write_json, write_jsonl
from wrbench.eval.d1.camera_intent import score_camera_intent_row
from wrbench.eval.d1.hydra_segments import load_pose_segment, prediction_eval_poses
from wrbench.eval.d1.nominal_target import camera_type

read_jsonl = read_records


def _camera_type(row: dict[str, Any]) -> str:
    return camera_type(row)


def _pose_file(row: dict[str, Any], cache_root: Path, poses_file: str) -> Path:
    return cache_root / "pose" / safe_video_id(row.get("video_id")) / poses_file


def score_camalign_row(
    row: dict[str, Any],
    *,
    cache_root: Path,
    poses_file: str,
    scorer_version: str = "v2_lite",
) -> dict[str, Any]:
    camera = _camera_type(row)
    if camera == "uncontrolled":
        out = dict(row)
        out.update(
            {
                "d1_camalign_score": None,
                "d1_camalign_status": "excluded_uncontrolled",
                "d1_camalign_metric_scope": "excluded",
            }
        )
        return out

    pose_file = _pose_file(row, cache_root, poses_file)
    if not pose_file.exists():
        out = dict(row)
        out.update(
            {
                "d1_camalign_score": None,
                "d1_camalign_status": "missing_output",
                "d1_camalign_metric_scope": "missing_pose",
            }
        )
        return out

    try:
        predicted = np.load(pose_file, allow_pickle=False)
        eval_poses, hydra_meta = prediction_eval_poses(load_pose_segment(row, pose_file), predicted)
        if hydra_meta.get("hydra_segment_status") == "unresolved":
            out = dict(row)
            out.update(hydra_meta)
            out.update(
                {
                    "d1_camalign_score": None,
                    "d1_camalign_status": "unresolved_hydra_segment",
                    "d1_camalign_metric_scope": "unresolved",
                }
            )
            return out
        scored_input = dict(row)
        if hydra_meta:
            scored_input.update(hydra_meta)
        return score_camera_intent_row(
            scored_input,
            eval_poses,
            scorer_version=scorer_version,
        )
    except Exception:
        out = dict(row)
        out.update(
            {
                "d1_camalign_score": None,
                "d1_camalign_status": "error",
                "d1_camalign_metric_scope": "error",
            }
        )
        return out


def summarize_camalign_rows(rows: list[dict[str, Any]], *, scorer_version: str) -> dict[str, Any]:
    counts: dict[str, int] = {}
    scores: list[float] = []
    v2_lite_scores: list[float] = []
    for row in rows:
        status = str(row.get("d1_camalign_status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
        score = row.get("d1_camalign_score")
        if score is not None:
            scores.append(float(score))
        v2_score = row.get("d1_camalign_v2_score")
        if v2_score is not None:
            v2_lite_scores.append(float(v2_score))
    return {
        "scorer_version": scorer_version,
        "total_rows": len(rows),
        "scored_rows": len(scores),
        "mean_score": (sum(scores) / len(scores)) if scores else None,
        "mean_v2_lite_score": (sum(v2_lite_scores) / len(v2_lite_scores)) if v2_lite_scores else None,
        "status_counts": counts,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score D1 prompt-camera alignment (CamAlign).")
    parser.add_argument("--input-jsonl", "--input_jsonl", dest="input_jsonl", required=True)
    parser.add_argument("--output-jsonl", "--output_jsonl", dest="output_jsonl", required=True)
    parser.add_argument(
        "--pose-cache-root",
        "--pose_cache_root",
        dest="pose_cache_root",
        required=True,
    )
    parser.add_argument("--poses-file", "--poses_file", dest="poses_file", default="poses.npy")
    parser.add_argument(
        "--scorer-version",
        "--scorer_version",
        dest="scorer_version",
        choices=("v2_lite", "v1"),
        default="v2_lite",
    )
    parser.add_argument("--summary-json", "--summary_json", dest="summary_json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache_root = Path(args.pose_cache_root)
    rows = [
        score_camalign_row(
            row,
            cache_root=cache_root,
            poses_file=args.poses_file,
            scorer_version=args.scorer_version,
        )
        for row in read_jsonl(Path(args.input_jsonl))
    ]
    write_jsonl(rows, Path(args.output_jsonl))
    if args.summary_json:
        write_json(
            Path(args.summary_json),
            summarize_camalign_rows(rows, scorer_version=args.scorer_version),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
