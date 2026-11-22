from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wrbench.eval import plugins
from wrbench.eval.d1 import d1_camalign, d1_camera, hydra_segments, run_d1_vggt_omega_batch


@pytest.mark.parametrize(
    ("model", "condition_frames", "generated_frames", "max_frames"),
    [("hydra", 77, 77, None), ("custom-concat", 4, 8, 6), ("manifest-concat", 3, 6, None)],
)
def test_pose_export_hands_off_generated_segment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    condition_frames: int,
    generated_frames: int,
    max_frames: int | None,
) -> None:
    if model == "custom-concat":
        policy = plugins.ConcatSegmentPolicy("custom_concat", frozenset({model}), condition_frames, generated_frames)
        monkeypatch.setitem(plugins._SEGMENT_POLICIES, policy.name, policy)

    row = {"video_id": "case", "model": model, "video_path": "clips/case.mp4", "camera_type": "static"}
    if model == "manifest-concat":
        row.update(
            scoring_video_surface="source_generated_concat",
            condition_frame_count=condition_frames,
            generated_frame_count=generated_frames,
        )
    mirror = tmp_path / "mirror"
    video = mirror / row["video_path"]
    video.parent.mkdir(parents=True)
    video.write_bytes(b"source plus generated")
    manifest = tmp_path / "videos.json"
    manifest.write_text(json.dumps([row]))
    output = tmp_path / "output"
    cache = output / "cache"
    pose_file = cache / "pose" / "case" / "poses.npy"
    actual_count = min(generated_frames, max_frames or generated_frames)
    poses = np.repeat(np.eye(4)[None], actual_count, axis=0)
    cut_calls = []
    inference_calls = []

    def video_frame_count(path: Path) -> int:
        return condition_frames + generated_frames if path == video else generated_frames

    def cut(src: Path, dst: Path, *, start_frame: int, frame_count: int) -> None:
        assert src == video
        cut_calls.append((start_frame, frame_count))
        dst.write_bytes(b"generated")

    def infer(argv: list[str], **_: object) -> SimpleNamespace:
        clip = Path(argv[argv.index("--video_path") + 1])
        assert clip.read_bytes() == b"generated"
        inference_calls.append(clip)
        pose_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(pose_file, poses)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(hydra_segments, "video_frame_count", video_frame_count)
    monkeypatch.setattr(hydra_segments, "materialize_generated_only_clip", cut)
    monkeypatch.setattr(run_d1_vggt_omega_batch.subprocess, "run", infer)
    args = run_d1_vggt_omega_batch.parse_args([
        "--input-jsonl", str(manifest),
        "--output-root", str(output),
        "--cache-root", str(cache),
        "--video-root", str(mirror),
        "--vggt-repo", str(tmp_path),
        "--checkpoint", str(tmp_path / "weights.pt"),
        "--python", "python",
        "--cwd", str(tmp_path),
        "--gpu-id", "0",
        "--shard-index", "0",
        "--num-shards", "1",
        "--image-resolution", "512",
        "--preprocess-mode", "balanced",
        "--execution-mode", "subprocess",
        "--poses-file", "poses.npy",
    ])
    args.max_frames = max_frames

    # Without provenance, a cache the right size doesn't prove what clip produced it.
    pose_file.parent.mkdir(parents=True)
    np.save(pose_file, poses)
    run = run_d1_vggt_omega_batch.run_batch
    assert run(args)["rows"][0]["status"] == "ok"
    assert run(args)["rows"][0]["status"] == "skipped_existing_pose"
    assert cut_calls == [(condition_frames, generated_frames)]
    assert len(inference_calls) == 1

    enriched = hydra_segments.load_pose_segment(row, pose_file)
    evaluated, info = hydra_segments.prediction_eval_poses(enriched, np.load(pose_file))
    np.testing.assert_array_equal(evaluated, poses)
    assert info["evaluated_segment"] == "generated_only"
    assert info["eval_video_frame_start"] == condition_frames
    assert enriched["video_path"] == row["video_path"]
    assert Path(info["original_concat_path"]) == video

    camalign = d1_camalign.score_camalign_row(row, cache_root=cache, poses_file=pose_file.name)
    assert camalign["d1_camalign_status"] == "ok"
    assert camalign["d1_camalign_score"] == 1.0
    camprec = d1_camera.score_d1_row(row, {
        **plugins.DEFAULT_D1_SCORER_CONFIG,
        "cache_root": cache,
        "require_main_profile": True,
        "target_fallback_mode": "nominal_profile",
    })
    assert camprec["d1_status"] == "ok"
    assert camprec["generated_frame_count"] == generated_frames

    # Even a replacement of the same length must not inherit an older stack's provenance.
    replacement = poses.copy()
    replacement[:, 0, 3] = 1
    np.save(pose_file, replacement)
    assert hydra_segments.load_pose_segment(row, pose_file) == row

    # A legacy full stack replacing the generated stack must ignore the old receipt and slice once.
    full = np.repeat(np.eye(4)[None], condition_frames + generated_frames, axis=0)
    np.save(pose_file, full)
    assert hydra_segments.load_pose_segment(row, pose_file) == row
    evaluated, info = hydra_segments.prediction_eval_poses(row, full)
    assert len(evaluated) == generated_frames
    assert info["hydra_segment_status"] == "generated_only_from_concat"


def test_full_concat_pose_mode_preserves_hydra_154_and_caps_non_hydra_to_49(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under paper-default --max-frames 49 + --concat-pose-mode full_concat:
    - Non-HyDRA videos pass --max_frames 49 to VGGT (saving (49, 4, 4) poses).
    - HyDRA 154-frame concat videos run full 154-frame VGGT without truncation
      or pre-slicing, and D1/CamAlign slice [77:154] during scoring.
    """
    mirror = tmp_path / "mirror"
    (mirror / "clips").mkdir(parents=True)
    hydra_video = mirror / "clips" / "hydra_clip.mp4"
    wan_video = mirror / "clips" / "wan_clip.mp4"
    hydra_video.write_bytes(b"hydra-154-concat")
    wan_video.write_bytes(b"wan-81-clip")

    rows = [
        {"video_id": "hydra_case", "model": "hydra", "video_path": "clips/hydra_clip.mp4", "camera_type": "static"},
        {"video_id": "wan_case", "model": "wan_2_2", "video_path": "clips/wan_clip.mp4", "camera_type": "static"},
    ]
    manifest = tmp_path / "videos.json"
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    output = tmp_path / "output"
    cache = output / "cache"

    captured_cmds: dict[str, list[str]] = {}

    def video_frame_count(path: Path) -> int:
        return 154 if path == hydra_video else 81

    def forbid_cut(*_a: object, **_kw: object) -> None:
        raise AssertionError("full_concat mode must not materialize generated_only clips for full HyDRA videos")

    def infer(argv: list[str], **_: object) -> SimpleNamespace:
        scene = argv[argv.index("--scene_name") + 1]
        out_dir = Path(argv[argv.index("--output_dir") + 1])
        captured_cmds[scene] = list(argv)
        n_frames = int(argv[argv.index("--max_frames") + 1]) if "--max_frames" in argv else 154
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "poses.npy", np.repeat(np.eye(4)[None], n_frames, axis=0))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(hydra_segments, "video_frame_count", video_frame_count)
    monkeypatch.setattr(hydra_segments, "materialize_generated_only_clip", forbid_cut)
    monkeypatch.setattr(run_d1_vggt_omega_batch.subprocess, "run", infer)

    args = run_d1_vggt_omega_batch.parse_args([
        "--input-jsonl", str(manifest),
        "--output-root", str(output),
        "--cache-root", str(cache),
        "--video-root", str(mirror),
        "--vggt-repo", str(tmp_path),
        "--checkpoint", str(tmp_path / "weights.pt"),
        "--python", "python",
        "--cwd", str(tmp_path),
        "--gpu-id", "0",
        "--shard-index", "0",
        "--num-shards", "1",
        "--max-frames", "49",
        "--concat-pose-mode", "full_concat",
        "--image-resolution", "512",
        "--preprocess-mode", "balanced",
        "--execution-mode", "subprocess",
        "--poses-file", "poses.npy",
    ])

    result = run_d1_vggt_omega_batch.run_batch(args)
    by_id = {r["video_id"]: r for r in result["rows"]}
    assert by_id["hydra_case"]["status"] == "ok"
    assert by_id["wan_case"]["status"] == "ok"

    # HyDRA ran on full concat MP4 without --max_frames 49 truncation -> (154, 4, 4)
    assert "--max_frames" not in captured_cmds["hydra_case"]
    assert Path(captured_cmds["hydra_case"][captured_cmds["hydra_case"].index("--video_path") + 1]) == hydra_video
    hydra_poses = np.load(cache / "pose" / "hydra_case" / "poses.npy")
    assert hydra_poses.shape == (154, 4, 4)

    # Wan ran with --max_frames 49 -> (49, 4, 4)
    assert captured_cmds["wan_case"][captured_cmds["wan_case"].index("--max_frames") + 1] == "49"
    wan_poses = np.load(cache / "pose" / "wan_case" / "poses.npy")
    assert wan_poses.shape == (49, 4, 4)

    # Re-running skips both cached pose stacks cleanly
    result2 = run_d1_vggt_omega_batch.run_batch(args)
    assert all(r["status"] == "skipped_existing_pose" for r in result2["rows"])

    # D1 and CamAlign slice HyDRA [77:154] cleanly
    camalign = d1_camalign.score_camalign_row(rows[0], cache_root=cache, poses_file="poses.npy")
    assert camalign["d1_camalign_status"] == "ok"
    assert camalign["hydra_segment_status"] == "generated_only_from_concat"
    assert camalign["eval_video_frame_start"] == 77
    assert camalign["eval_video_frame_end_exclusive"] == 154

    camprec = d1_camera.score_d1_row(rows[0], {
        **plugins.DEFAULT_D1_SCORER_CONFIG,
        "cache_root": cache,
        "require_main_profile": True,
        "target_fallback_mode": "requested_profile",
    })
    assert camprec["d1_status"] == "ok"
    assert camprec["hydra_segment_status"] == "generated_only_from_concat"
    assert camprec["d1_camera_accuracy"] == pytest.approx(1.0)
