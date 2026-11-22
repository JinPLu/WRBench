from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wrbench.cli import _build_parser
from wrbench.eval.aggregate import build_main_table as table_builder
from wrbench.eval.d1 import d1_camalign, d1_camera, hydra_segments
from wrbench.eval.d1.camera_intent import score_camera_intent_row, score_camera_intent_v2_lite
from wrbench.eval.d1.nominal_target import (
    build_script_nominal_target,
    rot_y,
    yaw_target_trajectory,
)
from wrbench.registry import all_records


def _yaw_curve_poses(peaks_deg: list[float]) -> np.ndarray:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(peaks_deg), axis=0)
    for idx, deg in enumerate(peaks_deg):
        poses[idx, :3, :3] = rot_y(float(deg))
    return poses


def test_camalign_v2_lite_scaling_and_empirical_denoms() -> None:
    # Matches frozen paper benchmark ratios to machine precision:
    # 60-deg profile uses denominator (60 - 2) = 58.0
    poses_gen3c_60 = _yaw_curve_poses([0.0, 25.0, 48.99793771985508, 20.0, 0.0])
    res_60 = score_camera_intent_v2_lite(
        poses_gen3c_60,
        intent="yaw_RL",
        row={"model": "gen3c", "profile": "TV2V60", "camera_type": "yaw_RL"},
    )
    assert res_60["d1_camalign_status"] == "ok"
    assert res_60["d1_camalign_score"] == pytest.approx(0.8103092710319841, abs=1e-12)
    assert res_60["d1_camalign_v2_denominator"]["yaw_RL"] == 60.0

    # pressure_level=yaw60 without profile also uses 60.0
    res_pressure = score_camera_intent_v2_lite(
        poses_gen3c_60,
        intent="yaw_RL",
        row={"model": "wan_2_2", "pressure_level": "yaw60", "camera_type": "yaw_RL"},
    )
    assert res_pressure["d1_camalign_v2_denominator"]["yaw_RL"] == 60.0

    # video_id profile suffix fallback without profile/pressure_level also uses 60.0
    res_vid = score_camera_intent_v2_lite(
        poses_gen3c_60,
        intent="yaw_RL",
        row={"video_id": "wan22__scene__T2_div_a__none__yaw_RL__profile_TI2V60"},
    )
    assert res_vid["d1_camalign_v2_denominator"]["yaw_RL"] == 60.0

    # 30-deg profile uses (30 - 2) = 28.0
    poses_30 = _yaw_curve_poses([0.0, 12.0, 24.241902816373207, 10.0, 0.0])
    res_30 = score_camera_intent_v2_lite(
        poses_30,
        intent="yaw_RL",
        row={"model": "gen3c", "profile": "TV2V30", "camera_type": "yaw_RL"},
    )
    assert res_30["d1_camalign_v2_denominator"]["yaw_RL"] == 30.0
    assert res_30["d1_camalign_score"] == pytest.approx((24.241902816373207 - 2.0) / 28.0, abs=1e-12)

    # API prompt-only row uses default 30.0 denominator (matching HF / paper numbers)
    poses_api = _yaw_curve_poses([0.0, 2.5, 4.981268134684671, 2.0, 0.0])
    res_api = score_camera_intent_row(
        {"model": "hailuo_2_3", "camera_type": "yaw_RL", "profile": "", "pressure_level": ""},
        poses_api,
    )
    assert res_api["d1_camalign_status"] == "ok"
    assert res_api["d1_camalign_score"] == pytest.approx(0.10647386195302398, abs=1e-12)
    assert res_api["d1_camalign_v2_denominator"]["yaw_RL"] == 30.0

    # Wrong sign yields direction_mismatch with score 0.0
    res_mismatch = score_camera_intent_v2_lite(
        poses_30,
        intent="yaw_LR",
        row={"model": "gen3c", "profile": "TV2V30"},
    )
    assert res_mismatch["d1_camalign_status"] == "direction_mismatch"
    assert res_mismatch["d1_camalign_score"] == 0.0

    # Weak motion (< 2 deg) yields weak_motion with score 0.0
    res_weak = score_camera_intent_v2_lite(
        _yaw_curve_poses([0.0, 1.0, 0.5, 0.0]),
        intent="yaw_RL",
        row={"model": "gen3c", "profile": "TV2V30"},
    )
    assert res_weak["d1_camalign_status"] == "weak_motion"
    assert res_weak["d1_camalign_score"] == 0.0

    # Static intent
    res_static = score_camera_intent_v2_lite(
        _yaw_curve_poses([0.0, 0.0, 0.0]),
        intent="static",
        row={"model": "gen3c", "camera_type": "static"},
    )
    assert res_static["d1_camalign_status"] == "ok"
    assert res_static["d1_camalign_score"] == 1.0


def test_nominal_target_builder_shapes_and_exclusions() -> None:
    poses_60, meta_60 = build_script_nominal_target(
        {"model": "gen3c", "profile": "TV2V60", "camera_type": "yaw_LR"},
        frames=61,
    )
    assert poses_60 is not None
    assert poses_60.shape == (61, 4, 4)
    assert meta_60["d1_target_abs_yaw_deg"] == 60.0
    assert meta_60["d1_target_signed_yaw_deg"] == -60.0
    assert meta_60["target_source"] == "nominal_profile_fallback"

    poses_cam, meta_cam = build_script_nominal_target(
        {"model": "recammaster", "camera_type": "yaw_RL"},
        frames=81,
    )
    assert poses_cam is not None
    assert poses_cam.shape == (81, 4, 4)
    assert meta_cam["d1_target_abs_yaw_deg"] == 30.0
    assert meta_cam["d1_target_signed_yaw_deg"] == 30.0

    poses_api, meta_api = build_script_nominal_target(
        {"model": "hailuo_2_3", "camera_type": "yaw_RL"},
        frames=100,
    )
    assert poses_api is None
    assert meta_api["status"] == "excluded_prompt_only"


def test_hydra_canonical_fallback_and_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    concat_poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 154, axis=0)
    concat_poses[77:] = yaw_target_trajectory(77, -30.0)
    row = {"model": "hydra", "camera_type": "yaw_LR", "path": "hydra/missing_sidecar.mp4"}

    sliced, meta = hydra_segments.prediction_eval_poses(row, concat_poses)
    assert sliced.shape == (77, 4, 4)
    assert meta["hydra_segment_status"] == "generated_only_from_concat"
    assert meta["hydra_segment_policy"] == "canonical_77_plus_77_default"
    assert meta["eval_video_frame_start"] == 77
    assert meta["eval_video_frame_end_exclusive"] == 154

    # 120 frames without sidecar must still fail closed
    _, bad_120 = hydra_segments.prediction_eval_poses(row, concat_poses[:120])
    assert bad_120["hydra_segment_status"] == "unresolved"

    # Unannotated 77 frames without sidecar or evaluated_segment=generated_only must fail closed
    _, bad_77 = hydra_segments.prediction_eval_poses(row, concat_poses[:77])
    assert bad_77["hydra_segment_status"] == "unresolved"

    # Video resolution with --video-root & missing sidecar canonical 77+77 fallback
    rel_video = tmp_path / "videos" / "hydra" / "clip.mp4"
    rel_video.parent.mkdir(parents=True, exist_ok=True)
    rel_video.write_bytes(b"fake-mp4")

    monkeypatch.setattr(hydra_segments, "video_frame_count", lambda _p: 154)
    materialized_calls: list[tuple[int, int]] = []

    def _fake_materialize(_src: Path, dst: Path, *, start_frame: int, frame_count: int) -> None:
        materialized_calls.append((start_frame, frame_count))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"generated-only")

    monkeypatch.setattr(hydra_segments, "materialize_generated_only_clip", _fake_materialize)
    eval_path, vmeta = hydra_segments.resolve_eval_video_for_row(
        {"video_id": "hydra_clip", "model": "hydra", "path": "hydra/clip.mp4"},
        clip_root=tmp_path / "clips",
        video_root=tmp_path / "videos",
    )
    assert materialized_calls == [(77, 77)]
    assert eval_path.exists()
    assert vmeta["hydra_segment_status"] == "generated_only_materialized"
    assert vmeta["hydra_segment_policy"] == "canonical_77_plus_77_default"

    # Explicit malformed/mismatched sidecar must NOT be overridden by 77+77 fallback
    sidecar_path = Path(str(rel_video) + ".camera.json")
    sidecar_path.write_text(json.dumps({"condition_frame_count": 50, "generated_frame_count": 50}), encoding="utf-8")
    _, explicit_bad = hydra_segments.resolve_eval_video_for_row(
        {"video_id": "hydra_clip2", "model": "hydra", "path": "hydra/clip.mp4"},
        clip_root=tmp_path / "clips2",
        video_root=tmp_path / "videos",
    )
    assert explicit_bad["hydra_segment_status"] == "unresolved"


def test_d1_nominal_profile_camalign_and_table_end_to_end(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    manifest_path = tmp_path / "manifest.json"
    d1_out = tmp_path / "d1_scored.jsonl"
    d1_summary = tmp_path / "d1_summary.csv"
    d1_failclosed = tmp_path / "d1_failclosed.jsonl"
    camalign_out = tmp_path / "camalign.jsonl"
    camalign_summary = tmp_path / "camalign_summary.json"

    rows = [
        {"video_id": "gen3c__s1__yaw_RL__profile_TV2V60", "model": "gen3c", "camera_type": "yaw_RL", "profile": "TV2V60"},
        {"video_id": "hydra__s1__yaw_LR__profile_TV2V30", "model": "hydra", "camera_type": "yaw_LR", "profile": "TV2V30"},
        {"video_id": "hailuo__s1__yaw_RL", "model": "hailuo_2_3", "camera_type": "yaw_RL"},
        {"video_id": "hailuo__s2__yaw_LR", "model": "hailuo_2_3", "camera_type": "yaw_LR"},
        {"video_id": "hailuo__s3__static", "model": "hailuo_2_3", "camera_type": "static"},
        {"video_id": "gamecraft__s1__static", "model": "hunyuan_game_craft", "camera_type": "static"},
    ]
    manifest_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Save predicted poses in pose cache
    def _save_pose(vid: str, arr: np.ndarray) -> None:
        p = cache_root / "pose" / vid / "poses.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, arr)

    _save_pose("gen3c__s1__yaw_RL__profile_TV2V60", yaw_target_trajectory(121, 60.0))
    hydra_concat = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 154, axis=0)
    hydra_concat[77:] = yaw_target_trajectory(77, -30.0)
    _save_pose("hydra__s1__yaw_LR__profile_TV2V30", hydra_concat)
    _save_pose("hailuo__s1__yaw_RL", yaw_target_trajectory(61, 16.0))
    # hailuo__s2 moves opposite direction (yaw_RL when asked for yaw_LR) -> direction_mismatch = 0.0
    _save_pose("hailuo__s2__yaw_LR", yaw_target_trajectory(61, 20.0))
    _save_pose("hailuo__s3__static", np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 61, axis=0))
    _save_pose("gamecraft__s1__static", np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 61, axis=0))

    common_d1_args = [
        "--input-jsonl", str(manifest_path),
        "--pose-cache-root", str(cache_root),
        "--pose-backend", "vggt_omega",
        "--poses-file", "poses.npy",
        "--default-frames", "121",
        "--predicted-pose-type", "c2w",
        "--predicted-camera-convention", "opencv",
        "--target-camera-convention", "opencv",
        "--rot-scale-deg", "30.0",
        "--trans-scale", "0.25",
        "--yaw-weak-threshold-deg", "2.0",
        "--pan-weak-threshold", "0.03",
        "--static-rot-threshold-deg", "3.0",
        "--static-trans-threshold", "0.05",
        "--sidecar-profile-gate", "main",
    ]
    # Default fail-closed stays missing_target_pose without sidecars
    assert d1_camera.main([*common_d1_args, "--output-jsonl", str(d1_failclosed), "--summary-csv", str(tmp_path / "s0.csv")]) == 0
    fail_rows = d1_camera.read_jsonl(d1_failclosed)
    assert all(r["d1_status"] == "missing_target_pose" for r in fail_rows)

    # Nominal fallback succeeds for gen3c & hydra, excludes hailuo_2_3 as prompt-only
    assert d1_camera.main(
        [
            *common_d1_args,
            "--output-jsonl", str(d1_out),
            "--summary-csv", str(d1_summary),
            "--target-fallback-mode", "nominal_profile",
        ]
    ) == 0
    d1_rows = d1_camera.read_jsonl(d1_out)
    by_id = {r["video_id"]: r for r in d1_rows}
    assert by_id["gen3c__s1__yaw_RL__profile_TV2V60"]["d1_status"] == "ok"
    assert by_id["gen3c__s1__yaw_RL__profile_TV2V60"]["d1_camera_accuracy"] is None
    assert by_id["gen3c__s1__yaw_RL__profile_TV2V60"]["d1_gt_valid"] is False
    assert by_id["gen3c__s1__yaw_RL__profile_TV2V60"]["d1_nominal_camera_accuracy"] == pytest.approx(1.0, abs=1e-6)
    assert by_id["hydra__s1__yaw_LR__profile_TV2V30"]["d1_status"] == "ok"
    assert by_id["hydra__s1__yaw_LR__profile_TV2V30"]["hydra_segment_status"] == "generated_only_from_concat"
    assert by_id["hydra__s1__yaw_LR__profile_TV2V30"]["d1_nominal_camera_accuracy"] == pytest.approx(1.0, abs=1e-6)
    assert by_id["hailuo__s1__yaw_RL"]["d1_status"] == "excluded_prompt_only"

    # CamAlign v2_lite scores including zero-score direction_mismatch row
    assert d1_camalign.main(
        [
            "--input-jsonl", str(manifest_path),
            "--output-jsonl", str(camalign_out),
            "--pose-cache-root", str(cache_root),
            "--poses-file", "poses.npy",
            "--scorer-version", "v2_lite",
            "--summary-json", str(camalign_summary),
        ]
    ) == 0
    cam_rows = d1_camalign.read_jsonl(camalign_out)
    cam_by_id = {r["video_id"]: r for r in cam_rows}
    assert cam_by_id["hailuo__s1__yaw_RL"]["d1_camalign_status"] == "ok"
    assert cam_by_id["hailuo__s1__yaw_RL"]["d1_camalign_score"] == pytest.approx(0.5, abs=1e-6)
    assert cam_by_id["hailuo__s2__yaw_LR"]["d1_camalign_status"] == "direction_mismatch"
    assert cam_by_id["hailuo__s2__yaw_LR"]["d1_camalign_score"] == 0.0

    # Table builder keeps direction_mismatch=0.0 for D1-CamAlign and excludes prompt-only & gamecraft static for D1
    table, summary = table_builder.build_table(
        runtime_records=[],
        d1_records=d1_rows,
        d2_records=[],
        d1_camalign_records=cam_rows,
    )
    table_by_model = {row["model"]: row for row in table}
    assert table_by_model["gen3c"]["D1_camera_pose"] is None
    assert table_by_model["gen3c"]["D1_camera_pose_eligible"] == 1
    assert table_by_model["gen3c"]["D1_camera_pose_nominal"] == pytest.approx(1.0, abs=1e-6)
    assert table_by_model["gen3c"]["D1_camalign"] == pytest.approx(1.0, abs=1e-6)
    assert table_by_model["hydra"]["D1_camera_pose"] is None
    assert table_by_model["hydra"]["D1_camera_pose_nominal_n"] == 1
    assert table_by_model["hailuo_2_3"]["D1_camera_pose_eligible"] == 0
    assert table_by_model["hailuo_2_3"]["D1_camalign_n"] == 2
    assert table_by_model["hailuo_2_3"]["D1_camalign_eligible"] == 2
    assert table_by_model["hailuo_2_3"]["D1_camalign"] == pytest.approx(0.25, abs=1e-6)
    assert table_by_model["hailuo_2_3"]["D1_camalign_static_hold"] == pytest.approx(1.0, abs=1e-6)
    assert table_by_model["hailuo_2_3"]["D1_camalign_static_hold_n"] == 1
    assert table_by_model["gen3c"]["D1_camalign_static_hold_eligible"] == 0
    assert table_by_model["hunyuan_game_craft"]["D1_camera_pose_eligible"] == 0
    assert summary["main_table_ready"] is False


def test_nominal_fallback_rejects_invalid_explicit_target(tmp_path: Path) -> None:
    row = {
        "video_id": "gen3c__s1__yaw_RL__profile_TV2V60",
        "model": "gen3c",
        "camera_type": "yaw_RL",
        "profile": "TV2V60",
        "target_pose_path": str(tmp_path / "target.npy"),
        "target_coordinate_convention": "unsupported",
    }
    config = d1_camera.d1_scoring_config_with_defaults(
        {"cache_root": tmp_path, "require_main_profile": True, "target_fallback_mode": "nominal_profile"}
    )
    pose_file = tmp_path / "pose" / row["video_id"] / "poses.npy"
    pose_file.parent.mkdir(parents=True)
    np.save(pose_file, yaw_target_trajectory(77, 60.0))
    output = d1_camera.score_d1_row(row, config)
    assert output["d1_status"] == "coordinate_mismatch"
    assert output["d1_target_source"] == "target_pose_path"
    assert output["d1_gt_valid"] is False
    assert output["d1_nominal_camera_accuracy"] is None


def test_requested_profile_scores_paper_controls_through_cli(tmp_path: Path) -> None:
    from wrbench.cli import main

    rows = [
        {"video_id": "static_still", "camera_type": "static"},
        {"video_id": "static_moving", "camera_type": "static"},
        {"video_id": "yaw", "camera_type": "yaw_RL", "profile": "TV2V60", "target_pose_path": "invalid.npy"},
        {"video_id": "prompt_yaw", "camera_type": "yaw_RL", "profile": "TV2V60", "viewpoint_condition": "prompt-only"},
    ]
    for row, peak in zip(rows[:3], (0.0, 3.0, 60.0), strict=True):
        pose_file = tmp_path / "pose" / row["video_id"] / "poses.npy"
        pose_file.parent.mkdir(parents=True)
        np.save(pose_file, yaw_target_trajectory(77, peak))

    manifest = tmp_path / "input.jsonl"
    d1_camera.write_jsonl(rows, manifest)
    scored = tmp_path / "scored.jsonl"
    assert main([
        "eval", "d1", "--input-jsonl", str(manifest), "--output-jsonl", str(scored),
        "--summary-csv", str(tmp_path / "summary.csv"), "--pose-cache-root", str(tmp_path),
        "--target-mode", "requested_profile",
    ]) == 0
    static_still, static_moving, yaw, prompt_yaw = d1_camera.read_jsonl(scored)
    for row, score in zip((static_still, static_moving, yaw), (1.0, 0.0, 1.0), strict=True):
        assert row["d1_camera_accuracy"] == pytest.approx(score, abs=1e-6)
        assert row["d1_nominal_camera_accuracy"] is None
        assert row["d1_gt_valid"] is True
        assert row["d1_target_source"] == row["d1_evidence_level"] == "requested_control_profile"
    assert prompt_yaw["d1_status"] == "excluded_prompt_only"
    assert prompt_yaw["d1_gt_valid"] is False


def test_all_registered_models_declare_viewpoint_condition_type() -> None:
    expected_keys = {
        "easyanimate-v51-camera",
        "gen3c",
        "hailuo-2-3",
        "happyhorse-1-0-i2v",
        "hunyuan-game-craft",
        "hunyuan-worldplay",
        "hydra",
        "inspatio-world",
        "kling-v2-6",
        "lingbot-world",
        "lingbot-world-act",
        "liveworld",
        "magicworld",
        "minwm-hy-action2v",
        "minwm-wan-action2v",
        "recammaster",
        "sana-wm",
        "spatia",
        "versecrafter",
        "wan2-2-i2v-plus",
        "wan2-6-i2v",
        "wan2-7-i2v",
        "wan21-fun-14b-cam",
        "wan21-fun-1p3b-cam",
        "wan22-fun-5b-cam",
        "wan22-fun-a14b-cam",
        "wanx2-1-i2v-turbo",
    }
    by_key = {m.key: m for m in all_records()}
    missing = sorted(k for k in expected_keys if not by_key[k].viewpoint_condition_type)
    assert missing == []


def test_cli_eval_parser_supports_new_repro_flags() -> None:
    parser = _build_parser()
    args_d1 = parser.parse_args(
        [
            "eval", "d1",
            "--input-jsonl", "in.jsonl",
            "--output-jsonl", "out.jsonl",
            "--summary-csv", "s.csv",
            "--pose-cache-root", "cache",
            "--pose-backend", "vggt_omega",
            "--poses-file", "poses.npy",
            "--default-frames", "121",
            "--predicted-pose-type", "extrinsics_c2w",
            "--predicted-camera-convention", "opencv_c2w",
            "--target-camera-convention", "opencv_c2w",
            "--rot-scale-deg", "30",
            "--trans-scale", "0.25",
            "--yaw-weak-threshold-deg", "2",
            "--pan-weak-threshold", "0.03",
            "--static-rot-threshold-deg", "3",
            "--static-trans-threshold", "0.05",
            "--sidecar-profile-gate", "main",
            "--target-fallback-mode", "nominal_profile",
        ]
    )
    assert args_d1.target_fallback_mode == "nominal_profile"

    args_cam = parser.parse_args(
        [
            "eval", "d1-camalign",
            "--input-jsonl", "in.jsonl",
            "--output-jsonl", "out.jsonl",
            "--pose-cache-root", "cache",
            "--poses-file", "poses.npy",
            "--scorer-version", "v2_lite",
            "--summary-json", "summary.json",
        ]
    )
    assert args_cam.scorer_version == "v2_lite"
    assert args_cam.summary_json == "summary.json"

    args_d2 = parser.parse_args(
        ["eval", "d2", "--videos-manifest", "v.json", "--out-jsonl", "d2.jsonl", "--video-root", "/data"]
    )
    assert args_d2.video_root == "/data"


def test_pluggable_segment_policy_and_minimal_eval_d1_defaults(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from wrbench.cli import main as cli_main
    from wrbench.eval.plugins import (
        ConcatSegmentPolicy,
        SegmentPolicyPlugin,
        register_segment_policy,
        registered_segment_policies,
    )

    custom_policy = ConcatSegmentPolicy(
        name="toy_concat_49_49",
        models=frozenset({"toy_concat_model"}),
        condition_frames=49,
        generated_frames=49,
    )
    assert isinstance(custom_policy, SegmentPolicyPlugin)
    register_segment_policy("toy_concat_49_49", custom_policy)
    try:
        assert "toy_concat_49_49" in registered_segment_policies()
        concat_poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 98, axis=0)
        concat_poses[49:] = yaw_target_trajectory(49, 30.0)
        sliced, meta = hydra_segments.prediction_eval_poses(
            {"model": "toy_concat_model", "camera_type": "yaw_RL"},
            concat_poses,
        )
        assert sliced.shape == (49, 4, 4)
        assert meta["hydra_segment_status"] == "generated_only_from_concat"
        assert meta["hydra_segment_policy"] == "toy_concat_49_49"
    finally:
        from wrbench.eval import plugins as plugins_mod

        plugins_mod._SEGMENT_POLICIES.pop("toy_concat_49_49", None)

    assert cli_main(["eval", "plugins", "--json"]) == 0
    plugins_out = json.loads(capsys.readouterr().out)
    assert "vggt_omega" in plugins_out["eval"]["pose_backends"]
    assert "hydra_concat_77_77" in plugins_out["eval"]["segment_policies"]
    assert "dinov2_fullfov_v2" in plugins_out["eval"]["d2_scorers"]
    assert "qwen_runtime_v2" in plugins_out["eval"]["vlm_scorers"]
    assert "gen3c" in plugins_out["generation"]["adapter_models"]

    cache_root = tmp_path / "cache"
    pose_dir = cache_root / "pose" / "row_min"
    pose_dir.mkdir(parents=True)
    np.save(pose_dir / "poses.npy", yaw_target_trajectory(121, -30.0))
    manifest = tmp_path / "min.json"
    manifest.write_text(
        json.dumps([{"video_id": "row_min", "model": "gen3c", "camera_type": "yaw_LR", "profile": "TV2V30"}]),
        encoding="utf-8",
    )
    out_jsonl = tmp_path / "d1_min.jsonl"
    rc = cli_main(
        [
            "eval",
            "d1",
            "--input-jsonl",
            str(manifest),
            "--output-jsonl",
            str(out_jsonl),
            "--summary-csv",
            str(tmp_path / "sum.csv"),
            "--pose-cache-root",
            str(cache_root),
            "--target-fallback-mode",
            "nominal_profile",
        ]
    )
    assert rc == 0
    row = json.loads(out_jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert row["d1_status"] == "ok"
    assert row["d1_nominal_camera_accuracy"] == pytest.approx(1.0, abs=1e-6)
