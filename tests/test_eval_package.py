"""Eval package tests: imports, contract, D1 math, scoring helpers, table builder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_eval_imports_clean() -> None:
    import wrbench.eval  # noqa: F401
    import wrbench.eval.aggregate.latest_d1_d6_metrics as contract  # noqa: F401
    import wrbench.eval.d1.d1_camera  # noqa: F401
    import wrbench.eval.scoring.runtime_common  # noqa: F401


def test_metric_contract_schema_pinned() -> None:
    from wrbench.eval.aggregate import latest_d1_d6_metrics as contract

    assert contract.SCHEMA_VERSION == "wrbench_latest_d1_d6_metrics_v3"
    specs = contract.latest_metric_specs()
    assert {spec.dimension_id for spec in specs} == {
        "D1",
        "D1-CamAlign",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
    }


def test_contract_json_matches_source() -> None:
    from wrbench.eval.runtime import contract_path

    payload = json.loads(contract_path().read_text(encoding="utf-8"))
    assert payload["schema_version"] == "wrbench_latest_d1_d6_metrics_v3"
    assert "metrics" in payload
    assert len(payload["metrics"]) == 7


def test_d1_pose_score_identity_trajectory() -> None:
    from wrbench.eval.d1.pose import score_trajectory_pose

    poses = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 10, axis=0)
    target = poses.copy()
    result = score_trajectory_pose(
        target,
        poses,
        rot_scale_deg=45.0,
        trans_scale=1.0,
        predicted_pose_type="c2w",
        predicted_camera_convention="opencv",
        target_camera_convention="opencv",
    )
    assert result["pose_status"] == "ok"
    assert result["pose_reward"] == pytest.approx(1.0)


def test_scoring_video_path_prefers_eval_video_path() -> None:
    from wrbench.eval.scoring.runtime_common import scoring_video_path

    assert (
        scoring_video_path(
            {
                "eval_video_path": "/tmp/generated_only.mp4",
                "path": "/tmp/concat.mp4",
                "video_path": "/tmp/legacy.mp4",
            }
        )
        == "/tmp/generated_only.mp4"
    )


def test_eval_runtime_reads_nested_scorers_block(tmp_path: Path) -> None:
    from wrbench.eval.runtime import load_eval_runtime

    runtime_path = tmp_path / "wrbench.runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "eval": {
                    "scorers": {
                        "gpu_id": 2,
                        "vggt_python_bin": "/tmp/vggt/bin/python",
                        "vggt_repo": "/tmp/vggt",
                        "vggt_checkpoint": "/tmp/vggt.ckpt",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = load_eval_runtime(runtime_path)
    assert cfg is not None
    assert cfg.scorers.gpu_id == 2
    assert cfg.scorers.vggt_python_bin == "/tmp/vggt/bin/python"


def test_eval_runtime_requires_scorer_gpu_id(tmp_path: Path) -> None:
    from wrbench.eval.runtime import load_eval_runtime
    from wrbench.runtime import RuntimeConfigError

    runtime_path = tmp_path / "wrbench.runtime.json"
    runtime_path.write_text(
        json.dumps({"schema_version": 1, "eval": {"scorers": {"vggt_python_bin": "/tmp/python"}}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeConfigError, match="eval.scorers.*gpu_id"):
        load_eval_runtime(runtime_path)


def test_d2_extract_requires_explicit_runtime_fields(tmp_path: Path) -> None:
    from wrbench.eval.runtime import EvalRuntimeConfig, EvalScorerRuntime, d2_extract

    py = tmp_path / "python"
    py.write_text("#!/bin/sh\n", encoding="utf-8")
    py.chmod(0o755)
    runtime = EvalRuntimeConfig(
        schema_version=1,
        scorers=EvalScorerRuntime(dinov2_python_bin=str(py), gpu_id=0),
    )

    with pytest.raises(FileNotFoundError, match="dinov2_model_path"):
        d2_extract(
            eval_runtime=runtime,
            videos_manifest=tmp_path / "videos.jsonl",
            out_jsonl=tmp_path / "d2.jsonl",
        )


def test_d3d6_env_requires_explicit_runtime_fields(tmp_path: Path) -> None:
    from wrbench.eval.runtime import EvalRuntimeConfig, EvalScorerRuntime, d3d6_env

    scorer_py = tmp_path / "python"
    scorer_py.write_text("#!/bin/sh\n", encoding="utf-8")
    scorer_py.chmod(0o755)
    runtime = EvalRuntimeConfig(
        schema_version=1,
        scorers=EvalScorerRuntime(qwen_scorer_python=str(scorer_py), gpu_id=0),
    )

    with pytest.raises(FileNotFoundError, match="qwen35_model"):
        d3d6_env(
            eval_runtime=runtime,
            manifest=tmp_path / "manifest.json",
            out_dir=tmp_path / "out",
            scorer_profile="wrbench_default",
        )


def test_eval_d1_does_not_require_eval_scorers(tmp_path: Path) -> None:
    """`wrbench eval d1` is pure-numpy scoring and must run without an eval.scorers block."""
    import numpy as np

    from wrbench.cli import main

    runtime_path = tmp_path / "wrbench.runtime.json"
    runtime_path.write_text(
        json.dumps({"schema_version": 1, "models": {"foo": {"python_bin": "/x"}}}),
        encoding="utf-8",
    )
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 10, axis=0)
    cache = tmp_path / "cache" / "pose" / "clip"
    cache.mkdir(parents=True)
    np.save(cache / "poses.npy", poses)
    np.save(str(video) + ".target_c2w.npy", poses)
    (Path(str(video) + ".camera_trajectory.json")).write_text("{}", encoding="utf-8")
    sidecar = {
        "coordinate_convention": "opencv_c2w",
        "target_role": "benchmark_intended_control",
        "control_direction": "yaw_LR",
        "control_family": "rotation",
        "control_profile": "canonical_60deg",
        "evidence_level": "benchmark_intent",
        "yaw_peak_deg": 60.0,
        "num_frames": 10,
        "fps": 24,
        "image_size": [512, 512],
        "fov": 60.0,
        "trajectory_sampling_rule": "uniform",
        "adapter_provenance": "deterministic_adapter",
        "target_pose_path": str(video) + ".target_c2w.npy",
        "camera_trajectory_path": str(video) + ".camera_trajectory.json",
    }
    (Path(str(video) + ".camera.json")).write_text(json.dumps(sidecar), encoding="utf-8")
    inp = tmp_path / "in.jsonl"
    inp.write_text(
        json.dumps({"video_id": "clip", "path": str(video), "camera_type": "yaw_LR"}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"
    rc = main(
        [
            "eval",
            "--runtime-config",
            str(runtime_path),
            "d1",
            "--input-jsonl",
            str(inp),
            "--output-jsonl",
            str(out),
            "--summary-csv",
            str(tmp_path / "sum.csv"),
            "--pose-cache-root",
            str(tmp_path / "cache"),
            "--pose-backend",
            "vggt_omega",
            "--poses-file",
            "poses.npy",
            "--default-frames",
            "10",
            "--sidecar-profile-gate",
            "main",
            "--predicted-pose-type",
            "c2w",
            "--predicted-camera-convention",
            "opencv",
            "--target-camera-convention",
            "opencv",
            "--rot-scale-deg",
            "45.0",
            "--trans-scale",
            "1.0",
            "--yaw-weak-threshold-deg",
            "2.0",
            "--pan-weak-threshold",
            "0.0001",
            "--static-rot-threshold-deg",
            "2.0",
            "--static-trans-threshold",
            "0.05",
        ]
    )
    assert rc == 0
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["d1_status"] == "ok"


def test_scoring_modules_do_not_import_workplace() -> None:
    scoring_dir = Path(__file__).resolve().parents[1] / "src" / "wrbench" / "eval" / "scoring"
    offenders = []
    for path in scoring_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "workplace." in text or "metric.scoring" in text:
            offenders.append(path.name)
    assert offenders == []


def test_contract_version_is_paper_facing() -> None:
    from wrbench.eval.aggregate import latest_d1_d6_metrics as contract

    assert contract.CONTRACT_VERSION == "wrbench_paper_v1"
    payload = contract.metric_contract_payload()
    assert payload["policy"]["paper_reference"].startswith("WRBench:")


def test_eval_runtime_exposes_no_scorer_profile_alias_layer() -> None:
    import wrbench.eval.runtime as runtime

    assert not hasattr(runtime, "normalize_scorer_profile")


def test_main_table_includes_viewpoint_and_reobservation() -> None:
    from wrbench.eval.aggregate.build_wrbench_vnext_main_table import build_table

    runtime_records = [
        {
            "video_id": "v1",
            "model": "wan22-fun-5b-cam",
            "vlm_spatial_fidelity": 0.8,
            "vlm_state_fidelity": 0.7,
            "vlm_spatial_reasoning": 0.6,
            "vlm_state_reasoning": 0.5,
            "runtime_v2_evidence_shared_oov_applicable": True,
        }
    ]
    d1_records = [
        {
            "video_id": "v1",
            "model": "wan22-fun-5b-cam",
            "camera_type": "yaw_LR",
            "d1_status": "ok",
            "d1_camera_accuracy": 0.9,
        }
    ]
    d2_records = [
        {
            "video_id": "v1",
            "model": "wan22-fun-5b-cam",
            "d2_status": "ok",
            "d2_selected_visual_integrity_score": 0.85,
        }
    ]
    table, _summary = build_table(
        runtime_records=runtime_records,
        d1_records=d1_records,
        d2_records=d2_records,
    )
    assert len(table) == 1
    row = table[0]
    assert row["viewpoint_condition_type"] == "model-inferred"
    assert row["reobservation_support"] == pytest.approx(1.0)
    assert row["D1_camera_pose"] == pytest.approx(0.9)


def test_natural25_dataset_paths_exist() -> None:
    import json

    from wrbench.datasets import (
        build_natural25_candidates,
        load_natural25_families,
        load_jsonl,
        natural25_first_frame_path,
        natural25_first_frames_manifest_path,
        natural25_families_path,
        natural25_variants_path,
        published_results_csv,
    )

    assert natural25_families_path().is_file()
    assert natural25_variants_path().is_file()
    assert natural25_first_frames_manifest_path().is_file()
    assert published_results_csv().is_file()
    families = load_natural25_families()
    candidates = build_natural25_candidates()
    assert len(families) == 25
    assert len(candidates) == 25
    variants = list(load_jsonl(natural25_variants_path()))
    assert len(variants) == 400
    manifest = json.loads(natural25_first_frames_manifest_path().read_text(encoding="utf-8"))
    assert {row["family_id"] for row in manifest} == set(families)
    assert all(natural25_first_frame_path(fid).is_file() for fid in families)


def test_published_23model_model_input_column() -> None:
    """The frozen 23-model table must expose a model_input column with values in
    {TI2V, TV2V}, consistent between CSV and JSON, and row-aligned.

    Prompt-only T2V rows live in the separate T2V addendum, not the 23-model
    main table.
    """
    import csv

    from wrbench.datasets import published_results_csv, published_results_json

    allowed = {"TI2V", "TV2V"}

    with published_results_csv().open(encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert csv_rows, "CSV has no rows"
    assert "model_input" in csv_rows[0], "CSV missing model_input column"
    for row in csv_rows:
        assert row["model_input"] in allowed, (row["model_id"], row["model_input"])

    payload = json.loads(published_results_json().read_text(encoding="utf-8"))
    json_rows = payload["rows"]
    assert len(json_rows) == len(csv_rows) == 23
    for row in json_rows:
        assert row.get("model_input") in allowed, (row["model_id"], row.get("model_input"))

    csv_map = {row["model_id"]: row["model_input"] for row in csv_rows}
    json_map = {row["model_id"]: row["model_input"] for row in json_rows}
    assert csv_map == json_map, "CSV and JSON disagree on model_input"
    assert csv_map["liveworld"] == "TI2V"
    assert csv_map["spatia"] == "TI2V"


def test_published_23model_viewpoint_condition_type_column() -> None:
    """The frozen 23-model table must expose viewpoint_condition_type consistent
    with build_wrbench_vnext_main_table and aligned between CSV and JSON."""
    import csv

    from wrbench.datasets import published_results_csv, published_results_json
    from wrbench.eval.aggregate.build_wrbench_vnext_main_table import _viewpoint_condition

    allowed = {"source-video", "geometry-cache", "model-inferred", "prompt-only"}

    with published_results_csv().open(encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert csv_rows, "CSV has no rows"
    assert "viewpoint_condition_type" in csv_rows[0], "CSV missing viewpoint_condition_type"
    for row in csv_rows:
        assert row["viewpoint_condition_type"] in allowed, (
            row["model_id"],
            row["viewpoint_condition_type"],
        )
        assert row["viewpoint_condition_type"] == _viewpoint_condition(row["model_id"])

    payload = json.loads(published_results_json().read_text(encoding="utf-8"))
    json_rows = payload["rows"]
    for row in json_rows:
        assert row.get("viewpoint_condition_type") in allowed, (
            row["model_id"],
            row.get("viewpoint_condition_type"),
        )

    csv_map = {row["model_id"]: row["viewpoint_condition_type"] for row in csv_rows}
    json_map = {row["model_id"]: row["viewpoint_condition_type"] for row in json_rows}
    assert csv_map == json_map, "CSV and JSON disagree on viewpoint_condition_type"


def test_eval_run_cli_help() -> None:
    from wrbench.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["eval", "run", "--help"])
    assert exc.value.code == 0


def test_camalign_yaw_direction_match() -> None:
    from wrbench.eval.d1.camera_intent import score_camera_intent

    poses = np.repeat(np.eye(4, dtype=np.float64)[None, ...], 10, axis=0)
    yaw = np.deg2rad(-20.0)
    rot = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    poses[-1, :3, :3] = rot[:3, :3]
    ok = score_camera_intent(poses, intent="yaw_LR")
    bad = score_camera_intent(poses, intent="yaw_RL")
    assert ok["d1_camalign_status"] == "ok"
    assert ok["d1_camalign_score"] == pytest.approx(0.6, abs=0.05)
    assert bad["d1_camalign_status"] == "direction_mismatch"
    assert bad["d1_camalign_score"] == 0.0


def test_camalign_static_hold() -> None:
    from wrbench.eval.d1.camera_intent import score_camera_intent

    poses = np.repeat(np.eye(4, dtype=np.float64)[None, ...], 10, axis=0)
    result = score_camera_intent(poses, intent="static")
    assert result["d1_camalign_status"] == "ok"
    assert result["d1_camalign_score"] == pytest.approx(1.0)


def test_main_table_camalign_prompt_only() -> None:
    from wrbench.eval.aggregate.build_wrbench_vnext_main_table import build_table

    runtime_records = [
        {
            "video_id": "v1",
            "model": "hailuo-2.3",
            "camera_type": "yaw_LR",
            "vlm_spatial_fidelity": 0.8,
            "vlm_state_fidelity": 0.7,
            "vlm_spatial_reasoning": 0.6,
            "vlm_state_reasoning": 0.5,
            "runtime_v2_evidence_shared_oov_applicable": False,
        }
    ]
    camalign_records = [
        {
            "video_id": "v1",
            "model": "hailuo-2.3",
            "camera_type": "yaw_LR",
            "d1_camalign_status": "ok",
            "d1_camalign_score": 0.4,
        }
    ]
    table, _ = build_table(
        runtime_records=runtime_records,
        d1_records=[],
        d2_records=[],
        d1_camalign_records=camalign_records,
    )
    assert table[0]["viewpoint_condition_type"] == "prompt-only"
    assert table[0]["D1_camalign"] == pytest.approx(0.4)
    assert table[0]["D1_camera_pose"] is None


def test_prompt_task_rejects_partial_natural25_override(tmp_path: Path) -> None:
    from wrbench.cli import main

    candidates = tmp_path / "candidates.json"
    candidates.write_text("[]", encoding="utf-8")
    rc = main(
        [
            "prompt",
            "task",
            "--deterministic",
            "--candidates-json",
            str(candidates),
        ]
    )
    assert rc == 2
