from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from wrbench.paper_sources import prepare_paper_tv2v_sources


def test_run_natural25_generation_manifest_is_eval_ready(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_natural25_generation.py"
    scope = root / "src" / "wrbench" / "data" / "natural25" / "camera_scopes" / "t2v_rotation_stress_30_60.json"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model",
            "minwm-wan-action2v",
            "--out-dir",
            str(tmp_path),
            "--camera-scope",
            str(scope),
            "--prompt-profile",
            "t2v_layout_anchor",
            "--dry-run",
            "--overwrite-existing",
            "--fail-fast",
            "--limit",
            "1",
            "--shard-index",
            "0",
            "--num-shards",
            "1",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    manifest = tmp_path / "minwm-wan-action2v" / "manifest.shard00.jsonl"
    row = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    assert set(row) >= {
        "schema_version",
        "model",
        "display_name",
        "video_id",
        "variant_id",
        "family_id",
        "reasoning_tier",
        "event_delta",
        "camera",
        "camera_type",
        "camera_preset",
        "camera_scope_id",
        "path",
        "world_state_prompt",
        "expected_state",
        "prompt_profile_id",
        "ti2v_prompt",
        "prompt",
        "model_input",
        "status",
    }
    assert "output_id" not in row
    assert "event_tier" not in row
    assert "output_path" not in row
    assert "video_path" not in row
    assert "prompt_text" not in row
    assert "generation_prompt" not in row
    assert "generation_manifest_status" not in row
    assert row["world_state_prompt"]
    assert row["expected_state"]
    assert row["ti2v_prompt"]
    assert row["prompt_profile_id"] == "t2v_layout_anchor"
    assert row["prompt"]
    assert row["prompt"] != row["ti2v_prompt"]
    assert "Realistic photography" not in row["prompt"]
    assert "left third of the frame" in row["prompt"]
    assert row["model_input"] == "T2V"
    assert row["status"] == "ok"
    assert row["control_family"] == "static"
    assert row["control_direction"] == "static"
    assert row["control_profile"] == "canonical_static"
    assert row["target_coordinate_convention"] == "opencv_c2w"
    assert Path(row["target_pose_path"]).is_file()


@pytest.mark.parametrize(
    ("model", "expected_model_input", "expected_source_usage"),
    [
        ("gen3c", "TV2V", "temporal_conditioning_via_gen3c_vipe_cache"),
        ("hydra", "TV2V", "temporal_conditioning_source_video"),
        ("inspatio-world", "TV2V", "temporal_conditioning_source_video"),
        ("liveworld", "TI2V", "first_frame_extraction"),
        ("recammaster", "TV2V", "temporal_conditioning_source_video"),
        ("spatia", "TI2V", "first_frame_extraction"),
    ],
)
def test_prepared_source_task_map_runs_all_source_video_wrappers(
    tmp_path: Path,
    model: str,
    expected_model_input: str,
    expected_source_usage: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_natural25_generation.py"
    task_id = "bedroom_adult_bed_sit__T0__none"
    payload = b"runner-source-video"
    source_root = tmp_path / "sources"
    source_path = source_root / "wan2_7_i2v" / "static" / "source.mp4"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(payload)
    source_row = {
        "bytes": len(payload),
        "decoded_frame_count": 81,
        "event_tier": "T0",
        "family_id": "bedroom_adult_bed_sit",
        "fps": 16.0,
        "height": 720,
        "hf_path": f"videos/wan2_7_i2v/static/wan2_7_i2v__{task_id}__static.mp4",
        "hf_repo_id": "WRBench/wrbench-videos",
        "hf_revision": "8a927b9322c5d8af6474399ce4840ef4148f8e39",
        "local_relpath": "wan2_7_i2v/static/source.mp4",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source_catalog_variant_id": "bedroom_adult_bed_sit__T0__static",
        "source_video_id": f"wan2_7_i2v__{task_id}__static",
        "source_video_origin": "wan2.7_static_camera_tv2v",
        "task_variant_id": task_id,
        "video_asset_id": "runner-source-asset",
        "width": 1280,
    }
    prepare_paper_tv2v_sources(
        [source_row],
        release_id="paper_main_20260608",
        source_root=source_root,
        mode="download",
        resume=True,
        downloader=lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected download")),
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model",
            model,
            "--out-dir",
            str(tmp_path / "out"),
            "--camera-scope",
            str(
                root
                / "src"
                / "wrbench"
                / "data"
                / "natural25"
                / "releases"
                / "paper_main_20260608"
                / "camera_scopes"
                / "local_dual_angle.json"
            ),
            "--prompt-profile",
            "ti2v_active",
            "--source-video-task-map",
            str(source_root / "source_video_task_map.json"),
            "--source-video-root",
            str(source_root),
            "--dry-run",
            "--overwrite-existing",
            "--fail-fast",
            "--limit",
            "1",
            "--shard-index",
            "0",
            "--num-shards",
            "1",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    manifest = tmp_path / "out" / model / "manifest.shard00.jsonl"
    row = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    assert row["status"] == "ok"
    assert row["camera"] == "yaw_LR_30"
    assert row["stress_yaw_deg"] == 30.0
    assert row["model_input"] == expected_model_input
    assert row["source_video_usage"] == expected_source_usage
    assert row["source_video_task_variant_id"] == task_id
    assert row["source_catalog_variant_id"] == "bedroom_adult_bed_sit__T0__static"
    assert row["source_video_sha256"] == source_row["sha256"]
    sidecar = json.loads(Path(row["camera_sidecar_path"]).read_text(encoding="utf-8"))
    provenance = sidecar["source_video_provenance"]
    assert provenance["source_video_task_variant_id"] == task_id
    assert provenance["source_video_usage"] == expected_source_usage


def test_runner_rejects_model_outside_paper_scope_applicability(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "run_natural25_generation.py"),
            "--model",
            "gen3c",
            "--out-dir",
            str(tmp_path / "out"),
            "--variants",
            str(
                root
                / "src"
                / "wrbench"
                / "data"
                / "natural25"
                / "releases"
                / "paper_main_20260608"
                / "variants.local_ti2v_tv2v.jsonl"
            ),
            "--camera-scope",
            str(
                root
                / "src"
                / "wrbench"
                / "data"
                / "natural25"
                / "releases"
                / "paper_main_20260608"
                / "camera_scopes"
                / "local_static.json"
            ),
            "--prompt-profile",
            "ti2v_active",
            "--dry-run",
            "--overwrite-existing",
            "--fail-fast",
            "--limit",
            "0",
            "--shard-index",
            "0",
            "--num-shards",
            "1",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "is not included in camera scope" in result.stderr
