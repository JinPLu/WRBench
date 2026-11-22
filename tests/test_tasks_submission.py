"""Tests for benchmark task export and video submission manifest validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from wrbench.datasets import NATURAL25_PAPER_RELEASE_ID
from wrbench.registry import register_model_record, reload
from wrbench.tasks_submission import export_benchmark_tasks, validate_video_submission


@pytest.fixture(autouse=True)
def _reset_registry():
    reload()
    yield
    reload()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# export_benchmark_tasks
# ---------------------------------------------------------------------------

def test_export_benchmark_tasks_without_model(tmp_path: Path):
    out_dir = tmp_path / "tasks_plain"
    summary = export_benchmark_tasks(out_dir=out_dir, limit=5)

    assert summary["ok"] is True
    assert summary["tasks_count"] == 5
    tasks_path = Path(summary["tasks_jsonl"])
    assert tasks_path.is_file()
    rows = _read_jsonl(tasks_path)
    assert len(rows) == 5

    required = {
        "video_id",
        "task_id",
        "family_id",
        "variant_id",
        "camera_type",
        "camera_script",
        "profile",
        "world_state_prompt",
        "prompt_to_send",
        "first_frame_path",
    }
    for row in rows:
        assert required.issubset(row)
        assert Path(row["first_frame_path"]).is_file()
        assert "target_pose_path" not in row


def test_export_benchmark_tasks_with_model_compiles_targets(tmp_path: Path):
    out_dir = tmp_path / "tasks_wan"
    summary = export_benchmark_tasks(
        out_dir=out_dir,
        model="wan22-fun-5b-cam",
        release_id=NATURAL25_PAPER_RELEASE_ID,
        camera_presets=["yaw_LR", "yaw_RL"],
        limit=2,
        compile_targets=True,
    )

    assert summary["ok"] is True
    assert summary["tasks_count"] == 2
    rows = _read_jsonl(Path(summary["tasks_jsonl"]))
    assert len(rows) == 2
    assert [r["camera_type"] for r in rows] == ["yaw_LR", "yaw_RL"]
    for row in rows:
        assert row["model"] == "wan22-fun-5b-cam"
        assert row["video_id"].startswith("wan22-fun-5b-cam__")
        assert row["profile"] == "canonical_60deg"
        assert Path(row["target_pose_path"]).is_file()
        assert Path(row["payload_json_path"]).is_file()
        assert str(out_dir / "compiled") in row["target_pose_path"]


def test_export_benchmark_tasks_prompt_camera_model(tmp_path: Path):
    register_model_record(
        {
            "key": "byo-prompt-api",
            "aliases": ["byo-prompt-api"],
            "status": "active",
            "input_kind": "none",
            "model_input": "T2V",
            "source_video_usage": "none",
            "viewpoint_condition_type": "prompt-only",
            "adapter": "prompt_camera",
            "payload_type": "prompt_camera",
            "default_width": 832,
            "default_height": 480,
            "default_frames": 25,
            "default_fps": 16,
            "amplitude": {
                "rotation_gain": 1.0,
                "translation_gain": 1.0,
                "max_amount": 0.5,
                "translation_unit": "none",
                "calibration_status": "prompt_only",
            },
            "capabilities": {"supports_static": True},
        }
    )
    out_dir = tmp_path / "tasks_prompt"
    summary = export_benchmark_tasks(
        out_dir=out_dir,
        model="byo-prompt-api",
        camera_presets=["yaw_LR"],
        limit=1,
        compile_targets=True,
    )
    rows = _read_jsonl(Path(summary["tasks_jsonl"]))
    assert len(rows) == 1
    assert "camera slowly moves left" in rows[0]["prompt_to_send"].lower()
    assert Path(rows[0]["target_pose_path"]).is_file()
    assert Path(rows[0]["payload_json_path"]).is_file()


def test_export_benchmark_tasks_skips_static_when_model_forbids_static(tmp_path: Path):
    out_dir = tmp_path / "tasks_recam"
    summary = export_benchmark_tasks(
        out_dir=out_dir,
        model="recammaster",
        limit=2,
        compile_targets=True,
    )
    rows = _read_jsonl(Path(summary["tasks_jsonl"]))
    assert len(rows) == 2
    assert all(row["camera_type"] != "static" for row in rows)


def test_export_benchmark_tasks_unregistered_byo_model_no_compile(tmp_path: Path):
    out_dir = tmp_path / "tasks_byo"
    summary = export_benchmark_tasks(
        out_dir=out_dir,
        model="my-unregistered-model",
        limit=2,
        compile_targets=False,
    )
    rows = _read_jsonl(Path(summary["tasks_jsonl"]))
    assert len(rows) == 2
    assert all(row["model"] == "my-unregistered-model" for row in rows)

    with pytest.raises(KeyError):
        export_benchmark_tasks(
            out_dir=tmp_path / "tasks_byo_fail",
            model="my-unregistered-model",
            limit=1,
            compile_targets=True,
        )


def test_export_benchmark_tasks_validates_args(tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown camera preset"):
        export_benchmark_tasks(out_dir=tmp_path / "bad", camera_presets=["not_a_preset"])
    with pytest.raises(ValueError, match="non-negative"):
        export_benchmark_tasks(out_dir=tmp_path / "bad_lim", limit=-1)


# ---------------------------------------------------------------------------
# validate_video_submission
# ---------------------------------------------------------------------------

def test_validate_video_submission_valid_jsonl_and_json(tmp_path: Path):
    vid1 = tmp_path / "videos" / "v1.mp4"
    vid2 = tmp_path / "videos" / "v2.mp4"
    vid1.parent.mkdir(parents=True)
    vid1.write_bytes(b"video-bytes-1")
    vid2.write_bytes(b"video-bytes-2")
    sha1 = hashlib.sha256(b"video-bytes-1").hexdigest()
    sha2 = hashlib.sha256(b"video-bytes-2").hexdigest()

    rows = [
        {
            "video_id": "m1__v1",
            "model": "my-model-a",
            "path": "videos/v1.mp4",
            "camera_type": "yaw_LR",
            "world_state_prompt": "A cat jumps on the bed.",
            "sha256": sha1,
        },
        {
            "video_id": "m2__v2",
            "model": "my-model-b",
            "video_path": "videos/v2.mp4",
            "camera_type": "static",
            "world_state_prompt": "A book stays on the desk.",
            "sha256": sha2,
        },
    ]

    jsonl_path = tmp_path / "manifest.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    res_jsonl = validate_video_submission(jsonl_path, video_root=tmp_path, check_files=True, check_sha256=True)
    assert res_jsonl == {
        "ok": True,
        "row_count": 2,
        "errors": [],
        "warnings": [],
        "models": ["my-model-a", "my-model-b"],
    }

    json_path = tmp_path / "manifest.json"
    json_path.write_text(json.dumps({"records": rows}), encoding="utf-8")
    res_json = validate_video_submission(json_path, video_root=tmp_path, check_files=True, check_sha256=True)
    assert res_json["ok"] is True
    assert res_json["row_count"] == 2
    assert res_json["models"] == ["my-model-a", "my-model-b"]


def test_validate_video_submission_errors_and_warnings(tmp_path: Path):
    empty_vid = tmp_path / "empty.mp4"
    empty_vid.write_bytes(b"")
    good_vid = tmp_path / "good.mp4"
    good_vid.write_bytes(b"nonempty")

    rows = [
        {"video_id": "dup", "model": "m1", "path": str(good_vid)},  # missing camera_type & world_state_prompt -> warnings
        {"video_id": "dup", "model": "m1", "path": str(empty_vid), "camera_type": "static", "world_state_prompt": "ok"},
        {"video_id": "missing_file", "model": "", "path": "does_not_exist.mp4", "camera_type": "yaw_LR", "world_state_prompt": "ok"},
    ]
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    res = validate_video_submission(manifest, video_root=tmp_path, check_files=True)
    assert res["ok"] is False
    assert res["row_count"] == 3
    assert res["models"] == ["m1"]
    assert any("duplicate video_id" in e for e in res["errors"])
    assert any("empty (0 bytes)" in e for e in res["errors"])
    assert any("missing non-empty 'model'" in e for e in res["errors"])
    assert any("video file not found" in e for e in res["errors"])
    assert any("missing 'camera_type'" in w for w in res["warnings"])
    assert any("missing 'world_state_prompt'" in w for w in res["warnings"])


def test_validate_video_submission_check_files_false_and_sha256_mismatch(tmp_path: Path):
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"actual-data")

    manifest_missing = tmp_path / "no_files.jsonl"
    manifest_missing.write_text(
        json.dumps(
            {
                "video_id": "v1",
                "model": "m1",
                "path": "not_here.mp4",
                "camera_type": "static",
                "world_state_prompt": "prompt",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    res_skip = validate_video_submission(manifest_missing, video_root=tmp_path, check_files=False)
    assert res_skip["ok"] is True

    manifest_bad_sha = tmp_path / "bad_sha.jsonl"
    manifest_bad_sha.write_text(
        json.dumps(
            {
                "video_id": "v1",
                "model": "m1",
                "path": str(vid),
                "camera_type": "static",
                "world_state_prompt": "prompt",
                "sha256": "0" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    res_sha = validate_video_submission(manifest_bad_sha, check_files=True, check_sha256=True)
    assert res_sha["ok"] is False
    assert any("sha256 mismatch" in e for e in res_sha["errors"])

    res_not_found = validate_video_submission(tmp_path / "nonexistent.jsonl")
    assert res_not_found["ok"] is False
    assert res_not_found["row_count"] == 0
