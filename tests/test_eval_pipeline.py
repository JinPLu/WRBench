"""Exercise the public runner without loading model weights."""

import csv
import json
from pathlib import Path

from wrbench.cli import main


def test_cli_dispatches_external_backends_with_resolved_manifest(tmp_path: Path, monkeypatch) -> None:
    # A plugin outside the checkout can run the standard pipeline with its own
    # runtime and produce the same artifacts as a built-in backend.
    (tmp_path / "external_backend.py").write_text(
        """\
import json
import numpy as np

def pose(*, input_jsonl, cache_root, **kwargs):
    row = json.loads(input_jsonl.read_text())
    assert row["video_path"].startswith("/") and row["target_pose_path"].startswith("/")
    dst = cache_root / "pose" / row["video_id"] / "poses.npy"
    dst.parent.mkdir(parents=True)
    np.save(dst, np.repeat(np.eye(4)[None], 3, axis=0))
    return 0

def d2(*, videos_manifest, out_jsonl, **kwargs):
    row, = json.loads(videos_manifest.read_text())
    assert row["video_path"].startswith("/")
    row.update(d2_selected_visual_integrity_score=0.75, d2_status="ok")
    out_jsonl.write_text(json.dumps(row) + "\\n")
    return 0

def vlm(*, manifest, scores_path, stage, scorer_profile, **kwargs):
    row, = json.loads(manifest.read_text())
    assert stage == "all" and scorer_profile == "custom"
    with open(row["video_path"], "rb") as f:
        assert f.read() == b"clip"
    row.update(
        vlm_spatial_fidelity=0.8, vlm_state_fidelity=0.6,
        vlm_spatial_reasoning=None, vlm_state_reasoning=None,
        runtime_v2_evidence_shared_oov_applicable=False,
    )
    scores_path.parent.mkdir(parents=True)
    scores_path.write_text(json.dumps([row]))
    return 0
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "sample.mp4").write_bytes(b"clip")
    config = tmp_path / "runtime.json"
    config.write_text(json.dumps({
        "schema_version": 1,
        "eval": {
            "scorers": {"gpu_id": 0},
            "backends": {stage: f"external_backend:{stage}" for stage in ("pose", "d2", "vlm")},
        },
    }))
    manifest = tmp_path / "videos.jsonl"
    row = {
        "video_id": "example",
        "model": "external_model",
        "camera_type": "yaw_RL",
        "video_path": "sample.mp4",
        "world_state_prompt": "A mug sits on a table.",
        "target_pose_path": "targets/sample.npy",
    }
    original = json.dumps(row) + "\n"
    manifest.write_text(original)
    out = tmp_path / "scores"
    argv = [
        "eval", "--runtime-config", str(config), "run", "--manifest", str(manifest),
        "--video-root", str(mirror), "--out-dir", str(out), "--sidecar-profile-gate", "main",
    ]
    assert main([*argv, "--scorer-profile", "wrbench_default"]) == 1
    assert not out.exists()
    assert main([*argv, "--scorer-profile", "default"]) == 1
    assert not out.exists()
    manifest.write_text(json.dumps([row, None]))
    assert main([*argv, "--scorer-profile", "custom"]) == 1
    assert not out.exists()
    manifest.write_text(original)
    # Omitting --scorer-profile and --sidecar-profile-gate auto-resolves custom for eval.backends
    assert main([
        "eval", "--runtime-config", str(config), "run", "--manifest", str(manifest),
        "--video-root", str(mirror), "--out-dir", str(out),
    ]) == 0
    assert manifest.read_text() == original
    with (out / "main_table.csv").open() as stream:
        record, = csv.DictReader(stream)
    assert record["model"] == "external_model"
    assert record["viewpoint_condition_type"] == "custom"
    assert record["model_input"] == "unknown"
    assert record["D1_camalign"] == "0.0"
    assert record["D2_visual_integrity"] == "0.75"
    assert record["D3_spatial_in"] == "0.8"


def test_cli_custom_backends_without_eval_scorers_block(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "custom_only_backend.py").write_text(
        """\
import json
import numpy as np

def pose(*, input_jsonl, cache_root, **kwargs):
    row = json.loads(input_jsonl.read_text())
    dst = cache_root / "pose" / row["video_id"] / "poses.npy"
    dst.parent.mkdir(parents=True)
    np.save(dst, np.repeat(np.eye(4)[None], 3, axis=0))
    return 0

def d2(*, videos_manifest, out_jsonl, **kwargs):
    row, = json.loads(videos_manifest.read_text())
    row.update(d2_selected_visual_integrity_score=0.5, d2_status="ok")
    out_jsonl.write_text(json.dumps(row) + "\\n")
    return 0

def vlm(*, manifest, scores_path, **kwargs):
    row, = json.loads(manifest.read_text())
    row.update(
        vlm_spatial_fidelity=0.9, vlm_state_fidelity=0.7,
        vlm_spatial_reasoning=None, vlm_state_reasoning=None,
        runtime_v2_evidence_shared_oov_applicable=False,
    )
    scores_path.parent.mkdir(parents=True)
    scores_path.write_text(json.dumps([row]))
    return 0
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"clip")
    config = tmp_path / "runtime.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "eval": {
                    "backends": {stage: f"custom_only_backend:{stage}" for stage in ("pose", "d2", "vlm")},
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "videos.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "video_id": "custom_vid",
                "model": "brand_new_third_party_model",
                "model_input": "TI2V",
                "viewpoint_condition_type": "model-inferred",
                "camera_type": "yaw_LR",
                "video_path": str(video),
                "world_state_prompt": "A cup rests on a wooden desk.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    rc = main(
        [
            "eval",
            "--runtime-config",
            str(config),
            "run",
            "--manifest",
            str(manifest),
            "--out-dir",
            str(out),
        ]
    )
    assert rc == 0
    with (out / "main_table.csv").open(encoding="utf-8") as stream:
        record, = csv.DictReader(stream)
    assert record["model"] == "brand_new_third_party_model"
    assert record["viewpoint_condition_type"] == "model-inferred"
    assert record["model_input"] == "TI2V"
