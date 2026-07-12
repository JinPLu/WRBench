from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import wrbench
from wrbench.datasets import (
    load_jsonl,
    natural25_families_path,
    natural25_prompt_profile_path,
    natural25_t2v_event_tails_path,
    natural25_t2v_layout_anchors_path,
    natural25_variants_path,
    resolve_variant_prompt,
)
from wrbench.t2v import (
    assess_t2v_gates,
    expected_yaw_peak_deg,
    validate_subject_anchored_prompt,
    verify_minwm_rotation_calibration,
)


def test_validate_subject_anchored_prompt() -> None:
    assert validate_subject_anchored_prompt(
        "Scene. background. Immediately, the adult person walks toward the bed."
    )
    assert not validate_subject_anchored_prompt(
        "Scene. background. He remains perfectly still in place."
    )


def test_expected_yaw_peak_deg() -> None:
    assert expected_yaw_peak_deg("yaw:left:60@40,yaw:right:60@41") == 60.0
    assert expected_yaw_peak_deg("yaw:left:30@38,yaw:right:30@39") == 30.0
    assert expected_yaw_peak_deg("static@77") == 0.0


def test_verify_minwm_rotation_calibration_ok() -> None:
    ok, details = verify_minwm_rotation_calibration(
        {
            "requested_yaw_peak_deg": 60.0,
            "effective_yaw_peak_deg": 60.0,
            "runtime_yaw_deg_per_token": 6.0,
            "requires_runtime_rot_step_patch": True,
        }
    )
    assert ok
    assert details["delta_deg"] == 0.0


def test_assess_t2v_gates() -> None:
    result = assess_t2v_gates(
        subject_present=True,
        scene_present=True,
        action_judgeable=True,
        camera_visible=True,
        camera_amplitude_ok=True,
    )
    assert result["passed"]
    assert result["failed_gates"] == []


def test_bundled_ti2v_variants_keep_legacy_pronoun_contract() -> None:
    variants = list(load_jsonl(natural25_variants_path()))
    assert len(variants) == 400
    sample = next(row for row in variants if row["variant_id"] == "bedroom_adult_bed_sit__T1__none")
    assert "Immediately, he walks toward the bed." in sample["ti2v_prompt"]
    assert "Immediately, the adult person walks toward the bed." not in sample["ti2v_prompt"]


def test_t2v_event_tails_hold_subject_anchored_prompt_profile_text() -> None:
    tails = {row["variant_id"]: row["t2v_event_tail"] for row in load_jsonl(natural25_t2v_event_tails_path())}
    assert len(tails) == 400
    sample = tails["bedroom_adult_bed_sit__T1__none"]
    assert sample == "Immediately, the adult person walks toward the bed."
    assert validate_subject_anchored_prompt(f"Scene. background. {sample}")


def test_t2v_layout_anchor_profile_contract() -> None:
    profile = json.loads(natural25_prompt_profile_path("t2v_layout_anchor").read_text(encoding="utf-8"))
    assert profile["profile_id"] == "t2v_layout_anchor"
    assert "version" not in profile
    assert "schema_version" not in profile

    family_ids = {row["family_id"] for row in load_jsonl(natural25_families_path())}
    anchors = list(load_jsonl(natural25_t2v_layout_anchors_path()))
    anchor_ids = [row["family_id"] for row in anchors]
    expected_keys = ["family_id", "scene", "subject", "interactor", "open_surface", "anchors"]

    assert len(anchors) == 25
    assert set(anchor_ids) == family_ids
    assert len(set(anchor_ids)) == len(anchor_ids)
    for row in anchors:
        assert list(row.keys()) == expected_keys
        assert isinstance(row["anchors"], list) and row["anchors"]
        blob = json.dumps(row, ensure_ascii=False).lower()
        for token in profile["banned_style_tokens"]:
            assert token.lower() not in blob


def test_t2v_layout_anchor_profile_rejects_camera_gap_prompt_rows() -> None:
    variant = next(
        row
        for row in load_jsonl(natural25_variants_path())
        if row["variant_id"] == "bedroom_adult_bed_sit__T1__yaw_LR"
    )
    with pytest.raises(ValueError, match="requires oov_gap 'none'"):
        resolve_variant_prompt(variant, prompt_profile="t2v_layout_anchor")


def test_minwm_wan_rotation_patch_materialized(tmp_path) -> None:
    out = tmp_path / "wan.mp4"
    result = wrbench.compile_camera(
        model="minwm-wan-action2v",
        camera="yaw:left:60@40,yaw:right:60@41",
        out=out,
        prompt="Unified prompt.",
        dry_run=True,
    )
    payload = result["payload"].payload
    patch = payload["rotation_step_patch"]
    assert patch is not None
    assert Path(patch["launcher_path"]).is_file()
    request = json.loads(open(payload["request_json"], encoding="utf-8").read())
    module_text = Path(patch["patch_root"], "wan_utils", "camera_trajectory.py").read_text(encoding="utf-8")
    assert "_ROT_STEP_DEG = 6.0" in module_text
    assert "runtime_env" not in request
    assert request["input_contract"]["prompt_file_format"] == (
        "one materialized WRBench generation prompt per line, aligned with trajectory_path"
    )
    assert request["runtime_patch"]["launcher_path"] == patch["launcher_path"]


def test_prepare_minwm_probe_requires_explicit_case(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "t2v" / "prepare_minwm_wan_acceptance_probe.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--out-dir",
            str(tmp_path),
            "--prompt-profile",
            "t2v_layout_anchor",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--case" in result.stderr


def test_prepare_minwm_probe_requires_explicit_prompt_profile(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "t2v" / "prepare_minwm_wan_acceptance_probe.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--out-dir",
            str(tmp_path),
            "--case",
            "bedroom_adult_bed_sit__T1__none:yaw_LR",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--prompt-profile" in result.stderr


def test_prepare_minwm_probe_writes_prompt_profile_manifest(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "t2v" / "prepare_minwm_wan_acceptance_probe.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--out-dir",
            str(tmp_path),
            "--prompt-profile",
            "t2v_layout_anchor",
            "--case",
            "bedroom_adult_bed_sit__T1__none:yaw_LR",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 1
    row = manifest[0]
    assert row["prompt_profile_id"] == "t2v_layout_anchor"
    assert row["camera_preset"] == "yaw_LR"
    assert row["generation_prompt"] == row["prompt"]
    assert row["generation_prompt"] != row["ti2v_prompt"]
    assert "Realistic photography" not in row["generation_prompt"]
    assert "left third of the frame" in row["generation_prompt"]
    assert "camera slowly moves" not in row["generation_prompt"]
    assert (tmp_path / "bedroom_adult_bed_sit__T1__none__yaw_LR" / "prompt.txt").is_file()


def test_minwm_probe_runner_has_no_implicit_runtime_defaults() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "t2v"
        / "run_minwm_wan_acceptance_probe.sh"
    )
    text = script.read_text(encoding="utf-8")
    assert "CONFIG=\"${CONFIG:-" not in text
    assert "--num_output_frames 20" not in text
    assert "${PYTHON_BIN%/python}/torchrun" not in text
    assert "--sp_size 1" not in text
    assert "--nproc_per_node=1" not in text
    assert "29500 + RANDOM" not in text
    assert "TORCHRUN_BIN=\"${TORCHRUN_BIN:?" in text
    assert "SP_SIZE=\"${SP_SIZE:?" in text
    assert "NPROC_PER_NODE=\"${NPROC_PER_NODE:?" in text
    assert "MASTER_PORT=\"${MASTER_PORT:?" in text
