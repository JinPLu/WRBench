"""Tests for wrcam.eval.scoring."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from wrcam.eval.scoring import export_runtime_v2_evidence_first as export_mod
from wrcam.eval.scoring import runtime_common
from wrcam.eval.scoring import run_local_qwen35_probe_logprob_scorer as qwen35_mod
from wrcam.eval.scoring import run_local_qwen3vl_video_evidence as qwen3vl_mod

DUMMY_QWEN35_MODEL = "/tmp/qwen35-model-dir"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_scoring_video_path_prefers_eval_video_path_then_path_then_video_path() -> None:
    assert (
        runtime_common.scoring_video_path(
            {
                "eval_video_path": "/tmp/generated_only.mp4",
                "path": "/tmp/concat.mp4",
                "video_path": "/tmp/legacy.mp4",
            }
        )
        == "/tmp/generated_only.mp4"
    )
    assert runtime_common.scoring_video_path({"path": "/tmp/concat.mp4", "video_path": "/tmp/legacy.mp4"}) == "/tmp/concat.mp4"
    assert runtime_common.scoring_video_path({"video_path": "/tmp/legacy.mp4"}) == "/tmp/legacy.mp4"
    assert runtime_common.scoring_video_path({}) == ""


def test_scoring_video_path_full_continuation_uses_concat_surface() -> None:
    assert (
        runtime_common.scoring_video_path(
            {
                "scoring_video_surface": "full_continuation",
                "eval_video_path": "/tmp/generated_only.mp4",
                "original_concat_path": "/tmp/source_plus_generated.mp4",
                "path": "/tmp/path.mp4",
                "video_path": "/tmp/video_path.mp4",
            }
        )
        == "/tmp/source_plus_generated.mp4"
    )
    assert (
        runtime_common.scoring_video_path(
            {
                "scoring_video_surface": "full_continuation",
                "scoring_video_path": "/tmp/explicit_full_continuation.mp4",
                "eval_video_path": "/tmp/generated_only.mp4",
                "original_concat_path": "/tmp/source_plus_generated.mp4",
            }
        )
        == "/tmp/explicit_full_continuation.mp4"
    )


def test_qwen35_standalone_manifest_dry_run_without_source_scores(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"not-a-real-video")
    manifest = [
        {
            "video_id": "vid-1",
            "path": str(video),
            "world_state_prompt": "A red cube moves behind a wall and returns.",
            "model": "toy_model",
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    out_dir = tmp_path / "qwen35"
    write_json(manifest_path, manifest)

    assert (
        qwen35_mod.main(
            [
                "--experiment-id",
                "standalone_dry_run",
                "--manifest-path",
                str(manifest_path),
                "--output-dir",
                str(out_dir),
                "--model-path",
                DUMMY_QWEN35_MODEL,
                "--dry-run",
            ]
        )
        == 0
    )

    score_available = out_dir / "scores_v7_candidate_runtime_v2_probe_score_available.json"
    gate_masked = out_dir / "scores_v7_candidate_runtime_v2_probe_gate_masked.json"
    assert score_available.is_file()
    assert gate_masked.is_file()
    rows = json.loads(score_available.read_text(encoding="utf-8"))
    assert rows[0]["video_id"] == "vid-1"
    assert "vlm_spatial_fidelity" in rows[0]
    assert "vlm_state_fidelity" in rows[0]


def test_qwen35_legacy_p9_manifest_metadata_context_remains_available() -> None:
    args = qwen35_mod.parse_args(
        [
            "--experiment-id",
            "best_profile_contract",
            "--prompt-mode",
            qwen35_mod.PROMPT_MODE_P9_D4_P8_D5_P6_COMBINED,
            "--task-context-mode",
            "all_manifest_metadata",
            "--strict-manifest-contract",
            "--manifest-path",
            "/tmp/manifest.json",
        "--output-dir",
        "/tmp/out",
        "--model-path",
        DUMMY_QWEN35_MODEL,
    ]
)

    assert args.task_context_mode == "all_manifest_metadata"
    task_context = qwen35_mod.build_task_context(
        {
            "video_id": "vid-1",
            "world_state_prompt": "prompt",
            "model": "toy_model",
            "variant_id": "variant-a",
            "camera_type": "yaw_LR",
        },
        task_context_mode=args.task_context_mode,
    )
    assert task_context["model"] == "toy_model"
    assert task_context["variant_id"] == "variant-a"
    assert task_context["camera_type"] == "yaw_LR"
    scorer = qwen35_mod.LocalQwen35ProbeLogprobScorer(
        model_path=Path("/tmp/model"),
        fps="2",
        dtype="bfloat16",
        attn_implementation="flash_attention_2",
        local_rank=0,
        dry_run=True,
        prompt_mode=qwen35_mod.PROMPT_MODE_P9_D4_P8_D5_P6_COMBINED,
    )
    assert scorer.task_context_mode == "all_manifest_metadata"


def test_qwen35_v7_export_preserves_task_context_mode() -> None:
    row = qwen35_mod.export_v7_style_probe_record(
        {
            "video_id": "vid-1",
            "path": "/tmp/video.mp4",
            "world_state_prompt": "A red cube moves behind a wall and returns.",
            "d3_spatial_in_view_score": 0.7,
            "d4_state_in_view_score": 0.8,
            "d5_spatial_oov_score": 0.6,
            "d6_state_oov_score": 0.5,
            "runtime_v2_task_context_mode": "none",
        },
        export_policy="score_available",
    )

    assert row["runtime_v2_task_context_mode"] == "none"


def test_qwen35_camera_motion_context_only_renders_camera_block() -> None:
    item = {
        "video_id": "model_a__task__T1__none__yaw_RL",
        "world_state_prompt": "A red cube moves behind a wall and returns.",
        "model": "leaky_model_name",
        "variant_id": "leaky_variant",
        "camera_type": "yaw_RL",
        "expected_state": "This must not be injected.",
    }
    task_context = qwen35_mod.build_task_context(
        item,
        task_context_mode=qwen35_mod.TASK_CONTEXT_MODE_CAMERA_MOTION,
    )
    assert task_context == {qwen35_mod.CAMERA_MOTION_CONTEXT_KEY: "yaw_RL"}

    probe = next(
        probe
        for probe in qwen35_mod.RUNTIME_V2_PROBE_CATALOG
        if probe.probe_id == "D3_POSITION_RELATION"
    )
    prompt = qwen35_mod.build_runtime_v2_probe_prompt(
        world_state_prompt=item["world_state_prompt"],
        video_id=item["video_id"],
        probe=probe,
        task_context=task_context,
    )

    assert "Text prompt used to generate the video:\nA red cube moves behind a wall and returns." in prompt
    assert "How camera move:\nyaw_RL" in prompt
    assert "Video id:" not in prompt
    assert item["video_id"] not in prompt
    assert "Task context from manifest" not in prompt
    assert "leaky_model_name" not in prompt
    assert "leaky_variant" not in prompt
    assert "This must not be injected." not in prompt


def test_qwen3vl_clean_gate_prompt_omits_vlm_facing_identifiers() -> None:
    video_id = "model_a__source_x__family_y__variant_z__D6_PROBE"
    prompt = qwen3vl_mod.build_judgeability_prompt(
        world_state_prompt="A red cube moves behind a wall and returns.",
        video_id=video_id,
        prompt_schema=qwen3vl_mod.PROMPT_SCHEMA_GUARDED_CLEAN,
    )

    assert "strict D5/D6 judgeability gate auditor" in prompt
    assert "Video id:" not in prompt
    assert "video_id" not in prompt
    assert video_id not in prompt
    assert "model_a" not in prompt
    assert "source_x" not in prompt
    assert "family_y" not in prompt
    assert "variant_z" not in prompt
    assert "D6_PROBE" not in prompt


def test_qwen3vl_direct3q_clean_gate_prompt_omits_vlm_facing_identifiers() -> None:
    video_id = "model_a__source_x__family_y__variant_z__D6_PROBE"
    prompt = qwen3vl_mod.build_judgeability_prompt(
        world_state_prompt="A red cube moves behind a wall and returns.",
        video_id=video_id,
        prompt_schema=qwen3vl_mod.PROMPT_SCHEMA_SHARED_DIRECT3Q_CLEAN,
    )

    assert "Your job is to answer three judgeability questions" in prompt
    assert "Video id:" not in prompt
    assert "video_id" not in prompt
    assert "D5" not in prompt
    assert "D6" not in prompt
    assert "scoreable" not in prompt
    assert video_id not in prompt
    assert "model_a" not in prompt
    assert "source_x" not in prompt
    assert "family_y" not in prompt
    assert "variant_z" not in prompt
    assert "D6_PROBE" not in prompt
    assert '"d5_applicable"' not in prompt
    assert '"d6_applicable"' not in prompt


def test_qwen3vl_visible_bool_clean_gate_prompt_omits_vlm_facing_identifiers() -> None:
    video_id = "model_a__source_x__family_y__variant_z__D6_PROBE"
    prompt = qwen3vl_mod.build_judgeability_prompt(
        world_state_prompt="A red cube moves behind a wall and returns.",
        video_id=video_id,
        prompt_schema=qwen3vl_mod.PROMPT_SCHEMA_VISIBLE_BOOL_CLEAN,
    )

    assert "has_visible_after_gap_evidence_to_judge" in prompt
    assert "The text request itself is not evidence" in prompt
    assert "Do not answer true for ordinary continuously visible evidence with no such gap." in prompt
    assert "Video id:" not in prompt
    assert "video_id" not in prompt
    assert "D5" not in prompt
    assert "D6" not in prompt
    assert "WRBench" not in prompt
    assert "benchmark" not in prompt
    assert video_id not in prompt
    assert "model_a" not in prompt
    assert "source_x" not in prompt
    assert "family_y" not in prompt
    assert "variant_z" not in prompt
    assert "D6_PROBE" not in prompt
    assert '"d5_applicable"' not in prompt
    assert '"d6_applicable"' not in prompt


def test_qwen3vl_oov_gap_bool_clean_gate_prompt_omits_vlm_facing_identifiers() -> None:
    video_id = "model_a__source_x__family_y__variant_z__D6_PROBE"
    prompt = qwen3vl_mod.build_judgeability_prompt(
        world_state_prompt="A red cube moves behind a wall and returns.",
        video_id=video_id,
        prompt_schema=qwen3vl_mod.PROMPT_SCHEMA_OOV_GAP_BOOL_CLEAN,
    )

    assert "has_oov_gap_with_later_comparable_evidence" in prompt
    assert "The text request itself is not evidence" in prompt
    assert "Later clear visibility is necessary for a true answer, but it is not sufficient." in prompt
    assert "remains continuously visible enough to judge" in prompt
    assert "ordinary action progress" in prompt
    assert "Video id:" not in prompt
    assert "video_id" not in prompt
    assert "D5" not in prompt
    assert "D6" not in prompt
    assert "WRBench" not in prompt
    assert "benchmark" not in prompt
    assert "has_visible_after_gap_evidence_to_judge" not in prompt
    assert video_id not in prompt
    assert "model_a" not in prompt
    assert "source_x" not in prompt
    assert "family_y" not in prompt
    assert "variant_z" not in prompt
    assert "D6_PROBE" not in prompt
    assert '"d5_applicable"' not in prompt
    assert '"d6_applicable"' not in prompt


def test_qwen3vl_oov_gap_scan_clean_gate_prompt_omits_vlm_facing_identifiers() -> None:
    video_id = "model_a__source_x__family_y__variant_z__D6_PROBE"
    prompt = qwen3vl_mod.build_judgeability_prompt(
        world_state_prompt="A red cube moves behind a wall and returns.",
        video_id=video_id,
        prompt_schema=qwen3vl_mod.PROMPT_SCHEMA_OOV_GAP_SCAN_CLEAN,
    )

    assert "Fill visibility_scan before deciding final_oov_applicable" in prompt
    assert "Return exactly 8 checkpoints" in prompt
    assert "Wrong, absent, replaced, reset, failed, or changed later evidence is still judgeable" in prompt
    assert "Do not answer false merely because the evidence is visible in the first and last parts" in prompt
    assert '"visibility_scan"' in prompt
    assert '"judgeable": "<yes|no|unclear>"' in prompt
    assert "The text request itself is not evidence" in prompt
    assert "Video id:" not in prompt
    assert "video_id" not in prompt
    assert "D5" not in prompt
    assert "D6" not in prompt
    assert "WRBench" not in prompt
    assert "benchmark" not in prompt
    assert "has_visible_after_gap_evidence_to_judge" not in prompt
    assert video_id not in prompt
    assert "model_a" not in prompt
    assert "source_x" not in prompt
    assert "family_y" not in prompt
    assert "variant_z" not in prompt
    assert "D6_PROBE" not in prompt
    assert '"d5_applicable"' not in prompt
    assert '"d6_applicable"' not in prompt


def test_qwen3vl_oov_gap_triplet_clean_gate_prompt_omits_vlm_facing_identifiers() -> None:
    video_id = "model_a__source_x__family_y__variant_z__D6_PROBE"
    prompt = qwen3vl_mod.build_judgeability_prompt(
        world_state_prompt="A red cube moves behind a wall and returns.",
        video_id=video_id,
        prompt_schema=qwen3vl_mod.PROMPT_SCHEMA_OOV_GAP_TRIPLET_CLEAN,
    )

    assert "Fill the three evidence fields before deciding final_oov_applicable" in prompt
    assert '"before_reference"' in prompt
    assert '"middle_unjudgeable_gap"' in prompt
    assert '"later_comparable_evidence"' in prompt
    assert '"present": "<yes|no|unclear>"' in prompt
    assert "Do not require the exact same subject or object to reappear" in prompt
    assert "The text request itself is not evidence" in prompt
    assert "Video id:" not in prompt
    assert "video_id" not in prompt
    assert "D5" not in prompt
    assert "D6" not in prompt
    assert "WRBench" not in prompt
    assert "benchmark" not in prompt
    assert "has_visible_after_gap_evidence_to_judge" not in prompt
    assert video_id not in prompt
    assert "model_a" not in prompt
    assert "source_x" not in prompt
    assert "family_y" not in prompt
    assert "variant_z" not in prompt
    assert "D6_PROBE" not in prompt
    assert '"d5_applicable"' not in prompt
    assert '"d6_applicable"' not in prompt


def test_qwen3vl_oov_gap_triplet_sheet_clean_gate_prompt_omits_vlm_facing_identifiers() -> None:
    video_id = "model_a__source_x__family_y__variant_z__D6_PROBE"
    prompt = qwen3vl_mod.build_judgeability_prompt(
        world_state_prompt="A red cube moves behind a wall and returns.",
        video_id=video_id,
        prompt_schema=qwen3vl_mod.PROMPT_SCHEMA_OOV_GAP_TRIPLET_SHEET_CLEAN,
    )

    assert "contact sheet made from one video" in prompt
    assert "Refer to those labels in your notes" in prompt
    assert "Judge only the frames that are shown" in prompt
    assert "three-part visual pattern" in prompt
    assert "scan the sampled frames one by one" in prompt
    assert "Do not summarize a range as visible" in prompt
    assert '"critical_evidence"' in prompt
    assert '"frame_visibility"' in prompt
    assert '"early_reference"' in prompt
    assert '"middle_cannot_follow"' in prompt
    assert '"later_comparison"' in prompt
    assert '"final_applicable": <true|false>' in prompt
    assert "Video id:" not in prompt
    assert "video_id" not in prompt
    assert "metadata" not in prompt
    assert "filename" not in prompt
    assert "filenames" not in prompt
    assert "model" not in prompt
    assert "dataset" not in prompt
    assert "outside knowledge" not in prompt
    assert "OOV" not in prompt
    assert "OoV" not in prompt
    assert "final_oov_applicable" not in prompt
    assert "unreasonable" not in prompt
    assert "D5" not in prompt
    assert "D6" not in prompt
    assert "WRBench" not in prompt
    assert "benchmark" not in prompt
    assert "has_visible_after_gap_evidence_to_judge" not in prompt
    assert video_id not in prompt
    assert "model_a" not in prompt
    assert "source_x" not in prompt
    assert "family_y" not in prompt
    assert "variant_z" not in prompt
    assert "D6_PROBE" not in prompt
    assert '"d5_applicable"' not in prompt
    assert '"d6_applicable"' not in prompt


def test_qwen3vl_oov_gap_triplet_sheet_clean_parser_allows_missing_later_type() -> None:
    payload = {
        "critical_evidence": "cup and table area",
        "early_reference": {
            "present": "yes",
            "note": "idx0 shows the cup on the table",
        },
        "frame_visibility": [
            {"label": "idx0", "visible": "yes", "note": "cup is visible"},
            {"label": "idx8", "visible": "bad-value", "note": "cup is hidden"},
        ],
        "middle_cannot_follow": {
            "present": "no",
            "note": "idx8 shows the cup remains visible",
        },
        "later_comparison": {
            "present": "no",
            "note": "idx16 is unrelated",
        },
        "final_applicable": False,
        "brief_reason": "The cup remains visible, so the three-part pattern is absent.",
    }

    parsed = qwen3vl_mod.parse_judgeability_response(
        json.dumps(payload),
        expected_video_id="vid-a",
        prompt_schema=qwen3vl_mod.PROMPT_SCHEMA_OOV_GAP_TRIPLET_SHEET_CLEAN,
    )

    assert parsed["later_comparable_evidence"]["type"] == "unclear"
    assert parsed["frame_visibility"] == [
        {"label": "idx0", "visible": "yes", "note": "cup is visible"},
        {"label": "idx8", "visible": "unclear", "note": "cup is hidden"},
    ]
    assert parsed["oov_applicable"] is False


def test_qwen3vl_oov_gap_per_second_audit_bool_prompt_omits_vlm_facing_identifiers() -> None:
    video_id = "model_a__source_x__family_y__variant_z__D6_PROBE"
    prompt = qwen3vl_mod.build_judgeability_prompt(
        world_state_prompt="A waiter places a tray on a table.",
        video_id=video_id,
        prompt_schema=qwen3vl_mod.PROMPT_SCHEMA_OOV_GAP_PER_SECOND_AUDIT_BOOL_CLEAN,
    )

    assert "For each whole second you can inspect" in prompt
    assert "The per_second list is audit evidence" in prompt
    assert "A support surface counts only when it helps judge the requested thing" in prompt
    assert '"per_second"' in prompt
    assert '"status"' in prompt
    assert '"can_judge_after_gap"' in prompt
    assert "clear_relevant_evidence" not in prompt
    assert '"early_reference"' not in prompt
    assert '"middle_cannot_follow"' not in prompt
    assert '"later_comparison"' not in prompt
    assert "Video id:" not in prompt
    assert "video_id" not in prompt
    assert "metadata" not in prompt
    assert "filename" not in prompt
    assert "filenames" not in prompt
    assert "model" not in prompt
    assert "dataset" not in prompt
    assert "outside knowledge" not in prompt
    assert "OOV" not in prompt
    assert "OoV" not in prompt
    assert "D5" not in prompt
    assert "D6" not in prompt
    assert "WRBench" not in prompt
    assert "benchmark" not in prompt
    assert video_id not in prompt


def test_qwen3vl_oov_gap_per_second_audit_bool_keeps_raw_gate_when_scan_conflicts() -> None:
    payload = {
        "look_for": "waiter tray and table",
        "per_second": [
            {"sec": 0, "status": "visible", "note": "waiter and tray visible"},
            {"sec": 1, "status": "visible", "note": "waiter still visible"},
            {"sec": 2, "status": "visible", "note": "table area visible"},
        ],
        "can_judge_after_gap": True,
        "reason_code": "yes_after_gap",
        "brief_reason": "The video has enough visible evidence after the interruption.",
    }

    parsed = qwen3vl_mod.parse_judgeability_response(
        json.dumps(payload),
        expected_video_id="vid-a",
        prompt_schema=qwen3vl_mod.PROMPT_SCHEMA_OOV_GAP_PER_SECOND_AUDIT_BOOL_CLEAN,
    )

    assert parsed["schema_version"] == qwen3vl_mod.OOV_GAP_PER_SECOND_AUDIT_BOOL_CLEAN_SCHEMA_VERSION
    assert parsed["look_for"] == "waiter tray and table"
    assert parsed["can_judge_after_gap"] is True
    assert parsed["oov_applicable"] is True
    assert parsed["scan_derived_pattern"] is False
    assert parsed["gate_consistent_with_scan"] is False
    assert "raw_true_scan_no_unjudgeable" in parsed["conflict_types"]


def test_qwen3vl_oov_gap_per_second_audit_bool_flags_scan_positive_without_override() -> None:
    payload = {
        "look_for": "person chair and sitting result",
        "per_second": [
            {"sec": 0, "status": "visible", "note": "person and chair visible"},
            {"sec": 1, "status": "not_visible", "note": "person outside the frame"},
            {"sec": 2, "status": "visible", "note": "chair area visible again"},
        ],
        "can_judge_after_gap": False,
        "reason_code": "no_later_visible",
        "brief_reason": "The later evidence is not enough to judge the request.",
    }

    parsed = qwen3vl_mod.parse_judgeability_response(
        json.dumps(payload),
        expected_video_id="vid-a",
        prompt_schema=qwen3vl_mod.PROMPT_SCHEMA_OOV_GAP_PER_SECOND_AUDIT_BOOL_CLEAN,
    )

    assert parsed["oov_applicable"] is False
    assert parsed["scan_derived_pattern"] is True
    assert parsed["gate_consistent_with_scan"] is False
    assert "raw_false_scan_has_clear_unjudgeable_clear" in parsed["conflict_types"]


def test_qwen3vl_oov_gap_per_second_strict_collapse_prompt_omits_vlm_facing_identifiers() -> None:
    video_id = "model_a__source_x__family_y__variant_z__D6_PROBE"
    prompt = qwen3vl_mod.build_judgeability_prompt(
        world_state_prompt="A worker places a cardboard box onto a pallet.",
        video_id=video_id,
        prompt_schema=qwen3vl_mod.PROMPT_SCHEMA_OOV_GAP_PER_SECOND_STRICT_COLLAPSE,
    )

    assert "For each whole second you can inspect" in prompt
    assert "background target area alone" in prompt
    assert "scene collapse" in prompt
    assert '"per_second"' in prompt
    assert '"status"' in prompt
    assert '"can_judge_after_gap"' not in prompt
    assert '"reason_code"' not in prompt
    assert "Video id:" not in prompt
    assert "video_id" not in prompt
    assert "metadata" not in prompt
    assert "filename" not in prompt
    assert "filenames" not in prompt
    assert "model" not in prompt
    assert "dataset" not in prompt
    assert "outside knowledge" not in prompt
    assert "OOV" not in prompt
    assert "OoV" not in prompt
    assert "D5" not in prompt
    assert "D6" not in prompt
    assert "WRBench" not in prompt
    assert "benchmark" not in prompt
    assert video_id not in prompt


def test_qwen3vl_oov_gap_per_second_strict_collapse_derives_visible_after_gap() -> None:
    payload = {
        "look_for": "worker and cardboard box being placed on pallet",
        "per_second": [
            {"sec": 0, "status": "visible", "note": "worker and box are visible"},
            {"sec": 1, "status": "not_visible", "note": "worker and box move outside the frame"},
            {"sec": 2, "status": "visible", "note": "box is visible on the pallet"},
        ],
        "brief_reason": "The requested object is visible, absent, then visible again.",
    }

    parsed = qwen3vl_mod.parse_judgeability_response(
        json.dumps(payload),
        expected_video_id="vid-a",
        prompt_schema=qwen3vl_mod.PROMPT_SCHEMA_OOV_GAP_PER_SECOND_STRICT_COLLAPSE,
    )

    assert parsed["schema_version"] == qwen3vl_mod.OOV_GAP_PER_SECOND_STRICT_COLLAPSE_SCHEMA_VERSION
    assert parsed["oov_applicable"] is True
    assert parsed["reason_code"] == "visible_after_gap"
    assert parsed["scan_derived_positive_reason"] == "visible_after_gap"
    assert parsed["oov_interval"] == {
        "start_sec": 1.0,
        "end_sec": 2.0,
        "status": "present",
        "problem_status": "not_visible",
    }
    assert parsed["collapse_present"] is False


def test_qwen3vl_oov_gap_per_second_strict_collapse_derives_collapse_after_visible() -> None:
    payload = {
        "look_for": "adult sitting on chair",
        "per_second": [
            {"sec": 0, "status": "visible", "note": "adult and chair are visible"},
            {"sec": 1, "status": "collapsed", "note": "body and chair smear into a warped artifact"},
            {"sec": 2, "status": "collapsed", "note": "scene remains melted and unjudgeable"},
        ],
        "brief_reason": "The visible requested scene breaks into severe generated artifacts.",
    }

    parsed = qwen3vl_mod.parse_judgeability_response(
        json.dumps(payload),
        expected_video_id="vid-a",
        prompt_schema=qwen3vl_mod.PROMPT_SCHEMA_OOV_GAP_PER_SECOND_STRICT_COLLAPSE,
    )

    assert parsed["oov_applicable"] is True
    assert parsed["reason_code"] == "collapse_after_visible"
    assert parsed["scan_derived_positive_reason"] == "collapse_after_visible"
    assert parsed["collapse_present"] is True
    assert parsed["oov_interval"] == {
        "start_sec": 1.0,
        "end_sec": 3.0,
        "status": "present",
        "problem_status": "collapsed",
    }


def test_qwen3vl_oov_gap_per_second_strict_collapse_requires_later_visible_after_gap() -> None:
    payload = {
        "look_for": "waiter placing tray on table",
        "per_second": [
            {"sec": 0, "status": "visible", "note": "waiter and tray are visible"},
            {"sec": 1, "status": "not_visible", "note": "waiter and tray leave the frame"},
            {"sec": 2, "status": "not_visible", "note": "only an empty table is visible"},
        ],
        "brief_reason": "The requested thing does not become visible again after the gap.",
    }

    parsed = qwen3vl_mod.parse_judgeability_response(
        json.dumps(payload),
        expected_video_id="vid-a",
        prompt_schema=qwen3vl_mod.PROMPT_SCHEMA_OOV_GAP_PER_SECOND_STRICT_COLLAPSE,
    )

    assert parsed["oov_applicable"] is False
    assert parsed["reason_code"] == "no_later_visible"
    assert parsed["scan_derived_na_reason"] == "no_later_visible"
    assert parsed["collapse_present"] is False
    assert parsed["oov_interval"] == {"start_sec": None, "end_sec": None, "status": "unclear"}


def test_qwen3vl_oov_subject_result_integrity_prompt_omits_vlm_facing_identifiers() -> None:
    video_id = "model_a__source_x__family_y__variant_z__D6_PROBE"
    prompt = qwen3vl_mod.build_judgeability_prompt(
        world_state_prompt="A worker places a cardboard box onto a pallet.",
        video_id=video_id,
        prompt_schema=qwen3vl_mod.PROMPT_SCHEMA_OOV_SUBJECT_RESULT_INTEGRITY,
    )

    assert "Your job is to describe what can actually be judged from the video" in prompt
    assert "subject_per_second" in prompt
    assert "result_evidence_per_second" in prompt
    assert "scene_integrity_per_second" in prompt
    assert '"judgeable": "<yes|no|unclear>"' in prompt
    assert '"status": "<coherent|broken|unclear>"' in prompt
    assert "Do not decide whether the requested outcome is correct" in prompt
    assert "Video id:" not in prompt
    assert "video_id" not in prompt
    assert "metadata" not in prompt
    assert "filename" not in prompt
    assert "filenames" not in prompt
    assert "model" not in prompt
    assert "dataset" not in prompt
    assert "outside knowledge" not in prompt
    assert "OOV" not in prompt
    assert "OoV" not in prompt
    assert "D5" not in prompt
    assert "D6" not in prompt
    assert "WRBench" not in prompt
    assert "benchmark" not in prompt
    assert video_id not in prompt


def test_qwen3vl_oov_subject_result_integrity_derives_result_after_subject_gap() -> None:
    payload = {
        "main_subject": "person carrying a box",
        "result_evidence": "box on pallet",
        "subject_per_second": [
            {"sec": 0, "judgeable": "yes", "note": "person and box are visible"},
            {"sec": 1, "judgeable": "no", "note": "person leaves frame"},
            {"sec": 2, "judgeable": "no", "note": "person remains outside the frame"},
        ],
        "result_evidence_per_second": [
            {"sec": 0, "judgeable": "yes", "note": "box starts in the person's hands"},
            {"sec": 1, "judgeable": "no", "note": "box is not visible"},
            {"sec": 2, "judgeable": "yes", "note": "box is clearly visible on the pallet"},
        ],
        "scene_integrity_per_second": [
            {"sec": 0, "status": "coherent", "note": "scene is stable"},
            {"sec": 1, "status": "coherent", "note": "scene remains stable"},
            {"sec": 2, "status": "coherent", "note": "scene remains stable"},
        ],
        "brief_reason": "The person disappears and the resulting box placement is visible later.",
    }

    parsed = qwen3vl_mod.parse_judgeability_response(
        json.dumps(payload),
        expected_video_id="vid-a",
        prompt_schema=qwen3vl_mod.PROMPT_SCHEMA_OOV_SUBJECT_RESULT_INTEGRITY,
    )

    assert parsed["schema_version"] == qwen3vl_mod.OOV_SUBJECT_RESULT_INTEGRITY_SCHEMA_VERSION
    assert parsed["oov_applicable"] is True
    assert parsed["reason_code"] == "result_evidence_after_subject_gap"
    assert parsed["shared_oov_applicable"] is True
    assert parsed["evidence_d5_applicable"] is True
    assert parsed["evidence_d6_applicable"] is True


def test_qwen35_manifest_video_preflight_fails_on_missing_video(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    out_dir = tmp_path / "out"
    missing_video = tmp_path / "missing.mp4"
    write_json(
        manifest_path,
        [
            {
                "video_id": "vid-missing",
                "path": str(missing_video),
                "world_state_prompt": "A red cube moves behind a wall and returns.",
            }
        ],
    )

    with pytest.raises(FileNotFoundError, match="vid-missing"):
        qwen35_mod.main(
            [
                "--experiment-id",
                "missing_video_preflight",
                "--manifest-path",
                str(manifest_path),
                "--output-dir",
                str(out_dir),
                "--model-path",
                DUMMY_QWEN35_MODEL,
                "--dry-run",
            ]
        )


def test_evidence_export_standalone_required_inputs(tmp_path: Path) -> None:
    manifest = [
        {
            "video_id": "vid-1",
            "path": "/tmp/vid-1.mp4",
            "scoring_video_surface": "full_continuation",
            "scoring_video_path": "/tmp/vid-1.concat.mp4",
            "eval_video_path": "/tmp/vid-1.generated.mp4",
            "original_concat_path": "/tmp/vid-1.concat.mp4",
            "evaluated_segment": "generated_only",
            "condition_frame_count": 81,
            "generated_frame_count": 49,
            "eval_video_frame_start": 81,
            "eval_video_frame_end_exclusive": 130,
            "source_video_path": "/tmp/vid-1.source.mp4",
            "world_state_prompt": "A red cube moves behind a wall and returns.",
            "camera_type": "static",
        }
    ]
    scores = [
        {
            "video_id": "vid-1",
            "path": "/tmp/vid-1.mp4",
            "vlm_spatial_fidelity": 0.8,
            "vlm_state_fidelity": 0.7,
            "vlm_spatial_reasoning": 0.6,
            "vlm_state_reasoning": 0.5,
            "runtime_v2_d5_raw_score": 0.6,
            "runtime_v2_d6_raw_score": 0.5,
            "vlm_dimension_applicable": {
                "spatial_fidelity": True,
                "state_fidelity": True,
                "spatial_reasoning": True,
                "state_reasoning": True,
            },
        }
    ]
    evidence = [
        {
            "video_id": "vid-1",
            "schema_version": "qwen3vl_guarded_teacher_gate_v3",
            "evidence_shared_oov_applicable": False,
            "evidence_shared_oov_na_reason": "no_return",
            "evidence_d5_applicable": False,
            "evidence_d6_applicable": True,
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    scores_path = tmp_path / "scores.json"
    evidence_path = tmp_path / "evidence.jsonl"
    out_dir = tmp_path / "exports"
    write_json(manifest_path, manifest)
    write_json(scores_path, scores)
    write_jsonl(evidence_path, evidence)

    assert (
        export_mod.main(
            [
                "--scores-v7",
                str(scores_path),
                "--evidence-jsonl",
                str(evidence_path),
                "--manifest-path",
                str(manifest_path),
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )

    masked = json.loads(
        (out_dir / "scores_v7_runtime_v2_evidence_first_gate_masked_export.json").read_text(
            encoding="utf-8"
        )
    )
    assert masked[0]["camera_type"] == "static"
    assert masked[0]["scoring_video_surface"] == "full_continuation"
    assert masked[0]["scoring_video_path"] == "/tmp/vid-1.concat.mp4"
    assert masked[0]["eval_video_path"] == "/tmp/vid-1.generated.mp4"
    assert masked[0]["original_concat_path"] == "/tmp/vid-1.concat.mp4"
    assert masked[0]["evaluated_segment"] == "generated_only"
    assert masked[0]["condition_frame_count"] == 81
    assert masked[0]["generated_frame_count"] == 49
    assert masked[0]["eval_video_frame_start"] == 81
    assert masked[0]["eval_video_frame_end_exclusive"] == 130
    assert masked[0]["source_video_path"] == "/tmp/vid-1.source.mp4"
    assert masked[0]["vlm_spatial_reasoning"] is None
    assert masked[0]["vlm_state_reasoning"] == 0.5


def test_metric_scoring_python_files_do_not_import_workplace() -> None:
    scoring_dir = Path(__file__).resolve().parents[1] / "src" / "wrcam" / "eval" / "scoring"
    offenders = []
    for path in scoring_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "workplace." in text or "workplace/" in text:
            offenders.append(path.name)
    assert offenders == []


def test_score_runtime_v2_shell_preflight_accepts_manifest_aliases(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"not-a-real-video")
    manifest_path = tmp_path / "manifest.json"
    out_dir = tmp_path / "out"
    write_json(
        manifest_path,
        [
            {
                "video_id": "vid-1",
                "video_path": str(video),
                "prompt_text": "A red cube moves behind a wall and returns.",
            }
        ],
    )

    script = Path(__file__).resolve().parents[1] / "scripts" / "eval" / "score_runtime_v2_d3d6.sh"
    env = {
        **os.environ,
        "MANIFEST": str(manifest_path),
        "OUT_DIR": str(out_dir),
    }
    result = subprocess.run(
        ["bash", str(script), "preflight"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "[preflight] manifest_records=1" in result.stdout


def test_overlay_install_accepts_wrcam_repo_root() -> None:
    from wrcam.eval.runtime import wrcam_repo_root
    from wrcam.eval.scoring import prompts_v2_probe
    from wrcam.eval.scoring import run_qwen35_p22_overlay as p22
    from wrcam.eval.scoring import run_qwen35_p25_d3d4_slot_parse_overlay as p25

    repo_root = wrcam_repo_root()
    p22.install_p22_overlay(repo_root)
    p25.install_p25_overlay(repo_root)
    assert p22.P22_PROMPT_MODE in prompts_v2_probe.SUPPORTED_PROMPT_MODES
    assert p25.P25_PROMPT_MODE in prompts_v2_probe.SUPPORTED_PROMPT_MODES


def test_score_runtime_v2_shell_defaults_to_wrbench_default_profile() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "eval" / "score_runtime_v2_d3d6.sh"
    text = script.read_text(encoding="utf-8")

    assert 'SCORER_PROFILE="${SCORER_PROFILE:-wrbench_default}"' in text
    assert "wrbench_default|current_benchmark_p25_p22_e14)" in text
    assert 'PROMPT_MODE="${PROMPT_MODE:-runtime_v2_probe_logprob_p25_d3d4_slot_parse}"' in text
    assert 'TASK_CONTEXT_MODE="${TASK_CONTEXT_MODE:-none}"' in text
    assert "run_qwen35_p25_d3d4_slot_parse_overlay.py" in text


def test_score_runtime_v2_shell_keeps_legacy_p9_profile_explicit() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "eval" / "score_runtime_v2_d3d6.sh"
    text = script.read_text(encoding="utf-8")

    assert "legacy_p9_all_manifest_metadata|ablation_manifest_metadata)" in text
    assert 'PROMPT_MODE="${PROMPT_MODE:-runtime_v2_probe_logprob_p9_d4_p8_d5_p6_combined}"' in text
    assert 'TASK_CONTEXT_MODE="${TASK_CONTEXT_MODE:-all_manifest_metadata}"' in text
