"""Tests for wrbench.eval.scoring."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from wrbench.eval.scoring import export_runtime_v2_evidence_first as export_mod
from wrbench.eval.scoring import runtime_common
from wrbench.eval.scoring import run_local_qwen35_probe_logprob_scorer as qwen35_mod
from wrbench.eval.scoring import run_local_qwen3vl_video_evidence as qwen3vl_mod

DUMMY_QWEN35_MODEL = "/tmp/qwen35-model-dir"


def qwen35_required_args(
    *,
    prompt_mode: str = qwen35_mod.DEFAULT_PROMPT_MODE,
    task_context_mode: str = "all_manifest_metadata",
) -> list[str]:
    return [
        "--prompt-mode",
        prompt_mode,
        "--task-context-mode",
        task_context_mode,
        "--num-shards",
        "1",
        "--shard-id",
        "0",
        "--vlm-name",
        "local_qwen35_runtime_v2_probe_logprob",
        "--loader-family",
        "auto",
        "--fps",
        "2",
        "--max-videos",
        "0",
        "--dtype",
        "bfloat16",
        "--attn-implementation",
        "flash_attention_2",
        "--local-rank",
        "0",
        "--num-samples",
        "1",
        "--progress-every",
        "1",
        "--evidence-context-mode",
        "visibility_v1",
    ]


def d3d6_shell_required_env() -> dict[str, str]:
    return {
        "QWEN35_VLM_NAME": "local_qwen35_runtime_v2_probe_logprob",
        "QWEN35_LOADER_FAMILY": "auto",
        "QWEN35_DTYPE": "bfloat16",
        "QWEN35_ATTN_IMPLEMENTATION": "flash_attention_2",
        "QWEN35_NUM_SAMPLES": "1",
        "QWEN35_MAX_VIDEOS": "0",
        "QWEN35_EVIDENCE_CONTEXT_MODE": "visibility_v1",
        "QWEN3VL_DTYPE": "bfloat16",
        "QWEN3VL_ATTN_IMPLEMENTATION": "flash_attention_2",
        "QWEN3VL_MAX_NEW_TOKENS": "48",
        "QWEN3VL_MAX_VIDEOS": "0",
        "QWEN3VL_BINARY_PROMPT_SCHEMA": "subject_judgeability_v1",
        "QWEN3VL_RESCUE_PROMPT_SCHEMA": "guarded_teacher_gate_v3",
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_single_gpu_model_loading_does_not_require_accelerate_device_map() -> None:
    class FakeModel:
        calls: list[tuple[str, dict[str, object]]] = []

        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> "FakeModel":
            cls.calls.append((path, kwargs))
            return cls()

        def to(self, device: str) -> "FakeModel":
            self.device = device
            return self

    model = runtime_common.load_model_explicit_cuda(
        FakeModel,
        model_path="/models/qwen",
        dtype="bf16-sentinel",
        attn_implementation="flash_attention_2",
        local_rank=3,
    )

    assert model.device == "cuda:3"
    assert FakeModel.calls == [
        (
            "/models/qwen",
            {
                "dtype": "bf16-sentinel",
                "trust_remote_code": True,
                "local_files_only": True,
                "attn_implementation": "flash_attention_2",
            },
        )
    ]
    assert "device_map" not in FakeModel.calls[0][1]
    assert qwen35_mod.load_model_explicit_cuda is runtime_common.load_model_explicit_cuda
    assert qwen3vl_mod.load_model_explicit_cuda is runtime_common.load_model_explicit_cuda
    assert runtime_common.explicit_cuda_placement_metadata(3) == {
        "device_map": None,
        "device_placement": "explicit_model_to",
        "model_device": "cuda:3",
    }


def test_single_gpu_model_loading_keeps_torch_dtype_compatibility_retry() -> None:
    class LegacyModel:
        calls: list[dict[str, object]] = []

        @classmethod
        def from_pretrained(cls, _path: str, **kwargs: object) -> "LegacyModel":
            cls.calls.append(kwargs)
            if "dtype" in kwargs:
                raise TypeError("legacy model expects torch_dtype")
            return cls()

        def to(self, device: str) -> "LegacyModel":
            self.device = device
            return self

    model = runtime_common.load_model_explicit_cuda(
        LegacyModel,
        model_path="/models/legacy-qwen",
        dtype="bf16-sentinel",
        attn_implementation="flash_attention_2",
        local_rank=0,
    )

    assert model.device == "cuda:0"
    assert "device_map" not in LegacyModel.calls[0]
    assert "device_map" not in LegacyModel.calls[1]
    assert LegacyModel.calls[1]["torch_dtype"] == "bf16-sentinel"


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
                *qwen35_required_args(),
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


def test_qwen35_manifest_metadata_context_remains_available_in_default_prompt_mode() -> None:
    args = qwen35_mod.parse_args(
        [
            "--experiment-id",
            "metadata_context_contract",
            "--manifest-path",
            "/tmp/manifest.json",
            "--output-dir",
            "/tmp/out",
            "--model-path",
            DUMMY_QWEN35_MODEL,
            *qwen35_required_args(
                prompt_mode=qwen35_mod.DEFAULT_PROMPT_MODE,
                task_context_mode="all_manifest_metadata",
            ),
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
        prompt_mode=qwen35_mod.DEFAULT_PROMPT_MODE,
    )
    assert scorer.task_context_mode == "all_manifest_metadata"


def test_qwen35_current_default_profile_strict_manifest_dry_run(tmp_path: Path) -> None:
    from wrbench.eval.scoring import prompts_v2_probe

    video = tmp_path / "video.mp4"
    video.write_bytes(b"not-a-real-video")
    manifest_path = tmp_path / "manifest.json"
    out_dir = tmp_path / "qwen35-current"
    write_json(
        manifest_path,
        [
            {
                "video_id": "vid-current",
                "path": str(video),
                "world_state_prompt": "A person places a cup on a table.",
            }
        ],
    )

    result = qwen35_mod.main(
        [
            "--experiment-id",
            "wrbench_default_dry_run",
            "--manifest-path",
            str(manifest_path),
            "--output-dir",
            str(out_dir),
            "--model-path",
            DUMMY_QWEN35_MODEL,
            *qwen35_required_args(
                prompt_mode=prompts_v2_probe.PROMPT_MODE_P25_D3D4_SLOT_PARSE,
                task_context_mode="none",
            ),
            "--strict-manifest-contract",
            "--dry-run",
        ]
    )

    assert result == 0
    config = json.loads((out_dir / "run_config.json").read_text())
    assert config["prompt_mode"] == prompts_v2_probe.PROMPT_MODE_P25_D3D4_SLOT_PARSE
    assert config["task_context_mode"] == "none"
    assert config["strict_manifest_contract"] is True


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


def test_qwen3vl_active_gate_schemas_build_and_parse() -> None:
    subject_prompt = qwen3vl_mod.build_judgeability_prompt(
        world_state_prompt="A red cube moves behind a wall and returns.",
        video_id="vid-subj",
        prompt_schema=qwen3vl_mod.PROMPT_SCHEMA_SUBJECT,
    )
    assert "strict video judgeability auditor" in subject_prompt
    parsed_subject = qwen3vl_mod.parse_judgeability_response(
        json.dumps(
            {
                "video_id": "vid-subj",
                "main_subject": "red cube",
                "per_second": [
                    {"sec": 0, "judgeable": "yes", "note": "visible before occlusion"},
                    {"sec": 1, "judgeable": "no", "note": "behind wall"},
                    {"sec": 2, "judgeable": "yes", "note": "reappeared on right"},
                ],
            }
        ),
        expected_video_id="vid-subj",
        prompt_schema=qwen3vl_mod.PROMPT_SCHEMA_SUBJECT,
    )
    assert parsed_subject["schema_version"] == qwen3vl_mod.SUBJECT_SCHEMA_VERSION
    assert parsed_subject["evidence_shared_oov_applicable"] is True
    assert parsed_subject["evidence_d5_applicable"] is True
    assert parsed_subject["evidence_d6_applicable"] is True

    guarded_prompt = qwen3vl_mod.build_judgeability_prompt(
        world_state_prompt="A red cube moves behind a wall and returns.",
        video_id="vid-guard",
        prompt_schema=qwen3vl_mod.PROMPT_SCHEMA_GUARDED,
    )
    assert "strict D5/D6 judgeability gate auditor" in guarded_prompt
    parsed_guarded = qwen3vl_mod.parse_judgeability_response(
        json.dumps(
            {
                "video_id": "vid-guard",
                "prompt_critical_evidence": "red cube",
                "expected_motion_or_state": "moves behind wall and returns",
                "static_anchors": ["wall"],
                "before_hidden_evidence": "cube visible left",
                "after_return_evidence": "cube visible in wrong corner",
                "per_second": [
                    {"sec": 0, "judgeable": "yes", "note": "visible"},
                    {"sec": 1, "judgeable": "no", "note": "hidden"},
                    {"sec": 2, "judgeable": "yes", "note": "wrong corner but judgeable"},
                ],
                "gate": {"d5_applicable": True, "d6_applicable": True, "na_reason": None},
                "audit": {"wrong_but_judgeable": True, "anchor_available": True, "after_return_available": True},
            }
        ),
        expected_video_id="vid-guard",
        prompt_schema=qwen3vl_mod.PROMPT_SCHEMA_GUARDED,
    )
    assert parsed_guarded["schema_version"] == qwen3vl_mod.GUARDED_SCHEMA_VERSION
    assert parsed_guarded["evidence_shared_oov_applicable"] is True
    assert parsed_guarded["evidence_d5_applicable"] is True
    assert parsed_guarded["evidence_d6_applicable"] is True


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
                *qwen35_required_args(),
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


@pytest.mark.parametrize("disabled", [False, True])
def test_evidence_export_uses_video_gate_unless_disabled(tmp_path: Path, disabled: bool) -> None:
    manifest = [
        {
            "video_id": "vid-no-oov",
            "path": "/tmp/vid-no-oov.mp4",
            "world_state_prompt": "A child stands beside a chair.",
            "camera_type": "yaw_RL",
            "oov_gap": "none",
        }
    ]
    if disabled:
        manifest[0]["reobservation_enabled"] = False
    scores = [
        {
            "video_id": "vid-no-oov",
            "path": "/tmp/vid-no-oov.mp4",
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
            "video_id": "vid-no-oov",
            "schema_version": "qwen3vl_guarded_teacher_gate_v3",
            "evidence_shared_oov_applicable": True,
            "evidence_d5_applicable": True,
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
    row = masked[0]
    if not disabled:
        assert row["runtime_v2_evidence_shared_oov_applicable"] is True
        assert row["vlm_spatial_reasoning"] == 0.6
        assert row["vlm_state_reasoning"] == 0.5
        return
    assert row["runtime_v2_evidence_shared_oov_applicable"] is False
    assert row["runtime_v2_evidence_shared_oov_na_reason"] == "no_oov"
    assert row["runtime_v2_evidence_d5_applicable"] is False
    assert row["runtime_v2_evidence_d5_na_reason"] == "no_oov"
    assert row["runtime_v2_evidence_d6_applicable"] is False
    assert row["runtime_v2_evidence_d6_na_reason"] == "no_oov"
    assert row["vlm_spatial_reasoning"] is None
    assert row["vlm_state_reasoning"] is None


def test_metric_scoring_python_files_do_not_import_workplace() -> None:
    scoring_dir = Path(__file__).resolve().parents[1] / "src" / "wrbench" / "eval" / "scoring"
    offenders = []
    for path in scoring_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "workplace." in text or "workplace/" in text:
            offenders.append(path.name)
    assert offenders == []


@pytest.mark.parametrize(
    ("legacy_entry", "overrides", "bad_setting"),
    [
        (False, {}, None),
        (True, {}, None),
        (False, {"SCORER_PROFILE": "default", "PROMPT_MODE": "wrbench_paper"}, None),
        (False, {"PROMPT_MODE": qwen35_mod.DEFAULT_PROMPT_MODE}, "PROMPT_MODE"),
        (False, {"TASK_CONTEXT_MODE": "all_manifest_metadata"}, "TASK_CONTEXT_MODE"),
        (False, {"QWEN3VL_BINARY_PROMPT_SCHEMA": qwen3vl_mod.PROMPT_SCHEMA_GUARDED}, "QWEN3VL_BINARY_PROMPT_SCHEMA"),
        (False, {"QWEN3VL_RESCUE_PROMPT_SCHEMA": "incorrect"}, "QWEN3VL_RESCUE_PROMPT_SCHEMA"),
        (
            False,
            {"SCORER_PROFILE": "ablation_manifest_metadata",
             "PROMPT_MODE": qwen35_mod.DEFAULT_PROMPT_MODE,
             "TASK_CONTEXT_MODE": "all_manifest_metadata"},
            None,
        ),
    ],
)
def test_score_runtime_v2_shell_preflight(
    tmp_path: Path, legacy_entry: bool, overrides: dict[str, str], bad_setting: str | None,
) -> None:
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

    if legacy_entry:
        script = Path(__file__).resolve().parents[1] / "scripts/eval/score_runtime_v2_d3d6.sh"
    else:
        script = Path(runtime_common.__file__).resolve().parents[1] / "score_runtime_v2_d3d6.sh"
    env = {
        **os.environ,
        "FORCE_QWENVL_VIDEO_READER": "decord",
        "WORLD_STATE_VIDEO_BACKEND": "decord",
        "PY_SCORER": sys.executable,
        "PY_HELPER": sys.executable,
        "MANIFEST": str(manifest_path),
        "OUT_DIR": str(out_dir),
        "QWEN35_MODEL": str(tmp_path),
        "QWEN3VL_MODEL": str(tmp_path),
        "SCORER_PROFILE": "wrbench_default",
        "RUN_TAG": "test_run",
        "PROMPT_MODE": "runtime_v2_probe_logprob_p25_d3d4_slot_parse",
        "TASK_CONTEXT_MODE": "none",
        "NUM_SHARDS": "1",
        "SHARD_IDS": "0",
        "CUDA_DEVICES_CSV": "0",
        "FPS": "2",
        "PROGRESS_EVERY": "1",
        "SKIP_EXISTING": "0",
        **d3d6_shell_required_env(),
        **overrides,
    }
    result = subprocess.run(
        ["bash", str(script), "preflight"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )

    if bad_setting is None:
        assert result.returncode == 0, result.stderr
        assert "[preflight] manifest_records=1" in result.stdout
    else:
        assert result.returncode == 2
        assert f"wrbench_default requires {bad_setting}=" in result.stderr
        assert not out_dir.exists()


def test_runtime_example_matches_wrbench_default_d3d6_profile() -> None:
    from wrbench.eval import runtime as eval_runtime_mod
    from wrbench.eval.scoring import prompts_v2_probe

    runtime_example = Path(__file__).resolve().parents[1] / "wrbench.runtime.example.json"
    cfg = eval_runtime_mod.load_eval_runtime(runtime_example)
    assert cfg is not None
    assert cfg.scorers.env == {}

    env = eval_runtime_mod.DEFAULT_D3D6_ENV
    assert env["PROMPT_MODE"] == prompts_v2_probe.PROMPT_MODE_WRBENCH_PAPER
    assert (
        prompts_v2_probe.validate_prompt_mode(env["PROMPT_MODE"])
        == prompts_v2_probe.PROMPT_MODE_P25_D3D4_SLOT_PARSE
    )
    assert env["TASK_CONTEXT_MODE"] == "none"
    assert int(env["QWEN3VL_MAX_NEW_TOKENS"]) == qwen3vl_mod.DEFAULT_MAX_NEW_TOKENS


def test_current_qwen35_profile_is_declared_and_guarded_without_overlay() -> None:
    from wrbench.eval.scoring import prompts_v2_probe

    assert prompts_v2_probe.PROMPT_MODE_WRBENCH_PAPER in prompts_v2_probe.SUPPORTED_PROMPT_MODES
    assert prompts_v2_probe.PROMPT_MODE_P22_NATIVE_D5D6 in prompts_v2_probe.SUPPORTED_PROMPT_MODES
    assert prompts_v2_probe.PROMPT_MODE_P25_D3D4_SLOT_PARSE in prompts_v2_probe.SUPPORTED_PROMPT_MODES
    assert (
        prompts_v2_probe.validate_prompt_mode(prompts_v2_probe.PROMPT_MODE_WRBENCH_PAPER)
        == prompts_v2_probe.PROMPT_MODE_P25_D3D4_SLOT_PARSE
    )
    assert (
        prompts_v2_probe.validate_prompt_mode(prompts_v2_probe.PROMPT_MODE_P25_D3D4_SLOT_PARSE)
        == prompts_v2_probe.PROMPT_MODE_P25_D3D4_SLOT_PARSE
    )

    # Confirm wrbench_paper produces 100% bit-identical prompts across all probes
    for probe in prompts_v2_probe.RUNTIME_V2_PROBE_CATALOG:
        p_alias = prompts_v2_probe.build_runtime_v2_probe_prompt(
            world_state_prompt="A person places a cup on a table.",
            video_id="vid-alias",
            probe=probe,
            fps="2",
            prompt_mode=prompts_v2_probe.PROMPT_MODE_WRBENCH_PAPER,
        )
        p_canon = prompts_v2_probe.build_runtime_v2_probe_prompt(
            world_state_prompt="A person places a cup on a table.",
            video_id="vid-alias",
            probe=probe,
            fps="2",
            prompt_mode=prompts_v2_probe.PROMPT_MODE_P25_D3D4_SLOT_PARSE,
        )
        assert p_alias == p_canon

    # parse_args normalizes wrbench_paper to canonical mode and enforces the same guards
    parsed = qwen35_mod.parse_args(
        [
            "--experiment-id", "alias_ok",
            "--manifest-path", "/tmp/m.json",
            "--output-dir", "/tmp/o",
            "--model-path", DUMMY_QWEN35_MODEL,
            "--strict-manifest-contract",
            *qwen35_required_args(
                prompt_mode=prompts_v2_probe.PROMPT_MODE_WRBENCH_PAPER,
                task_context_mode="none",
            ),
        ]
    )
    assert parsed.prompt_mode == prompts_v2_probe.PROMPT_MODE_P25_D3D4_SLOT_PARSE

    with pytest.raises(SystemExit):
        qwen35_mod.parse_args(
            [
                "--experiment-id", "p25_guard",
                "--manifest-path", "/tmp/m.json",
                "--output-dir", "/tmp/o",
                "--model-path", DUMMY_QWEN35_MODEL,
                *qwen35_required_args(
                    prompt_mode=prompts_v2_probe.PROMPT_MODE_WRBENCH_PAPER,
                    task_context_mode="all_manifest_metadata",
                ),
            ]
        )

    with pytest.raises(SystemExit):
        qwen35_mod.parse_args(
            [
                "--experiment-id", "p25_guard_ev",
                "--manifest-path", "/tmp/m.json",
                "--output-dir", "/tmp/o",
                "--model-path", DUMMY_QWEN35_MODEL,
                "--evidence-jsonl", "/tmp/ev.jsonl",
                *qwen35_required_args(
                    prompt_mode=prompts_v2_probe.PROMPT_MODE_P25_D3D4_SLOT_PARSE,
                    task_context_mode="none",
                ),
            ]
        )
