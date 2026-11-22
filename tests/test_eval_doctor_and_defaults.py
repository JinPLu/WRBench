"""Tests for eval defaults, pure-Python D3-D6 pipeline, Issue #11 guard, eval doctor, plugins, and custom table rows."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from wrbench.eval import d3d6_pipeline, doctor, plugins as eval_plugins, runtime as eval_runtime
from wrbench.eval.aggregate import build_main_table
from wrbench.eval.d2.extract_d2_dinov2_fullfov_candidate import PAPER_SAMPLING
from wrbench.eval.scoring import run_local_qwen35_probe_logprob_scorer as qwen35_mod
from wrbench.eval.scoring import runtime_common
from wrbench.registry import RegistryError
from wrbench.runtime import RuntimeConfigError


def _make_executable(path: Path, body: str = "#!/usr/bin/env python3\nimport sys\n") -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _make_minimal_scorer_paths(tmp_path: Path, *, qwen_python: Path | None = None) -> dict[str, str]:
    vggt_py = _make_executable(tmp_path / "vggt_py")
    vggt_repo = tmp_path / "vggt_repo"
    vggt_repo.mkdir()
    vggt_ckpt = tmp_path / "vggt.pt"
    vggt_ckpt.write_bytes(b"ckpt")
    dino_py = _make_executable(tmp_path / "dino_py")
    dino_dir = tmp_path / "dinov2"
    dino_dir.mkdir()
    qwen_py = qwen_python or _make_executable(tmp_path / "qwen_py")
    qwen35_dir = tmp_path / "qwen35"
    qwen35_dir.mkdir()
    qwen3vl_dir = tmp_path / "qwen3vl"
    qwen3vl_dir.mkdir()
    return {
        "vggt_python_bin": str(vggt_py),
        "vggt_repo": str(vggt_repo),
        "vggt_checkpoint": str(vggt_ckpt),
        "dinov2_python_bin": str(dino_py),
        "dinov2_model_path": str(dino_dir),
        "qwen_scorer_python": str(qwen_py),
        "qwen35_model": str(qwen35_dir),
        "qwen3vl_model": str(qwen3vl_dir),
    }


def test_builtin_paper_defaults_exported_and_consistent() -> None:
    assert eval_runtime.DEFAULT_D2_EXTRA["d2_device"] == "cuda:0"
    assert eval_runtime._d2_sampling(eval_runtime.EvalScorerRuntime()) == PAPER_SAMPLING

    assert eval_runtime.DEFAULT_VGGT_EXTRA["d1_vggt_execution_mode"] == "subprocess"
    assert eval_runtime.DEFAULT_VGGT_EXTRA["d1_vggt_image_resolution"] == "512"
    assert eval_runtime.DEFAULT_VGGT_EXTRA["d1_vggt_preprocess_mode"] == "balanced"
    assert eval_runtime.DEFAULT_VGGT_EXTRA["d1_vggt_max_frames"] == "49"
    assert eval_runtime.DEFAULT_VGGT_EXTRA["d1_vggt_shard_index"] == "0"
    assert eval_runtime.DEFAULT_VGGT_EXTRA["d1_vggt_num_shards"] == "1"

    example_path = Path(__file__).resolve().parents[1] / "wrbench.runtime.example.json"
    example_cfg = eval_runtime.load_eval_runtime(example_path)
    assert example_cfg is not None
    assert example_cfg.scorers.extra_paths == {}
    assert example_cfg.scorers.env == {}
    assert eval_runtime._d2_sampling(example_cfg.scorers) == PAPER_SAMPLING
    assert eval_runtime.DEFAULT_D3D6_ENV["PROMPT_MODE"] == "wrbench_paper"
    assert eval_runtime.DEFAULT_D3D6_ENV["TASK_CONTEXT_MODE"] == "none"
    assert eval_runtime.DEFAULT_D3D6_ENV["QWEN35_VISION_BOUNDARY_MODE"] == "warn"


def test_minimal_runtime_config_works_without_extra_paths_or_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer_paths = _make_minimal_scorer_paths(tmp_path)
    cfg_path = tmp_path / "wrbench.runtime.json"
    cfg_path.write_text(
        json.dumps({"schema_version": 1, "eval": {"scorers": scorer_paths}}),
        encoding="utf-8",
    )
    cfg = eval_runtime.load_eval_runtime(cfg_path)
    assert cfg is not None
    assert cfg.scorers.gpu_id == 0
    assert cfg.scorers.extra_paths == {}
    assert cfg.scorers.env == {}

    # VGGT falls back to DEFAULT_VGGT_EXTRA
    vggt_argvs: list[list[str]] = []
    monkeypatch.setattr(
        "wrbench.eval.d1.run_d1_vggt_omega_batch.main",
        lambda argv: (vggt_argvs.append(argv) or 0),
    )
    assert (
        eval_runtime.d1_vggt_batch(
            eval_runtime=cfg,
            input_jsonl=tmp_path / "in.jsonl",
            output_root=tmp_path / "pose_out",
            cache_root=tmp_path / "cache",
        )
        == 0
    )
    assert "--image-resolution" in vggt_argvs[0] and "512" in vggt_argvs[0]
    assert "--preprocess-mode" in vggt_argvs[0] and "balanced" in vggt_argvs[0]
    assert vggt_argvs[0][vggt_argvs[0].index("--max-frames") + 1] == "49"
    assert vggt_argvs[0][vggt_argvs[0].index("--concat-pose-mode") + 1] == "full_concat"

    # Explicit max_frames=0 omits --max-frames for custom unconstrained runs
    assert (
        eval_runtime.d1_vggt_batch(
            eval_runtime=cfg,
            input_jsonl=tmp_path / "in.jsonl",
            output_root=tmp_path / "pose_out",
            cache_root=tmp_path / "cache",
            max_frames=0,
        )
        == 0
    )
    assert "--max-frames" not in vggt_argvs[1]

    # DINOv2 falls back to DEFAULT_D2_EXTRA
    d2_calls: list[tuple[str, str, list[str]]] = []
    monkeypatch.setattr(
        eval_runtime,
        "run_scorer_python",
        lambda py, mod, argv, *, env=None: (d2_calls.append((py, mod, argv)) or 0),
    )
    assert (
        eval_runtime.d2_extract(
            eval_runtime=cfg,
            videos_manifest=tmp_path / "manifest.json",
            out_jsonl=tmp_path / "d2.jsonl",
        )
        == 0
    )
    assert "--device" in d2_calls[0][2] and "cuda:0" in d2_calls[0][2]
    assert "--sample-fps" in d2_calls[0][2] and "3.0" in d2_calls[0][2]

    # D3-D6 falls back to DEFAULT_D3D6_ENV
    env = eval_runtime.d3d6_env(
        eval_runtime=cfg,
        manifest=tmp_path / "manifest.json",
        out_dir=tmp_path / "d3d6",
        scorer_profile="default",
    )
    assert env["SCORER_PROFILE"] == "wrbench_default"
    assert env["PROMPT_MODE"] == "wrbench_paper"
    assert env["TASK_CONTEXT_MODE"] == "none"
    assert env["QWEN35_VISION_BOUNDARY_MODE"] == "warn"
    assert env["CUDA_DEVICES_CSV"] == "0"


def test_wrbench_default_still_rejects_explicit_d2_vggt_and_camalign_overrides(tmp_path: Path) -> None:
    scorer_paths = _make_minimal_scorer_paths(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")

    bad_d2_cfg = eval_runtime.EvalRuntimeConfig(
        schema_version=1,
        scorers=eval_runtime.EvalScorerRuntime(
            **scorer_paths,
            extra_paths={"d2_sample_fps": "1.0"},
        ),
    )
    with pytest.raises(ValueError, match="wrbench_default requires D2"):
        eval_runtime.eval_run(
            eval_runtime=bad_d2_cfg,
            manifest=manifest,
            out_dir=tmp_path / "out1",
            scorer_profile="wrbench_default",
            sidecar_profile_gate="main",
        )

    bad_vggt_cfg = eval_runtime.EvalRuntimeConfig(
        schema_version=1,
        scorers=eval_runtime.EvalScorerRuntime(
            **scorer_paths,
            extra_paths={"d1_vggt_max_frames": "81"},
        ),
    )
    with pytest.raises(ValueError, match="wrbench_default requires d1_vggt_max_frames=49"):
        eval_runtime.eval_run(
            eval_runtime=bad_vggt_cfg,
            manifest=manifest,
            out_dir=tmp_path / "out_vggt",
        )

    good_cfg = eval_runtime.EvalRuntimeConfig(
        schema_version=1,
        scorers=eval_runtime.EvalScorerRuntime(**scorer_paths),
    )
    with pytest.raises(ValueError, match="CamAlign v2_lite"):
        eval_runtime.eval_run(
            eval_runtime=good_cfg,
            manifest=manifest,
            out_dir=tmp_path / "out2",
            scorer_profile="default",
            sidecar_profile_gate="main",
            camalign_scorer_version="v1",
        )


def test_eval_entry_point_plugin_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyPolicy:
        name = "ep_policy"

        def applies(self, row: dict[str, Any]) -> bool:
            return False

        def slice_poses(self, row: dict[str, Any], poses: Any) -> tuple[Any, dict[str, Any]]:
            return poses, {}

    def _register_ep() -> None:
        eval_plugins.register_pose_backend("ep_pose", lambda **kw: 0)
        eval_plugins.register_d2_scorer("ep_d2", lambda **kw: 0)
        eval_plugins.register_vlm_scorer("ep_vlm", lambda **kw: 0)
        eval_plugins.register_segment_policy("ep_policy", DummyPolicy())

    class FakeEP:
        name = "fake_eval_plugin"

        def load(self) -> Any:
            return _register_ep

    def fake_entry_points(*, group: str | None = None) -> list[FakeEP]:
        return [FakeEP()] if group == "wrbench.eval_plugins" else []

    monkeypatch.setattr(eval_plugins.importlib_metadata, "entry_points", fake_entry_points)
    monkeypatch.setattr(eval_plugins, "_ENTRY_POINTS_LOADED", False)
    try:
        assert "ep_pose" in eval_plugins.registered_pose_backends()
        assert "ep_d2" in eval_plugins.registered_d2_scorers()
        assert "ep_vlm" in eval_plugins.registered_vlm_scorers()
        assert "ep_policy" in eval_plugins.registered_segment_policies()
        assert eval_plugins.get_backend("pose", "ep_pose")() == 0
    finally:
        eval_plugins._BACKENDS["pose"].pop("ep_pose", None)
        eval_plugins._BACKENDS["d2"].pop("ep_d2", None)
        eval_plugins._BACKENDS["vlm"].pop("ep_vlm", None)
        eval_plugins._SEGMENT_POLICIES.pop("ep_policy", None)
        eval_plugins._ENTRY_POINTS_LOADED = True


def test_pure_python_d3d6_pipeline_all_stages_and_no_bash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"mp4")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps([{"video_id": "v1", "video_path": str(video), "world_state_prompt": "A ball rolls."}]),
        encoding="utf-8",
    )
    scorer_paths = _make_minimal_scorer_paths(tmp_path)
    cfg = eval_runtime.EvalRuntimeConfig(
        schema_version=1,
        scorers=eval_runtime.EvalScorerRuntime(**scorer_paths),
    )
    out_dir = tmp_path / "d3d6_out"

    sharded_calls: list[tuple[str, list[str]]] = []
    cmd_calls: list[list[str]] = []

    def fake_sharded(*, name: str, py: str, module_args: list[str], **kwargs: Any) -> int:
        sharded_calls.append((name, list(module_args)))
        return 0

    def fake_cmd(cmd: list[str], *, env: Any) -> int:
        cmd_calls.append(list(cmd))
        if any("build_qwen3vl_binary_na_rescue_manifest" in part for part in cmd):
            rescue_manifest = Path(cmd[cmd.index("--out-manifest") + 1])
            rescue_manifest.write_text("[]", encoding="utf-8")
        if any("export_runtime_v2_evidence_first" in part for part in cmd):
            final_dir = Path(cmd[cmd.index("--out-dir") + 1])
            final_dir.mkdir(parents=True, exist_ok=True)
            (final_dir / "scores_v7_runtime_v2_evidence_first_gate_masked_export.json").write_text(
                "[]", encoding="utf-8"
            )
        return 0

    def forbid_bash(*args: Any, **kwargs: Any) -> int:
        raise AssertionError("_run_qwen_runtime must not invoke run_shell_script/bash")

    monkeypatch.setattr(d3d6_pipeline, "_run_sharded", fake_sharded)
    monkeypatch.setattr(d3d6_pipeline, "_run_command", fake_cmd)
    monkeypatch.setattr(eval_runtime, "run_shell_script", forbid_bash)

    rc = eval_runtime.d3d6_score(
        eval_runtime=cfg,
        manifest=manifest,
        out_dir=out_dir,
        stage="all",
        scorer_profile="wrbench_default",
    )
    assert rc == 0
    assert [name for name, _ in sharded_calls] == ["qwen35", "qwen3vl_binary"]
    assert "--strict-manifest-contract" in sharded_calls[0][1]
    assert (out_dir / "qwen3vl_oov_guarded_rescue" / "evidence.jsonl").is_file()
    modules_run = [cmd[2] for cmd in cmd_calls if len(cmd) > 2 and cmd[1] == "-m"]
    assert modules_run == [
        "wrbench.eval.scoring.run_local_qwen35_probe_logprob_scorer",
        "wrbench.eval.scoring.run_local_qwen3vl_video_evidence",
        "wrbench.eval.scoring.build_qwen3vl_binary_na_rescue_manifest",
        "wrbench.eval.scoring.merge_qwen3vl_rescue_evidence",
        "wrbench.eval.scoring.export_runtime_v2_evidence_first",
    ]


def test_d3d6_pipeline_main_help_and_missing_stage(capsys: pytest.CaptureFixture[str]) -> None:
    assert d3d6_pipeline.main(["--help"]) == 0
    assert "Stages:" in capsys.readouterr().out
    assert d3d6_pipeline.main([]) == 2
    assert "stage argument is required" in capsys.readouterr().err


def test_vision_boundary_tokens_detect_warn_strip_and_preserve() -> None:
    VS = runtime_common.DEFAULT_VISION_START_TOKEN_ID
    VE = runtime_common.DEFAULT_VISION_END_TOKEN_ID
    VP = runtime_common.DEFAULT_VIDEO_PAD_TOKEN_ID

    # Historical transformers 5.9.0 layout (no outer markers around timestamped frames)
    clean_ids = np.array([[10, 11, 12, 900, VS, VP, VP, VE, 901, VS, VP, VP, VE, 20, 21]], dtype=np.int64)
    clean_inputs = {
        "input_ids": clean_ids,
        "attention_mask": np.ones_like(clean_ids),
        "mm_token_type_ids": np.zeros_like(clean_ids),
    }
    aligned_clean, clean_info = runtime_common.detect_or_align_vision_boundary_tokens(
        clean_inputs,
        mode="strip_outer_markers",
        transformers_version="5.9.0",
    )
    assert clean_info["extra_boundary_tokens_detected"] is False
    assert clean_info["vision_boundary_aligned"] is False
    assert clean_info["warning"] is None
    assert np.array_equal(aligned_clean["input_ids"], clean_ids)

    # Transformers >=5.22 layout: outer VS at index 3 and outer VE at index 14 (right after inner VE at 13)
    issue_ids = np.array([[10, 11, 12, VS, 900, VS, VP, VP, VE, 901, VS, VP, VP, VE, VE, 20, 21]], dtype=np.int64)
    issue_mask = np.ones_like(issue_ids)
    issue_mm = np.arange(issue_ids.shape[1], dtype=np.int64)[None, :]
    issue_inputs = {
        "input_ids": issue_ids,
        "attention_mask": issue_mask,
        "mm_token_type_ids": issue_mm,
    }

    # Mode = warn
    warned_out, warn_info = runtime_common.detect_or_align_vision_boundary_tokens(
        issue_inputs,
        mode="warn",
        transformers_version="5.22.1",
    )
    assert warn_info["extra_boundary_tokens_detected"] is True
    assert warn_info["extra_boundary_indices"] == [3, 14]
    assert warn_info["vision_boundary_aligned"] is False
    assert "docs/eval/hunyuan-d3-replay.md" in str(warn_info["warning"])
    assert "QWEN35_VISION_BOUNDARY_MODE=strip_outer_markers" in str(warn_info["warning"])
    assert "5.9.0" in str(warn_info["warning"])
    assert np.array_equal(warned_out["input_ids"], issue_ids)

    # Mode = strip_outer_markers
    stripped_out, strip_info = runtime_common.detect_or_align_vision_boundary_tokens(
        issue_inputs,
        mode="strip_outer_markers",
        transformers_version="5.22.1",
    )
    assert strip_info["extra_boundary_tokens_detected"] is True
    assert strip_info["vision_boundary_aligned"] is True
    assert strip_info["original_seq_len"] == 17
    assert strip_info["aligned_seq_len"] == 15
    assert np.array_equal(stripped_out["input_ids"], clean_ids)
    assert stripped_out["attention_mask"].shape == (1, 15)
    assert stripped_out["mm_token_type_ids"].tolist() == [[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16]]

    # Mode = preserve
    preserved_out, preserve_info = runtime_common.detect_or_align_vision_boundary_tokens(
        issue_inputs,
        mode="preserve",
        transformers_version="5.22.1",
    )
    assert preserve_info["extra_boundary_tokens_detected"] is True
    assert preserve_info["vision_boundary_aligned"] is False
    assert preserve_info["warning"] is None
    assert np.array_equal(preserved_out["input_ids"], issue_ids)


def test_qwen35_run_config_records_version_and_boundary_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QWEN35_VISION_BOUNDARY_MODE", "strip_outer_markers")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"clip")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps([{"video_id": "v1", "path": str(video), "world_state_prompt": "A cat jumps."}]),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    rc = qwen35_mod.main(
        [
            "--experiment-id",
            "meta_check",
            "--manifest-path",
            str(manifest),
            "--output-dir",
            str(out_dir),
            "--model-path",
            str(tmp_path),
            "--prompt-mode",
            qwen35_mod.PROMPT_MODE_P25_D3D4_SLOT_PARSE,
            "--task-context-mode",
            "none",
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
            "--dry-run",
        ]
    )
    assert rc == 0
    cfg = json.loads((out_dir / "run_config.json").read_text(encoding="utf-8"))
    assert "transformers_version" in cfg
    assert "torch_version" in cfg
    assert "tokenizers_version" in cfg
    assert "decord_version" in cfg
    assert cfg["vision_boundary_mode"] == "strip_outer_markers"
    assert cfg["vision_boundary_aligned"] is False


def test_eval_doctor_probes_transformers_version_and_respects_custom_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fake_qwen_python(name: str, tf_version: str) -> Path:
        script = (
            f"#!{sys.executable}\n"
            "import json\n"
            f"print(json.dumps({{'transformers': {tf_version!r}, 'torch': '2.11.0', 'tokenizers': '0.22.2', 'decord': '0.6.0'}}))\n"
        )
        return _make_executable(tmp_path / name, script)

    # Case 1: transformers 5.22.1 -> ok=True with Issue #11 warning
    run522 = tmp_path / "run522"
    run522.mkdir()
    paths_522 = _make_minimal_scorer_paths(run522, qwen_python=_fake_qwen_python("py522", "5.22.1"))
    cfg_522 = tmp_path / "rt_522.json"
    cfg_522.write_text(json.dumps({"schema_version": 1, "eval": {"scorers": paths_522}}), encoding="utf-8")
    ok_522, checks_522 = doctor.check_eval_runtime(cfg_522)
    assert ok_522 is True
    tf_check = next(c for c in checks_522 if c["name"] == "vlm.transformers_version")
    assert tf_check["status"] == "warn"
    assert "Issue #11" in tf_check["message"]
    assert "QWEN35_VISION_BOUNDARY_MODE=strip_outer_markers" in tf_check["message"]
    report = doctor.format_eval_doctor_report(checks_522)
    assert "WRBench Eval Doctor" in report
    assert "[WARN] vlm.transformers_version" in report
    assert "Summary: PASS" in report

    # Case 2: transformers 5.9.0 -> ok=True and vlm.transformers_version == ok
    run59 = tmp_path / "run59"
    run59.mkdir()
    paths_59 = _make_minimal_scorer_paths(run59, qwen_python=_fake_qwen_python("py59", "5.9.0"))
    cfg_59 = tmp_path / "rt_59.json"
    cfg_59.write_text(json.dumps({"schema_version": 1, "eval": {"scorers": paths_59}}), encoding="utf-8")
    ok_59, checks_59 = doctor.check_eval_runtime(cfg_59)
    assert ok_59 is True
    assert next(c for c in checks_59 if c["name"] == "vlm.transformers_version")["status"] == "ok"

    # Case 3: custom backends skip built-in scorer paths, and bad backend fails doctor
    (tmp_path / "doc_backend.py").write_text("def ok_fn(**kw):\n    return 0\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    cfg_custom = tmp_path / "rt_custom.json"
    cfg_custom.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "eval": {
                    "backends": {
                        "pose": "doc_backend:ok_fn",
                        "d2": "doc_backend:ok_fn",
                        "vlm": "doc_backend:ok_fn",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    ok_custom, checks_custom = doctor.check_eval_runtime(cfg_custom)
    assert ok_custom is True
    assert all(c["status"] == "ok" for c in checks_custom)

    cfg_broken = tmp_path / "rt_broken.json"
    cfg_broken.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "eval": {
                    "backends": {
                        "pose": "nonexistent_mod_xyz:fn",
                        "d2": "doc_backend:ok_fn",
                        "vlm": "doc_backend:ok_fn",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    ok_broken, checks_broken = doctor.check_eval_runtime(cfg_broken)
    assert ok_broken is False
    assert next(c for c in checks_broken if c["name"] == "pose.backend")["status"] == "error"


def test_build_main_table_handles_unknown_and_registry_error_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_records = [
        {
            "video_id": "v_custom_1",
            "model": "unregistered_model_alpha",
            "vlm_spatial_fidelity": 0.7,
            "vlm_state_fidelity": 0.6,
            "vlm_spatial_reasoning": None,
            "vlm_state_reasoning": None,
            "runtime_v2_evidence_shared_oov_applicable": False,
        },
        {
            "video_id": "v_custom_2",
            "model": "unregistered_model_beta",
            "viewpoint_condition_type": "geometry-cache",
            "model_input": "TV2V",
            "vlm_spatial_fidelity": 0.8,
            "vlm_state_fidelity": 0.75,
            "vlm_spatial_reasoning": None,
            "vlm_state_reasoning": None,
            "runtime_v2_evidence_shared_oov_applicable": False,
        },
    ]
    table, _ = build_main_table.build_table(
        runtime_records=runtime_records,
        d1_records=[],
        d2_records=[],
    )
    by_model = {row["model"]: row for row in table}
    assert by_model["unregistered_model_alpha"]["viewpoint_condition_type"] == "custom"
    assert by_model["unregistered_model_alpha"]["model_input"] == "unknown"
    assert by_model["unregistered_model_beta"]["viewpoint_condition_type"] == "geometry-cache"
    assert by_model["unregistered_model_beta"]["model_input"] == "TV2V"

    # Also verify RegistryError from viewpoint_condition_type/model_input is caught cleanly
    def raise_registry_error(_name: str) -> str:
        raise RegistryError("not declared")

    monkeypatch.setattr(build_main_table, "viewpoint_condition_type", raise_registry_error)
    monkeypatch.setattr(build_main_table, "registry_model_input", raise_registry_error)
    assert build_main_table._viewpoint_condition("partial_model", []) == "custom"
    assert build_main_table._model_input("partial_model", []) == "unknown"


def test_d3d6_pipeline_nonempty_rescue_and_skip_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"mp4")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps([{"video_id": "v1", "video_path": str(video), "world_state_prompt": "A cup sits."}]),
        encoding="utf-8",
    )
    scorer_paths = _make_minimal_scorer_paths(tmp_path)
    cfg = eval_runtime.EvalRuntimeConfig(
        schema_version=1,
        scorers=eval_runtime.EvalScorerRuntime(
            **scorer_paths,
            env={"SKIP_EXISTING": "1"},
        ),
    )
    out_dir = tmp_path / "d3d6_rescue"
    sharded_calls: list[tuple[str, list[str]]] = []
    cmd_calls: list[list[str]] = []

    def fake_sharded(*, name: str, py: str, module_args: list[str], **kwargs: Any) -> int:
        sharded_calls.append((name, list(module_args)))
        return 0

    def fake_cmd(cmd: list[str], *, env: Any) -> int:
        cmd_calls.append(list(cmd))
        if any("build_qwen3vl_binary_na_rescue_manifest" in part for part in cmd):
            rescue_manifest = Path(cmd[cmd.index("--out-manifest") + 1])
            rescue_manifest.write_text(json.dumps([{"video_id": "v1"}]), encoding="utf-8")
        if any("export_runtime_v2_evidence_first" in part for part in cmd):
            final_dir = Path(cmd[cmd.index("--out-dir") + 1])
            final_dir.mkdir(parents=True, exist_ok=True)
            (final_dir / "scores_v7_runtime_v2_evidence_first_gate_masked_export.json").write_text(
                "[]", encoding="utf-8"
            )
        return 0

    monkeypatch.setattr(d3d6_pipeline, "_run_sharded", fake_sharded)
    monkeypatch.setattr(d3d6_pipeline, "_run_command", fake_cmd)

    rc = eval_runtime.d3d6_score(
        eval_runtime=cfg,
        manifest=manifest,
        out_dir=out_dir,
        stage="all",
        scorer_profile="wrbench_default",
    )
    assert rc == 0
    assert [name for name, _ in sharded_calls] == ["qwen35", "qwen3vl_binary", "qwen3vl_rescue"]
    for _name, args in sharded_calls:
        assert "--skip-existing" in args
    assert "--strict-manifest-contract" not in sharded_calls[0][1]
    assert any(
        any("run_local_qwen3vl_video_evidence" in part for part in cmd)
        and "--merge-only" in cmd
        and "guarded_teacher_gate_v3" in cmd
        for cmd in cmd_calls
    )


def test_vision_boundary_tokens_custom_tokenizer_and_version_mismatch_warning() -> None:
    class FakeTokenizer:
        unk_token_id = 0

        def convert_tokens_to_ids(self, token: str) -> int:
            return {"<|vision_start|>": 101, "<|vision_end|>": 102, "<|video_pad|>": 103}.get(token, 0)

    class FakeProcessor:
        tokenizer = FakeTokenizer()

    # Custom token IDs with plain Python lists
    seq = [[1, 2, 101, 50, 101, 103, 103, 102, 102, 9]]
    out, info = runtime_common.detect_or_align_vision_boundary_tokens(
        {"input_ids": seq, "attention_mask": [[1] * 10], "mm_token_type_ids": [[0] * 10]},
        processor=FakeProcessor(),
        mode="strip_outer_markers",
        transformers_version="5.22.1",
    )
    assert info["vision_boundary_aligned"] is True
    assert info["extra_boundary_indices"] == [2, 8]
    assert out["input_ids"] == [[1, 2, 50, 101, 103, 103, 102, 9]]
    assert len(out["attention_mask"][0]) == 8
    assert len(out["mm_token_type_ids"][0]) == 8

    # Version mismatch warning even when sequence has no extra boundary tokens
    _, warn_info = runtime_common.detect_or_align_vision_boundary_tokens(
        {"input_ids": [[1, 2, 50, 101, 103, 103, 102, 9]]},
        processor=FakeProcessor(),
        mode="warn",
        transformers_version="5.22.1",
    )
    assert warn_info["extra_boundary_tokens_detected"] is False
    assert warn_info["transformers_version_differs_from_5_9"] is True
    assert "5.22.1" in str(warn_info["warning"])
    assert "docs/eval/hunyuan-d3-replay.md" in str(warn_info["warning"])

