"""Tests for modular CLI, --model-spec, plugins, doctor, tasks export, and eval validate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import wrbench
from wrbench.cli import _build_parser, _doctor_check_model, build_parser, main


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _pose_model_spec(key: str = "ext-pose-model") -> dict:
    return {
        "key": key,
        "aliases": [f"{key}-alias"],
        "status": "active",
        "input_kind": "image",
        "model_input": "TI2V",
        "viewpoint_condition_type": "model-inferred",
        "adapter": "declarative_pose",
        "payload_type": "declarative_pose_npy",
        "default_width": 832,
        "default_height": 480,
        "default_fps": 16,
        "default_frames": 17,
        "amplitude": {
            "rotation_gain": 1.0,
            "translation_gain": 1.0,
            "max_amount": 1.0,
            "translation_unit": "canonical_scene",
            "calibration_status": "initial_manual",
            "metadata": {},
        },
        "capabilities": {
            "rotation": True,
            "translation": True,
            "supports_static": True,
        },
        "execution_contract": {
            "pose_type": "c2w",
            "convention": "opencv",
            "format": "npy",
        },
        "notes": "External declarative pose test model.",
    }


def _prompt_model_spec(key: str = "ext-prompt-model") -> dict:
    return {
        "key": key,
        "status": "active",
        "input_kind": "none",
        "model_input": "T2V",
        "viewpoint_condition_type": "prompt-only",
        "adapter": "prompt_camera",
        "payload_type": "prompt_camera",
        "default_width": 1280,
        "default_height": 720,
        "default_fps": 24,
        "default_frames": 17,
        "amplitude": {
            "rotation_gain": 1.0,
            "translation_gain": 1.0,
            "max_amount": 1.0,
            "translation_unit": "none",
            "calibration_status": "not_applicable",
            "metadata": {},
        },
        "capabilities": {
            "rotation": True,
            "translation": True,
            "supports_static": True,
        },
        "execution_contract": {
            "prompt_templates": {
                "static": "{prompt} Hold the camera still.",
                "yaw_LR": "{prompt} Yaw left then return right.",
            }
        },
        "notes": "External prompt-camera test model.",
    }


@pytest.fixture(autouse=True)
def _reset_registry_after_test():
    yield
    from wrbench import registry

    registry.reload()


def test_cli_reexports_and_package_exports() -> None:
    assert build_parser is _build_parser
    assert callable(_doctor_check_model)
    assert callable(wrbench.load_model_spec_file)
    assert callable(wrbench.register_model_record)


def test_plugins_command_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["plugins", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert "wan22-fun-5b-cam" in out["models"]["active"]
    assert "declarative_pose" in out["generation"]["adapters"]
    assert "declarative_action" in out["generation"]["adapters"]
    assert "prompt_camera" in out["generation"]["adapters"]
    assert "dry_run" in out["generation"]["backends"]
    assert "local_subprocess" in out["generation"]["backends"]
    assert "command_template" in out["generation"]["backends"]
    assert "vggt_omega" in out["eval"]["pose_backends"]
    assert "dinov2_fullfov_v2" in out["eval"]["d2_scorers"]
    assert "qwen_runtime_v2" in out["eval"]["vlm_scorers"]


def test_model_spec_root_and_subcommand_doctor_and_generate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pose_spec = _write_json(tmp_path / "pose_model.json", _pose_model_spec())
    prompt_spec = _write_json(tmp_path / "prompt_model.json", _prompt_model_spec())
    dummy_img = tmp_path / "first_frame.png"
    dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n")

    # Root-level repeatable --model-spec works with models --json
    rc = main(
        [
            "--model-spec",
            str(pose_spec),
            "--model-spec",
            str(prompt_spec),
            "models",
            "--json",
        ]
    )
    assert rc == 0
    models_json = json.loads(capsys.readouterr().out)
    keys = {m["key"] for m in models_json}
    assert "ext-pose-model" in keys
    assert "ext-prompt-model" in keys

    # Doctor checks both external declarative specs with --json
    rc = main(
        [
            "--model-spec",
            str(pose_spec),
            "doctor",
            "--model",
            "ext-pose-model",
            "--json",
        ]
    )
    assert rc == 0
    doc_out = json.loads(capsys.readouterr().out)
    assert doc_out["ok"] is True
    assert doc_out["models"][0]["key"] == "ext-pose-model"

    # Subcommand-level --model-spec also works for generate
    out_mp4 = tmp_path / "out_pose.mp4"
    rc = main(
        [
            "generate",
            "--model-spec",
            str(pose_spec),
            "--model",
            "ext-pose-model",
            "--camera",
            "preset:yaw_LR",
            "--image",
            str(dummy_img),
            "--prompt",
            "A cat sits on a rug.",
            "--out",
            str(out_mp4),
        ]
    )
    assert rc == 0
    gen_stdout = capsys.readouterr().out
    assert "model    : ext-pose-model" in gen_stdout
    assert "dry_run  : True" in gen_stdout


def test_generate_no_dry_run_precheck_applies_to_image_models(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression test: --no-dry-run pre-check must run for image models (not only input_kind='none')."""
    dummy_img = tmp_path / "first_frame.png"
    dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n")
    empty_rt = _write_json(tmp_path / "empty_runtime.json", {"schema_version": 1, "models": {}})

    rc = main(
        [
            "generate",
            "--model",
            "wan22-fun-5b-cam",
            "--camera",
            "static@17",
            "--image",
            str(dummy_img),
            "--prompt",
            "A cat sits on a rug.",
            "--out",
            str(tmp_path / "out.mp4"),
            "--runtime-config",
            str(empty_rt),
            "--no-dry-run",
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "dry-run backend is available" in err


def test_generate_no_dry_run_with_command_template_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pose_spec = _write_json(tmp_path / "pose_cmd.json", _pose_model_spec("ext-cmd-model"))
    dummy_img = tmp_path / "first_frame.png"
    dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n")
    writer_script = tmp_path / "fake_gen.py"
    writer_script.write_text(
        "import pathlib, sys\n"
        "out = pathlib.Path(sys.argv[1])\n"
        "out.parent.mkdir(parents=True, exist_ok=True)\n"
        "out.write_bytes(b'fake-mp4-bytes')\n",
        encoding="utf-8",
    )
    rt_path = _write_json(
        tmp_path / "runtime.json",
        {
            "schema_version": 1,
            "model_spec_files": [str(pose_spec)],
            "models": {
                "ext-cmd-model": {
                    "backend": "command_template",
                    "python_bin": sys.executable,
                    "repo_root": str(tmp_path),
                    "gpu_id": 0,
                    "command_template": [
                        "{python_bin}",
                        str(writer_script),
                        "{output_path}",
                    ],
                }
            },
        },
    )

    # Doctor checks backend readiness when --runtime-config is provided
    rc_doc = main(["doctor", "--model", "ext-cmd-model", "--runtime-config", str(rt_path), "--json"])
    assert rc_doc == 0
    doc_payload = json.loads(capsys.readouterr().out)
    assert doc_payload["ok"] is True

    # Reset registry so ext-cmd-model must be discovered purely from runtime.json's model_spec_files
    from wrbench import registry

    registry.reload()

    out_mp4 = tmp_path / "generated.mp4"
    rc = main(
        [
            "generate",
            "--model",
            "ext-cmd-model",
            "--camera",
            "preset:yaw_LR",
            "--image",
            str(dummy_img),
            "--prompt",
            "A cat sits on a rug.",
            "--out",
            str(out_mp4),
            "--runtime-config",
            str(rt_path),
            "--no-dry-run",
        ]
    )
    assert rc == 0
    assert out_mp4.read_bytes() == b"fake-mp4-bytes"


def test_tasks_export_and_eval_validate_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks_dir = tmp_path / "exported_tasks"
    rc = main(
        [
            "tasks",
            "export",
            "--out-dir",
            str(tasks_dir),
            "--model",
            "wan22-fun-5b-cam",
            "--cameras",
            "static,yaw_LR",
            "--limit",
            "2",
            "--json",
        ]
    )
    assert rc == 0
    export_res = json.loads(capsys.readouterr().out)
    assert export_res["ok"] is True
    assert export_res["tasks_count"] == 2
    tasks_jsonl = Path(export_res["tasks_jsonl"])
    assert tasks_jsonl.is_file()

    # Create a valid 2-row submission manifest
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    rows = []
    for idx, line in enumerate(tasks_jsonl.read_text(encoding="utf-8").splitlines()):
        task = json.loads(line)
        rel_mp4 = f"clip_{idx}.mp4"
        (videos_dir / rel_mp4).write_bytes(b"0000ftypmp42")
        rows.append(
            {
                "video_id": task["video_id"],
                "model": "my-byo-model",
                "video_path": rel_mp4,
                "camera_type": task["camera_type"],
                "world_state_prompt": task["world_state_prompt"],
            }
        )
    manifest_path = tmp_path / "submission.jsonl"
    manifest_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    rc_val = main(
        [
            "eval",
            "validate",
            "--manifest",
            str(manifest_path),
            "--video-root",
            str(videos_dir),
            "--json",
        ]
    )
    assert rc_val == 0
    val_out = json.loads(capsys.readouterr().out)
    assert val_out["ok"] is True
    assert val_out["row_count"] == 2
    assert val_out["models"] == ["my-byo-model"]

    # Missing video file must exit 1
    bad_manifest = _write_json(
        tmp_path / "bad_manifest.json",
        [
            {
                "video_id": "v1",
                "model": "my-byo-model",
                "video_path": "does_not_exist.mp4",
                "camera_type": "static",
                "world_state_prompt": "A cat sits.",
            }
        ],
    )
    rc_bad = main(
        [
            "eval",
            "validate",
            "--manifest",
            str(bad_manifest),
            "--video-root",
            str(videos_dir),
            "--json",
        ]
    )
    assert rc_bad == 1
    bad_out = json.loads(capsys.readouterr().out)
    assert bad_out["ok"] is False
    assert bad_out["errors"]


def test_doctor_eval_runtime_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_repo = tmp_path / "vggt_repo"
    fake_repo.mkdir()
    fake_ckpt = tmp_path / "vggt.pt"
    fake_ckpt.write_bytes(b"ckpt")
    fake_dino = tmp_path / "dino"
    fake_dino.mkdir()
    fake_q35 = tmp_path / "qwen35"
    fake_q35.mkdir()
    fake_q3vl = tmp_path / "qwen3vl"
    fake_q3vl.mkdir()

    rt_path = _write_json(
        tmp_path / "eval_runtime.json",
        {
            "schema_version": 1,
            "eval": {
                "scorers": {
                    "vggt_python_bin": sys.executable,
                    "vggt_repo": str(fake_repo),
                    "vggt_checkpoint": str(fake_ckpt),
                    "dinov2_python_bin": sys.executable,
                    "dinov2_model_path": str(fake_dino),
                    "qwen_scorer_python": sys.executable,
                    "qwen35_model": str(fake_q35),
                    "qwen3vl_model": str(fake_q3vl),
                }
            },
        },
    )

    rc = main(["doctor", "--eval", "--runtime-config", str(rt_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert "eval" in payload
    assert payload["eval"]["ok"] is True
    assert isinstance(payload["eval"]["checks"], list)
    assert "models" not in payload

    # Without --eval, --model + --runtime-config containing an eval block runs both model and eval checks
    rc_both = main(["doctor", "--model", "wan22-fun-5b-cam", "--runtime-config", str(rt_path), "--json"])
    assert rc_both == 0
    both_payload = json.loads(capsys.readouterr().out)
    assert both_payload["ok"] is True
    assert "models" in both_payload and len(both_payload["models"]) == 1
    assert "eval" in both_payload and both_payload["eval"]["ok"] is True


def test_declarative_action_and_prompt_camera_cli_flow(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    action_spec = _write_json(
        tmp_path / "action_model.json",
        {
            "key": "ext-action-model",
            "status": "active",
            "input_kind": "image",
            "adapter": "declarative_action",
            "payload_type": "declarative_action_timeline",
            "default_width": 832,
            "default_height": 480,
            "default_fps": 16,
            "default_frames": 17,
            "amplitude": {
                "rotation_gain": 1.0,
                "translation_gain": 1.0,
                "max_amount": 1.0,
                "translation_unit": "custom:action_steps",
                "calibration_status": "initial_manual",
                "metadata": {},
            },
            "capabilities": {"rotation": True, "translation": True, "supports_static": True},
            "execution_contract": {
                "action_map": {
                    "static": "STAY",
                    "yaw_left": "LEFT",
                    "yaw_right": "RIGHT",
                }
            },
        },
    )
    prompt_spec = _write_json(tmp_path / "prompt_model.json", _prompt_model_spec("ext-prompt-cli"))

    assert main(["--model-spec", str(action_spec), "doctor", "--model", "ext-action-model"]) == 0
    assert main(["--model-spec", str(prompt_spec), "doctor", "--model", "ext-prompt-cli"]) == 0
    capsys.readouterr()

    out_prompt = tmp_path / "prompt_out.mp4"
    rc = main(
        [
            "--model-spec",
            str(prompt_spec),
            "generate",
            "--model",
            "ext-prompt-cli",
            "--camera",
            "preset:yaw_LR",
            "--prompt",
            "A dog runs across the yard.",
            "--out",
            str(out_prompt),
        ]
    )
    assert rc == 0
    assert "model    : ext-prompt-cli" in capsys.readouterr().out

    # Invalid --model-spec returns exit code 1 cleanly
    rc_missing = main(["--model-spec", str(tmp_path / "missing.json"), "models"])
    assert rc_missing == 1
    assert "failed to load --model-spec" in capsys.readouterr().err


def test_eval_validate_sha256_and_no_check_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import hashlib

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video-content-bytes")
    digest = hashlib.sha256(b"video-content-bytes").hexdigest()

    good_manifest = _write_json(
        tmp_path / "good.json",
        [
            {
                "video_id": "v1",
                "model": "custom-model",
                "video_path": "clip.mp4",
                "camera_type": "static",
                "world_state_prompt": "A cat sits.",
                "sha256": digest,
            }
        ],
    )
    rc = main(
        [
            "eval",
            "validate",
            "--manifest",
            str(good_manifest),
            "--video-root",
            str(tmp_path),
            "--check-sha256",
            "--json",
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    # Missing file passes when --no-check-files is set
    schema_only = _write_json(
        tmp_path / "schema_only.json",
        [
            {
                "video_id": "v2",
                "model": "custom-model",
                "video_path": "nonexistent.mp4",
                "camera_type": "static",
                "world_state_prompt": "A cat sits.",
            }
        ],
    )
    rc_nocheck = main(
        ["eval", "validate", "--manifest", str(schema_only), "--no-check-files", "--json"]
    )
    assert rc_nocheck == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_docs_model_record_snippets_register_cleanly() -> None:
    """Verify copy-pasteable model record snippets in docs/plugins.md and docs/adding-a-model.md."""
    import ast
    import re

    root = Path(__file__).resolve().parents[1]

    # 1. Check register_model_record({...}) dict literal in docs/plugins.md
    plugins_md = (root / "docs" / "plugins.md").read_text(encoding="utf-8")
    py_blocks = re.findall(r"```python\n(.*?)```", plugins_md, flags=re.DOTALL)
    assert py_blocks, "expected Python snippet in docs/plugins.md"
    tree = ast.parse(py_blocks[-1])
    record_dict_node = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "register_model_record"
            and node.args
        ):
            record_dict_node = node.args[0]
            break
    assert record_dict_node is not None, "expected register_model_record({...}) call in docs/plugins.md"
    payload = ast.literal_eval(record_dict_node)
    rec = wrbench.register_model_record(payload, overwrite=True)
    assert rec.key == "my-plugin-model"
    assert rec.default_width == 832 and rec.default_frames == 49

    # 2. Check all 3 declarative JSON model specs in docs/plugins.md
    json_blocks = re.findall(r"```json\n(.*?)```", plugins_md, flags=re.DOTALL)
    model_json_blocks = [json.loads(b) for b in json_blocks if '"key"' in b and '"adapter"' in b]
    assert len(model_json_blocks) == 3
    for spec in model_json_blocks:
        registered = wrbench.register_model_record(spec, overwrite=True)
        assert registered.key == spec["key"]

    # 3. Check JSON model record snippet in docs/adding-a-model.md
    adding_md = (root / "docs" / "adding-a-model.md").read_text(encoding="utf-8")
    adding_json = re.findall(r"```json\n(.*?)```", adding_md, flags=re.DOTALL)
    assert adding_json, "expected JSON snippet in docs/adding-a-model.md"
    adding_spec = json.loads(adding_json[0])
    rec_adding = wrbench.register_model_record(adding_spec, overwrite=True)
    assert rec_adding.key == "my-model-7b-cam"


def test_eval_cli_zero_jargon_defaults_and_flag_aliases() -> None:
    parser = _build_parser()

    # eval run works without --scorer-profile or --sidecar-profile-gate
    args_run = parser.parse_args(["eval", "run", "--manifest", "m.jsonl", "--out-dir", "out"])
    assert args_run.scorer_profile is None
    assert args_run.sidecar_profile_gate == "main"

    # --scorer-profile default alias accepted on eval run and eval d3d6
    args_run_default = parser.parse_args(
        ["eval", "run", "--manifest", "m.jsonl", "--out-dir", "out", "--scorer-profile", "default"]
    )
    assert args_run_default.scorer_profile == "default"

    args_d3d6 = parser.parse_args(["eval", "d3d6", "--manifest", "m.json", "--out-dir", "out"])
    assert args_d3d6.stage == "all"
    assert args_d3d6.scorer_profile == "wrbench_default"

    args_d3d6_alias = parser.parse_args(
        ["eval", "d3d6", "--manifest", "m.json", "--out-dir", "out", "--scorer-profile", "default"]
    )
    assert args_d3d6_alias.scorer_profile == "default"

    # eval d1-camalign defaults --poses-file to poses.npy
    args_cam = parser.parse_args(
        [
            "eval", "d1-camalign",
            "--input-jsonl", "in.jsonl",
            "--output-jsonl", "out.jsonl",
            "--pose-cache-root", "cache",
        ]
    )
    assert args_cam.poses_file == "poses.npy"

    # eval pose accepts optional --max-frames
    args_pose = parser.parse_args(
        [
            "eval", "pose",
            "--input-jsonl", "in.jsonl",
            "--output-root", "out",
            "--cache-root", "cache",
            "--max-frames", "49",
        ]
    )
    assert args_pose.max_frames == 49

    # eval table accepts --d3d6-scores, --vlm-scores, and legacy --runtime-scores
    for flag in ("--d3d6-scores", "--vlm-scores", "--runtime-scores"):
        args_tbl = parser.parse_args(
            [
                "eval", "table",
                flag, "scores.json",
                "--out-csv", "t.csv",
                "--out-md", "t.md",
                "--out-summary", "s.json",
            ]
        )
        assert args_tbl.runtime_scores == "scores.json"


