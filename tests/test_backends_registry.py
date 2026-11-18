"""Extended backend and runtime tests."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from subprocess import CompletedProcess

import numpy as np
import pytest

import wrbench
from wrbench.backends.base import GenerationRequest
from wrbench.backends.launchers.easyanimate import build_easyanimate_command
from wrbench.backends.launchers.minwm_hy import build_minwm_hy_command
from wrbench.backends.launchers.minwm_wan import build_minwm_wan_command
from wrbench.backends.launchers.recammaster import build_recammaster_command, validate_recammaster_runtime
from wrbench.backends.launchers import launcher_spec, supported_models
from wrbench.backends.registry import list_backends, resolve_backend
from wrbench.backends.local_subprocess import LocalSubprocessBackend
from wrbench.runtime import RuntimeConfig, ModelRuntime, load_runtime_config


def test_local_launcher_registry_contains_only_supported_models():
    expected = frozenset(
        {
            "easyanimate-v51-camera",
            "minwm-hy-action2v",
            "minwm-wan-action2v",
            "recammaster",
            "spatia",
        }
    )
    assert supported_models() == expected
    assert all(launcher_spec(model).key == model for model in expected)
    assert launcher_spec("worldcam") is None


@pytest.fixture
def recammaster_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModelRuntime:
    monkeypatch.chdir(tmp_path)
    repo = Path("ReCamMaster")
    (repo / "diffsynth" / "pipelines").mkdir(parents=True)
    (repo / "diffsynth" / "pipelines" / "wan_video_recammaster.py").write_text("# stub\n")
    py = Path("venv/bin/python")
    py.parent.mkdir(parents=True)
    py.symlink_to(sys.executable)
    checkpoint = Path("step20000.ckpt")
    checkpoint.write_bytes(b"checkpoint")
    wan = Path("Wan2.1-T2V-1.3B")
    (wan / "google" / "umt5-xxl").mkdir(parents=True)
    for name in ("diffusion_pytorch_model.safetensors", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"):
        (wan / name).write_bytes(b"weights")
    Path("source.mp4").write_bytes(b"video")
    return ModelRuntime(
        key="recammaster", python_bin=str(py), repo_root=str(repo),
        model_path=str(checkpoint), gpu_id=3,
        extra_paths={
            "wan_model_dir": str(wan), "negative_prompt": "",
            "cfg_scale": "5.0", "num_inference_steps": "50", "seed": "42", "tiled": "true",
        },
    )


@pytest.mark.parametrize("negative_prompt", ["", "--bad quality"])
def test_build_recammaster_command_uses_compiled_embedding(recammaster_runtime, negative_prompt):
    runtime = replace(
        recammaster_runtime,
        extra_paths={**recammaster_runtime.extra_paths, "negative_prompt": negative_prompt},
    )
    embedding = np.zeros((1, 21, 12), dtype=np.float32)
    assert validate_recammaster_runtime(runtime) == []
    cmd, cwd, env, camera_path = build_recammaster_command(
        model="recammaster", payload={"camera_embedding": embedding},
        runtime=runtime, source_video_path=Path("source.mp4"), prompt="A scene.", output_path=Path("out.mp4"),
        width=832, height=480, num_frames=81, fps=16,
    )
    assert cwd == Path.cwd() / runtime.repo_root
    assert cmd[0] == str(Path.cwd() / runtime.python_bin)
    assert cmd[cmd.index("--wan-model-dir") + 1] == str(Path.cwd() / runtime.extra_paths["wan_model_dir"])
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    np.testing.assert_array_equal(np.load(camera_path), embedding)
    assert f"--negative-prompt={negative_prompt}" in cmd


def test_recammaster_generation_preserves_fps_and_distinguishes_output_from_control_frames(
    recammaster_runtime, monkeypatch,
):
    runtime_config = Path("runtime.json")
    runtime_config.write_text(json.dumps({"schema_version": 1, "models": {"recammaster": asdict(recammaster_runtime)}}))

    def fake_run(cmd, *, cwd, env, capture_output, text):
        assert Path(cwd) == Path.cwd() / recammaster_runtime.repo_root
        assert cmd[cmd.index("--fps") + 1] == "24"
        assert cmd[cmd.index("--num-frames") + 1] == "81"
        assert "--prompt=A scene." in cmd
        Path(cmd[cmd.index("--output") + 1]).write_bytes(b"generated video")
        return CompletedProcess(cmd, 0)

    monkeypatch.setattr("wrbench.backends.local_subprocess.subprocess.run", fake_run)
    result = wrbench.compile_camera(
        model="recammaster", camera="yaw:left:30@81", source_video="source.mp4", prompt="A scene.",
        out="output.mp4", fps=24, runtime_config=runtime_config, dry_run=False,
    )
    sidecar = json.loads(Path("output.mp4.camera.json").read_text())
    assert sidecar["fps"] == 24
    assert sidecar["model_frame_count"] == sidecar["generated_frame_count"] == 81
    assert sidecar["model_control_timeline"]["control_sample_count"] == 21
    assert np.load(result["generation"]["artifacts"]["camera_embedding"]).shape == (1, 21, 12)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [({"width": 960}, "requires 832x480 and 81 frames"),
     ({"num_frames": 97}, "requires 832x480 and 81 frames"),
     ({"source_video": "missing.mp4"}, "source video not found")],
)
def test_recammaster_rejects_invalid_generation_inputs(recammaster_runtime, overrides, message):
    runtime_config = Path("runtime.json")
    runtime_config.write_text(json.dumps({"schema_version": 1, "models": {"recammaster": asdict(recammaster_runtime)}}))
    inputs = {"source_video": "source.mp4", **overrides}
    with pytest.raises(ValueError, match=message):
        wrbench.compile_camera(
            model="recammaster", camera="yaw:left:30@81", prompt="A scene.", out="output.mp4",
            runtime_config=runtime_config, dry_run=False, **inputs,
        )


def test_recammaster_runtime_requires_wan_tokenizer(recammaster_runtime):
    (Path(recammaster_runtime.extra_paths["wan_model_dir"]) / "google" / "umt5-xxl").rmdir()
    assert validate_recammaster_runtime(recammaster_runtime) == ["extra_paths.wan_model_dir (google/umt5-xxl tokenizer)"]


def test_recammaster_preserves_existing_video_and_sidecar_on_retry(recammaster_runtime):
    output = Path("already-generated.mp4")
    sidecar = Path(f"{output}.camera.json")
    output.write_bytes(b"original video")
    sidecar.write_text('{"camera_script": "yaw:left:30@81"}')
    with pytest.raises(FileExistsError, match="Choose a new --out path"):
        wrbench.compile_camera(
            model="recammaster", camera="yaw:right:30@81", source_video="source.mp4",
            prompt="A scene.", out=output, dry_run=False,
        )
    assert output.read_bytes() == b"original video"
    assert json.loads(sidecar.read_text())["camera_script"] == "yaw:left:30@81"


def test_recammaster_rejects_nonfinite_camera_embedding(recammaster_runtime):
    with pytest.raises(ValueError, match="21 finite relative poses"):
        build_recammaster_command(
            model="recammaster", payload={"camera_embedding": np.full((1, 21, 12), np.nan)},
            runtime=recammaster_runtime, source_video_path=Path("source.mp4"), prompt="A scene.",
            output_path=Path("out.mp4"), width=832, height=480, num_frames=81, fps=16,
        )


def test_resolve_backend_defaults_to_dry_run_without_runtime():
    backend = resolve_backend("wan22-fun-5b-cam")
    assert backend.name == "dry_run"


def test_real_generation_fails_closed_when_only_dry_run_backend_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out.mp4"
    image = tmp_path / "first.png"
    image.write_bytes(b"png")

    with pytest.raises(RuntimeError, match="only the dry-run backend is available"):
        wrbench.compile_camera(
            model="easyanimate-v51-camera",
            camera="static@49",
            out=out,
            image=str(image),
            prompt="A WRBench prompt.",
            dry_run=False,
        )


def test_resolve_backend_local_when_runtime_configured(tmp_path: Path):
    runtime_path = tmp_path / "wrbench.runtime.json"
    repo = tmp_path / "EasyAnimate"
    repo.mkdir()
    (repo / "predict_v2v_control.py").write_text("# stub\n", encoding="utf-8")
    py = tmp_path / "python"
    py.write_text("#!/bin/sh\n", encoding="utf-8")
    py.chmod(0o755)
    weights = tmp_path / "weights"
    weights.mkdir()
    runtime_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": {
                    "easyanimate-v51-camera": {
                        "python_bin": str(py),
                        "repo_root": str(repo),
                        "model_path": str(weights),
                        "gpu_id": 0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    runtime = load_runtime_config(runtime_path)
    assert runtime is not None
    backend = resolve_backend("easyanimate-v51-camera", runtime=runtime)
    assert backend.name == "local_subprocess"
    ok, _ = backend.available_for("easyanimate-v51-camera")
    assert ok


def test_local_subprocess_spatia_requires_configured_ffmpeg(tmp_path: Path):
    repo = tmp_path / "Spatia"
    repo.mkdir()
    (repo / "inference.py").write_text("# stub\n", encoding="utf-8")
    py = tmp_path / "python"
    py.write_text("#!/bin/sh\n", encoding="utf-8")
    py.chmod(0o755)
    vace = tmp_path / "vace.safetensors"
    vace.write_bytes(b"vace")
    lora = tmp_path / "lora.safetensors"
    lora.write_bytes(b"lora")
    runtime = RuntimeConfig(
        schema_version=1,
        models={
            "spatia": ModelRuntime(
                key="spatia",
                python_bin=str(py),
                repo_root=str(repo),
                gpu_id=0,
                extra_paths={
                    "vace_path": str(vace),
                    "lora_path": str(lora),
                    "num_inference_steps": "40",
                    "cfg_scale": "3.5",
                    "sigma_shift": "5.0",
                    "seed": "20917",
                },
            )
        },
    )
    backend = LocalSubprocessBackend(runtime)

    ok, reason = backend.available_for("spatia")

    assert not ok
    assert "extra_paths.ffmpeg_bin" in reason


def test_build_easyanimate_command(tmp_path: Path):
    repo = tmp_path / "EasyAnimate"
    repo.mkdir()
    (repo / "predict_v2v_control.py").write_text(
        "\n".join(
            [
                "GPU_memory_mode         = \"model_cpu_offload\"",
                "enable_teacache         = True",
                "teacache_threshold      = 0.08",
                "config_path             = \"config/easyanimate.yaml\"",
                "model_name              = \"models/default\"",
                "sampler_name            = \"Flow\"",
                "sample_size             = [384, 672]",
                "video_length            = 49",
                "fps                     = 8",
                "weight_dtype            = \"torch.float16\"",
                "control_video           = None",
                "control_camera_txt      = \"control.txt\"",
                "ref_image               = \"ref.png\"",
                "prompt                  = \"prompt\"",
                "guidance_scale          = 6.0",
                "seed                    = 43",
                "num_inference_steps     = 50",
                "save_path               = \"samples/out\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    py = tmp_path / "python"
    py.write_text("#!/bin/sh\n", encoding="utf-8")
    py.chmod(0o755)
    weights = tmp_path / "weights"
    weights.mkdir()
    cam_txt = tmp_path / "control.txt"
    cam_txt.write_text("0\n", encoding="utf-8")
    image = tmp_path / "first.png"
    image.write_bytes(b"png")
    out = tmp_path / "out.mp4"

    runtime = ModelRuntime(
        key="easyanimate-v51-camera",
        python_bin=str(py),
        repo_root=str(repo),
        model_path=str(weights),
        gpu_id=0,
    )
    cmd, cwd, env = build_easyanimate_command(
        model="easyanimate-v51-camera",
        payload={
            "control_camera_txt": str(cam_txt),
            "sample_size": [384, 672],
            "video_length": 49,
            "fps": 8,
        },
        runtime=runtime,
        image_path=image,
        prompt="A scene.",
        output_path=out,
    )[:3]
    assert cmd[0] == str(py)
    assert cmd[1].endswith("_easyanimate_run.py")
    assert cwd == repo
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    materialized = Path(cmd[1])
    assert materialized.is_file()
    assert str(weights) in materialized.read_text(encoding="utf-8")


def _fake_minwm_runtime(tmp_path: Path) -> ModelRuntime:
    repo = tmp_path / "minWM"
    (repo / "Wan21").mkdir(parents=True)
    (repo / "Wan21" / "wan_inference.py").write_text("# fake minWM entrypoint\n", encoding="utf-8")
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    py = bin_dir / "python"
    py.write_text("#!/usr/bin/env sh\n", encoding="utf-8")
    py.chmod(0o755)
    torchrun = bin_dir / "torchrun"
    torchrun.write_text(
        "#!/usr/bin/env sh\n"
        "out=''\n"
        "prev=''\n"
        "for arg in \"$@\"; do\n"
        "  if [ \"$prev\" = '--output_folder' ]; then out=\"$arg\"; fi\n"
        "  prev=\"$arg\"\n"
        "done\n"
        "mkdir -p \"$out\"\n"
        "printf fake > \"$out/fake.mp4\"\n",
        encoding="utf-8",
    )
    torchrun.chmod(0o755)
    ckpt = tmp_path / "dmd.pt"
    ckpt.write_bytes(b"ckpt")
    config = tmp_path / "wan_config.yaml"
    config.write_text("fake: true\n", encoding="utf-8")
    return ModelRuntime(
        key="minwm-wan-action2v",
        python_bin=str(py),
        repo_root=str(repo),
        model_path=str(ckpt),
        gpu_id=2,
        extra_paths={
            "torchrun_bin": str(torchrun),
            "config_path": str(config),
            "num_output_frames": "20",
            "sp_size": "1",
        },
    )


def _fake_minwm_hy_runtime(tmp_path: Path) -> ModelRuntime:
    repo = tmp_path / "minWM-HY"
    (repo / "HY15").mkdir(parents=True, exist_ok=True)
    (repo / "HY15" / "hy15_inference.py").write_text(
        "import numpy as np\n_ROT_STEP = np.radians(3.0)\nprint('fake hy')\n",
        encoding="utf-8",
    )
    bin_dir = tmp_path / "hy-venv" / "bin"
    bin_dir.mkdir(parents=True)
    py = bin_dir / "python"
    py.write_text("#!/usr/bin/env sh\n", encoding="utf-8")
    py.chmod(0o755)
    torchrun = bin_dir / "torchrun"
    torchrun.write_text(
        "#!/usr/bin/env sh\n"
        "out=''\n"
        "prev=''\n"
        "for arg in \"$@\"; do\n"
        "  if [ \"$prev\" = '--output_dir' ]; then out=\"$arg\"; fi\n"
        "  prev=\"$arg\"\n"
        "done\n"
        "mkdir -p \"$out\"\n"
        "printf fake-hy > \"$out/fake_hy.mp4\"\n",
        encoding="utf-8",
    )
    torchrun.chmod(0o755)
    transformer = tmp_path / "HY15" / "Action2V" / "dmd"
    transformer.mkdir(parents=True)
    base = tmp_path / "HunyuanVideo-1.5"
    base.mkdir()
    return ModelRuntime(
        key="minwm-hy-action2v",
        python_bin=str(py),
        repo_root=str(repo),
        model_path=str(transformer),
        gpu_id=3,
        extra_paths={
            "torchrun_bin": str(torchrun),
            "base_model_path": str(base),
            "mode": "i2v",
            "chunk_latent_frames": "4",
            "num_inference_steps": "8",
            "shift": "5.0",
            "guidance_scale": "1.0",
            "stabilization_level": "0",
        },
    )


def test_build_minwm_hy_command_materializes_example_and_rotstep_patch(tmp_path: Path):
    runtime = _fake_minwm_hy_runtime(tmp_path)
    image = tmp_path / "first.png"
    image.write_bytes(b"png")
    out = tmp_path / "hy.mp4"
    result = wrbench.compile_camera(
        model="minwm-hy-action2v",
        camera="yaw:left:60@40,yaw:right:60@41",
        out=out,
        image=str(image),
        prompt="A HY WRBench prompt.",
        dry_run=True,
    )

    cmd, cwd, env, output_dir = build_minwm_hy_command(
        model="minwm-hy-action2v",
        payload=result["payload"].payload,
        runtime=runtime,
        image_path=image,
        prompt="A HY WRBench prompt.",
        output_path=out,
    )

    assert cwd == Path(runtime.repo_root)
    assert cmd[0] == runtime.extra_paths["torchrun_bin"]
    assert "--use_camera" in cmd
    entrypoint = Path(cmd[cmd.index("--nproc_per_node=1") + 1])
    assert entrypoint.name == "hy15_inference_wrbench_rotstep.py"
    assert "np.radians(6.0)" in entrypoint.read_text(encoding="utf-8")
    example_json = Path(cmd[cmd.index("--example_json") + 1])
    rows = json.loads(example_json.read_text(encoding="utf-8"))
    assert rows[0]["image"] == str(image.resolve())
    assert rows[0]["caption"] == "A HY WRBench prompt."
    assert rows[0]["trajectory"] == "j*10,l*9"
    assert cmd[cmd.index("--transformer_dir") + 1] == str(runtime.model_path)
    assert cmd[cmd.index("--model_path") + 1] == runtime.extra_paths["base_model_path"]
    assert output_dir == Path(str(out.resolve().with_suffix("")) + "_minwm_hy_output")
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert str(Path(runtime.repo_root) / "HY15") in env["PYTHONPATH"]
    assert str(Path(runtime.repo_root) / "shared") in env["PYTHONPATH"]
    assert str(runtime.repo_root) in env["PYTHONPATH"]


def test_build_minwm_hy_command_requires_configured_torchrun(tmp_path: Path):
    fake = _fake_minwm_hy_runtime(tmp_path)
    runtime = replace(fake, extra_paths={k: v for k, v in fake.extra_paths.items() if k != "torchrun_bin"})
    image = tmp_path / "first.png"
    image.write_bytes(b"png")
    out = tmp_path / "hy.mp4"
    result = wrbench.compile_camera(
        model="minwm-hy-action2v",
        camera="static@77",
        out=out,
        image=str(image),
        prompt="A HY WRBench prompt.",
        dry_run=True,
    )

    with pytest.raises(ValueError, match="runtime.extra_paths.torchrun_bin is required"):
        build_minwm_hy_command(
            model="minwm-hy-action2v",
            payload=result["payload"].payload,
            runtime=runtime,
            image_path=image,
            prompt="A HY WRBench prompt.",
            output_path=out,
        )


def test_local_subprocess_minwm_hy_requires_all_runtime_fields(tmp_path: Path):
    fake = _fake_minwm_hy_runtime(tmp_path)
    runtime = replace(fake, extra_paths={k: v for k, v in fake.extra_paths.items() if k != "mode"})
    backend = LocalSubprocessBackend(RuntimeConfig(schema_version=1, models={"minwm-hy-action2v": runtime}))

    ok, reason = backend.available_for("minwm-hy-action2v")

    assert not ok
    assert "extra_paths.mode" in reason


def test_local_subprocess_minwm_wan_requires_all_runtime_fields(tmp_path: Path):
    fake = _fake_minwm_runtime(tmp_path)
    runtime = replace(fake, extra_paths={k: v for k, v in fake.extra_paths.items() if k != "config_path"})
    backend = LocalSubprocessBackend(RuntimeConfig(schema_version=1, models={"minwm-wan-action2v": runtime}))

    ok, reason = backend.available_for("minwm-wan-action2v")

    assert not ok
    assert "extra_paths.config_path" in reason


def test_build_minwm_hy_command_requires_model_path_for_transformer_dir(tmp_path: Path):
    runtime = replace(_fake_minwm_hy_runtime(tmp_path), model_path=None)
    image = tmp_path / "first.png"
    image.write_bytes(b"png")
    out = tmp_path / "hy.mp4"
    result = wrbench.compile_camera(
        model="minwm-hy-action2v",
        camera="static@77",
        out=out,
        image=str(image),
        prompt="A HY WRBench prompt.",
        dry_run=True,
    )

    with pytest.raises(ValueError, match="runtime.model_path is required"):
        build_minwm_hy_command(
            model="minwm-hy-action2v",
            payload=result["payload"].payload,
            runtime=runtime,
            image_path=image,
            prompt="A HY WRBench prompt.",
            output_path=out,
        )


def test_build_minwm_wan_command_uses_prompt_trajectory_and_patch(tmp_path: Path):
    runtime = _fake_minwm_runtime(tmp_path)
    out = tmp_path / "wan.mp4"
    result = wrbench.compile_camera(
        model="minwm-wan-action2v",
        camera="yaw:left:60@40,yaw:right:60@41",
        out=out,
        prompt="A WRBench prompt.",
        dry_run=True,
    )
    payload = result["payload"].payload
    cmd, cwd, env, output_dir = build_minwm_wan_command(
        model="minwm-wan-action2v",
        payload=payload,
        runtime=runtime,
        prompt="A WRBench prompt.",
        output_path=out,
    )

    assert cwd == Path(runtime.repo_root)
    assert cmd[0] == payload["rotation_step_patch"]["launcher_path"]
    assert str(Path(runtime.python_bin).with_name("torchrun")) in cmd
    assert "--data_path" in cmd
    assert cmd[cmd.index("--data_path") + 1] == payload["prompt_txt"]
    assert "--trajectory_path" in cmd
    assert cmd[cmd.index("--trajectory_path") + 1] == payload["trajectory_txt"]
    assert "--checkpoint_path" in cmd
    assert str(runtime.model_path) in cmd
    assert output_dir == Path(str(out.resolve().with_suffix("")) + "_minwm_wan_output")
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert str(runtime.repo_root) in env["PYTHONPATH"]


def test_build_minwm_wan_command_requires_configured_torchrun(tmp_path: Path):
    runtime = replace(_fake_minwm_runtime(tmp_path), extra_paths={})
    out = tmp_path / "wan.mp4"
    result = wrbench.compile_camera(
        model="minwm-wan-action2v",
        camera="yaw:left:60@40,yaw:right:60@41",
        out=out,
        prompt="A WRBench prompt.",
        dry_run=True,
    )

    with pytest.raises(ValueError, match="runtime.extra_paths.torchrun_bin is required"):
        build_minwm_wan_command(
            model="minwm-wan-action2v",
            payload=result["payload"].payload,
            runtime=runtime,
            prompt="A WRBench prompt.",
            output_path=out,
        )


def test_local_subprocess_generates_minwm_wan_output_with_fake_torchrun(tmp_path: Path):
    runtime = RuntimeConfig(
        schema_version=1,
        models={"minwm-wan-action2v": _fake_minwm_runtime(tmp_path)},
    )
    backend = LocalSubprocessBackend(runtime)
    ok, reason = backend.available_for("minwm-wan-action2v")
    assert ok, reason

    out = tmp_path / "generated.mp4"
    compiled = wrbench.compile_camera(
        model="minwm-wan-action2v",
        camera="yaw:left:60@40,yaw:right:60@41",
        out=out,
        prompt="A WRBench prompt.",
        dry_run=True,
    )
    result = backend.generate(
        GenerationRequest(
            model="minwm-wan-action2v",
            prompt="A WRBench prompt.",
            payload=compiled["payload"],
            output_path=out,
            work_dir=tmp_path,
        )
    )

    assert result.success, result.message
    assert out.read_bytes() == b"fake"
    assert result.artifacts["output_mp4"] == str(out.resolve())


@pytest.mark.parametrize(
    ("camera", "prompt"),
    [
        ("yaw:left:60@40,yaw:right:60@41", "A HY WRBench prompt."),
        ("static@77", "A static HY WRBench prompt."),
    ],
)
def test_local_subprocess_generates_minwm_hy_output_with_fake_torchrun(
    tmp_path: Path,
    camera: str,
    prompt: str,
):
    runtime = RuntimeConfig(
        schema_version=1,
        models={"minwm-hy-action2v": _fake_minwm_hy_runtime(tmp_path)},
    )
    backend = LocalSubprocessBackend(runtime)
    ok, reason = backend.available_for("minwm-hy-action2v")
    assert ok, reason

    image = tmp_path / "first.png"
    image.write_bytes(b"png")
    out = tmp_path / "generated_hy.mp4"
    compiled = wrbench.compile_camera(
        model="minwm-hy-action2v",
        camera=camera,
        out=out,
        image=str(image),
        prompt=prompt,
        dry_run=True,
    )
    result = backend.generate(
        GenerationRequest(
            model="minwm-hy-action2v",
            prompt=prompt,
            payload=compiled["payload"],
            output_path=out,
            image_path=image,
            work_dir=tmp_path,
        )
    )

    assert result.success, result.message
    assert out.read_bytes() == b"fake-hy"
    assert result.artifacts["output_mp4"] == str(out.resolve())


def test_list_backends_without_runtime():
    rows = list_backends("wan22-fun-5b-cam", runtime=None)
    names = [name for name, _, _ in rows]
    assert "dry_run" in names
