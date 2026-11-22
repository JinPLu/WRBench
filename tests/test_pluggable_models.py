"""Tests for external model plugins, declarative adapters, and pluggable backends."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import wrbench
from wrbench.adapters.base import (
    _DYNAMIC_ADAPTER_SPECS,
    _REGISTRY,
    adapter_for_model,
    registered_model_keys,
    registered_named_adapters,
)
from wrbench.backends.base import GenerationRequest, GenerationResult
from wrbench.backends.command_template import CommandTemplateBackend
from wrbench.backends.launchers import LaunchSpec, PreparedLaunch, _SPECS, launcher_spec, register_launcher, supported_models
from wrbench.backends.registry import (
    _GENERATION_BACKENDS,
    list_backends,
    register_generation_backend,
    registered_generation_backends,
    resolve_backend,
)
from wrbench.eval.d1.pose import _normalize_predicted_poses
from wrbench.payload import CameraPayload
from wrbench.registry import (
    VALID_TRANSLATION_UNITS,
    RegistryError,
    canonical_model_key,
    load_model_spec_file,
    model_record,
    register_model_record,
    register_translation_unit,
    reload,
)
from wrbench.runtime import ModelRuntime, RuntimeConfig, RuntimeConfigError, load_runtime_config


@pytest.fixture(autouse=True)
def _clean_registries(monkeypatch: pytest.MonkeyPatch):
    """Ensure dynamic registrations do not leak across tests."""
    orig_units = set(VALID_TRANSLATION_UNITS)
    orig_backends = dict(_GENERATION_BACKENDS)
    orig_launchers = dict(_SPECS)
    orig_adapters = dict(_REGISTRY)
    orig_dyn_specs = dict(_DYNAMIC_ADAPTER_SPECS)
    monkeypatch.delenv("WRBENCH_MODEL_DIRS", raising=False)
    reload()
    yield
    VALID_TRANSLATION_UNITS.clear()
    VALID_TRANSLATION_UNITS.update(orig_units)
    _GENERATION_BACKENDS.clear()
    _GENERATION_BACKENDS.update(orig_backends)
    _SPECS.clear()
    _SPECS.update(orig_launchers)
    _REGISTRY.clear()
    _REGISTRY.update(orig_adapters)
    _DYNAMIC_ADAPTER_SPECS.clear()
    _DYNAMIC_ADAPTER_SPECS.update(orig_dyn_specs)
    reload()


def _base_model_payload(key: str, **overrides) -> dict:
    payload = {
        "key": key,
        "aliases": [key, key.replace("-", "_")],
        "status": "active",
        "input_kind": "image",
        "model_input": "TI2V",
        "source_video_usage": "none",
        "viewpoint_condition_type": "model-inferred",
        "adapter": "declarative_pose",
        "payload_type": "declarative_pose_npy",
        "default_width": 832,
        "default_height": 480,
        "default_frames": 25,
        "default_fps": 16,
        "amplitude": {
            "rotation_gain": 1.0,
            "translation_gain": 1.0,
            "max_amount": 0.5,
            "translation_unit": "canonical_scene",
            "calibration_status": "custom_test",
        },
        "capabilities": {"supports_static": True},
        "notes": "Test plugin model",
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Registry: dynamic registration, spec files, WRBENCH_MODEL_DIRS, entry points
# ---------------------------------------------------------------------------

def test_register_model_record_and_reload():
    rec = register_model_record(_base_model_payload("custom-pose-v1"))
    assert rec.key == "custom-pose-v1"
    assert canonical_model_key("custom_pose_v1") == "custom-pose-v1"

    with pytest.raises(RegistryError, match="Duplicate model key"):
        register_model_record(_base_model_payload("custom-pose-v1"), overwrite=False)

    updated = register_model_record(
        _base_model_payload("custom-pose-v1", aliases=["custom-pose-v1", "cpv1"], default_frames=33),
        overwrite=True,
    )
    assert updated.default_frames == 33
    assert canonical_model_key("cpv1") == "custom-pose-v1"

    reload()
    with pytest.raises(KeyError):
        canonical_model_key("custom-pose-v1")


def test_load_model_spec_file_and_wrbench_model_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    spec_file = tmp_path / "ext_model.json"
    spec_file.write_text(json.dumps(_base_model_payload("spec-file-model")), encoding="utf-8")
    rec = load_model_spec_file(spec_file)
    assert rec.key == "spec-file-model"

    dir_a = tmp_path / "models_a"
    dir_b = tmp_path / "models_b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "m1.json").write_text(json.dumps(_base_model_payload("env-dir-model-1")), encoding="utf-8")
    (dir_b / "m2.json").write_text(json.dumps(_base_model_payload("env-dir-model-2")), encoding="utf-8")
    monkeypatch.setenv("WRBENCH_MODEL_DIRS", os.pathsep.join([str(dir_a), str(dir_b)]))
    reload()

    assert canonical_model_key("env-dir-model-1") == "env-dir-model-1"
    assert canonical_model_key("env-dir-model-2") == "env-dir-model-2"
    with pytest.raises(KeyError):
        canonical_model_key("spec-file-model")


def test_entry_point_plugin_discovery_zero_and_one_arg(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    def plugin_zero_arg():
        calls.append("zero")
        register_model_record(_base_model_payload("ep-zero-model"))

    def plugin_one_arg(api):
        calls.append("one")
        api.register_model_record(_base_model_payload("ep-one-model"))

    fake_eps = SimpleNamespace(
        select=lambda group: (
            [
                SimpleNamespace(name="p0", load=lambda: plugin_zero_arg),
                SimpleNamespace(name="p1", load=lambda: plugin_one_arg),
            ]
            if group == "wrbench.plugins"
            else []
        )
    )
    monkeypatch.setattr("wrbench.registry.importlib.metadata.entry_points", lambda: fake_eps)
    reload()

    assert canonical_model_key("ep-zero-model") == "ep-zero-model"
    assert canonical_model_key("ep-one-model") == "ep-one-model"
    assert calls == ["zero", "one"]


def test_extensible_translation_units():
    assert "none" in VALID_TRANSLATION_UNITS
    rec_custom = register_model_record(
        _base_model_payload(
            "custom-unit-model",
            amplitude={
                "rotation_gain": 1.0,
                "translation_gain": 1.0,
                "max_amount": 0.5,
                "translation_unit": "custom:meters_v2",
                "calibration_status": "uncalibrated",
            },
        )
    )
    assert rec_custom.amplitude.translation_unit == "custom:meters_v2"

    with pytest.raises(RegistryError, match="unsupported translation_unit"):
        register_model_record(
            _base_model_payload(
                "bad-custom-unit-model",
                amplitude={
                    "rotation_gain": 1.0,
                    "translation_gain": 1.0,
                    "max_amount": 0.5,
                    "translation_unit": "custom:",
                    "calibration_status": "uncalibrated",
                },
            )
        )

    register_translation_unit("my_registered_unit")
    rec_reg = register_model_record(
        _base_model_payload(
            "registered-unit-model",
            amplitude={
                "rotation_gain": 1.0,
                "translation_gain": 1.0,
                "max_amount": 0.5,
                "translation_unit": "my_registered_unit",
                "calibration_status": "calibrated",
            },
        )
    )
    assert rec_reg.amplitude.translation_unit == "my_registered_unit"


# ---------------------------------------------------------------------------
# Declarative adapters & module:AttrName dynamic adapters
# ---------------------------------------------------------------------------

def test_named_adapters_registered():
    names = registered_named_adapters()
    assert "declarative_pose" in names
    assert "declarative_action" in names
    assert "prompt_camera" in names


@pytest.mark.parametrize(
    ("pose_type", "convention", "fmt"),
    [
        ("c2w", "opencv", "npy"),
        ("w2c", "opengl", "npz"),
        ("c2w", "blender", "json"),
        ("w2c", "opencv", "txt_3x4"),
        ("c2w", "opengl", "txt_4x4"),
    ],
)
def test_declarative_pose_adapter_formats_and_roundtrip(
    tmp_path: Path,
    pose_type: str,
    convention: str,
    fmt: str,
):
    key = f"pose-{pose_type}-{convention}-{fmt}".replace("_", "-")
    register_model_record(
        _base_model_payload(
            key,
            adapter="declarative_pose",
            payload_type=f"declarative_pose_{fmt}",
            execution_contract={
                "pose_type": pose_type,
                "convention": convention,
                "format": fmt,
                "npz_key": "extrinsics_custom",
            },
        )
    )
    assert key in registered_model_keys()
    img = tmp_path / "first.png"
    img.write_bytes(b"png")
    out = tmp_path / f"{key}.mp4"
    res = wrbench.compile_camera(
        model=key,
        camera="yaw:left:45@12,yaw:right:45@13",
        out=out,
        image=str(img),
        dry_run=True,
    )
    payload = res["payload"]
    pose_file = Path(payload.payload["pose_file"])
    assert pose_file.is_file()

    if fmt == "npy":
        loaded = np.load(pose_file)
    elif fmt == "npz":
        archive = np.load(pose_file)
        assert "extrinsics_custom" in archive
        assert "intrinsics" in archive
        loaded = archive["extrinsics_custom"]
    elif fmt == "json":
        doc = json.loads(pose_file.read_text(encoding="utf-8"))
        loaded = np.asarray(doc["poses"], dtype=np.float32)
    elif fmt == "txt_3x4":
        loaded = np.loadtxt(pose_file).reshape(-1, 3, 4)
    else:
        loaded = np.loadtxt(pose_file).reshape(-1, 4, 4)

    norm_conv = "opengl" if convention == "blender" else convention
    recovered_c2w = _normalize_predicted_poses(loaded, pose_type, norm_conv, "opencv")
    np.testing.assert_allclose(recovered_c2w, payload.target_trajectory.to_c2w(), atol=1e-4)


def test_declarative_action_adapter_tokens_and_vectors(tmp_path: Path):
    # Token timeline with segment_frames
    register_model_record(
        _base_model_payload(
            "action-token-model",
            adapter="declarative_action",
            payload_type="declarative_action_tokens",
            default_frames=24,
            execution_contract={
                "action_map": {
                    "static": "S",
                    "yaw_left": "L",
                    "yaw_right": "R",
                    "pan_left": "PL",
                    "pan_right": "PR",
                },
                "segment_frames": [12, 12],
                "granularity": "per_segment",
            },
        )
    )
    img = tmp_path / "first.png"
    img.write_bytes(b"png")
    res_tok = wrbench.compile_camera(
        model="action-token-model",
        camera="yaw:left:60@12,yaw:right:60@12",
        out=tmp_path / "tok.mp4",
        image=str(img),
        dry_run=True,
    )
    assert res_tok["payload"].payload["timeline"] == ["L", "R"]
    assert Path(res_tok["payload"].payload["actions_json"]).is_file()

    # Numeric action vectors on arbitrary multi-segment script
    register_model_record(
        _base_model_payload(
            "action-vec-model",
            adapter="declarative_action",
            payload_type="declarative_action_vectors",
            default_frames=30,
            execution_contract={
                "action_map": {
                    "static": [0.0, 0.0],
                    "yaw_left": [-1.0, 0.0],
                    "pan_right": [0.0, 1.0],
                },
                "granularity": "per_frame",
            },
        )
    )
    res_vec = wrbench.compile_camera(
        model="action-vec-model",
        camera="yaw:left:30@10,static@10,pan:right:0.5@10",
        out=tmp_path / "vec.mp4",
        image=str(img),
        dry_run=True,
    )
    arr = np.load(res_vec["payload"].payload["actions_npy"])
    assert arr.shape == (30, 2)
    np.testing.assert_allclose(arr[0], [-1.0, 0.0])
    np.testing.assert_allclose(arr[15], [0.0, 0.0])
    np.testing.assert_allclose(arr[25], [0.0, 1.0])


def test_prompt_camera_adapter_t2v_api_and_templates(tmp_path: Path):
    # 1. Generic prompt-only T2V model with translation_unit="none"
    register_model_record(
        _base_model_payload(
            "my-t2v-api",
            input_kind="none",
            model_input="T2V",
            viewpoint_condition_type="prompt-only",
            adapter="prompt_camera",
            payload_type="prompt_camera",
            amplitude={
                "rotation_gain": 1.0,
                "translation_gain": 1.0,
                "max_amount": 0.5,
                "translation_unit": "none",
                "calibration_status": "prompt_only",
            },
        )
    )
    res = wrbench.compile_camera(
        model="my-t2v-api",
        camera="yaw:left:60@12,yaw:right:60@13",
        camera_type="yaw_LR",
        out=tmp_path / "t2v.mp4",
        prompt="A cat sits on a wooden table.",
        dry_run=True,
    )
    payload = res["payload"]
    assert payload.payload_type == "prompt_camera"
    assert payload.payload["camera_motion"] == "yaw_LR"
    assert "A cat sits on a wooden table." in payload.payload["prompt_to_send"]
    assert "camera slowly moves left" in payload.payload["prompt_to_send"].lower()

    # Empty prompt dry-run (used by wrbench doctor) succeeds cleanly
    res_empty = wrbench.compile_camera(
        model="my-t2v-api",
        camera="static@25",
        out=tmp_path / "t2v_empty.mp4",
        dry_run=True,
    )
    assert res_empty["payload"].payload["camera_motion"] == "static"
    assert res_empty["payload"].payload["prompt_to_send"]

    # 2. Hailuo command injection via prompt_build_model
    register_model_record(
        _base_model_payload(
            "hailuo-active-plugin",
            input_kind="image",
            viewpoint_condition_type="prompt-only",
            adapter="prompt_camera",
            payload_type="prompt_camera",
            amplitude={
                "rotation_gain": 1.0,
                "translation_gain": 1.0,
                "max_amount": 0.5,
                "translation_unit": "none",
                "calibration_status": "prompt_only",
            },
            execution_contract={"prompt_build_model": "hailuo-2.3"},
        )
    )
    img = tmp_path / "first.png"
    img.write_bytes(b"png")
    res_hailuo = wrbench.compile_camera(
        model="hailuo-active-plugin",
        camera="yaw:right:60@12,yaw:left:60@13",
        camera_type="yaw_RL",
        out=tmp_path / "hailuo.mp4",
        image=str(img),
        prompt="A dog sits on grass. It barks.",
        dry_run=True,
    )
    assert "[Pan right]" in res_hailuo["payload"].payload["prompt_to_send"]

    # 3. Custom prompt_templates map
    register_model_record(
        _base_model_payload(
            "templated-api-model",
            input_kind="none",
            model_input="T2V",
            viewpoint_condition_type="prompt-only",
            adapter="prompt_camera",
            payload_type="prompt_camera",
            amplitude={
                "rotation_gain": 1.0,
                "translation_gain": 1.0,
                "max_amount": 0.5,
                "translation_unit": "none",
                "calibration_status": "prompt_only",
            },
            execution_contract={
                "prompt_templates": {
                    "yaw_LR": "{prompt} <<CAM_LEFT_RETURN>>",
                    "static": "{prompt} <<CAM_LOCKED>>",
                }
            },
        )
    )
    res_tmpl = wrbench.compile_camera(
        model="templated-api-model",
        camera="yaw:left:60@12,yaw:right:60@13",
        camera_type="yaw_LR",
        out=tmp_path / "tmpl.mp4",
        prompt="A sunlit room.",
        dry_run=True,
    )
    assert res_tmpl["payload"].payload["prompt_to_send"] == "A sunlit room. <<CAM_LEFT_RETURN>>"


def test_dynamic_module_import_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    plugin_mod = tmp_path / "custom_adapter_pkg.py"
    plugin_mod.write_text(
        "from wrbench.adapters._utils import adapter_taxonomy_metadata, model_target_trajectory\n"
        "from wrbench.payload import CameraPayload\n"
        "class MyExternalAdapter:\n"
        "    name = 'my_external'\n"
        "    def compile(self, trajectory, *, model_name, width, height, num_frames, work_dir=None, device=None):\n"
        "        target, amp = model_target_trajectory(trajectory, model_name, num_frames)\n"
        "        return CameraPayload(\n"
        "            payload_type='ext_custom',\n"
        "            payload={'frames': target.frame_count},\n"
        "            target_trajectory=target,\n"
        "            official_camera_entrypoint='ext',\n"
        "            coordinate_notes='test',\n"
        "            calibration_status=amp.calibration_status,\n"
        "            metadata=adapter_taxonomy_metadata(model_name=model_name, amp=amp, target=target, requested_frames=num_frames, payload_type='ext_custom'),\n"
        "        )\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    register_model_record(
        _base_model_payload(
            "ext-class-model",
            adapter="custom_adapter_pkg:MyExternalAdapter",
            payload_type="ext_custom",
        )
    )
    assert "ext-class-model" in registered_model_keys()
    adapter = adapter_for_model("ext-class-model")
    assert adapter.name == "my_external"
    assert "ext-class-model" in _REGISTRY

    img = tmp_path / "first.png"
    img.write_bytes(b"png")
    res = wrbench.compile_camera(
        model="ext-class-model",
        camera="static@25",
        out=tmp_path / "ext.mp4",
        image=str(img),
        dry_run=True,
    )
    assert res["payload"].payload == {"frames": 25}


# ---------------------------------------------------------------------------
# Runtime, CommandTemplateBackend, custom backend & launcher registration
# ---------------------------------------------------------------------------

def test_command_template_backend_end_to_end_and_lightweight_runtime(tmp_path: Path):
    spec_file = tmp_path / "my_model.json"
    spec_file.write_text(
        json.dumps(
            _base_model_payload(
                "my-cmd-model",
                adapter="declarative_pose",
                payload_type="declarative_pose_npy",
                execution_contract={"pose_type": "c2w", "convention": "opencv", "format": "npy"},
            )
        ),
        encoding="utf-8",
    )
    infer_script = tmp_path / "fake_infer.py"
    infer_script.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
        "out = Path(args['--out'])\n"
        "out.write_bytes(b'FAKE_MP4_DATA')\n"
        "Path(args['--record']).write_text(json.dumps({\n"
        "    'args': args,\n"
        "    'cuda': os.environ.get('CUDA_VISIBLE_DEVICES'),\n"
        "    'custom_env': os.environ.get('MY_CUSTOM_FLAG'),\n"
        "}))\n",
        encoding="utf-8",
    )
    record_file = tmp_path / "invoked.json"
    runtime_path = tmp_path / "wrbench.runtime.json"
    # Note: omit python_bin, repo_root, gpu_id to verify lightweight defaults when command_template is set!
    runtime_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_spec_files": ["my_model.json"],
                "models": {
                    "my-cmd-model": {
                        "model_path": "/fake/ckpt",
                        "extra_paths": {"record_path": str(record_file)},
                        "env": {"MY_CUSTOM_FLAG": "yes"},
                        "command_template": [
                            "{python_bin}",
                            str(infer_script),
                            "--model",
                            "{model}",
                            "--ckpt",
                            "{model_path}",
                            "--prompt",
                            "{prompt}",
                            "--image",
                            "{image_path}",
                            "--poses",
                            "{target_c2w_path}",
                            "--payload",
                            "{payload_json_path}",
                            "--traj",
                            "{camera_trajectory_json_path}",
                            "--sidecar",
                            "{camera_sidecar_path}",
                            "--pose-file",
                            "{pose_file}",
                            "--record",
                            "{record_path}",
                            "--out",
                            "{output_path}",
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    img = tmp_path / "first.png"
    img.write_bytes(b"png")
    out = tmp_path / "generated.mp4"

    res = wrbench.compile_camera(
        model="my-cmd-model",
        camera="yaw:left:60@12,yaw:right:60@13",
        out=out,
        image=str(img),
        prompt="A bright kitchen.",
        runtime_config=runtime_path,
        dry_run=False,
    )
    assert res["generation"]["success"] is True
    assert out.read_bytes() == b"FAKE_MP4_DATA"

    recorded = json.loads(record_file.read_text(encoding="utf-8"))
    assert recorded["args"]["--model"] == "my-cmd-model"
    assert recorded["args"]["--ckpt"] == "/fake/ckpt"
    assert recorded["args"]["--prompt"] == "A bright kitchen."
    assert Path(recorded["args"]["--poses"]).is_file()
    assert Path(recorded["args"]["--payload"]).is_file()
    assert Path(recorded["args"]["--traj"]).is_file()
    assert Path(recorded["args"]["--sidecar"]).is_file()
    assert Path(recorded["args"]["--pose-file"]).is_file()
    assert recorded["cuda"] == "0"
    assert recorded["custom_env"] == "yes"

    rt = load_runtime_config(runtime_path)
    assert rt is not None
    backend = resolve_backend("my-cmd-model", runtime=rt)
    assert isinstance(backend, CommandTemplateBackend)
    listed = dict((name, ok) for name, ok, _ in list_backends("my-cmd-model", runtime=rt))
    assert listed["command_template"] is True


def test_command_template_rejects_unsafe_placeholder(tmp_path: Path):
    register_model_record(_base_model_payload("unsafe-tmpl-model"))
    rt = RuntimeConfig(
        schema_version=1,
        models={
            "unsafe-tmpl-model": ModelRuntime(
                key="unsafe-tmpl-model",
                backend="command_template",
                command_template=(sys.executable, "{model.__class__}"),
            )
        },
    )
    backend = CommandTemplateBackend(rt)
    img = tmp_path / "first.png"
    img.write_bytes(b"png")
    compiled = wrbench.compile_camera(
        model="unsafe-tmpl-model",
        camera="static@25",
        out=tmp_path / "out.mp4",
        image=str(img),
        dry_run=True,
    )
    result = backend.generate(
        GenerationRequest(
            model="unsafe-tmpl-model",
            prompt="p",
            payload=compiled["payload"],
            output_path=tmp_path / "out.mp4",
        )
    )
    assert not result.success
    assert "Unsafe or unsupported placeholder" in result.message


def test_register_custom_backend_module_callable_and_launcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    register_model_record(_base_model_payload("plug-backend-model"))

    class DummyBackend:
        name = "dummy_custom"

        def __init__(self, runtime=None):
            self.runtime = runtime

        def available(self):
            return True, "dummy ready"

        def generate(self, request: GenerationRequest) -> GenerationResult:
            Path(request.output_path).write_bytes(b"DUMMY_VIDEO")
            return GenerationResult(success=True, output_path=Path(request.output_path), message="ok")

    register_generation_backend("dummy_custom", DummyBackend)
    assert "dummy_custom" in registered_generation_backends()
    assert "command_template" in registered_generation_backends()

    resolved = resolve_backend("plug-backend-model", backend_name="dummy_custom")
    assert resolved.name == "dummy_custom"

    # Also test module:callable backend resolution via runtime config
    mod_file = tmp_path / "custom_gen_backend.py"
    mod_file.write_text(
        "from pathlib import Path\n"
        "from wrbench.backends.base import GenerationResult\n"
        "class RemoteApiBackend:\n"
        "    name = 'remote_api'\n"
        "    def __init__(self, runtime=None):\n"
        "        self.runtime = runtime\n"
        "    def available(self):\n"
        "        return True, 'remote api ready'\n"
        "    def generate(self, request):\n"
        "        Path(request.output_path).write_bytes(b'API_MP4')\n"
        "        return GenerationResult(success=True, output_path=Path(request.output_path), message='ok')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    rt_path = tmp_path / "api_runtime.json"
    rt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": {
                    "plug-backend-model": {
                        "backend": "custom_gen_backend:RemoteApiBackend",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    img = tmp_path / "first.png"
    img.write_bytes(b"png")
    out = tmp_path / "api_out.mp4"
    res = wrbench.compile_camera(
        model="plug-backend-model",
        camera="static@25",
        out=out,
        image=str(img),
        runtime_config=rt_path,
        dry_run=False,
    )
    assert res["generation"]["success"] is True
    assert out.read_bytes() == b"API_MP4"

    # Test register_launcher
    spec = LaunchSpec(
        key="plug-backend-model",
        validate_runtime=lambda runtime: [],
        prepare=lambda req, rt: PreparedLaunch(
            cmd=[sys.executable, "-c", "pass"],
            cwd=tmp_path,
            env={},
            finalize=lambda: GenerationResult(success=True, output_path=req.output_path, message="ok"),
        ),
    )
    register_launcher(spec)
    assert "plug-backend-model" in supported_models()
    assert launcher_spec("plug-backend-model") is spec


def test_refuse_overwrite_output_policy_driven(tmp_path: Path):
    register_model_record(
        _base_model_payload(
            "no-overwrite-model",
            execution_contract={
                "pose_type": "c2w",
                "convention": "opencv",
                "format": "npy",
                "wrbench_policy": {"refuse_overwrite_output": True},
            },
        )
    )
    img = tmp_path / "first.png"
    img.write_bytes(b"png")
    existing = tmp_path / "already.mp4"
    existing.write_bytes(b"keep-me")
    with pytest.raises(FileExistsError, match="Choose a new --out path"):
        wrbench.compile_camera(
            model="no-overwrite-model",
            camera="static@25",
            out=existing,
            image=str(img),
            dry_run=False,
        )
    assert existing.read_bytes() == b"keep-me"


def test_runtime_config_still_requires_fields_for_local_subprocess(tmp_path: Path):
    rt_path = tmp_path / "bad_runtime.json"
    rt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": {
                    "easyanimate-v51-camera": {
                        "gpu_id": 0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeConfigError, match="python_bin"):
        load_runtime_config(rt_path)
