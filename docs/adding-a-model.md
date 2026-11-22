# Adding a built-in model

You can integrate a model into WRBench in two ways:

- **External plugin (recommended for custom or third-party models):** load a standalone JSON spec via `--model-spec`, `WRBENCH_MODEL_DIRS`, `model_spec_files` in `wrbench.runtime.json`, or a Python `wrbench.plugins` entry point without editing WRBench source files. See [Integrating models and evaluators](plugins.md) for examples using the built-in `declarative_pose`, `declarative_action`, and `prompt_camera` adapters or `module:callable` adapters.
- **Built-in model (for upstream PRs into WRBench):** add one registry record, one adapter module (or reuse a declarative adapter), and one adapter import below.

Keep capabilities, calibration, aliases, and input metadata in the registry record rather than creating parallel maps.

## 1. Add the model record

Create `src/wrbench/models/<key>.json` (or an external JSON file passed via `--model-spec`). Use a lowercase, hyphen-separated canonical key.

```json
{
  "key": "my-model-7b-cam",
  "aliases": ["my-model-7b"],
  "status": "active",
  "input_kind": "image",
  "adapter": "my_model",
  "payload_type": "my_model_pose_txt",
  "default_width": 832,
  "default_height": 480,
  "default_fps": 16,
  "default_frames": 49,
  "amplitude": {
    "rotation_gain": 1.0,
    "translation_gain": 2.0,
    "max_amount": 0.8,
    "translation_unit": "canonical_scene",
    "calibration_status": "initial_manual",
    "metadata": {}
  },
  "capabilities": {
    "rotation": true,
    "translation": true,
    "supports_static": true
  },
  "notes": "My model camera-control variant."
}
```

Required choices:

- `status` is `active` or `deferred`. Deferred records do not register an adapter.
- `input_kind` is `image`, `source_video`, or `none`.
- `adapter` can be a registered built-in name, a declarative adapter (`declarative_pose`, `declarative_action`, `prompt_camera`), or a Python reference (`module.path:AdapterClass`).
- `amplitude` records the explicit rotation and translation calibration used to compile the target trajectory.
- `translation_unit` accepts any built-in unit, `"none"` (for prompt-only / API models), `"custom:<name>"`, or a unit registered via `wrbench.registry.register_translation_unit(...)`. Reuse an existing built-in unit whenever the native representation matches.
- `viewpoint_condition_type`, `model_input`, and source-video usage belong in this same record when the model participates in result tables.

Use existing active model JSON files as the schema reference. Registry validation rejects unknown input kinds, duplicate keys/aliases, missing active-model fields, and unsupported translation units.

## 2. Add the adapter (skip if using a declarative adapter)

If your model matches standard pose matrices, discrete action timelines, or prompt-based camera wording, use `declarative_pose`, `declarative_action`, or `prompt_camera` directly in your JSON spec (see [Integrating models and evaluators](plugins.md)).

Otherwise, create `src/wrbench/adapters/<module>.py` (or a class referenced as `my_pkg.adapter:MyModelAdapter`). The adapter receives a canonical `CameraTrajectory` and returns a `CameraPayload` containing both the model-native control and the exact target trajectory used for evaluation.

```python
from __future__ import annotations

from pathlib import Path

from wrbench.adapters._utils import adapter_taxonomy_metadata, model_target_trajectory
from wrbench.adapters.base import register
from wrbench.payload import CameraPayload
from wrbench.trajectory import CameraTrajectory


@register("my-model-7b-cam")
class MyModelAdapter:
    name = "my_model"

    def compile(
        self,
        trajectory: CameraTrajectory,
        *,
        model_name: str,
        width: int,
        height: int,
        num_frames: int,
        work_dir: str | Path | None = None,
        device: str | None = None,
    ) -> CameraPayload:
        target, amp = model_target_trajectory(trajectory, model_name, num_frames)
        native_payload = build_native_payload(target.to_c2w())
        payload_type = "my_model_pose_txt"
        return CameraPayload(
            payload_type=payload_type,
            payload=native_payload,
            target_trajectory=target,
            official_camera_entrypoint="control_camera_video",
            coordinate_notes="Document the native-to-OpenCV C2W conversion.",
            calibration_status=amp.calibration_status,
            metadata=adapter_taxonomy_metadata(
                model_name=model_name,
                amp=amp,
                target=target,
                requested_frames=num_frames,
                payload_type=payload_type,
                model_payload_summary={},
            ),
        )
```

Do not hide coordinate conversion, resampling, or gain changes in a backend launcher: the compiled payload and target sidecar must remain reproducible without running the model.

## 3. Import the adapter (built-in models only)

For a built-in model inside `src/wrbench/adapters/`, add the module name to `_ADAPTER_MODULES` in `src/wrbench/adapters/__init__.py`:

```python
_ADAPTER_MODULES = [
    "wrbench.adapters.my_model",
]
```

The import triggers `@register`. Registered keys must already exist as active model records. External plugins skip this step and use `module.path:AdapterClass`, `declarative_*`, or `wrbench.plugins` entry points instead.

## 4. Validate it

```bash
wrbench doctor --model my-model-7b-cam
wrbench generate \
  --model my-model-7b-cam \
  --camera preset:yaw_LR \
  --image /path/to/first_frame.png \
  --prompt "Describe the scene and event." \
  --out /tmp/my-model-yaw.mp4
```

For an external JSON spec, prefix with `--model-spec /path/to/my-model-7b-cam.json`. The `generate` command is compile-only by default: inspect its trajectory, payload, and metadata sidecars before wiring real generation.

If the upstream model can be launched from WRBench, continue with [Generation backends](backends/README.md). Machine-specific paths belong only in `wrbench.runtime.json`.
