# Adding a model

A generation-capable model has one registry record, one adapter module, and one adapter import. Keep capabilities, calibration, aliases, and input metadata in the registry record rather than creating parallel maps.

## 1. Add the model record

Create `src/wrbench/models/<key>.json`. Use a lowercase, hyphen-separated canonical key.

```json
{
  "key": "my-model-7b-cam",
  "aliases": ["my-model-7b"],
  "status": "active",
  "input_kind": "image",
  "adapter": "my_model",
  "payload_type": "my_model_pose_txt",
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
- `amplitude` records the explicit rotation and translation calibration used to compile the target trajectory.
- `translation_unit` must use a value already accepted by the registry. Check an existing model with the same native camera representation; do not add a synonym for an existing unit.
- `viewpoint_condition_type`, `model_input`, and source-video usage belong in this same record when the model participates in result tables.

Use existing active model JSON files as the schema reference. Registry validation rejects unknown input kinds, duplicate keys/aliases, missing active-model fields, and unsupported translation units.

## 2. Add the adapter

Create `src/wrbench/adapters/<module>.py`. The adapter receives a canonical `CameraTrajectory` and returns a `CameraPayload` containing both the model-native control and the exact target trajectory used for evaluation.

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

## 3. Import the adapter

Add the module name to `_ADAPTER_MODULES` in `src/wrbench/adapters/__init__.py`:

```python
_ADAPTER_MODULES = [
    "wrbench.adapters.my_model",
]
```

The import triggers `@register`. Registered keys must already exist as active model records.

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

The second command is compile-only by default. Inspect its trajectory, payload, and metadata sidecars before adding real generation support. Add focused registry, adapter, and payload tests, then run the project test suite.

If the upstream model can be launched from WRBench, continue with [Generation backends](backends/README.md). Machine-specific paths belong only in `wrbench.runtime.json`.
