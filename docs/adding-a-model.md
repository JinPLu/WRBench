# Adding a Model to wrcam

Adding a new model requires exactly **two files** plus **one import line**. There is no separate alias map, no capability contract file, no amplitude YAML, no shell list, and no run-spec to edit — all of that is consolidated into the registry JSON.

---

## Overview

| Step | What you create |
|---|---|
| 1 | `src/wrcam/models/<key>.json` — model registry record |
| 2 | `src/wrcam/adapters/<key>.py` — adapter class with `@register` |
| 3 | One import line in `src/wrcam/adapters/__init__.py` |
| 4 | `wrcam doctor --model <key>` to verify |

---

## Step 1 — Create the model registry JSON

Create `src/wrcam/models/<key>.json` where `<key>` is the canonical model identifier (lowercase, hyphen-separated, e.g. `my-model-7b-cam`).

### Full schema

```json
{
  "key": "my-model-7b-cam",
  "aliases": ["mymodel-7b", "my-model-7b"],
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
  "notes": "My model 7B camera-control variant."
}
```

### Field reference

| Field | Type | Required | Validation rule |
|---|---|---|---|
| `key` | string | yes | Must be non-empty; must be unique across all JSON files |
| `aliases` | list of strings | no | Additional names that resolve to this model; normalised to lowercase-hyphen |
| `status` | string | yes | Must be `"active"` or `"deferred"` |
| `input_kind` | string | yes for active | Must be `"image"` (TI2V) or `"source_video"` (V2V) |
| `adapter` | string | yes for active | Logical name of the adapter; used in sidecars for traceability |
| `payload_type` | string | no | Identifies the native payload format in sidecars and `model_control_samples.json` |
| `amplitude` | object | yes for active | See amplitude sub-fields below |
| `capabilities` | object | no | Free-form capability flags; surfaced in `wrcam models` output |
| `notes` | string | no | Human-readable description for `wrcam models` |

#### `amplitude` sub-fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `rotation_gain` | float | no (default 1.0) | Scales `degrees` into the model's rotation range |
| `translation_gain` | float | yes | Scales `amount` into the model's translation range |
| `max_amount` | float | yes | Soft ceiling on translated distance in the model's coordinate space |
| `translation_unit` | string | yes | Must be one of the values in `VALID_TRANSLATION_UNITS` (see below) |
| `calibration_status` | string | no (default `"uncalibrated"`) | How gains were determined; free string, e.g. `"initial_manual"` |
| `metadata` | object | no | Additional calibration notes; arbitrary key-value pairs |

#### Valid `translation_unit` values

```
canonical_scene
cameractrl_scene
blender_c2w_scene
cache3d_scene
relative_pose_embedding
w2c_scene
official_displacement_scale
trajectory_template_c2w
hydra_c2w_pre_div100
```

These are enforced by `registry.py`; loading a JSON with an unsupported value raises `RegistryError`.

#### Deferred models

Set `"status": "deferred"` for planned but not yet implemented models. Deferred models do not require `input_kind`, `adapter`, or a valid `amplitude` block — the registry supplies defaults. Deferred models cannot be registered by adapters and raise an error if passed to `compile_camera`.

---

## Step 2 — Create the adapter module

Create `src/wrcam/adapters/<key>.py`. The adapter class must implement the `CameraAdapter` protocol and be decorated with `@register(...)`.

```python
# src/wrcam/adapters/my_model.py

from __future__ import annotations

from pathlib import Path

from wrcam.adapters._utils import adapter_taxonomy_metadata, model_target_trajectory
from wrcam.adapters.base import register
from wrcam.payload import CameraPayload
from wrcam.trajectory import CameraTrajectory


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

        # Convert target C2W matrices into the model's native format here.
        # target.to_c2w() is a (F, 4, 4) numpy array in OpenCV C2W convention.
        native_payload = _build_native_payload(target)

        payload_type = "my_model_pose_txt"
        return CameraPayload(
            payload_type=payload_type,
            payload=native_payload,
            target_trajectory=target,
            official_camera_entrypoint="control_camera_video",
            coordinate_notes="Describe coordinate conversion here.",
            calibration_status=amp.calibration_status,
            metadata=adapter_taxonomy_metadata(
                model_name=model_name,
                amp=amp,
                target=target,
                requested_frames=int(num_frames),
                payload_type=payload_type,
                model_payload_summary={"pose_row_count": target.frame_count},
            ),
        )
```

### `@register` decorator

`@register("my-model-7b-cam")` instantiates the class and inserts the instance into the adapter registry under that canonical model key. Multiple keys can be registered to the same adapter class:

```python
@register("my-model-7b-cam", "my-model-14b-cam")
class MyModelAdapter:
    ...
```

The keys must already exist in `src/wrcam/models/` and must be `"active"` (deferred models cannot be registered). The decorator resolves aliases, so you can pass an alias — but using the canonical key is clearer.

### `compile` contract

`compile` receives a `CameraTrajectory` and returns a `CameraPayload`. The helper `model_target_trajectory(trajectory, model_name, num_frames)` applies the model's `rotation_gain` and `translation_gain` from the registry, resamples to `num_frames` if needed, and returns `(target_trajectory, amplitude)`.

---

## Step 3 — Register the module import

Open `src/wrcam/adapters/__init__.py` and add your module to `_ADAPTER_MODULES`:

```python
_ADAPTER_MODULES = [
    "wrcam.adapters.wan_fun",
    "wrcam.adapters.my_model",   # add this line
]
```

`wrcam.adapters` imports every module in this list on package import, which triggers the `@register` decorators and populates the adapter registry. Without this line the adapter is never loaded and `compile_camera` will raise `KeyError`.

---

## Step 4 — Verify with `wrcam doctor`

```bash
wrcam doctor --model my-model-7b-cam
```

This checks that:

- The registry JSON parses and validates without `RegistryError`
- The adapter module loads and registers the key
- The `CameraAdapter` protocol is satisfied (the class has a `compile` method)
- Amplitude fields are within expected bounds

To check all registered models at once:

```bash
wrcam doctor --all
```

---

## Design note

WRCam collapses per-model configuration into **one JSON + one Python module + one import line**, with a single validation pass at load time. Alias maps, capability flags, amplitude calibration, and adapter wiring all live in the model JSON; `wrcam doctor` validates the pair at load time.
