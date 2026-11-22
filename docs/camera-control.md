# Camera-Control Grammar and Pipeline

Reference documentation for the `wrbench` frame-action camera grammar, the three-layer compilation pipeline, and amplitude calibration.

---

## Frame-action grammar

### Compact string format

The grammar is a comma-separated list of **segments**:

```
kind:direction:value@frames[,kind:direction:value@frames,...]
```

Each segment covers a contiguous block of frames and expresses one camera intent. The full string is passed directly to `wrbench.compile_camera(camera=...)` or to `wrbench.parse_camera_script(...)`.

### Compound segments — simultaneous motion

To combine multiple motions **within the same time window**, join them with `+` inside a comma-separated segment. The `@frames` suffix applies to the whole group:

```
yaw:left:60+dolly:forward:1.0@40
```

This yaws the camera 60° left **while** pushing forward 1.0 units — both over the same 40 frames. The `+` notation is fully composable with the regular comma-separated chaining:

```
yaw:left:60+dolly:forward:1.0@40,yaw:right:60+dolly:back:1.0@41
```

### Action kinds and valid directions

| Kind | Type | Valid directions | Value field | Notes |
|---|---|---|---|---|
| `yaw` | rotation | `left`, `right` | `degrees` (any float) | Horizontal pan around vertical axis |
| `pitch` | rotation | `up`, `down` | `degrees` (any float) | Tilt up or down |
| `roll` | rotation | `left`, `right`, `cw`, `ccw` | `degrees` (any float) | Rotate around optical axis |
| `pan` | translation | `left`, `right` | `amount` (any float) | Lateral slide |
| `dolly` | translation | `forward`, `back`, `backward` | `amount` (any float) | Push in or pull out along viewing axis |
| `crane` | translation | `up`, `down` | `amount` (any float) | Vertical lift |
| `static` | — | *(none)* | *(none)* | Hold the camera still |

### `@frames` semantics

The `@frames` suffix sets the number of frames this segment covers. It is optional for a single-segment script when `num_frames` is supplied to `compile_camera`; for multi-segment scripts each segment should carry its own `@frames` so the total is unambiguous.

**Segment frame counts are additive.** A script with two segments of 40 and 41 frames produces 81 total frames. Compound `+` actions within a segment share the same window and do **not** add to the total.

---

## Python API

### Individual motion methods

```python
from wrbench.actions import CameraScript

script = (
    CameraScript()
    .yaw("left", degrees=60, frames=40)
    .yaw("right", degrees=60, frames=41)
)
```

### `segment()` — simultaneous motions

Use `.segment(frames, **kwargs)` to combine multiple motions into one window. Each keyword argument is `kind_direction=value`:

```python
# Arc shot: yaw left while pushing forward
script = CameraScript().segment(40, yaw_left=60, dolly_forward=1.0)

# Diagonal look: yaw + pitch together
script = CameraScript().segment(40, yaw_left=30, pitch_down=15)

# Crane while panning
script = CameraScript().segment(40, crane_up=0.3, pan_right=0.5)

# Chained compound segments (go-return arc)
script = (
    CameraScript()
    .segment(40, yaw_left=60, dolly_forward=1.0)
    .segment(41, yaw_right=60, dolly_back=1.0)
)
```

Valid `kind` prefixes: `yaw`, `pitch`, `roll`, `pan`, `dolly`, `crane`.  
Valid `direction` suffixes follow the direction table above (e.g. `yaw_left`, `dolly_forward`, `crane_up`, `roll_cw`).

`.segment()` and the individual methods (`.yaw()`, `.pan()`, …) can be mixed freely in a chain.

---

## Worked examples

### 1. 60-degree go-return yaw over 81 frames

```
yaw:left:60@40,yaw:right:60@41
```

The camera yaws 60° left across 40 frames, then yaws 60° right across 41 frames, returning close to the original pose. Equivalent to `wrbench.presets.yaw_LR(peak_deg=60, frames=81)`.

```python
import wrbench
from wrbench.datasets import natural25_first_frame_path

result = wrbench.compile_camera(
    model="wan22-fun-5b-cam",
    camera="yaw:left:60@40,yaw:right:60@41",
    image=natural25_first_frame_path("bedroom_cat_bed_jump"),
    out="out.mp4",
)
```

### 2. Arc shot — yaw left while pushing forward, then return

```
yaw:left:60+dolly:forward:1.0@40,yaw:right:60+dolly:back:1.0@41
```

The camera orbits leftward around a subject while approaching it. Using the built-in preset:

```python
result = wrbench.compile_camera(
    model="wan22-fun-5b-cam",
    camera=wrbench.presets.arc_LR(peak_deg=60, dolly_amount=1.0),
    image=natural25_first_frame_path("bedroom_cat_bed_jump"),
    out="out.mp4",
)
```

Or building manually with `segment()`:

```python
from wrbench.actions import CameraScript

script = (
    CameraScript()
    .segment(40, yaw_left=60, dolly_forward=1.0)
    .segment(41, yaw_right=60, dolly_back=1.0)
)
```

### 3. Diagonal look — yaw and pitch together

```
yaw:left:30+pitch:down:15@40
```

A simultaneous horizontal and vertical pan — the camera looks diagonally:

```python
script = CameraScript().segment(40, yaw_left=30, pitch_down=15)
```

### 4. Arbitrary 37-degree one-way sweep

```
yaw:left:37@49
```

A one-directional yaw sweep of any angle. Using the preset builder:

```python
script = wrbench.presets.sweep("yaw", "left", 37, frames=49)
result = wrbench.compile_camera(
    model="wan22-fun-5b-cam",
    camera=script,
    image=natural25_first_frame_path("bedroom_cat_bed_jump"),
    out="out.mp4",
)
```

`sweep` works for all rotation and translation kinds:

```python
wrbench.presets.sweep("pan", "right", 0.4, frames=49)
wrbench.presets.sweep("dolly", "forward", 1.0, frames=49)
```

### 5. Pan go-return

```
pan:left:0.5@40,pan:right:0.5@41
```

Slides the camera left for 40 frames then back right for 41 frames. Equivalent to `wrbench.presets.pan_LR(amount=0.5, frames=81)`.

### 6. Multi-segment composite: yaw left, hold, pan right

```
yaw:left:60@27,static@27,pan:right:0.5@27
```

Three segments of 27 frames each, totalling 81 frames. The camera yaws left, holds still, then pans right. Composed in Python:

```python
from wrbench.actions import CameraScript

script = (
    CameraScript()
    .yaw("left", degrees=60, frames=27)
    .static(frames=27)
    .pan("right", amount=0.5, frames=27)
)
```

---

## Go-return semantics

Go-return presets (`yaw_LR`, `yaw_RL`, `pan_LR`, `pan_RL`, `arc_LR`, `arc_RL`, `orbit`) divide the total frame budget into two halves using:

```
half = max(1, frames // 2)
rest = max(1, frames - half)
```

For `frames=81`: `half=40`, `rest=41`. The first segment goes in the primary direction; the second returns in the opposite direction with the same value, so the pose at the end is close to the starting pose. Frame counts are integers, so for odd totals the return leg gets one extra frame.

Use `wrbench.presets.go_return(kind, first_direction, second_direction, value, frames)` for a generic go-return on any rotation or translation kind:

```python
# Pitch down then up
wrbench.presets.go_return("pitch", "down", "up", 20.0, frames=81)

# Dolly forward then back
wrbench.presets.go_return("dolly", "forward", "back", 1.0, frames=81)
```

---

## Compound preset summary

| Preset | Motion | String equivalent |
|---|---|---|
| `arc_LR(peak_deg, dolly_amount)` | Yaw left + dolly forward, return | `yaw:left:P+dolly:forward:D@H,yaw:right:P+dolly:back:D@R` |
| `arc_RL(peak_deg, dolly_amount)` | Yaw right + dolly forward, return | `yaw:right:P+dolly:forward:D@H,...` |
| `orbit(peak_deg, pan_amount)` | Yaw left + pan right, return | `yaw:left:P+pan:right:A@H,...` |

---

## Three-layer contract

WRBench compiles camera control through three named layers:

### Layer 1 — `frame_action_script`

The compact string or `CameraScript` object. This is the benchmark / researcher intent: "yaw left 60° for 40 frames, then return 60° right for 41 frames". It is model-agnostic.

### Layer 2 — `target_c2w` (OpenCV C2W)

The builder converts the action sequence into a smooth OpenCV camera-to-world (C2W) matrix trajectory, shape `(F, 4, 4)`. This is the evaluation / sidecar trajectory: the `.target_c2w.npy` sidecar records exactly this array. All adapters receive this representation, so the same trajectory can be re-inspected or re-used regardless of which model was targeted.

### Layer 3 — `model_control_timeline`

The adapter translates the C2W trajectory into the model's native control representation and records it in `model_control_timeline` inside `.model_control_samples.json`. **Unified does not mean every model consumes per-frame action tokens.** Depending on the model, the native representation may be:

- **Pose matrices** — per-frame W2C or C2W rows (e.g. Wan Fun CameraCtrl pose text)
- **Latent pose embeddings** — e.g. 21 relative poses for sparse models
- **Geometry NPZ** — Blender C2W scene geometry
- **Action tokens** — numeric action segments
- **Pose text / JSONL** — per-frame JSON rows

Dense models have one control sample per target frame; sparse models may control every few frames. The `model_control_samples.json` sidecar records `control_sample_count`, `payload_type`, and `target_frame_count` so the mapping is always auditable.

---

## Amplitude and calibration

Each model has an `amplitude` block in its registry JSON that scales the model-agnostic motion into the model's native coordinate range:

| Field | Type | Meaning |
|---|---|---|
| `rotation_gain` | float | Multiplier applied to relative rotation in the resampled C2W trajectory |
| `translation_gain` | float | Multiplier applied to translation components in the resampled C2W trajectory |
| `max_amount` | float | Soft ceiling on translated distance in the model's coordinate space |
| `translation_unit` | string | Coordinate space identifier (e.g. `canonical_scene`, `cameractrl_scene`) |
| `calibration_status` | string | How the gains were determined (e.g. `initial_manual`, `uncalibrated`) |

These values are applied during `compile_camera` inside the adapter via `model_target_trajectory`. You can inspect them with:

```python
record = wrbench.model_record("wan22-fun-5b-cam")
print(record.amplitude)
```

Built-in `translation_unit` values are defined in `registry.py` (`VALID_TRANSLATION_UNITS`), along with `"none"`, `"custom:<name>"`, and units registered via `register_translation_unit(...)`. Adapters may interpret the translation unit to pick the correct coordinate conversion path.
