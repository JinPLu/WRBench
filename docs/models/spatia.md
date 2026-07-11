# spatia

| Field | Value |
|---|---|
| Runtime input | `source_video` wrapper |
| Benchmark model input | `TI2V` (`first_frame_extraction`) |
| Adapter | `spatia` |
| Payload type | `spatia_w2c_trajectory_files` |
| Default frames | `121` |
| Default fps | `24` |
| Resolution | `1248x704` |

## Dry-run compile (out of the box)

```bash
wrbench generate --model spatia --camera preset:yaw_LR --source-video source.mp4 --out out/spatia.mp4
```

## Python

```python
import wrbench
wrbench.compile_camera(model="spatia", camera="yaw:left:60@40,yaw:right:60@41", source_video='source.mp4', out="out/spatia.mp4")
```

## Real generation

WRBench compiles the model-native payload and sidecars locally. Real video generation requires the model's own environment (weights, GPU, venv). See the upstream model repository and use the compiled `.payload.json` / sidecars as inputs.

Spatia's wrapper extracts frame 0 for the official image path. It is not a
temporal TV2V model for WRBench taxonomy, even though the runtime argument is a
source-video path.

Use `wrbench doctor --model spatia` to inspect the current backend status and required runtime fields.
