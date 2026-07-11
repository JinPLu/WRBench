# hydra

| Field | Value |
|---|---|
| Input | `source_video` |
| Benchmark model input | `TV2V` (`temporal_conditioning_source_video`) |
| Adapter | `hydra` |
| Payload type | `hydra_split_camera_json` |
| Default frames | `77` |
| Default fps | `15` |
| Resolution | `832x480` |

## Dry-run compile (out of the box)

```bash
wrbench generate --model hydra --camera preset:yaw_LR --source-video source.mp4 --out out/hydra.mp4
```

## Python

```python
import wrbench
wrbench.compile_camera(model="hydra", camera="yaw:left:60@40,yaw:right:60@41", source_video='source.mp4', out="out/hydra.mp4")
```

## Real generation

WRBench compiles the model-native payload and sidecars locally. Real video generation requires the model's own environment (weights, GPU, venv). See the upstream model repository and use the compiled `.payload.json` / sidecars as inputs.

Use `wrbench doctor --model hydra` to inspect the current backend status and required runtime fields.

## Paper-release frame and metric policy

HyDRA consumes 77 source frames and submits a 154-frame concatenation: source
`[0,77)` followed by generated continuation `[77,154)`. The current D1 policy
crops the generated segment before pose inference, while D2-D6 score all 154
frames. Camera matrices alone are stride-sampled to 20 embeddings; RGB frames
are decoded in order, padded by repeating the last frame when short, and
center-cropped/resized to 480x832.

The published CamAlign surface uses the repaired generated-only path. The
frozen CamPrec value retains legacy full-concat-pose then post-hoc-slice
provenance. See
`src/wrbench/data/natural25/releases/paper_main_20260608/hydra_evaluation_policy.json`.
