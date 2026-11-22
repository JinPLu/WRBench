# gen3c

| Field | Value |
|---|---|
| Input | `source_video` |
| Adapter | `gen3c` |
| Payload type | `gen3c_w2c_intrinsics` |
| Default frames | `81` |
| Default fps | `16` |
| Resolution | `832x480` |

## Dry-run compile (out of the box)

```bash
wrbench generate --model gen3c --camera preset:yaw_LR --source-video source.mp4 --prompt "A house cat waits on the bedroom floor, facing the bed." --out out/gen3c.mp4
```

## Python

```python
import wrbench
wrbench.compile_camera(model="gen3c", camera="yaw:left:60@40,yaw:right:60@41", source_video='source.mp4', out="out/gen3c.mp4")
```

## Real generation

WRBench compiles the model-native payload and sidecars locally. Real video generation requires the model's own environment (weights, GPU, venv): configure `command_template` or `local_subprocess` in `wrbench.runtime.json` (see [Generation backends](../backends/README.md) and [Integrating models and evaluators](../plugins.md)) or pass the compiled `.payload.json` / sidecars to the upstream model repository.

Use `wrbench doctor --model gen3c` to inspect the current backend status and required runtime fields.
