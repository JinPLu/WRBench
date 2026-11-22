# hunyuan-game-craft

| Field | Value |
|---|---|
| Input | `image` |
| Adapter | `hunyuan` |
| Payload type | `hunyuan_gamecraft_input_pose_txt` |
| Default frames | `81` |
| Default fps | `16` |
| Resolution | `832x480` |

## Dry-run compile (out of the box)

```bash
IMAGE="$(python - <<'PY2'
from wrbench.datasets import natural25_first_frame_path
print(natural25_first_frame_path("bedroom_cat_bed_jump"))
PY2
)"

wrbench generate --model hunyuan-game-craft --camera preset:yaw_LR --image "$IMAGE" --prompt "A house cat waits on the bedroom floor, facing the bed." --out out/hunyuan-game-craft.mp4
```

## Python

```python
import wrbench
from wrbench.datasets import natural25_first_frame_path

wrbench.compile_camera(model="hunyuan-game-craft", camera="yaw:left:60@40,yaw:right:60@41", image=natural25_first_frame_path("bedroom_cat_bed_jump"), out="out/hunyuan-game-craft.mp4")
```

## Real generation

WRBench compiles the model-native payload and sidecars locally. Real video generation requires the model's own environment (weights, GPU, venv): configure `command_template` or `local_subprocess` in `wrbench.runtime.json` (see [Generation backends](../backends/README.md) and [Integrating models and evaluators](../plugins.md)) or pass the compiled `.payload.json` / sidecars to the upstream model repository.

Use `wrbench doctor --model hunyuan-game-craft` to inspect the current backend status and required runtime fields.
