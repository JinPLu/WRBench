# Compile examples

These examples only compile camera requests; they do not load model weights or require a GPU. Run them from the repository root after installation.

## Inspect available controls

```bash
wrbench models
wrbench presets
wrbench actions --camera "yaw:left:37@49" --fps 16
```

## Compile a preset

Use a bundled Natural-25 first frame:

```bash
IMAGE="$(python - <<'PY'
from wrbench.datasets import natural25_first_frame_path
print(natural25_first_frame_path("bedroom_cat_bed_jump"))
PY
)"

wrbench generate \
  --model wan22-fun-5b-cam \
  --camera preset:yaw_LR \
  --image "$IMAGE" \
  --prompt "A house cat waits on the bedroom floor, facing the bed." \
  --out /tmp/wrbench_demo/yaw_lr_demo.mp4
```

The default command writes the compiled payload and sidecars without generating `yaw_lr_demo.mp4`.

## Change the angle or frame count

```bash
wrbench generate \
  --model wan22-fun-5b-cam \
  --camera preset:yaw_LR \
  --peak-deg 45 \
  --frames 81 \
  --image "$IMAGE" \
  --prompt "A house cat waits on the bedroom floor, facing the bed." \
  --out /tmp/wrbench_demo/yaw_lr_45.mp4
```

Use the camera grammar directly when a preset is not enough:

```bash
wrbench generate \
  --model wan22-fun-5b-cam \
  --camera "yaw:left:37@49" \
  --image "$IMAGE" \
  --prompt "A house cat waits on the bedroom floor, facing the bed." \
  --out /tmp/wrbench_demo/yaw_37.mp4
```

Run `wrbench doctor --model wan22-fun-5b-cam` to validate the model record and adapter. For real generation, continue with [the backend guide](../docs/backends/README.md). For the frozen paper source contract, use the commands maintained in its [versioned release README](../src/wrbench/data/natural25/releases/paper_main_20260608/README.md).
