# WRBench Examples

Equivalent CLI commands for the three actions shown in `quickstart.py`.

---

## (a) List supported models

```bash
# Plain table (active models only)
wrbench models

# Include deferred models
wrbench models --deferred

# Full JSON output
wrbench models --json
```

---

## (b) Compile the `yaw_LR` preset for `wan22-fun-5b-cam`

Use one of the bundled Natural-25 first frames:

```bash
python - <<'PY'
from wrbench.datasets import natural25_first_frame_path
print(natural25_first_frame_path("bedroom_cat_bed_jump"))
PY
```

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
  --out /tmp/wrbench_demo/yaw_lr_demo.mp4

# With custom peak angle and frame count:
wrbench generate \
  --model wan22-fun-5b-cam \
  --camera preset:yaw_LR \
  --peak-deg 45 \
  --frames 81 \
  --image "$IMAGE" \
  --out /tmp/wrbench_demo/yaw_lr_demo.mp4
```

---

## (c) Compile an arbitrary sweep script

```bash
# Using the raw script grammar directly:
wrbench generate \
  --model wan22-fun-5b-cam \
  --camera "yaw:left:37@49" \
  --image "$IMAGE" \
  --out /tmp/wrbench_demo/sweep_demo.mp4

# Inspect / validate the script before generating:
wrbench actions --camera "yaw:left:37@49"
```

---

## Other useful commands

```bash
# List preset names with default expansion
wrbench presets

# Validate registry and adapter wiring for all models
wrbench doctor --all

# Validate a specific model
wrbench doctor --model wan22-fun-5b-cam
```

---

## (d) Reproduce the frozen paper source contract

Plan source preparation without creating the destination directory:

```bash
python scripts/prepare_paper_tv2v_sources.py \
  --source-video-root /tmp/wrbench_paper_sources \
  --plan
```

Download or resume verified files, then pass the emitted task map and the same
root explicitly to the Natural-25 runner:

```bash
python scripts/prepare_paper_tv2v_sources.py \
  --source-video-root /tmp/wrbench_paper_sources \
  --download \
  --resume

python scripts/run_natural25_generation.py \
  --model hydra \
  --out-dir /tmp/wrbench_paper_run \
  --variants src/wrbench/data/natural25/releases/paper_main_20260608/variants.local_ti2v_tv2v.jsonl \
  --camera-scope src/wrbench/data/natural25/releases/paper_main_20260608/camera_scopes/local_dual_angle.json \
  --prompt-profile ti2v_active \
  --source-video-task-map /tmp/wrbench_paper_sources/source_video_task_map.json \
  --source-video-root /tmp/wrbench_paper_sources \
  --dry-run --overwrite-existing --fail-fast \
  --limit 1 --shard-index 0 --num-shards 1
```

The paper scope supplies the exact first cell as `yaw_LR_30`; remove
`--limit 1` to compile all four LR/RL × 30/60 cells for the selected model.
The release's `api_prompt_camera.json` is inspect-only because API rows have no
requested degree or target C2W.
