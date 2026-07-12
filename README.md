# WRBench

**Official toolkit and release artifacts for [WRBench](https://jinplu.github.io/WRBench/): a camera-controlled benchmark for testing whether video world models keep a persistent world state.**

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2606.20545-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2606.20545)
[![HF Paper](https://img.shields.io/badge/Hugging%20Face-Paper-yellow?logo=huggingface)](https://huggingface.co/papers/2606.20545)
[![Project Page](https://img.shields.io/badge/Project-jinplu.github.io%2FWRBench-green?logo=googlechrome)](https://jinplu.github.io/WRBench/)
[![Artifacts](https://img.shields.io/badge/Hugging%20Face-Artifacts-yellow?logo=huggingface)](https://huggingface.co/collections/WRBench/wrbench-current-world-models-lack-a-persistent-state-core-6a365c717251293c9fc2cc26)
[![Leaderboard](https://img.shields.io/badge/Hugging%20Face-Leaderboard-yellow?logo=huggingface)](https://huggingface.co/spaces/WRBench/wrbench-leaderboard)
[![GitHub](https://img.shields.io/github/stars/JinPLu/WRBench?style=social)](https://github.com/JinPLu/WRBench)

> **Current World Models Lack a Persistent State Core.**
> WRBench is a diagnostic benchmark for asking whether a generated video world remains consistent when the camera moves away and returns.

<p align="center">
  <img src="docs/assets/overview.png" alt="WRBench overview: Natural-25 prompts, controlled camera paths, model-native generation, and D1-D6 evaluation" width="900">
</p>

## What This Repository Provides

WRBench is both a benchmark release and a Python toolkit:

| Component | What it is for |
|-----------|----------------|
| Natural-25 | Scene/event prompts and first frames for controlled viewpoint-intervention tests. |
| Camera compiler | Converts one camera intent into model-native controls: pose paths, prompt text, action tokens, or backend payloads. |
| Evaluation toolkit | Runs the D1-D6 diagnostic contract for camera control, visual integrity, visible consistency, and re-observation consistency. |
| Re-observation gate | Marks whether a video actually brings out-of-view content back into frame before D5/D6 are scored. |
| Published artifacts | Bundled result tables, public Hugging Face datasets, benchmark videos, human annotations, and leaderboard links. |

WRBench is a diagnostic benchmark rather than a single video-quality leaderboard. It separates camera compliance, visible scene consistency, and returned-state consistency so those failure modes stay legible.

## Typical Use

1. Choose the model input mode: first-frame image, source video, prompt plus camera controls, or API prompt-camera control.
2. Compile a camera-controlled generation request for that model.
3. Run the model through a configured backend, or inspect the dry-run payload.
4. Evaluate generated videos with the D1-D6 scoring contract.
5. Compare against the released 23-model main table, or publish a model entry in the matching track.

Detailed D1-D6, re-observation, and prompt-only T2V policy lives in [docs/eval/README.md](docs/eval/README.md).

## Model Input Modes

All WRBench generations use a text prompt. The model `input_kind` records the extra visual input, if any.

| Mode | Extra input | How camera control is provided | Examples |
|------|-------------|--------------------------------|----------|
| First-frame / TI2V | Natural-25 first-frame image | Pose path, action tokens, or model-specific camera payload | Wan-Fun, EasyAnimate, VerseCrafter, MagicWorld, Hunyuan WorldPlay, Lingbot, minWM-HY |
| Source-video / TV2V | Temporal source video clip | Re-render or camera-conditioned source-video contract | Hydra, Gen3C, ReCamMaster, InSpatio World |
| Source-wrapped first-frame / TI2V | Source wrapper extracts frame 0 | First-frame image plus geometry/camera controls | LiveWorld, Spatia |
| Prompt plus controls / T2V | None beyond prompt | Native camera tokens or model-specific prompt/control files | minWM Wan Action2V, reported in a separate T2V track rather than the 23-model main table |
| API prompt-camera | No explicit pose input | Camera instruction is written into the prompt | Kling, Hailuo, Wan API, HappyHorse |

D1-CamPrec applies to models with explicit pose or trajectory targets. API and prompt-camera models use D1-CamAlign instead.

## Table of Contents

- [What This Repository Provides](#what-this-repository-provides)
- [Typical Use](#typical-use)
- [Model Input Modes](#model-input-modes)
- [Release artifacts](#release-artifacts)
- [Benchmark Results](#benchmark-results)
- [Installation](#installation)
- [Quick start - compile](#quick-start--compile-no-gpu)
- [Quick start - evaluate](#quick-start--evaluate)
- [Quick start - Natural-25 prompts](#quick-start--natural-25-prompts)
- [Evaluation dimensions](#evaluation-dimensions)
- [Supported models](#supported-models)
- [Adding a model](#adding-a-model)
- [Documentation](#documentation)
- [Citation](#citation)

---

## Release artifacts

| Surface | Link |
|---------|------|
| Paper | [arXiv](https://arxiv.org/abs/2606.20545) · [Hugging Face paper page](https://huggingface.co/papers/2606.20545) |
| Project page | [jinplu.github.io/WRBench](https://jinplu.github.io/WRBench/) |
| All Hugging Face artifacts | [WRBench collection](https://huggingface.co/collections/WRBench/wrbench-current-world-models-lack-a-persistent-state-core-6a365c717251293c9fc2cc26) |
| Leaderboard | [WRBench/wrbench-leaderboard](https://huggingface.co/spaces/WRBench/wrbench-leaderboard) |
| Natural-25 prompts and first frames | [WRBench/wrbench-natural25](https://huggingface.co/datasets/WRBench/wrbench-natural25) |
| Published 23-model results | [WRBench/wrbench-results](https://huggingface.co/datasets/WRBench/wrbench-results) |
| Human annotation verdicts | [WRBench/wrbench-human-annotations](https://huggingface.co/datasets/WRBench/wrbench-human-annotations) |
| Benchmark videos and per-video scores | [WRBench/wrbench-videos](https://huggingface.co/datasets/WRBench/wrbench-videos) |
| Frozen paper reproducibility contract | [`paper_main_20260608`](src/wrbench/data/natural25/releases/paper_main_20260608/README.md) |

The compatibility Hugging Face configs load directly with `datasets`:
`variants`, `model_scores`, `pairs`, and `videos_master`. The immutable local
release directory pins the exact paper prompts, first-frame generation catalog
and bytes, camera denominator, TV2V source assets, and HyDRA evaluation policy.
The 9,600-row paper table stays frozen. Corrections to non-paper metadata or
generation assets, and newly evaluated models, are versioned in the separate
11,100-row rolling dataset so they cannot silently rewrite the paper table.

---

## Benchmark Results

Results for 23 models on the WRBench diagnostic profile (9,600 generated videos, 2,073 re-observation-supported rows for D5/D6). Full CSV at `src/wrbench/data/results/wrbench_23model_results.csv`.

This is the frozen 23-model main table. Prompt-only T2V addenda are tracked separately; see [docs/eval/README.md](docs/eval/README.md) for the public scope and promotion rules.

Validate the paper contract and inspect its bundled paths from an installed
package:

```python
from wrbench.datasets import natural25_release_dir
from wrbench.release_validation import validate_natural25_release

print(natural25_release_dir("paper_main_20260608"))
print(validate_natural25_release("paper_main_20260608"))
```

Prepare the 100 pinned source videos without guessing local paths:

```bash
python scripts/prepare_paper_tv2v_sources.py \
  --source-video-root ./paper_tv2v_sources \
  --plan
```

<details open>
<summary><b>Camera-trained and controlled-view models</b></summary>

| Model | CamPrec ↑ | CamAlign ↑ | D2 ↑ | D3 ↑ | D4 ↑ | D5 ↑ | D6 ↑ |
|-------|-----------|-----------|------|------|------|------|------|
| HyDRA‡ | 0.822 | 0.855 | 0.691 | 0.648 | 0.500 | 0.509 | 0.445 |
| LiveWorld | 0.812 | 0.856 | 0.775 | 0.703 | 0.541 | 0.661 | 0.600 |
| VerseCrafter | 0.781 | 0.667 | 0.846 | 0.707 | 0.508 | 0.607 | 0.584 |
| Wan-Fun 2.1-1.3B | 0.771 | 0.729 | 0.842 | 0.725 | 0.513 | 0.709 | 0.657 |
| Wan-Fun 2.2-A14B | 0.758 | 0.553 | **0.848** | **0.810** | **0.625** | 0.698 | 0.649 |
| Wan-Fun 2.1-14B | 0.757 | 0.526 | 0.846 | 0.733 | 0.530 | 0.659 | 0.621 |
| Wan-Fun 2.2-5B | 0.724 | 0.335 | 0.812 | 0.805 | 0.607 | 0.709 | 0.664 |
| ReCamMaster | 0.717 | 0.729 | 0.740 | 0.715 | 0.535 | 0.665 | 0.616 |
| Spatia | 0.704 | 0.482 | 0.763 | 0.731 | 0.541 | 0.600 | 0.586 |
| Gen3C | 0.699 | 0.764 | 0.749 | 0.723 | 0.558 | 0.681 | 0.640 |
| InSpatio World 14B | 0.693 | 0.661 | 0.824 | 0.821 | 0.668 | **0.734** | 0.664 |

‡ HyDRA's frozen CamPrec value retains legacy full-concatenation pose
reconstruction followed by a post-hoc slice. Its maintained CamAlign value
uses generated-only poses, so 0.822 is provenance rather than directly
comparable evidence of best camera execution.

</details>

<details>
<summary><b>Interactive and action-driven models</b></summary>

| Model | CamPrec ↑ | CamAlign ↑ | D2 ↑ | D3 ↑ | D4 ↑ | D5 ↑ | D6 ↑ |
|-------|-----------|-----------|------|------|------|------|------|
| MagicWorld | 0.764 | 0.720 | 0.543 | 0.623 | 0.458 | 0.584 | 0.574 |
| Hunyuan WorldPlay | 0.708 | 0.261 | 0.870 | 0.737 | 0.523 | 0.640 | 0.603 |
| Hunyuan GameCraft | 0.534 | 0.361 | 0.705 | 0.672 | 0.440 | 0.554 | 0.490 |
| Lingbot World | 0.513 | 0.175 | 0.870 | 0.876 | 0.735 | 0.717 | 0.663 |
| Lingbot Act | 0.468 | 0.168 | 0.856 | 0.874 | 0.719 | 0.771 | **0.725** |

</details>

<details>
<summary><b>API prompt-camera models (no pose input)</b></summary>

| Model | CamAlign ↑ | D2 ↑ | D3 ↑ | D4 ↑ | D5 ↑ | D6 ↑ |
|-------|-----------|------|------|------|------|------|
| Hailuo 2.3 | 0.075 | 0.829 | 0.891 | 0.759 | 0.719 | 0.642 |
| HappyHorse 1.0 I2V | 0.025 | 0.860 | 0.875 | 0.715 | 0.779 | 0.695 |
| Kling v2.6 | 0.094 | 0.864 | 0.854 | 0.674 | 0.711 | 0.617 |
| Wan2.2 I2V Plus | 0.013 | 0.800 | 0.829 | 0.644 | 0.714 | 0.610 |
| Wan2.6 I2V | 0.016 | 0.856 | 0.855 | 0.682 | 0.659 | 0.556 |
| Wan2.7 I2V | 0.020 | 0.750 | 0.848 | 0.676 | 0.715 | 0.638 |
| WanX2.1 I2V Turbo | 0.030 | 0.713 | 0.839 | 0.651 | **0.855** | **0.777** |

</details>

Want your model in WRBench? See [Adding a model](#adding-a-model) and submit it
to the matching surface: the 23-model main table for standard WRBench runs, or
the separate T2V track for prompt-only addenda such as minWM Wan Action2V.

---

## Installation

```bash
pip install -e .
# or with all optional extras (prompt generation, first-frame T2I, profiling):
pip install -e ".[all]"
```

**Requirements:** Python ≥ 3.10. Core dependency: `numpy>=1.23` (no GPU required for compilation).

For real generation and evaluation, configure backends in `wrbench.runtime.json` — copy from [`wrbench.runtime.example.json`](wrbench.runtime.example.json).

---

## Quick start — compile (no GPU)

Describe camera motion once with the `kind:direction:value@frames` grammar; wrbench compiles it into each model's native control format and writes auditable sidecars. Natural-25 first frames are bundled, so the example does not require generating an image first.

```python
import wrbench
from wrbench.datasets import natural25_first_frame_path

result = wrbench.compile_camera(
    model="wan22-fun-5b-cam",
    camera="yaw:left:60@40,yaw:right:60@41",  # look left 60° for 40 frames, then right
    image=natural25_first_frame_path("bedroom_cat_bed_jump"),
    out="out.mp4",
)
print(result["artifacts"])  # .target_c2w.npy, .camera_trajectory.json, .payload.json, ...
```

```bash
IMAGE="$(python - <<'PY'
from wrbench.datasets import natural25_first_frame_path
print(natural25_first_frame_path("bedroom_cat_bed_jump"))
PY
)"

# dry-run (no GPU): inspect compiled payload
wrbench generate --model wan22-fun-5b-cam --camera preset:yaw_LR --image "$IMAGE" --out out.mp4

# inspect all presets and models
wrbench presets
wrbench models
wrbench doctor --all
```

**Camera grammar cheatsheet:**

| Syntax | Meaning |
|--------|---------|
| `yaw:left:60@40` | Yaw left 60° over 40 frames |
| `pan:right:0.3@30` | Pan right 0.3 m over 30 frames |
| `preset:yaw_LR` | Built-in go-and-return yaw preset |
| `preset:static` | No camera motion |
| `yaw:left:60@40,yaw:right:60@41` | Compound: two actions concatenated |

Full grammar reference: [docs/camera-control.md](docs/camera-control.md).

---

## Quick start — evaluate

With scorers configured in `wrbench.runtime.json`:

```bash
# Run the full D1–D6 pipeline
wrbench eval run --manifest videos.json --out-dir eval_out/

# Print the metric contract (no config needed)
wrbench eval contract
```

`videos.json` is a list of records with `video_path`, `model`, `camera`, and optional sidecar paths. See [docs/eval/README.md](docs/eval/README.md) for the schema and granular stage commands (`d1-vggt`, `d1`, `d2`, `d3d6`, `table`).

---

## Quick start — Natural-25 prompts

The Natural-25 scene/event grid (25 scenes × 4 event categories) is bundled in the package:

```python
from wrbench.datasets import (
    build_natural25_candidates,
    load_jsonl,
    load_natural25_families,
    natural25_first_frame_path,
    natural25_variants_path,
)
from wrbench.prompts.task import generate_variants_deterministic

variants = generate_variants_deterministic(
    build_natural25_candidates(),
    load_natural25_families(),
)

# Or load the pre-generated 400-row prompt set directly:
variants = list(load_jsonl(natural25_variants_path()))
first_frame = natural25_first_frame_path("bedroom_cat_bed_jump")
```

```bash
wrbench prompt task --deterministic --output variants.jsonl
```

---

## Evaluation dimensions

WRBench is designed to be *separable*: each dimension can be scored independently, and models can achieve high D2-D4 with poor D1 camera compliance, or strong visual quality with weak returned-state consistency.

| Dim | Full name | Scorer | Requires |
|-----|-----------|--------|---------|
| D1-CamPrec | Requested-camera precision | VGGT-Omega pose estimation | Explicit pose/trajectory target |
| D1-CamAlign | Prompt-camera alignment | LLM intent parsing | Prompt/API camera-control models |
| D2 | Visual integrity | DINOv2 local/global features | — |
| D3 | Visible spatial consistency | Qwen-3.5B VLM | — |
| D4 | Visible state consistency | Qwen-3.5B VLM | — |
| D5 | Re-observation spatial consistency | Qwen-3.5B VLM | Re-observation-supported rows only |
| D6 | Re-observation event-state consistency | Qwen-3.5B VLM | Re-observation-supported rows only |

Detailed scorer profiles and configuration: [docs/eval/README.md](docs/eval/README.md).

---

## Supported models

The frozen public main table covers 23 models across four primary control paradigms. Run `wrbench models` for the full registry with capability flags and input requirements.

| Paradigm | Model input | Examples | Camera metric |
|----------|-------------|----------|---------------|
| **TV2V / temporal source-video** | Temporal source video + prompt | Hydra, Gen3C, ReCamMaster, InSpatio World | D1-CamPrec when a target trajectory is compiled |
| **TI2V / source-wrapped first frame** | Frame 0 extracted from a source wrapper + prompt | LiveWorld, Spatia | D1-CamPrec when a target trajectory is compiled |
| **Camera-conditioned / TI2V** | First-frame image + prompt | Wan-Fun series, EasyAnimate, VerseCrafter, Lingbot, minWM-HY | D1-CamPrec |
| **Interactive / action-driven** | First-frame image + prompt + native actions | Hunyuan WorldPlay, Hunyuan GameCraft, MagicWorld | D1-CamPrec or adapter-specific camera diagnostics |
| **API prompt-camera** | Prompt-only camera instruction | Kling, Hailuo, Wan API, HappyHorse | D1-CamAlign |
| **T2V addenda** | Prompt + native camera/control tokens | minWM Wan Action2V | Separate T2V diagnostic track; not part of the frozen 23-model main table |

Per-model guides: [`docs/models/`](docs/models/).

---

## Adding a model

Two files + one import line. See [docs/adding-a-model.md](docs/adding-a-model.md) for the walkthrough; `wrbench doctor --model <name>` validates your adapter before running.

---

## Documentation

| Topic | Link |
|-------|------|
| Camera-control grammar | [docs/camera-control.md](docs/camera-control.md) |
| Evaluation (D1–D6) | [docs/eval/README.md](docs/eval/README.md) |
| Adding a model | [docs/adding-a-model.md](docs/adding-a-model.md) |
| Backends (real generation) | [docs/backends/README.md](docs/backends/README.md) |
| Prompt generation | [docs/prompts.md](docs/prompts.md) |
| First-frame T2I | [docs/first-frame.md](docs/first-frame.md) |
| Cost profiling | [docs/cost-profiling.md](docs/cost-profiling.md) |

---

## Paper data

| Artifact | Location |
|----------|----------|
| Natural-25 scene/event prompts | `src/wrbench/data/natural25/` (bundled in package) |
| Frozen paper prompt/camera/source contract | `src/wrbench/data/natural25/releases/paper_main_20260608/` |
| Current deterministic toolkit prompt variants | `src/wrbench/data/natural25/variants.jsonl` |
| Natural-25 released first frames | `src/wrbench/data/natural25/first_frames/` |
| Published 23-model results | `src/wrbench/data/results/wrbench_23model_results.{csv,json}` |
| Hugging Face release hub | [WRBench collection](https://huggingface.co/collections/WRBench/wrbench-current-world-models-lack-a-persistent-state-core-6a365c717251293c9fc2cc26) |
| Natural-25 public dataset | [WRBench/wrbench-natural25](https://huggingface.co/datasets/WRBench/wrbench-natural25) |
| Published results public dataset | [WRBench/wrbench-results](https://huggingface.co/datasets/WRBench/wrbench-results) |
| Human annotation verdicts (2,547) | [WRBench/wrbench-human-annotations](https://huggingface.co/datasets/WRBench/wrbench-human-annotations) |
| Benchmark videos (9,600 frozen paper rows; 11,100 rolling public rows) | [WRBench/wrbench-videos](https://huggingface.co/datasets/WRBench/wrbench-videos) |
| Interactive leaderboard | [WRBench/wrbench-leaderboard](https://huggingface.co/spaces/WRBench/wrbench-leaderboard) |

---

## Citation

```bibtex
@article{wrbench2026,
  title   = {Current World Models Lack a Persistent State Core},
  author  = {Jinpeng Lu and Dexu Zhu and Haoyuan Shi and Yinda Chen and Linghan Cai and Guo Tang and Jie Cao and Yong Dai},
  journal = {arXiv preprint arXiv:2606.20545},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.20545},
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
