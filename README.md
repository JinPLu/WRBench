# WRBench

**Official toolkit for [WRBench](https://jinplu.github.io/WRBench/): camera-controlled generation and diagnostic evaluation of video world models.**

[![Paper](https://img.shields.io/badge/Paper-Preprint-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/TODO)
[![Project Page](https://img.shields.io/badge/Project-jinplu.github.io%2FWRBench-green?logo=googlechrome)](https://jinplu.github.io/WRBench/)
[![GitHub](https://img.shields.io/github/stars/JinPLu/WRBench?style=social)](https://github.com/JinPLu/WRBench)

---

> **Current World Models Lack a Persistent State Core**
> *Preprint*

---

## Table of Contents

- [Overview](#overview)
- [Benchmark Results](#benchmark-results)
- [Installation](#installation)
- [Quick start — compile](#quick-start--compile-no-gpu)
- [Quick start — evaluate](#quick-start--evaluate)
- [Quick start — Natural-25 prompts](#quick-start--natural-25-prompts)
- [Evaluation dimensions](#evaluation-dimensions)
- [Supported models](#supported-models)
- [Adding a model](#adding-a-model)
- [Documentation](#documentation)
- [Citation](#citation)

---

## Overview

Most video-generation benchmarks measure **single-scene quality** — aesthetics, text alignment, motion smoothness. They miss whether a model maintains a *consistent 4D world state* across viewpoint changes.

**WRBench** asks: if you look away and come back, is the world still there?

We evaluate models along **six separable diagnostic dimensions**, grouped by whether content is currently in view (*visible*) or had left the frame (*returned*):

| # | Dimension | Short name | What it measures |
|---|-----------|------------|-----------------|
| D1 | Requested-camera precision | CamPrec | Does the camera actually move as instructed? |
| D1 | Prompt-camera alignment | CamAlign | For API models: does the prompt map to correct motion? |
| D2 | Visual integrity | — | Is every frame free of collapse, blur, or distortion? |
| D3 | Visible spatial consistency | — | Do spatial relations hold while objects are in view? |
| D4 | Visible state consistency | — | Do object states remain coherent while in view? |
| D5 | Returned spatial consistency | — | Do spatial relations hold after re-observation? |
| D6 | Returned event-state consistency | — | Do event outcomes persist after re-observation? |

D5 and D6 require **re-observation support** — the model must actually bring content back into frame before they can be scored.

The evaluation uses the **Natural-25** scene/event grid: 25 scene types × 4 event categories, producing controlled viewpoint-intervention prompts. All data and the published 23-model results are **bundled with `pip install wrbench`**.

---

## Benchmark Results

Results for 23 models on the WRBench diagnostic profile (9,600 generated videos, 2,073 re-observation-supported rows for D5/D6). Full CSV at `src/wrbench/data/results/wrbench_23model_results.csv`.

> **D5/D6 are scored on shared judgeable re-observation rows**: the model must bring the relevant content back into frame before returned spatial or event-state consistency can be checked. Models without a D1-CamPrec score use API prompt-camera control only.

<details open>
<summary><b>Camera-trained and video-to-video models</b></summary>

| Model | CamPrec ↑ | CamAlign ↑ | D2 ↑ | D3 ↑ | D4 ↑ | D5 ↑ | D6 ↑ |
|-------|-----------|-----------|------|------|------|------|------|
| Hydra | **0.822** | 0.999 | 0.691 | 0.648 | 0.500 | 0.509 | 0.445 |
| LiveWorld | 0.812 | 0.977 | 0.775 | 0.703 | 0.541 | 0.661 | 0.600 |
| VerseCrafter | 0.781 | 0.904 | 0.846 | 0.707 | 0.508 | 0.607 | 0.584 |
| Wan-Fun 2.1-1.3B | 0.771 | 0.882 | 0.842 | 0.725 | 0.513 | 0.709 | 0.657 |
| Wan-Fun 2.2-A14B | 0.758 | 0.761 | **0.848** | **0.810** | **0.625** | 0.698 | 0.649 |
| Wan-Fun 2.1-14B | 0.757 | 0.740 | 0.846 | 0.733 | 0.530 | 0.659 | 0.621 |
| Wan-Fun 2.2-5B | 0.724 | 0.513 | 0.812 | 0.805 | 0.607 | 0.709 | 0.664 |
| ReCamMaster | 0.717 | 0.940 | 0.740 | 0.715 | 0.535 | 0.665 | 0.616 |
| Spatia | 0.704 | 0.620 | 0.763 | 0.731 | 0.541 | 0.600 | 0.586 |
| Gen3C | 0.699 | 0.902 | 0.749 | 0.723 | 0.558 | 0.681 | 0.640 |
| InSpatio World 14B | 0.693 | 0.835 | 0.824 | 0.821 | 0.668 | **0.734** | 0.664 |

</details>

<details>
<summary><b>Interactive and action-driven models</b></summary>

| Model | CamPrec ↑ | CamAlign ↑ | D2 ↑ | D3 ↑ | D4 ↑ | D5 ↑ | D6 ↑ |
|-------|-----------|-----------|------|------|------|------|------|
| MagicWorld | 0.764 | 0.851 | 0.543 | 0.623 | 0.458 | 0.584 | 0.574 |
| Hunyuan WorldPlay | 0.708 | 0.401 | 0.870 | 0.737 | 0.523 | 0.640 | 0.603 |
| Hunyuan GameCraft | 0.534 | 0.464 | 0.705 | 0.672 | 0.440 | 0.554 | 0.490 |
| Lingbot World | 0.513 | 0.220 | 0.870 | 0.876 | 0.735 | 0.717 | 0.663 |
| Lingbot Act | 0.468 | 0.326 | 0.856 | 0.874 | 0.719 | 0.771 | **0.725** |

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

Want your model on the list? See [Adding a model](#adding-a-model) and open a PR.

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

WRBench is designed to be *separable*: each dimension can be scored independently, and models can achieve high D2–D4 with poor D1 (camera compliance) or vice versa. D5/D6 are only valid when **re-observation support** applies — the model must actually bring previously-out-of-view content back into frame before the state can be checked.

| Dim | Full name | Scorer | Requires |
|-----|-----------|--------|---------|
| D1-CamPrec | Requested-camera precision | VGGT-Omega pose estimation | Pose-input models |
| D1-CamAlign | Prompt-camera alignment | LLM intent parsing | All models |
| D2 | Visual integrity | DINOv2 local/global features | — |
| D3 | Visible spatial consistency | Qwen-3.5B VLM | — |
| D4 | Visible state consistency | Qwen-3.5B VLM | — |
| D5 | Returned spatial consistency | Qwen-3.5B VLM | Re-observation support |
| D6 | Returned event-state consistency | Qwen-3.5B VLM | Re-observation support |

Detailed scorer profiles and configuration: [docs/eval/README.md](docs/eval/README.md).

---

## Supported models

23 models across four control paradigms. Run `wrbench models` for the full registry with capability flags.

| Paradigm | Examples |
|----------|---------|
| **V2V** (video-to-video re-render) | Hydra, VerseCrafter, ReCamMaster, Gen3C, Spatia, InSpatio World |
| **Camera-conditioned** (pose input) | Wan-Fun series, LiveWorld, Lingbot |
| **Interactive / action-driven** | Hunyuan WorldPlay, Hunyuan GameCraft, MagicWorld |
| **API prompt-camera** | Kling, Hailuo, Wan API, HappyHorse |

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
| Natural-25 pre-generated TI2V prompt variants | `src/wrbench/data/natural25/variants.jsonl` |
| Natural-25 released first frames | `src/wrbench/data/natural25/first_frames/` |
| Published 23-model results | `src/wrbench/data/results/wrbench_23model_results.{csv,json}` |
| Human annotation verdicts (2,547) | Separate release — see [project page](https://jinplu.github.io/WRBench/) |

---

## Citation

```bibtex
@article{wrbench2026,
  title   = {Current World Models Lack a Persistent State Core},
  author  = {Jinpeng Lu and Dexu Zhu and Haoyuan Shi and Yinda Chen and Linghan Cai and Guo Tang and Jie Cao and Yong Dai},
  year    = {2026},
  note    = {Preprint},
  url     = {https://github.com/JinPLu/WRBench},
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
