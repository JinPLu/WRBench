# WRBench

**Official toolkit and release artifacts for [WRBench](https://jinplu.github.io/WRBench/), a diagnostic benchmark for testing whether video world models preserve world state across camera motion.**

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2606.20545-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2606.20545)
[![Hugging Face Collection](https://img.shields.io/badge/Hugging%20Face-Collection-yellow?logo=huggingface)](https://huggingface.co/collections/WRBench/wrbench-current-world-models-lack-a-persistent-state-core)
[![Leaderboard](https://img.shields.io/badge/Hugging%20Face-Leaderboard-yellow?logo=huggingface)](https://huggingface.co/spaces/WRBench/wrbench-leaderboard)

<p align="center">
  <img src="docs/assets/overview.png" alt="WRBench overview: Natural-25 prompts, controlled camera paths, model-native generation, and D1-D6 evaluation" width="900">
</p>

WRBench provides three composable pieces:

- **Compile:** turn a camera instruction into a model-native payload and auditable sidecars. No GPU is required.
- **Generate (optional):** execute that payload through a configured model backend or command template.
- **Evaluate:** score existing videos with the separable D1–D6 diagnostic contract.

The repository also bundles Natural-25 prompts and first frames, the frozen 23-model paper table, and the versioned paper reproducibility contract.

## Release resources

The [Hugging Face Collection](https://huggingface.co/collections/WRBench/wrbench-current-world-models-lack-a-persistent-state-core) brings together the paper, datasets, and interactive leaderboard.

| Resource | What you'll find |
| --- | --- |
| [Paper](https://arxiv.org/abs/2606.20545) | Benchmark design, evaluation protocol, and reported results |
| [Natural-25 inputs](https://huggingface.co/datasets/WRBench/wrbench-natural25) | Test prompts, first frames, and camera specifications |
| [Results and addenda](https://huggingface.co/datasets/WRBench/wrbench-results) | Frozen paper tables and separately versioned rolling results |
| [Generated videos and scores](https://huggingface.co/datasets/WRBench/wrbench-videos) | Model outputs, video index, and per-video scores |
| [Human annotations](https://huggingface.co/datasets/WRBench/wrbench-human-annotations) | Released human judgments and annotation metadata |
| [Leaderboard](https://huggingface.co/spaces/WRBench/wrbench-leaderboard) | Interactive comparison of released model results |

For the paper snapshot, use the [frozen input release](src/wrbench/data/natural25/releases/paper_main_20260608/README.md)
and the [bundled 23-model results](src/wrbench/data/results/), which cover 9,600 videos. The input release documents
the exact prompts, first frames, camera scopes, and source mappings. Rolling updates are versioned separately on Hugging Face.

## Install

Python 3.10 or newer is required.

```bash
git clone https://github.com/JinPLu/WRBench.git
cd WRBench
pip install -e .
```

Install optional prompt, first-frame, and development dependencies with `pip install -e ".[all]"`. Model weights and GPU frameworks are intentionally not package dependencies.

## 1. Inspect and compile a camera request

List the available models and camera presets:

```bash
wrbench models
wrbench presets
```

Compile a go-and-return yaw request with a bundled Natural-25 first frame:

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
  --out out.mp4
```

`wrbench generate` is compile-only by default: it writes the target trajectory, native payload, and metadata without loading a model. Add `--runtime-config wrbench.runtime.json --no-dry-run` to run real generation through a configured backend. Use `wrbench actions --camera "yaw:left:37@49" --fps 16` to inspect a custom camera script. See [the camera-control reference](docs/camera-control.md), [generation backends](docs/backends/README.md), and [more compile examples](examples/README.md).

## 2. Evaluate a video set

First inspect the metric contract; this requires no model configuration:

```bash
wrbench eval contract
```

For D1–D6 scoring, copy [`wrbench.runtime.example.json`](wrbench.runtime.example.json) to an untracked `wrbench.runtime.json` and set your local scorer interpreter, repository, and checkpoint paths (canonical paper defaults for VGGT-Omega, DINOv2, and Qwen D3–D6 are applied automatically). Check scorer readiness, validate the manifest, and run the pipeline:

```bash
wrbench doctor --eval --runtime-config wrbench.runtime.json
wrbench eval validate --manifest videos.json
wrbench eval --runtime-config wrbench.runtime.json run \
  --manifest videos.json \
  --out-dir eval_out \
  --target-mode requested_profile
```

Scorer requirements, individual stage commands, and custom evaluator hooks are documented in [the evaluation guide](docs/eval/README.md). D1–D6 remain separate diagnostics; WRBench does not collapse them into one quality score.

## 3. Benchmark or integrate any model

WRBench supports three integration levels without modifying WRBench source files:

- **Evaluate existing videos:** export standardized Natural-25 tasks with `wrbench tasks export`, generate MP4s in your own pipeline, verify the manifest with `wrbench eval validate`, and score with `wrbench eval run`.
- **Declarative model spec + command template:** pass an external JSON spec (`--model-spec` or top-level `model_spec_files` in `wrbench.runtime.json`) using a built-in declarative adapter (`declarative_pose`, `declarative_action`, or `prompt_camera`), and run inference via `"backend": "command_template"`.
- **Python plugins:** register custom adapters, generation backends, launchers, or evaluators (`pose`, `d2`, `vlm`, and segment policies) via `module:callable` references or `pyproject.toml` entry points (`wrbench.plugins`, `wrbench.eval_plugins`). Inspect loaded extensions with `wrbench plugins`.

See [Integrating models and evaluators](docs/plugins.md) for templates and examples, or [Adding a built-in model](docs/adding-a-model.md) when contributing a model directly to this repository.

## Evaluation dimensions

| Dimension | What it measures | Applies when |
| --- | --- | --- |
| D1-CamPrec | Requested-camera trajectory precision | An explicit target trajectory exists |
| D1-CamAlign | Prompt-camera intent alignment | Camera control is prompt/API based |
| D2 | Visual integrity | All videos |
| D3 | Visible spatial consistency | All videos |
| D4 | Visible state consistency | All videos |
| D5 | Re-observation spatial consistency | The relevant content returns into view |
| D6 | Re-observation event-state consistency | The relevant content returns into view |

Rows without re-observation support are excluded from D5/D6 rather than counted as failures. The exact contract lives in [`src/wrbench/eval/contract/`](src/wrbench/eval/contract/).

## Documentation

- [Integrating models and evaluators](docs/plugins.md)
- [Camera-control grammar](docs/camera-control.md)
- [Evaluation and scorer configuration](docs/eval/README.md)
- [Generation backends](docs/backends/README.md)
- [Adding a built-in model](docs/adding-a-model.md)
- [Prompt generation](docs/prompts.md)
- [First-frame generation](docs/first-frame.md)
- [Generation resource profiling](docs/cost-profiling.md)

## Citation

```bibtex
@article{wrbench2026,
  title   = {Current World Models Lack a Persistent State Core},
  author  = {Jinpeng Lu and Dexu Zhu and Haoyuan Shi and Linghan Cai and Guo Tang and Yinda Chen and Jie Cao and Duyu Tang and Yi Zhang and Yong Dai and Xiaozhu Ju},
  journal = {arXiv preprint arXiv:2606.20545},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.20545},
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
