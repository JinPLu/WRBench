# WRBench

**Official toolkit and release artifacts for [WRBench](https://jinplu.github.io/WRBench/), a diagnostic benchmark for testing whether video world models preserve world state across camera motion.**

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2606.20545-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2606.20545)
[![Artifacts](https://img.shields.io/badge/Hugging%20Face-Artifacts-yellow?logo=huggingface)](https://huggingface.co/collections/WRBench/wrbench-current-world-models-lack-a-persistent-state-core-6a365c717251293c9fc2cc26)
[![Leaderboard](https://img.shields.io/badge/Hugging%20Face-Leaderboard-yellow?logo=huggingface)](https://huggingface.co/spaces/WRBench/wrbench-leaderboard)

<p align="center">
  <img src="docs/assets/overview.png" alt="WRBench overview: Natural-25 prompts, controlled camera paths, model-native generation, and D1-D6 evaluation" width="900">
</p>

WRBench provides three composable pieces:

- **Compile:** turn a camera instruction into a model-native payload and auditable sidecars. No GPU is required.
- **Generate (optional):** execute that payload through a configured model backend.
- **Evaluate:** score existing videos with the separable D1–D6 diagnostic contract.

The repository also bundles Natural-25 prompts and first frames, the frozen 23-model paper table, and the versioned paper reproducibility contract.

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

`wrbench generate` is compile-only by default: it writes the target trajectory, native payload, and metadata without loading a model. Use `wrbench actions --camera "yaw:left:37@49" --fps 16` to inspect a custom camera script. See [the camera-control reference](docs/camera-control.md) and [more compile examples](examples/README.md).

## 2. Evaluate an existing video set

First inspect the metric contract; this requires no model configuration:

```bash
wrbench eval contract
```

For D1–D6 scoring, copy [`wrbench.runtime.example.json`](wrbench.runtime.example.json) to `wrbench.runtime.json`, fill in the scorer paths, and run:

```bash
wrbench eval --runtime-config wrbench.runtime.json run \
  --manifest videos.json \
  --out-dir eval_out \
  --scorer-profile wrbench_default \
  --sidecar-profile-gate main
```

Scorer requirements and granular commands are documented in [the evaluation guide](docs/eval/README.md). D1–D6 remain separate diagnostics; WRBench does not collapse them into one quality score.

## 3. Generate videos (optional)

Real generation is an explicit opt-in because each upstream model has its own environment and weights. Copy [`wrbench.runtime.example.json`](wrbench.runtime.example.json), configure a supported backend, then add `--runtime-config /path/to/wrbench.runtime.json --no-dry-run` to the compile command above. See [the backend guide](docs/backends/README.md).

## 4. Add a model

A generation-capable model needs one registry record, one adapter module, and one adapter import. Follow [Adding a model](docs/adding-a-model.md), then validate it with:

```bash
wrbench doctor --model <model-key>
```

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

## Published artifacts

| Artifact | Source |
| --- | --- |
| Paper | [arXiv:2606.20545](https://arxiv.org/abs/2606.20545) |
| All releases | [Hugging Face collection](https://huggingface.co/collections/WRBench/wrbench-current-world-models-lack-a-persistent-state-core-6a365c717251293c9fc2cc26) |
| Natural-25 | [dataset](https://huggingface.co/datasets/WRBench/wrbench-natural25) · [bundled data](src/wrbench/data/natural25/) |
| Frozen paper prompts, first frames, camera scopes, and source mappings | [`paper_main_20260608`](src/wrbench/data/natural25/releases/paper_main_20260608/README.md) |
| Results | [rolling dataset](https://huggingface.co/datasets/WRBench/wrbench-results) · [bundled frozen 23-model tables](src/wrbench/data/results/) |
| Videos and per-video scores | [dataset](https://huggingface.co/datasets/WRBench/wrbench-videos) |
| Human annotations | [dataset](https://huggingface.co/datasets/WRBench/wrbench-human-annotations) |
| Leaderboard | [Space](https://huggingface.co/spaces/WRBench/wrbench-leaderboard) |

The 9,600-row paper table and its provenance are frozen. Corrections or newly verified assets belong to a separately versioned rolling release; they do not silently rewrite the paper aggregates. Exact prompt, first-frame, TV2V-source, and HyDRA policies are owned by the linked release contract rather than repeated here.

## Documentation

- [Camera-control grammar](docs/camera-control.md)
- [Evaluation and scorer configuration](docs/eval/README.md)
- [Generation backends](docs/backends/README.md)
- [Adding a model](docs/adding-a-model.md)
- [Prompt generation](docs/prompts.md)
- [First-frame generation](docs/first-frame.md)

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
