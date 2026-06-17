# wrcam

**Official open-source toolkit for [WRBench](https://github.com/wrbench): unified camera control and diagnostic evaluation for video-generation world models.**

`wrcam` lets you drive any supported video-generation model with the same `kind:direction:value@frames` grammar. You describe the camera motion once; wrcam compiles it into each model's native control payload and writes auditable sidecars alongside the output path.

WRBench evaluates world models on **six separable diagnostic dimensions** (requested-camera precision, visual integrity, visible spatial/state consistency, returned spatial/state consistency) plus **re-observation support** for returned-state metrics — not a single scalar leaderboard.

---

## Mental model

```
compile (numpy, no config)  →  generate (optional, GPU)  →  evaluate (optional, GPU)
```

| Layer | Commands | Config needed |
| --- | --- | --- |
| **Core / compile** | `wrcam generate` (dry-run default), `presets`, `actions`, `models`, `doctor` | none |
| **Core / evaluate** | `wrcam eval run`, `wrcam eval contract` | `eval.scorers` for scoring (contract works with zero config) |
| **Optional / generate** | `wrcam generate --no-dry-run` | `models` in `wrcam.runtime.json` |
| **Optional extras** | `wrcam prompt`, `wrcam firstframe`, `wrcam profile` | provider keys as needed |

One config file: [`wrcam.runtime.example.json`](wrcam.runtime.example.json) → `wrcam.runtime.json`.

---

## Features

- **Unified frame-action grammar** — one compact string format (`kind:direction:value@frames`) drives all supported models
- **Arbitrary rotation / translation** — yaw/pitch/roll and pan/dolly/crane at near-per-frame granularity
- **Preset combinations** — `yaw_LR`, `pan_LR`, `static`; or compose with `sweep` and `go_return`
- **One adapter per model** — isolated translation layer; new models are two files + one import line
- **Auditable sidecars** — `.target_c2w.npy`, `.camera_trajectory.json`, `.camera.json`, `.model_control_samples.json`, `.payload.json`
- **Numpy-only core** — payload compilation requires only numpy; no GPU needed
- **WRBench D1–D6 evaluation** — diagnostic profile + main table via `wrcam eval` ([docs/eval/README.md](docs/eval/README.md))
- **Natural-25 prompts** — bundled scene/event grid in the package (`wrcam.data.natural25`)
- **Published 23-model results** — reference table in the package (`wrcam.data.results`)
- **Optional extras** — prompt generation, first-frame T2I, resource profiling

---

## Install

```bash
pip install -e .
# or with optional extras:
pip install -e ".[all]"
```

Requires Python ≥ 3.10. Core dependency: `numpy>=1.23`.

---

## Quickstart — compile (no GPU)

```python
import wrcam

result = wrcam.compile_camera(
    model="wan22-fun-5b-cam",
    camera="yaw:left:60@40,yaw:right:60@41",
    image="first.png",
    out="out.mp4",
)
print(result["artifacts"])
```

```bash
wrcam generate --model wan22-fun-5b-cam --camera preset:yaw_LR --image first.png --out out.mp4
wrcam eval contract
wrcam doctor --all
```

---

## Quickstart — evaluate

Configure scorers in `wrcam.runtime.json`, then run the full pipeline:

```bash
wrcam eval run --manifest videos.json --out-dir eval_out/
```

Granular stages (`d1-vggt`, `d1`, `d2`, `d3d6`, `table`) remain available for power users — see [docs/eval/README.md](docs/eval/README.md).

---

## Quickstart — Natural-25 prompts

```python
from wrcam.datasets import build_natural25_candidates, load_natural25_families
from wrcam.prompts.task import generate_variants_deterministic

variants = generate_variants_deterministic(
    build_natural25_candidates(),
    load_natural25_families(),
)
```

```bash
wrcam prompt task --deterministic --output variants.jsonl
# defaults to bundled data/natural25/ when paths omitted
```

---

## Supported models

Run `wrcam models` for the current registry (23 active + deferred entries). Per-model guides: [`docs/models/`](docs/models/).

---

## Documentation

- [Camera-control grammar](docs/camera-control.md)
- [Adding a model](docs/adding-a-model.md)
- [Evaluation (D1–D6)](docs/eval/README.md)
- [Backends (real generation)](docs/backends/README.md)
- [Prompt generation](docs/prompts.md) · [First frames](docs/first-frame.md) · [Profiling](docs/cost-profiling.md)

---

## Paper data

| Artifact | Location |
| --- | --- |
| Natural-25 scene/event prompts | `src/wrcam/data/natural25/` (shipped with `pip install wrcam`) |
| Published 23-model diagnostic profile | `src/wrcam/data/results/wrbench_23model_results.csv` |

Human annotation verdicts (2,547) are released separately from this repository.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
