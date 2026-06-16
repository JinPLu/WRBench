# Generation resource profiling

`wrcam.profiling` records **wall time**, **stage-level timing**, and **peak GPU memory** for generation commands. This is producer-side observability for fair speed/cost comparison — not D1-D6 scoring.

## Fair headline metric

Default speed comparison uses:

```text
gpu_seconds_per_output_second = sum(benchmark_gpu_seconds) / sum(output_video_seconds)
```

Where:

- `benchmark_generation_seconds = preprocess_seconds + inference_seconds`
- `benchmark_gpu_seconds = benchmark_generation_seconds × gpu_width`
- **Model load is excluded** from the default comparison
- Only rows with `generation_status="generated"` count toward the denominator
- Aggregation uses **sum/sum**, not mean-of-means

## Stage taxonomy

| Stage | Default comparison? |
|---|---|
| `model_load` | No — cold-start only |
| `input_decode` | No |
| `preprocess` | Yes |
| `inference` | Yes |
| `postprocess` | No |
| `encode_write` | No |
| `sidecar_write` | No |

## CLI

```bash
# Profile any command
wrcam profile --out-dir profiles/ --model wan22-fun-5b-cam -- \
  python run_generation.py --scene demo

# Summarize profiles (directory, JSON, or JSONL)
wrcam profile-summary profiles/ --format markdown
```

## Python API

```python
from wrcam.profiling import StageRecorder, run_profiled_command, summarize, load_profiles

with StageRecorder("events.jsonl").stage("inference", item_id="out.mp4"):
    ...

profile = run_profiled_command(["python", "gen.py"], cwd=".", summary_path="p.json", ...)
rows = summarize(load_profiles(["profiles/"]))
```

## Environment variables

Child processes receive stage recording when the parent sets:

- `WRCAM_STAGE_EVENTS_PATH` — path to append `.stage_events.jsonl`
- `WRCAM_RESOURCE_COMMAND_ID` — command identity hash

Use `from wrcam.profiling import get_stage_recorder` inside the child.

## Dependencies

Profiling is **stdlib-only** plus optional `nvidia-smi` on PATH for GPU sampling. No torch required.

## Published cost table (2026-06-17)

Full-model resource profile summary (17 ACTIVE smoke models with generated rows; host
`10.40.1.28` + manual shard reruns; `hunyuan-game-craft` pending dedicated run):

- [resource_profile_summary.all.md](data/resource_profile_summary.all.md)
- [resource_profile_summary.all.json](data/resource_profile_summary.all.json)

Headline metric: `gpu_seconds_per_output_second` (preprocess+inference only; model load
reported separately). Rows with `generation_status != generated` are excluded from the
denominator.

**Out of scope for v1 cost table:** `sana-wm`, `minwm-hy-action2v`, `matrix-game-2`
(no smoke-task harness yet); `hunyuanworld-voyager` (research-only); `hyworld-worldgen`
(upstream blocked).

Summarize WRBenchLib profiles with WRCam:

```bash
wrcam profile-summary /path/to/qc/resource_profiles --format markdown
```

## Fairness verification

Camera direction/amplitude fairness uses **VGGT-Omega** pose estimation + D1 scoring
(see WRBenchLib `scripts/calibration/`). Report:

- [fairness_verification_report.md](data/fairness_verification_report.md)

Regenerate:

```bash
python scripts/generate_fairness_report.py --acceptance-csv docs/data/acceptance_summary.csv
```
