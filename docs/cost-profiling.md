# Generation resource profiling

`wrbench.profiling` records **wall time**, **stage-level timing**, and **peak GPU memory** for generation commands. This is producer-side observability for speed and cost comparison, separate from D1–D6 evaluation.

## Fair headline metric

WRBench headline speed comparison uses:

```text
gpu_seconds_per_output_second = sum(benchmark_gpu_seconds) / sum(output_video_seconds)
```

Where:

- `benchmark_generation_seconds = preprocess_seconds + inference_seconds`
- `benchmark_gpu_seconds = benchmark_generation_seconds × gpu_width`
- **Model load is excluded** from the headline comparison
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
wrbench profile --out-dir profiles/ --model wan22-fun-5b-cam -- \
  python run_generation.py --scene demo

# Summarize profiles (directory, JSON, or JSONL)
wrbench profile-summary profiles/ --format markdown
```

## Python API

```python
from wrbench.profiling import StageRecorder, run_profiled_command, summarize, load_profiles

with StageRecorder("events.jsonl").stage("inference", item_id="out.mp4"):
    ...

profile = run_profiled_command(["python", "gen.py"], cwd=".", summary_path="p.json", ...)
rows = summarize(load_profiles(["profiles/"]))
```

## Child-process instrumentation

For custom generation launchers, call `from wrbench.profiling import get_stage_recorder`
inside the child process and write spans through the returned recorder. Under
`wrbench profile`, the parent command wires the child recorder context for you.

## Dependencies

Profiling is **stdlib-only** plus optional `nvidia-smi` on PATH for GPU sampling. No torch required.

## Cost-table output

`wrbench profile` writes the resource summary for the runs you record. WRBench
does not bundle a hardware-specific cost table because GPU type, runtime stack,
and model revisions affect throughput.

Headline metric: `gpu_seconds_per_output_second` (preprocess + inference only;
model load is reported separately). Rows with `generation_status != "generated"`
are excluded from the denominator, and `gpu_seconds_per_output_second`
multiplies benchmark wall time by `gpu_width` (GPUs visible via
`CUDA_VISIBLE_DEVICES`). When profiling a small smoke sample rather than a full
benchmark sweep, treat cross-model ordering as indicative.

Summarize saved profiles with WRBench:

```bash
wrbench profile-summary /path/to/resource_profiles --format markdown
```
