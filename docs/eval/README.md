# WRBench evaluation in WRBench

WRBench ships the official **WRBench** diagnostic evaluation toolkit described in
*Current World Models Lack a Persistent State Core* (preprint). Evaluation is organized
as six separable dimensions plus re-observation support for returned-state metrics.

## Mental model

```
compile (numpy, no config)  →  generate (optional, GPU)  →  evaluate (optional, GPU)
```

- **Core / compile** — camera grammar → native payloads + sidecars (`wrbench generate`, dry-run default)
- **Core / evaluate** — WRBench D1–D6 diagnostic profile (`wrbench eval`)
- **Optional / generate** — real video via configured backends (`wrbench generate --no-dry-run`)
- **Optional extras** — prompts, first frames, cost profiling (`wrbench prompt`, `wrbench firstframe`, `wrbench profile`)

## Diagnostic dimensions

| ID | Paper name | Column | Score field |
| --- | --- | --- | --- |
| D1 | Requested-camera precision (CamPrec) | `D1_camera_pose` | `d1_camera_accuracy` |
| D1 | Prompt-camera alignment (CamAlign) | `D1_camalign` | `d1_camalign_score` |
| D2 | Visual integrity | `D2_visual_integrity` | `d2_selected_visual_integrity_score` |
| D3 | Visible spatial consistency | `D3_spatial_in` | `vlm_spatial_fidelity` |
| D4 | Visible state consistency | `D4_state_in` | `vlm_state_fidelity` |
| (gate) | Re-observation support | `reobservation_support` | judgeability gate rate |
| D5 | Returned spatial consistency | `D5_spatial_oov` | `vlm_spatial_reasoning` |
| D6 | Returned event-state consistency | `D6_state_oov` | `vlm_state_reasoning` |

**D1 has two separate diagnostics** (never merged):

- **CamPrec** — strict requested-control trajectory precision for models that receive explicit trajectories.
- **CamAlign** — common-yaw / static-hold intent alignment for prompt-only and API models (and as a separate paper column in the main table).

Contract: [`src/wrbench/eval/contract/`](../src/wrbench/eval/contract/)

## Configure scorers

Copy [`wrbench.runtime.example.json`](../../wrbench.runtime.example.json) to `wrbench.runtime.json`
and set `eval.scorers` (VGGT, DINOv2, Qwen paths). Heavy models are **not** pip
dependencies; they run via configured interpreters, like generation backends.

## CLI

```bash
# Print the D1-D6 contract (no config required)
wrbench eval contract
wrbench eval contract --json

# One-command full pipeline (recommended)
wrbench eval run \
  --manifest videos.json \
  --out-dir eval_out/

# Granular stages (power users)
wrbench eval d1-vggt --input-jsonl rows.jsonl --output-root /tmp/d1_vggt
wrbench eval d1 --input-jsonl rows.jsonl --output-jsonl d1.jsonl \
  --summary-csv d1.csv --pose-cache-root /tmp/d1_vggt/cache
wrbench eval d1-camalign --input-jsonl rows.jsonl --output-jsonl camalign.jsonl \
  --pose-cache-root /tmp/d1_vggt/cache
wrbench eval d2 --videos-manifest videos.json --out-jsonl d2.jsonl
wrbench eval d3d6 --manifest videos.json --out-dir /tmp/d3d6 --stage all
wrbench eval table \
  --runtime-scores /tmp/d3d6/final_exports/scores_v7_*_gate_masked_export.json \
  --d1-scores d1.jsonl --d1-camalign-scores camalign.jsonl --d2-scores d2.jsonl \
  --out-csv table.csv --out-md table.md --out-summary summary.json
```

Default scorer profile: `wrbench_default` (alias: `current_benchmark_p25_p22_e14`).

D3–D6 orchestration shell: [`scripts/eval/score_runtime_v2_d3d6.sh`](../../scripts/eval/score_runtime_v2_d3d6.sh)

## Package layout

```
src/wrbench/eval/
  d1/          CamPrec + CamAlign (pose recovery + intent scoring)
  d2/          visual integrity (DINOv2)
  scoring/     visible + returned VLM probe scorers
  aggregate/   main table builder + metric contract source
  contract/    exported contract json/md
  runtime.py   eval runtime loader + eval run orchestration
```

## Caveats

- D3–D6 requires multi-GB Qwen weights and a CUDA venv with torch/transformers/decord.
- D1 pose backend defaults to VGGT-Omega; configure `vggt_checkpoint` as the `.pt` file.
- Returned D5/D6 scores are conditional on re-observation support (judgeability gate).
- WRBench reports a **diagnostic profile**, not a single scalar leaderboard score.
