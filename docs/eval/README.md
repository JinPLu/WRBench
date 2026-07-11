# WRBench evaluation in WRBench

WRBench ships the official **WRBench** diagnostic evaluation toolkit described in
*Current World Models Lack a Persistent State Core* ([arXiv:2606.20545](https://arxiv.org/abs/2606.20545)). Evaluation is organized
as six separable dimensions plus re-observation support for D5/D6.

## Mental model

```
compile (numpy, no config)  →  generate (optional, GPU)  →  evaluate (optional, GPU)
```

- **Core / compile** — camera grammar → native payloads + sidecars (`wrbench generate`)
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
| D5 | Re-observation spatial consistency | `D5_spatial_oov` | `vlm_spatial_reasoning` |
| D6 | Re-observation event-state consistency | `D6_state_oov` | `vlm_state_reasoning` |

**D1 has two separate diagnostics** (never merged):

- **CamPrec** — strict requested-control trajectory precision for models that receive explicit trajectories.
- **CamAlign** — common-yaw / static-hold intent alignment for prompt-only and API models (and as a separate paper column in the main table).

Contract: [`src/wrbench/eval/contract/`](../src/wrbench/eval/contract/)

## Configure scorers

Copy [`wrbench.runtime.example.json`](../../wrbench.runtime.example.json) to `wrbench.runtime.json`
and fill every required `eval.scorers` field explicitly: `gpu_id`, the VGGT
paths, the DINOv2 paths, the Qwen paths, and the `env` entries consumed by the
D3-D6 scorers. Heavy models are **not** pip dependencies; they run via
configured interpreters, like generation backends.

## CLI

```bash
# Print the D1-D6 contract (no config required)
wrbench eval contract
wrbench eval contract --json

# One-command full pipeline (recommended)
wrbench eval --runtime-config wrbench.runtime.json run \
  --manifest videos.json \
  --out-dir eval_out/ \
  --scorer-profile wrbench_default \
  --sidecar-profile-gate main

# Granular stages (power users)
wrbench eval --runtime-config wrbench.runtime.json d1-vggt \
  --input-jsonl rows.jsonl \
  --output-root /tmp/d1_vggt \
  --cache-root /tmp/d1_vggt/cache \
  --execution-mode subprocess
wrbench eval d1 --input-jsonl rows.jsonl --output-jsonl d1.jsonl \
  --summary-csv d1.csv \
  --pose-cache-root /tmp/d1_vggt/cache \
  --pose-backend vggt_omega \
  --poses-file poses.npy \
  --default-frames 121 \
  --sidecar-profile-gate main \
  --predicted-pose-type c2w \
  --predicted-camera-convention opencv \
  --target-camera-convention opencv \
  --rot-scale-deg 45.0 \
  --trans-scale 1.0 \
  --yaw-weak-threshold-deg 2.0 \
  --pan-weak-threshold 0.0001 \
  --static-rot-threshold-deg 2.0 \
  --static-trans-threshold 0.05
wrbench eval d1-camalign --input-jsonl rows.jsonl --output-jsonl camalign.jsonl \
  --pose-cache-root /tmp/d1_vggt/cache \
  --poses-file poses.npy
wrbench eval --runtime-config wrbench.runtime.json d2 \
  --videos-manifest videos.json \
  --out-jsonl d2.jsonl
wrbench eval --runtime-config wrbench.runtime.json d3d6 \
  --manifest videos.json --out-dir /tmp/d3d6 --stage all \
  --scorer-profile wrbench_default
wrbench eval table \
  --runtime-scores /tmp/d3d6/final_exports/scores_v7_*_gate_masked_export.json \
  --d1-scores d1.jsonl --d1-camalign-scores camalign.jsonl --d2-scores d2.jsonl \
  --out-csv table.csv --out-md table.md --out-summary summary.json
```

Choose `--scorer-profile` explicitly. Published benchmark runs use `wrbench_default`.

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
- D1 pose scoring uses VGGT-Omega; set `eval.scorers.vggt_python_bin`,
  `eval.scorers.vggt_repo`, and `eval.scorers.vggt_checkpoint` explicitly.
- WRBench reports a **diagnostic profile**, not a single scalar leaderboard score.

## Re-observation and T2V policy

This section is the canonical public policy for D5/D6 judgeability and for prompt-only T2V addenda.

### Re-observation gate

D5 and D6 are scored only on **re-observation-supported** rows. A row is judgeable only when the video actually brings the relevant out-of-view target back into frame, so returned spatial or event-state consistency can be checked. If the target never returns, the row is excluded from D5/D6 rather than treated as negative evidence.

This gate is independent of model family. It depends on the generated video and applies the same way to TI2V, TV2V, API prompt-camera, and prompt-only T2V runs.

### T2V scope (text-only models)

T2V models (`input_kind: none`) do not use Natural-25 first frames. Generate
them with an explicit Natural-25 prompt profile so the generation prompt carries
the layout information that TI2V/TV2V runs get from the first frame.

| Artifact | Purpose |
| --- | --- |
| `src/wrbench/data/natural25/variants.jsonl` | Current deterministic toolkit `ti2v_prompt` rows (compatibility surface) |
| `src/wrbench/data/natural25/releases/paper_main_20260608/` | Frozen paper prompts, camera scope, TV2V sources, and HyDRA policy |
| `src/wrbench/data/natural25/t2v_layout_anchors.jsonl` | Text-only layout anchors for subject/interactor/open-surface/background facts |
| `src/wrbench/data/natural25/prompt_profiles/t2v_layout_anchor.json` | T2V prompt-profile policy; excludes T2I/TI2V style wording |
| `src/wrbench/data/natural25/camera_scopes/t2v_rotation_stress_30_60.json` | Formal T2V rotation-stress scope: `static`, `yaw30_LR`, `yaw30_RL`, `yaw60_LR`, `yaw60_RL` |
| `src/wrbench/data/natural25/variants.legacy_pronoun_20260620.jsonl` | Compatibility snapshot for released T2V-addendum metadata; not the paper prompt-of-record |
| `src/wrbench/data/results/wrbench_t2v_results.json` | T2V-only benchmark table (separate from the frozen 23-model main table) |
| `wrbench.t2v` | Intake acceptance gates + minWM rotation-step calibration checks |

Prompt-only T2V entries are tracked outside the frozen 23-model main table. Promote them through the dedicated T2V results surface rather than rewriting the released main-table artifacts.

Formal T2V intake checks (before promoting a model into `wrbench_t2v_results.json`):

1. `subject_present`, `scene_present`, `action_judgeable`, `camera_visible`
2. `camera_amplitude_ok` — minWM native-token yaw peaks match WRBench go-return targets (30°/60° via rotation-step patch metadata)

Use `wrbench.datasets.resolve_variant_prompt(variant, prompt_profile="t2v_layout_anchor")`
when materializing prompt files for prompt-only T2V models. Natural-language
prompts should not carry camera-control clauses for the rotation-stress scope;
camera control comes from the camera scope and the backend payload sidecar.

### HyDRA frame policy

The versioned paper contract distinguishes current evaluator policy from the
legacy provenance of one published D1 surface:

- Source decode reads frames 0 through 76 in order. Short inputs repeat the
  last frame; RGB is center-cropped/resized to 480x832.
- Camera matrices, not RGB frames, use stride-4 sampling to 20 embeddings.
- Submitted frames `[0,77)` are the source/condition segment and `[77,154)` are
  the generated continuation.
- Current D1 evaluation crops `[77,154)` before pose inference. D2-D6 score the
  full 154-frame submission.
- Current CamAlign uses the repaired generated-only path. The frozen CamPrec
  value retains legacy full-concat-pose then post-hoc-slice provenance and is
  not silently replaced in version 0.1.1.

The machine-readable source is
`src/wrbench/data/natural25/releases/paper_main_20260608/hydra_evaluation_policy.json`.
