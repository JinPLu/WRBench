# Evaluation

WRBench evaluates videos with the D1–D6 diagnostic contract from *Current World Models Lack a Persistent State Core* ([arXiv:2606.20545](https://arxiv.org/abs/2606.20545)). Generation is optional: evaluation accepts an existing video manifest.

## Metrics

| ID | Diagnostic | Output field |
| --- | --- | --- |
| D1-CamPrec | Requested-camera trajectory precision | `d1_camera_accuracy` |
| D1-CamAlign | Prompt-camera intent alignment | `d1_camalign_score` |
| D2 | Visual integrity | `d2_selected_visual_integrity_score` |
| D3 | Visible spatial consistency | `vlm_spatial_fidelity` |
| D4 | Visible state consistency | `vlm_state_fidelity` |
| D5 | Re-observation spatial consistency | `vlm_spatial_reasoning` |
| D6 | Re-observation event-state consistency | `vlm_state_reasoning` |

CamPrec and CamAlign are separate D1 diagnostics. CamPrec requires an explicit target trajectory; CamAlign tests camera intent for prompt/API controls.

D5 and D6 are scored only when the relevant out-of-view content returns into frame. Unsupported rows are excluded rather than treated as failures. This gate depends on the video, not on the model family.

The machine-readable and rendered contracts are in [`src/wrbench/eval/contract/`](../../src/wrbench/eval/contract/). Inspect the current contract without a runtime configuration:

```bash
wrbench eval contract
wrbench eval contract --json
```

## Configure scorers

Copy [`wrbench.runtime.example.json`](../../wrbench.runtime.example.json) to an untracked `wrbench.runtime.json`. Fill every required `eval.scorers` field explicitly, including the interpreter, checkpoint, repository, GPU, and environment entries used by VGGT-Omega, DINOv2, and Qwen.

Heavy scorer frameworks and weights are not pip dependencies. Each scorer runs through the interpreter configured in the runtime file.

## Run D1–D6

Published benchmark runs use the `wrbench_default` scorer profile:

```bash
wrbench eval --runtime-config wrbench.runtime.json run \
  --manifest videos.json \
  --out-dir eval_out \
  --scorer-profile wrbench_default \
  --sidecar-profile-gate main
```

The command runs pose recovery and D1, D2, D3–D6, then writes the aggregated table under `eval_out`. Use paths valid in the scorer environments; the runtime config does not infer checkpoints or upstream repositories.

## Run individual stages

The combined command is the normal entry point. Individual stages are available for resuming or debugging an existing run:

```bash
wrbench eval --runtime-config wrbench.runtime.json d1-vggt --help
wrbench eval d1 --help
wrbench eval d1-camalign --help
wrbench eval --runtime-config wrbench.runtime.json d2 --help
wrbench eval --runtime-config wrbench.runtime.json d3d6 --help
wrbench eval table --help
```

Each stage requires explicit input and output paths. Run its `--help` against the installed version instead of copying a release-specific command line.

## Prompt-only T2V runs

Models with `input_kind: none` do not use Natural-25 first frames. Materialize an explicit T2V prompt profile and camera scope; do not append camera wording to the Natural-25 scene prompt when the backend already receives camera controls.

Prompt-only results belong to [`wrbench_t2v_results.json`](../../src/wrbench/data/results/wrbench_t2v_results.json), not the frozen 23-model paper table. The bundled `t2v_layout_anchor` profile and `t2v_rotation_stress_30_60` scope define the public intake surface. D5/D6 remain null for a no-re-observation T2V scope.

## Paper-specific policies

Exact paper prompts, camera denominators, TV2V sources, first frames, and the HyDRA frame policy are versioned together in [`paper_main_20260608`](../../src/wrbench/data/natural25/releases/paper_main_20260608/README.md). They are provenance for that release, not defaults to duplicate in new evaluation code or documentation.

Reference aggregate tables and their column definitions are documented in [Published results](../../src/wrbench/data/results/README.md).
