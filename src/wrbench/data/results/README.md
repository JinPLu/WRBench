# Published results

This directory bundles the frozen WRBench reference tables for inspection and reproducibility. Run `wrbench eval run` on your own videos to generate fresh evaluation tables; do not append new rows to the frozen paper files.

| File | Purpose |
| --- | --- |
| `wrbench_23model_results.csv` | Frozen 23-model paper table (9,600 videos) |
| `wrbench_23model_results.json` | Machine-readable paper table with column metadata |
| `wrbench_t2v_results.json` | Separate prompt-only T2V addendum |

## Identity columns

| Column | Meaning |
| --- | --- |
| `model_id` | Canonical row identifier |
| `display_name` | Human-readable name |
| `viewpoint_condition_type` | How the requested viewpoint is supplied: `source-video`, `geometry-cache`, `model-inferred`, or `prompt-only` |
| `model_input` | Input modality: `T2V`, `TI2V`, or temporal `TV2V` |
| `paper_group` | Deprecated grouping retained only in the frozen paper artifact |
| `source_group` | Deprecated provenance bucket retained only in the frozen paper artifact |

`viewpoint_condition_type` and `model_input` capture independent axes. A wrapper that extracts only frame 0 is `TI2V` even if its runtime CLI accepts a source-video path. New result tables derive both fields from the model registry (or manifest metadata for custom models).

The remaining columns report D1-CamPrec, D1-CamAlign, D2–D6, sample counts, re-observation gate rate, low-count flags, metric notes, and aggregates. See [the evaluation guide](../../../../docs/eval/README.md) for metric definitions.

## Frozen provenance

The exact prompts, first frames, camera scopes, TV2V sources, and HyDRA frame policy for the 23-model paper table live in [`paper_main_20260608`](../natural25/releases/paper_main_20260608/README.md). Rolling dataset updates are versioned separately on Hugging Face without modifying the frozen paper tables.

Prompt-only T2V models use their own prompt profile, camera scope, and result file (`wrbench_t2v_results.json`) because they do not consume a first frame or temporal source video. Public downloads and the interactive leaderboard are linked from the repository [README](../../../../README.md).
