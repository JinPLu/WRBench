# Published results

This directory contains reference results, not inputs to a required generation workflow.

| File | Purpose |
| --- | --- |
| `wrbench_23model_results.csv` | Frozen 23-model paper table |
| `wrbench_23model_results.json` | The same table with machine-readable column metadata |
| `wrbench_t2v_results.json` | Separate prompt-only T2V addendum |

Run `wrbench eval run` on your own videos to produce fresh scores. Do not append new rows to the frozen paper table.

## Identity columns

| Column | Meaning |
| --- | --- |
| `model_id` | Canonical row identifier |
| `display_name` | Human-readable name |
| `viewpoint_condition_type` | How the requested viewpoint is supplied: `source-video`, `geometry-cache`, `model-inferred`, or `prompt-only` |
| `model_input` | Input modality: `T2V`, `TI2V`, or temporal `TV2V` |
| `paper_group` | Deprecated grouping retained only in the frozen artifact |
| `source_group` | Deprecated provenance bucket retained only in the frozen artifact |

`viewpoint_condition_type` and `model_input` describe different things. A source wrapper that only extracts frame 0 is `TI2V`; `input_kind: source_video` alone does not make a model temporal TV2V. New tables should derive both fields from the model registry rather than maintain a second mapping.

The remaining columns are D1-CamPrec, D1-CamAlign, D2–D6, their sample counts, re-observation gate rate, low-count flags, metric notes, and the published aggregate. See [the evaluation guide](../../../../docs/eval/README.md) for metric semantics.

## Frozen provenance

The prompt, first-frame, camera-scope, TV2V-source, and HyDRA policies for the paper table are owned by the versioned [`paper_main_20260608`](../natural25/releases/paper_main_20260608/README.md) contract. The top-level Natural-25 variants are a current toolkit surface and must not be substituted for the frozen prompt-of-record.

The paper aggregate and its 9,600-row attestation stay frozen. Corrected assets or per-video metadata must be published on a separately versioned rolling surface; they do not silently replace the paper bytes or scores.

Prompt-only T2V models use their own prompt profile, camera scope, and result file because they do not consume a first frame or temporal source video. They are not directly inserted into the frozen 23-model table.

Public downloads and the leaderboard are linked once from the repository [README](../../../../README.md).
