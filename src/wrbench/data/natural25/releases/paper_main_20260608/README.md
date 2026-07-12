# WRBench paper release `paper_main_20260608`

This directory is the immutable reproducibility contract for the frozen
23-model, 9,600-output WRBench paper table dated 2026-06-08. Within this frozen
historical attestation, the provenance repair changes neither the recorded
scores nor the attested video bytes.

The separate 11,100-row rolling public release may correct paper-associated
per-video metadata and may replace independently verified assets with newly
computed scores. Those rolling changes are versioned separately and do not
rewrite the frozen paper aggregates or present replacement assets as the
original paper bytes.

One confirmed correction is explicitly outside this frozen attestation: a
decoded-frame-0 lineage audit found 120 VerseCrafter TI2V30 rows (15 families
× 4 tiers × LR/RL at 30°) with an incorrect first frame. Their original bytes
and scores remain historical frozen provenance. Replacement assets, QC, and
new scores belong only to the separately tagged rolling release.

## Prompt surfaces

- `variants.local_ti2v_tv2v.jsonl` is the exact 400-row local-generation
  catalog. Local models used its 100 `oov_gap=none` content rows while camera
  control was supplied separately; the remaining 300 rows are retained because
  they are part of the exact historical catalog.
- `variants.api_source.jsonl` is the 400-row historical source catalog for API
  prompt-camera models. Provider-specific `prompt_to_send` materialization can
  differ from these source rows and is exact only where request evidence exists.
- `prompt_usage.json` is the tested 23-model catalog-selection contract.

The active top-level `natural25/variants.jsonl` remains available for toolkit
compatibility. It is a current deterministic toolkit surface, not the prompt
file of record for this frozen paper release.

## First-frame surface

`first_frame_generation_families.jsonl` is the exact 25-row catalog used to
generate the bundled Natural-25 first frames. The top-level
`first_frames_manifest.json` maps each `family_id` to that catalog prompt and
to the released PNG with separate prompt, catalog, and image SHA256 values.
The current top-level `families.jsonl` remains a toolkit surface and must not
be substituted as historical first-frame generation provenance.

## Camera and source-video surfaces

`camera_scope.json` freezes 6,400 local dual-angle outputs, 1,100 local static
outputs, and 2,100 API prompt-camera outputs. API yaw rows express prompt
intent and do not fabricate requested 30/60-degree targets or target C2W.
This 9,600-row paper surface supersedes the older 4,200-row type-specific
TI2V60/TV2V30 protocol as the paper-table contract.

`tv2v_sources.jsonl` maps each of the 100 `...__none` content-task IDs to one
`...__static` source-catalog row and one existing Wan2.7 I2V static conditioning
asset at an immutable dataset revision. These conditioning assets are not
additional benchmark outputs. Each row also preserves the exact provider
request sidecar prompt and its hash. Seventy-five requests are represented by
the frozen catalogs; the 25 `T2_div_a` requests predate both catalogs and are
explicitly labeled `exact_request_sidecar_not_represented_by_frozen_catalogs`.
The task-to-static ID mapping must not be interpreted as prompt-text equality.

**Maintainer rationale (not a comparative model claim):** development
screening found this Wan2.7 I2V static-camera pool's event completion and visual
quality sufficient for use as one consistent conditioning-input surface. This
selection does not establish Wan2.7 I2V superiority, and the 100 assets remain
conditioning inputs rather than additional benchmark outputs.

Use:

```bash
python scripts/prepare_paper_tv2v_sources.py \
  --source-video-root ./paper_tv2v_sources \
  --plan
```

Then rerun with `--download --resume`. The emitted
`source_video_task_map.json` is accepted by `scripts/run_natural25_generation.py`
through the explicit `--source-video-task-map` and `--source-video-root` flags.

The two local scope files are accepted directly by the runner. The dual-angle
scope expands one applicable model to 400 tasks (LR/RL at 30/60 degrees), and
the static scope expands one applicable model to 100 tasks. For example:

```bash
python scripts/run_natural25_generation.py \
  --model hydra \
  --out-dir ./paper_run \
  --variants src/wrbench/data/natural25/releases/paper_main_20260608/variants.local_ti2v_tv2v.jsonl \
  --camera-scope src/wrbench/data/natural25/releases/paper_main_20260608/camera_scopes/local_dual_angle.json \
  --prompt-profile ti2v_active \
  --source-video-task-map ./paper_tv2v_sources/source_video_task_map.json \
  --source-video-root ./paper_tv2v_sources \
  --dry-run --overwrite-existing --fail-fast \
  --limit 1 --shard-index 0 --num-shards 1
```

Remove `--limit 1` for the complete per-model scope. The runner rejects models
outside each scope's explicit model list. `api_prompt_camera.json` is
inspect-only: its API rows record prompt-camera intent without requested
degrees or target C2W, so the file is deliberately not accepted by
`--camera-scope`. Materialize those requests from `variants.api_source.jsonl`
subject to the provider-request evidence boundary above.

## HyDRA evaluation policy

`hydra_evaluation_policy.json` separates current evaluator policy from legacy
published-result provenance. The current D1 path crops the generated
continuation `[77,154)` before pose inference; D2-D6 use the full 154-frame
submission. The published CamPrec value retains legacy full-concat-pose then
post-hoc-slice provenance and is not silently replaced by this release.

See `release_manifest.json` for checksums, immutable revisions, known
limitations, and the non-circular publication handshake.
