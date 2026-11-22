# Evaluation

WRBench evaluates videos with the D1–D6 diagnostic contract from *Current World Models Lack a Persistent State Core* ([arXiv:2606.20545](https://arxiv.org/abs/2606.20545)). Generation is optional: evaluation accepts any existing video manifest, including externally generated video sets.

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

CamPrec compares recovered motion with the requested trajectory. For custom evaluations, the default reads a validated target from the manifest or a sidecar. The frozen paper evaluation builds the target from the camera request profile. CamAlign v2-lite measures common yaw across local and API models, with API static hold reported separately. Yaw uses the row's 30° or 60° target when available, or a 30° reference angle for prompt-only controls.

D5 and D6 are scored only when the video provides judgeable return evidence. The manifest's `oov_gap` describes the content prompt, so even `"none"` can yield re-observation when the camera moves. To explicitly omit D5/D6 for a separate evaluation track, set `"reobservation_enabled": false` on its rows.

The machine-readable and rendered contracts are in [`src/wrbench/eval/contract/`](../../src/wrbench/eval/contract/). Inspect the current contract without a runtime configuration:

```bash
wrbench eval contract
wrbench eval contract --json
```

## Configure scorers (minimal runtime config)

Heavy scorer frameworks and weights are not pip dependencies. Each scorer runs through the interpreter configured in an untracked `wrbench.runtime.json`.

WRBench applies the canonical paper defaults for VGGT-Omega (`49`-frame sampling, `512` resolution, `balanced` preprocessing), DINOv2 (`PAPER_SAMPLING`), Qwen D3–D6 environment flags, and `gpu_id: 0`, so a minimal `wrbench.runtime.json` needs only your local interpreter, repository, and checkpoint paths:

```json
{
  "schema_version": 1,
  "eval": {
    "scorers": {
      "vggt_python_bin": "/path/to/vggt-venv/bin/python",
      "vggt_repo": "/path/to/vggt-omega",
      "vggt_checkpoint": "/path/to/vggt_omega_1b_512.pt",
      "dinov2_python_bin": "/path/to/dinov2-venv/bin/python",
      "dinov2_model_path": "/path/to/dinov2-large",
      "qwen_scorer_python": "/path/to/qwen-venv/bin/python",
      "qwen35_model": "/path/to/Qwen3.5-9B",
      "qwen3vl_model": "/path/to/Qwen3-VL-8B-Instruct"
    }
  }
}
```

You can still override any `extra_paths` or `env` setting explicitly (see [`wrbench.runtime.example.json`](../../wrbench.runtime.example.json)). By default `wrbench eval run` uses `wrbench_default` (validating explicit overrides against the frozen paper protocol) and automatically switches to `custom` when `eval.backends` overrides a stage; stages replaced via `eval.backends` do not require built-in scorer paths. In container images where a system `LD_LIBRARY_PATH` (such as `/usr/local/cuda/lib64`) shadows PyTorch's bundled CUDA libraries (`CUBLAS_STATUS_INVALID_VALUE`), clear it via `"env": {"LD_LIBRARY_PATH": ""}` in `eval.scorers`.

Run preflight diagnostics before launching a scoring run:

```bash
wrbench doctor --eval --runtime-config wrbench.runtime.json
```

`wrbench doctor --eval` checks scorer path existence, custom backend importability, and probes `qwen_scorer_python` for `transformers` / `torch` / `decord` versions (warning on Qwen3.5 processor drift — see [Issue #11 boundary-token guard](#qwen35-transformers-boundary-token-guard-issue-11) below).

## Validate a manifest and run D1–D6

Validate your manifest and video files first:

```bash
wrbench eval validate --manifest videos.json --video-root /path/to/videos
```

Then run the full pipeline (`D1 pose -> D1 score -> D2 -> D3-D6 -> main table`). The D3–D6 orchestrator runs directly in Python (no `bash` required):

```bash
wrbench eval --runtime-config wrbench.runtime.json run \
  --manifest videos.json \
  --video-root /path/to/videos \
  --out-dir eval_out \
  --target-mode requested_profile
```

A manifest can be a JSON array or JSON Lines. For your own videos, give each row a unique ID, a model label, its camera intent, and the event to score:

```jsonl
{"video_id":"mug-001","model":"my_model","video_path":"mug.mp4","camera_type":"yaw_RL","profile":"TI2V60","world_state_prompt":"A person moves the mug from the counter to the table."}
```

Third-party/custom model names not in the built-in registry aggregate cleanly into `eval_out/main_table.csv` (using `viewpoint_condition_type` / `model_input` from manifest rows when provided, or falling back to `custom` / `unknown`).

Use `--video-root /path/to/mirror` to resolve relative video and trajectory paths for all stages. For your own target matrices, omit `--target-mode` and supply a validated manifest target or sidecar. Use `--target-mode nominal_profile` to add a separate `D1_camera_pose_nominal` diagnostic for rows with a known request profile and no supplied target. An invalid supplied target is reported as an error in the score record. Prompt-only rows use CamAlign.

## Qwen3.5 Transformers boundary-token guard (Issue #11)

Newer `transformers` releases (e.g. `5.22.1` vs historical `5.9.0`) retain two extra boundary tokens (`<|vision_start|>` and `<|vision_end|>`) around expanded video pads in the Qwen3.5 processor, shifting multimodal RoPE positions by 2 and lowering D3/D4 scores. See the full root-cause trace in [Hunyuan WorldPlay D3 and LiveWorld D4 replays](hunyuan-d3-replay.md).

WRBench controls this behavior via `QWEN35_VISION_BOUNDARY_MODE` (set in `eval.scorers.env` or the process environment):

| Value | Behavior |
| --- | --- |
| `warn` (default) | Detects extra outer vision boundary tokens (or non-`5.9.x` `transformers`), logs a one-time actionable warning, and preserves processor outputs |
| `strip_outer_markers` | Strips the redundant outer `<|vision_start|>` and `<|vision_end|>` positions from `input_ids`, `attention_mask`, and `mm_token_type_ids` to align with `transformers==5.9.0`, recording `vision_boundary_aligned: true` in metadata |
| `preserve` | Keeps raw processor outputs silently without warning or stripping |

For strict historical replay, prefer pinning `transformers==5.9.0` or setting `QWEN35_VISION_BOUNDARY_MODE=strip_outer_markers` and recording your package versions alongside the score export.

### Reproduce a released video

Select a row from `index/videos_master.jsonl` and mirror its MP4 at the row's relative path. Keep the index's `world_state_prompt` when rescoring. Filter `benchmark_date == "2026-06-08"` for the 9,600 paper rows. The [release metadata](../../src/wrbench/data/natural25/releases/paper_main_20260608/release_index.json) pins the frozen videos. The [updated per-video score index](https://huggingface.co/datasets/WRBench/wrbench-videos/blob/6257b7e665cdd4ff89af49dcee07e7524f154bfd/index/videos_master.jsonl) includes the static CamPrec scores and the corrected D5/D6 exports.

For paper CamPrec, use `--target-mode requested_profile`. This mode builds the OpenCV C2W target from the row's yaw request or uses identity for a static request. The updated index covers all 7,500 eligible local rows, including 1,100 static videos verified by fresh pose recovery. Static CamPrec passes when the estimated camera stays within 2° of its initial rotation and 0.05 pose units of its initial position. Both `wrbench eval run` and `wrbench eval pose` apply the paper's `49`-frame VGGT-Omega sampling (`512` resolution, `balanced` preprocessing) automatically. The replay used VGGT-Omega 1B-512 at [upstream commit `39a0cb8`](https://github.com/facebookresearch/vggt-omega/commit/39a0cb8af88554f15ddcb5354cd52bde588fa014) and a checkpoint with SHA-256 `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`.

```bash
wrbench eval d1 --input-jsonl sample.jsonl --pose-cache-root pose_cache \
  --target-mode requested_profile --output-jsonl camprec.jsonl --summary-csv camprec.csv
```

For visual integrity alone, run `wrbench eval --runtime-config wrbench.runtime.json d2 --videos-manifest sample.jsonl --video-root /path/to/mirror --out-jsonl d2.jsonl`. D2 verifies the MP4 when the row provides a `sha256`. The paper preset uses 3 fps, retains both endpoints, and uses 4–24 frames and batches of 4. DINOv2-Large resizes the short edge to 256 with bicubic interpolation and preserves aspect ratio. The scorer normalizes RGB before zero-padding the bottom and right to the patch grid and masks padding-only patches during local matching. Intentional overrides are available with a custom profile or the standalone command and are recorded with the scores.

For HyDRA's published 154-frame full-concatenation clips, `eval pose` automatically recovers motion from the generated continuation (`[77:154]`, zero-based) and records segment provenance alongside the poses, while D2–D6 use all 154 frames. For the frozen HyDRA CamPrec protocol, run `python -m wrbench.eval.d1.run_vggt_omega_pose --help` to extract poses from the full MP4 into a separate cache, then run `eval d1 --target-mode requested_profile` against the original row; D1 selects the last 77 poses from that cache. See the [release's HyDRA policy](../../src/wrbench/data/natural25/releases/paper_main_20260608/hydra_evaluation_policy.json).

## Run individual stages

The combined command is the normal entry point. Individual stages are available for resuming or debugging an existing run:

```bash
wrbench eval validate --help
wrbench eval --runtime-config wrbench.runtime.json pose --help
wrbench eval d1 --help
wrbench eval d1-camalign --help
wrbench eval --runtime-config wrbench.runtime.json d2 --help
wrbench eval --runtime-config wrbench.runtime.json d3d6 --help
wrbench eval table --help
```

Each stage requires explicit input and output paths. Run its `--help` against the installed version instead of copying a release-specific command line.

## Use another evaluator

Set any of `pose`, `d2`, or `vlm` under `eval.backends` in the runtime file. Omitted stages use the bundled backend, and replaced stages do not require built-in scorer paths. Values can be registered names (including plugins auto-discovered via the `wrbench.eval_plugins` entry-point group) or Python references in `module:callable` form. Install the plugin in the environment that runs `wrbench`; `wrbench eval run` automatically selects the `custom` scorer profile whenever `eval.backends` overrides a stage.

```json
"backends": {
  "pose": "my_package.evaluation:recover_poses"
}
```

Each backend is a callable that returns a process-style exit status. It receives `eval_runtime` plus the stage arguments:

| Stage | Arguments | Output |
| --- | --- | --- |
| Pose | `input_jsonl`, `output_root`, `cache_root`, `execution_mode`, `video_root` | OpenCV C2W matrices at `cache_root/pose/<video_id>/poses.npy` |
| D2 | `videos_manifest`, `out_jsonl`, `model_dir`, `video_root` | One row per video with the D2 field in the metric contract |
| VLM | `manifest`, `out_dir`, `scores_path`, `stage`, `scorer_profile` | Final JSON array at `scores_path` for `stage="all"`. Include the D3–D6 fields and explicit shared re-observation applicability. |

Keep the input `video_id` on every output row. Use `wrbench plugins` (or `wrbench eval plugins`) to inspect registered names. For a new concatenated format, set `scoring_video_surface: "source_generated_concat"`, `condition_frame_count`, and `generated_frame_count` on the manifest row. A Python integration can also register a reusable `ConcatSegmentPolicy` for its models.

## Prompt-only T2V runs

Models with `input_kind: none` do not use Natural-25 first frames. Materialize an explicit T2V prompt profile and camera scope; do not append camera wording to the Natural-25 scene prompt when the backend already receives camera controls.

Prompt-only results belong to [`wrbench_t2v_results.json`](../../src/wrbench/data/results/wrbench_t2v_results.json). The bundled `t2v_layout_anchor` profile and `t2v_rotation_stress_30_60` scope define the intake surface for this separate track. Use `"reobservation_enabled": false` when preparing its no-re-observation scoring manifest.

## Paper-specific policies

Exact paper prompts, camera denominators, TV2V sources, first frames, and the HyDRA frame policy are versioned together in [`paper_main_20260608`](../../src/wrbench/data/natural25/releases/paper_main_20260608/README.md). They are provenance for that release, not defaults to duplicate in new evaluation code or documentation.

Reference aggregate tables and their column definitions are documented in [Published results](../../src/wrbench/data/results/README.md).
