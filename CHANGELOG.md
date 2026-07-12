# Changelog

## 0.1.3 — 2026-07-13

### Changed

- Model input, source usage, and viewpoint-condition metadata now come from the
  model registry for all 23 paper rows, including reference-only API models.
- LiveWorld and Spatia machine-readable results now correctly report TI2V
  first-frame extraction rather than temporal TV2V.
- Removed the byte-identical legacy Natural-25 variants copy; `variants.jsonl`
  remains the only toolkit compatibility surface, while paper prompts stay in
  the versioned release directory.
- The current P25/P22 scorer profile is self-contained and no longer patches
  historical prompt modules at runtime.
- Single-GPU Qwen3.5 and Qwen3-VL loading no longer requires Accelerate through
  an implicit `device_map`; runtime metadata records explicit CUDA placement.
- The example Qwen3-VL output limit now matches the scorer default (`768`) so
  structured evidence is not truncated by the published configuration.
- Newcomer documentation now follows one install, compile, evaluate, and
  optional-generate path instead of duplicating release and result details.
- Restored the unmodified official Apache License 2.0 text and advanced the
  package to 0.1.3 with CLI and build metadata sharing one version owner.
- CI now runs the canonical OSS release gate on Python 3.12 while retaining
  focused compatibility tests on Python 3.10 and 3.11.
- The rolling videos index now carries the verified per-video D5/D6 values
  that reproduce the frozen aggregates; video bytes and paper totals are
  unchanged, and the release index records the immutable Hub tag.
- EasyAnimate's 500 rolling rows now use the same current D3-D6 scoring
  snapshot as the aggregate results; 59 index rows changed, while videos,
  prompts, applicability masks, and the frozen paper table remain unchanged.
- The human-annotation release now bundles the 106 exact reviewed videos needed
  to resolve all 1,170 endpoints in the current recheck.

## 0.1.2 — 2026-07-12

### Added

- The exact 25-row first-frame generation catalog and SHA256-bound mapping
  from every Natural-25 `family_id` to its bundled PNG.
- Validation that the frozen prompt catalogs, first-frame catalog and bytes,
  camera scope, and TV2V source-video map form one self-consistent release.

### Changed

- First-frame metadata now cites the actual generation prompt catalog instead
  of the current toolkit `families.jsonl` compatibility surface.
- Release documentation distinguishes the immutable 9,600-row paper table
  from corrected or newly added assets in the 11,100-row rolling dataset.
- The frozen VerseCrafter row now discloses the 120 TI2V30 assets identified by
  decoded-frame-0 lineage audit for replacement and rescoring on rolling v2.
- Both issue-4 preparation branches are recorded as ancestors of `main`;
  their already-landed content was not duplicated.

## 0.1.1 — 2026-07-12

### Added

- Immutable `paper_main_20260608` Natural-25 release contract with the exact
  local-generation catalog and historical API source catalog, frozen camera
  scope, a 100-row Wan2.7 I2V source manifest, and explicit HyDRA
  segment/evaluator provenance.
- Package loaders and validation for named Natural-25 paper releases.
- Resumable `scripts/prepare_paper_tv2v_sources.py` source preparation with
  plan, download, verify-only, SHA256 verification, and overwrite refusal.
- Strict `task_variant_id` source-video task maps for Natural-25 generation.
- Separate Natural-25 T2V prompt profile for prompt-only models that do not
  receive a first-frame image.
- Legacy pronoun-anchored prompt snapshot:
  `src/wrbench/data/natural25/variants.legacy_pronoun_20260620.jsonl`.
- T2V rotation-stress camera scope:
  `src/wrbench/data/natural25/camera_scopes/t2v_rotation_stress_30_60.json`.
- T2V intake acceptance helpers (`wrbench.t2v`) for subject/scene/action/camera
  gates and minWM rotation-step calibration metadata.
- T2V results placeholder table:
  `src/wrbench/data/results/wrbench_t2v_results.json`.

### Changed

- `variants.jsonl` remains available as the active deterministic toolkit
  surface, while the local prompt-of-record and bounded historical API source
  catalog live under `natural25/releases/paper_main_20260608/`.
- Natural-25 generation accepts explicit `--source-video-task-map` and
  `--source-video-root`, verifies each source once across camera cells, rejects
  repeated-first-frame benchmark inputs, and records source provenance in the
  generated row and camera sidecar.
- Spatia and LiveWorld are classified as first-frame TI2V wrappers rather than
  temporal TV2V; temporal source-video use remains explicit for Gen3C, HyDRA,
  InSpatio World, and ReCamMaster.
- Prompt-only T2V models use the separate `t2v_layout_anchor` prompt profile
  and T2V event tails instead of mutating the compatibility prompt file.
- Published 23-model results (`wrbench_23model_results.*`) now resolve prompt
  provenance through the versioned local/API paper catalogs; the legacy
  pronoun snapshot remains T2V-addendum compatibility metadata only.
- README now links to `docs/eval/README.md` as the canonical public policy for
  re-observation scoring and prompt-only T2V scope; per-model pages point users
  to `wrbench doctor --model ...` instead of stale backend-status boilerplate.

## 0.1.0 — 2026-06-17

### Added

- D1 prompt-camera alignment (CamAlign) scorer: `wrbench eval d1-camalign` and `D1_camalign` contract/table column.
- Backend dispatcher (`resolve_backend`) wired into `compile_camera` and CLI `--no-dry-run`.
- `LocalSubprocessBackend` with reference launchers for `easyanimate-v51-camera` and `spatia`.
- `wrbench.runtime.example.json` runtime configuration schema.
- WRBench D1–D6 evaluation package (`wrbench eval`) with metric contract, scorers, and `wrbench eval run` one-command pipeline.
- Bundled Natural-25 prompts and published 23-model results in `src/wrbench/data/` (install-safe via package data).
- Backend docs under `docs/backends/`.
- Open-source verification script `scripts/oss_verify.sh`.
- CI workflow, `CODE_OF_CONDUCT.md`, and `SECURITY.md`.

### Changed

- `.payload.json` is always written alongside sidecars (not dry-run only).
- EasyAnimate launcher materializes `predict_v2v_control.py` instead of passing ineffective CLI flags.
- Public docs adopt WRBench paper terminology (diagnostic dimensions, viewpoint condition types, re-observation support).
- D3–D6 overlay scorers resolve the installed `wrbench.eval.scoring` package layout (no legacy metric tree required).

### Verified

- Editable install + full pytest suite on Python 3.10–3.12.
- `wrbench doctor`, dry-run `wrbench generate`, and `wrbench eval contract` work without a tracked runtime config file.
