# Changelog

## Unreleased

### Added

- **External model specs and plugin discovery**: register models without editing WRBench source via `--model-spec`, `WRBENCH_MODEL_DIRS`, top-level `model_spec_files` in `wrbench.runtime.json`, `wrbench.load_model_spec_file` / `wrbench.register_model_record`, and Python entry points (`wrbench.plugins` and `wrbench.eval_plugins`).
- **Declarative and dynamic adapters**: built-in `declarative_pose`, `declarative_action`, and `prompt_camera` adapters configured in JSON specs, plus dynamic `module:callable` adapter resolution and extensible translation units (`none`, `custom:<name>`, `register_translation_unit`).
- **Command-template and pluggable generation backends**: invoke any local model or API wrapper via `backend: "command_template"` with compiled payload and trajectory placeholders, or register custom backends and launchers (`register_generation_backend`, `register_launcher`, or `module:callable`).
- **ReCamMaster video-to-video generation**: added the ReCamMaster adapter and subprocess launcher with compiled camera trajectories.
- **External video evaluation workflow**: added `wrbench tasks export` to write standardized Natural-25 task packages (with optional compiled camera targets) and `wrbench eval validate` to verify manifests, video files, and optional SHA-256 digests before scoring.
- **Plugin inspection and runtime diagnostics**: added `wrbench plugins` and extended `wrbench doctor --eval --runtime-config <path> [--json]` to check generation backends, evaluation scorer paths, custom backends, and Qwen/Transformers package versions.
- **Qwen3.5 Transformers boundary-token guard ([#11](https://github.com/JinPLu/WRBench/issues/11))**: detect and optionally align the extra `<|vision_start|>` and `<|vision_end|>` tokens retained by newer `transformers` releases via `QWEN35_VISION_BOUNDARY_MODE` (`warn`, `strip_outer_markers`, `preserve`), and record runtime package versions with scores.
- **Public paper prompt-mode alias**: added `--prompt-mode wrbench_paper` for the frozen D3–D6 paper probe prompt mode.
- **Repository templates**: added GitHub issue forms (bug report, D1–D6 reproduction, new model or leaderboard submission) and a pull request checklist.

### Changed

- **Paper evaluation defaults**: `wrbench eval run` and `wrbench eval pose` apply the paper defaults for VGGT-Omega (`d1_vggt_max_frames=49`, `512` resolution, `balanced` preprocessing), DINOv2 (`PAPER_SAMPLING`), Qwen D3–D6 environment settings, and `gpu_id: 0` automatically. Minimal `wrbench.runtime.json` files need only local interpreter, repository, and checkpoint paths, and stages overridden via `eval.backends` do not require built-in scorer paths.
- **Simplified `wrbench eval run` defaults**: `--scorer-profile` is optional (defaulting to `wrbench_default`, switching to `custom` when `eval.backends` overrides a stage, and accepting `default` as an alias), and `--sidecar-profile-gate` defaults to `main`.
- **Pure-Python D3–D6 orchestrator**: `wrbench.eval.d3d6_pipeline` executes all D3–D6 stages directly in Python without requiring `bash`.
- **Modular CLI layout**: split `wrbench.cli` into focused modules under `wrbench._cli/` while preserving existing CLI entry points and re-exports.
- **Custom-model table aggregation**: `wrbench.eval.aggregate.build_main_table` aggregates models absent from the built-in registry using manifest metadata or fallback labels.
- **Pluggable evaluators**: select custom `pose`, `d2`, and `vlm` backends in `eval.backends` under the same manifest and output contracts as built-in backends.
- Shared evaluation I/O and pose math, consolidated the result table builder, and removed obsolete scoring and table wrappers.

### Fixed

- **D2 visual integrity ([#10](https://github.com/JinPLu/WRBench/issues/10))**: use full-field DINOv2 with the paper's 3 fps / 24-frame preset, and verify the video SHA-256 digest when the manifest provides one.
- **D1 camera scoring ([#9](https://github.com/JinPLu/WRBench/issues/9))**: use CamAlign v2-lite, report common yaw across local and API models, keep API static hold and nominal target diagnostics separate, and record segment provenance with poses so published HyDRA clips work without camera sidecars.
- Replay paper CamPrec with `--target-mode requested_profile`, using the requested yaw or static target from the video manifest.
- Resolve relative paths across all stages of `eval run --video-root`, validate paper scorer settings, and bundle the D3–D6 runner in the package.
- **CLI `--no-dry-run` pre-check**: run backend availability checks for all `input_kind` values (`image`, `source_video`, `none`) and load `model_spec_files` before model lookup.
- **Open-source verification gate**: replaced the `rg`-dependent leak check in `scripts/oss_verify.sh` with a deterministic Python 3 scanner and isolated `PYTHONPATH` in standalone subprocess tests.

## 0.1.3 — 2026-07-13

### Changed

- Unified model input, source usage, and viewpoint-conditioning metadata in the model registry for all 23 paper models (including reference-only API models), and corrected LiveWorld and Spatia machine-readable results to report `TI2V` first-frame extraction.
- Removed duplicate legacy Natural-25 variants; `variants.jsonl` serves as the toolkit compatibility surface while frozen paper prompts live under the versioned release directory.
- Made the default D3–D6 scorer profile self-contained without runtime patching of historical prompt modules, switched single-GPU Qwen3.5 / Qwen3-VL loading to explicit CUDA placement, and aligned the example Qwen3-VL `max_new_tokens` setting with the scorer default (`768`).
- Streamlined newcomer documentation, restored the exact official Apache License 2.0 text, unified package/CLI version ownership, and wired CI to run `scripts/oss_verify.sh` on Python 3.12 with compatibility tests on Python 3.10–3.11.
- Updated the separately versioned Hugging Face rolling videos index (verified per-video D5/D6 exports and EasyAnimate D3–D6 alignment) and human-annotation release bundle while keeping the frozen 9,600-row paper table unchanged.

## 0.1.2 — 2026-07-12

### Added

- Bundled the 25-row first-frame generation catalog and SHA-256-bound mapping from every Natural-25 `family_id` to its PNG, plus cross-artifact release validation covering prompts, first frames, camera scope, and TV2V source-video mappings.

### Changed

- Updated first-frame provenance metadata to cite the exact generation prompt catalog rather than the toolkit `families.jsonl` compatibility surface.
- Clarified the boundary between the immutable 9,600-row paper table and separately versioned rolling dataset releases, including documented caveats for the frozen VerseCrafter row.

## 0.1.1 — 2026-07-12

### Added

- Immutable `paper_main_20260608` Natural-25 release contract (local and API prompt catalogs, frozen camera scope, 100-row Wan2.7 I2V source manifest, and HyDRA segment/evaluator provenance) with package loaders, validators, and resumable `scripts/prepare_paper_tv2v_sources.py`.
- Strict `task_variant_id` source-video task maps for Natural-25 generation.
- Separate prompt-only T2V intake surface: `t2v_layout_anchor` prompt profile, `t2v_rotation_stress_30_60` camera scope, `wrbench.t2v` acceptance helpers, and `wrbench_t2v_results.json`.

### Changed

- Distinguished `variants.jsonl` (active toolkit surface) from the versioned paper catalogs under `natural25/releases/paper_main_20260608/`, and classified Spatia and LiveWorld as first-frame `TI2V` wrappers rather than temporal `TV2V`.
- Added explicit `--source-video-task-map` / `--source-video-root` verification with provenance recording and rejected repeated-first-frame inputs for temporal source-video runs.
- Linked `README.md` to `docs/eval/README.md` for canonical evaluation policies and pointed per-model docs to `wrbench doctor --model <key>`.

## 0.1.0 — 2026-06-17

### Added

- D1 prompt-camera alignment (CamAlign) scorer (`wrbench eval d1-camalign`) and `D1_camalign` contract/table column.
- Backend dispatcher (`resolve_backend`) wired into `compile_camera` and CLI `--no-dry-run`, `LocalSubprocessBackend` with reference launchers for `easyanimate-v51-camera` and `spatia`, and `wrbench.runtime.example.json`.
- WRBench D1–D6 evaluation package (`wrbench eval`) with metric contract, scorers, and `wrbench eval run` one-command pipeline.
- Bundled Natural-25 prompts, first frames, and published 23-model results in `src/wrbench/data/`, backend docs under `docs/backends/`, `scripts/oss_verify.sh`, CI workflow, `CODE_OF_CONDUCT.md`, and `SECURITY.md`.

### Changed

- Always write `.payload.json` alongside sidecars, materialize `predict_v2v_control.py` for the EasyAnimate launcher, and resolve D3–D6 scorers from the installed `wrbench.eval.scoring` package layout.
- Adopted canonical WRBench paper terminology across public docs (diagnostic dimensions, viewpoint-conditioning paradigms, re-observation support).

### Verified

- Editable install and full pytest suite on Python 3.10–3.12; `wrbench doctor`, dry-run `wrbench generate`, and `wrbench eval contract` run out of the box without a tracked runtime config file.
