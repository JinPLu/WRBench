# Changelog

## 0.1.0 — 2026-06-17

### Added

- D1 prompt-camera alignment (CamAlign) scorer: `wrcam eval d1-camalign` and `D1_camalign` contract/table column.
- Backend dispatcher (`resolve_backend`) wired into `compile_camera` and CLI `--no-dry-run`.
- `LocalSubprocessBackend` with reference launchers for `easyanimate-v51-camera` and `spatia`.
- `wrcam.runtime.example.json` runtime configuration schema.
- WRBench D1–D6 evaluation package (`wrcam eval`) with metric contract, scorers, and `wrcam eval run` one-command pipeline.
- Bundled Natural-25 prompts and published 23-model results in `src/wrcam/data/` (install-safe via package data).
- Backend docs under `docs/backends/`.
- Open-source verification script `scripts/oss_verify.sh`.
- CI workflow, `CODE_OF_CONDUCT.md`, and `SECURITY.md`.

### Changed

- `.payload.json` is always written alongside sidecars (not dry-run only).
- EasyAnimate launcher materializes `predict_v2v_control.py` (default patching) instead of passing ineffective CLI flags.
- Public docs adopt WRBench paper terminology (diagnostic dimensions, viewpoint condition types, re-observation support).
- D3–D6 overlay scorers resolve the installed `wrcam.eval.scoring` package layout (no legacy metric tree required).

### Verified

- Editable install + full pytest suite on Python 3.10–3.12.
- `wrcam doctor`, dry-run `wrcam generate`, and `wrcam eval contract` work with zero internal runtime config.
