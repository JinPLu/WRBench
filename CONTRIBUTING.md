# Contributing

Contributions to WRBench are welcome. Most new models and evaluators can be integrated without editing core `wrbench` source files.

## Development setup

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python3 -m pytest -q
```

Running `python3 -m pytest -q` also works directly from a source checkout.

## Extending WRBench with plugins

You do not need to fork or edit `src/wrbench/` to add a model, generation backend, or custom evaluator:

1. **External model spec and command template:** write a standalone JSON spec using a built-in declarative adapter (`declarative_pose`, `declarative_action`, or `prompt_camera`), pair it with `"backend": "command_template"` in `wrbench.runtime.json`, and load it via `--model-spec <spec.json>` or `WRBENCH_MODEL_DIRS`.
2. **Python plugin packages (`entry_points` / `module:callable`):** expose `wrbench.plugins` (for model records, adapters, generation backends, or launchers) or `wrbench.eval_plugins` (for `pose`, `d2`, `vlm`, or `segment_policy` evaluators) in `pyproject.toml`, or reference `module:callable` directly in model specs and `eval.backends`.

See [Integrating models and evaluators](docs/plugins.md), [Adding a built-in model](docs/adding-a-model.md), and [Generation backends](docs/backends/README.md) for examples, and inspect loaded extensions with:

```bash
wrbench plugins
wrbench doctor --model <model-key>
```

## Evaluating externally generated videos

To evaluate a model using your own inference pipeline:

```bash
wrbench tasks export --out-dir tasks_out --model <model-key>
# Generate videos externally and write a manifest JSON/JSONL
wrbench eval validate --manifest videos.jsonl
wrbench eval --runtime-config wrbench.runtime.json run \
  --manifest videos.jsonl \
  --out-dir eval_out \
  --target-mode requested_profile
```

See [Evaluation](docs/eval/README.md) for minimal `wrbench.runtime.json` scorer configuration and `wrbench doctor --eval --runtime-config wrbench.runtime.json`.

## Open-source verification gate

Before opening a pull request or cutting a release, run:

```bash
bash scripts/oss_verify.sh
```

This runs the Python leak and legacy-jargon scanner, creates a clean virtual environment, installs `.[dev]`, runs the full test suite, and verifies the bundled Natural-25 and paper-release contracts.

## License

Contributions are accepted under the Apache-2.0 license.
