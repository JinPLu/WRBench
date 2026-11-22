# Generation backends

`wrbench generate` compiles payloads and sidecars by default. Real generation is enabled only with `--no-dry-run` and a runtime configuration.

## Built-in execution modes

| Mode | Behavior |
| --- | --- |
| `dry_run` (default) | Compiles model-native controls and audit sidecars; no model environment is required |
| `local_subprocess` | Launches a built-in upstream model via a registered launcher in its configured environment |
| `command_template` | Runs any user-defined shell command array with compiled payload/trajectory placeholders |
| Custom (`module:callable` or registered name) | Dispatches to a Python `GenerationBackend` registered via `register_generation_backend(...)` or `wrbench.plugins` entry points |

Inspect active backends, launchers, and adapters with `wrbench plugins`.

## Configure a built-in backend

1. Copy [`wrbench.runtime.example.json`](../../wrbench.runtime.example.json) to an untracked `wrbench.runtime.json`.
2. Fill in the upstream Python interpreter, repository, checkpoint, and model-specific paths.
3. Validate the model record, adapter, and backend readiness:

```bash
wrbench doctor --model easyanimate-v51-camera --runtime-config /path/to/wrbench.runtime.json
```

4. Run the compile command with `--runtime-config` and `--no-dry-run`:

```bash
IMAGE="$(python - <<'PY'
from wrbench.datasets import natural25_first_frame_path
print(natural25_first_frame_path("bedroom_cat_bed_jump"))
PY
)"

wrbench generate \
  --model easyanimate-v51-camera \
  --camera preset:yaw_LR \
  --image "$IMAGE" \
  --prompt "A living room." \
  --out out.mp4 \
  --runtime-config /path/to/wrbench.runtime.json \
  --no-dry-run
```

The runtime file is the only place for machine-specific paths. Do not add checkpoints, credentials, or upstream repository paths to model JSON files.

## Command-template generation

To run a local model or API script without writing a Python launcher, configure `"backend": "command_template"` (or provide `"command_template"`) in `wrbench.runtime.json`:

```json
{
  "schema_version": 1,
  "model_spec_files": ["./my_custom_model.json"],
  "models": {
    "my-custom-model": {
      "backend": "command_template",
      "python_bin": "/path/to/venv/bin/python",
      "repo_root": "/path/to/my-model-repo",
      "model_path": "/path/to/checkpoint",
      "gpu_id": 0,
      "command_template": [
        "{python_bin}",
        "{repo_root}/generate.py",
        "--checkpoint", "{model_path}",
        "--image", "{image_path}",
        "--target-c2w", "{target_c2w_path}",
        "--payload", "{payload_json_path}",
        "--prompt", "{prompt}",
        "--out", "{output_path}"
      ],
      "extra_paths": {},
      "env": {}
    }
  }
}
```

Available template placeholders:

- Runtime fields: `{python_bin}`, `{repo_root}`, `{model_path}`, `{gpu_id}`, plus any key in `extra_paths`
- Request fields: `{model}`, `{prompt}`, `{prompt_to_send}`, `{output_path}`, `{image_path}`, `{source_video_path}`, `{work_dir}`
- Compiled sidecars: `{payload_json_path}`, `{target_c2w_path}`, `{camera_trajectory_json_path}`, `{camera_sidecar_path}`, `{pose_file}`

When `backend` is explicitly `"command_template"` or a custom `module:callable` backend, `python_bin`, `repo_root`, and `gpu_id` are optional and fall back to lightweight defaults (`sys.executable`, `"."`, `0`).

## Add a Python backend or launcher

- **External plugin (no WRBench edits):** set `"backend": "my_pkg.backend:MyBackend"` in `wrbench.runtime.json`, or register via `wrbench.backends.registry.register_generation_backend(...)` / `wrbench.backends.launchers.register_launcher(...)` inside a `wrbench.plugins` entry point. See [Integrating models and evaluators](../plugins.md).
- **Built-in launcher (upstream PR):** add a `LaunchSpec` under `src/wrbench/backends/launchers/`, add focused tests under `tests/test_backends*.py`, and document any new runtime fields in `wrbench.runtime.example.json`.

Model registry and adapter work is covered by [Adding a built-in model](../adding-a-model.md).
