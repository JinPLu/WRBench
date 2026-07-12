# Generation backends

`wrbench generate` compiles payloads and sidecars by default. Real generation is enabled only with `--no-dry-run` and a runtime configuration.

## Configure a backend

1. Copy [`wrbench.runtime.example.json`](../../wrbench.runtime.example.json) to an untracked `wrbench.runtime.json`.
2. Fill in the upstream Python interpreter, repository, checkpoint, and model-specific paths.
3. Check the model record and adapter with `wrbench doctor --model <model-key>`.
4. Run the same compile command with `--runtime-config` and `--no-dry-run`.

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

## Built-in execution modes

| Mode | Behavior |
| --- | --- |
| Compile-only (default) | Writes model-native controls and audit sidecars; no model environment is required |
| `local_subprocess` | Launches a supported upstream model in its configured environment |

Supported launchers and their required runtime fields are defined by the model records and [`wrbench.runtime.example.json`](../../wrbench.runtime.example.json). Run `wrbench models` for the current registry instead of relying on a duplicated model list here.

## Add backend support

1. Add or reuse a `GenerationBackend` under `src/wrbench/backends/`.
2. For subprocess execution, add a launcher under `src/wrbench/backends/launchers/`.
3. Connect it to the model's execution contract and runtime-config fields.
4. Add focused tests under `tests/test_backends*.py`.
5. Document only the new runtime fields in `wrbench.runtime.example.json`.

Model registry and adapter work is covered by [Adding a model](../adding-a-model.md).
