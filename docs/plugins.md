# Integrating models and evaluators

WRBench supports evaluating and integrating external video world models without modifying WRBench source files. Choose the integration level that matches your workflow:

1. **[Evaluate existing videos](#1-evaluate-existing-videos)** — export standardized tasks, generate MP4s with your own scripts, validate the manifest, and run D1–D6.
2. **[Declarative model specs and command templates](#2-declarative-model-specs-and-command-templates)** — describe a model in an external JSON spec (`--model-spec`) and invoke inference via `command_template`.
3. **[Python plugins](#3-python-plugins)** — register custom adapters, generation backends, launchers, or evaluators via `module:callable` references or `pyproject.toml` entry points.

Inspect the models, adapters, backends, launchers, and evaluators currently loaded in your environment at any time:

```bash
wrbench plugins
wrbench plugins --json
```

---

## 1. Evaluate existing videos

If you already have an inference pipeline, you do not need to write an adapter or backend.

### Step 1: Export Natural-25 tasks

```bash
wrbench tasks export \
  --out-dir ./tasks_out \
  --cameras static,yaw_LR,yaw_RL \
  --json
```

Add `--model <model-key>` (with or without `--model-spec <spec.json>`) to also compile model-native camera targets and payload sidecars under `./tasks_out/compiled/`. Use `--limit N` for a quick subset or `--no-compile-targets` to export prompts and first frames only.

### Step 2: Generate videos and write a manifest

Generate one MP4 per task row using your own pipeline and write a JSON or JSONL manifest (`submission.jsonl`):

```jsonl
{"video_id":"bedroom_cat_bed_jump__T0__yaw_LR","model":"my-world-model","video_path":"videos/cat_yaw_lr.mp4","camera_type":"yaw_LR","profile":"TI2V60","world_state_prompt":"A house cat waits on the bedroom floor, facing the bed, then jumps onto the bed."}
```

### Step 3: Validate and evaluate

```bash
wrbench eval validate --manifest submission.jsonl --video-root ./my_videos --json
wrbench doctor --eval --runtime-config wrbench.runtime.json
wrbench eval --runtime-config wrbench.runtime.json run \
  --manifest submission.jsonl \
  --video-root ./my_videos \
  --out-dir ./eval_out \
  --target-mode requested_profile
```

Model names not present in the built-in registry aggregate cleanly into `eval_out/main_table.csv` without requiring a registry file.

---

## 2. Declarative model specs and command templates

Pass one or more external JSON model specs with `--model-spec <path.json>` (repeatable), set the `WRBENCH_MODEL_DIRS` environment variable (`os.pathsep`-separated directories of `*.json` files), or list `"model_spec_files": ["./my_model.json"]` at the top level of `wrbench.runtime.json`.

WRBench ships three built-in declarative adapters covering the main camera-control paradigms.

### Paradigm 1: Pose Matrix (`declarative_pose`)

Converts canonical OpenCV C2W trajectories into `.npy`, `.npz`, `.json`, `.txt_3x4`, or `.txt_4x4` pose files (`c2w` or `w2c`, `opencv` or `opengl`/`blender`):

```json
{
  "key": "my-pose-model",
  "aliases": ["my-pose"],
  "status": "active",
  "input_kind": "image",
  "model_input": "TI2V",
  "viewpoint_condition_type": "model-inferred",
  "adapter": "declarative_pose",
  "payload_type": "declarative_pose_npy",
  "default_width": 832,
  "default_height": 480,
  "default_fps": 16,
  "default_frames": 49,
  "amplitude": {
    "rotation_gain": 1.0,
    "translation_gain": 1.0,
    "max_amount": 1.0,
    "translation_unit": "canonical_scene",
    "calibration_status": "initial_manual",
    "metadata": {}
  },
  "capabilities": {
    "rotation": true,
    "translation": true,
    "supports_static": true
  },
  "execution_contract": {
    "pose_type": "c2w",
    "convention": "opencv",
    "format": "npy"
  },
  "notes": "External pose-conditioned TI2V model."
}
```

### Paradigm 2: Action Timeline (`declarative_action`)

Maps canonical camera segments (`static`, `yaw_LR`, `yaw_RL`, `pan_LR`, `pan_RL`, or frame-level yaw/pan scripts) into discrete action tokens or numeric action vectors:

```json
{
  "key": "my-action-model",
  "status": "active",
  "input_kind": "image",
  "model_input": "TI2V",
  "viewpoint_condition_type": "model-inferred",
  "adapter": "declarative_action",
  "payload_type": "declarative_action_timeline",
  "default_width": 832,
  "default_height": 480,
  "default_fps": 16,
  "default_frames": 49,
  "amplitude": {
    "rotation_gain": 1.0,
    "translation_gain": 1.0,
    "max_amount": 1.0,
    "translation_unit": "custom:action_steps",
    "calibration_status": "initial_manual",
    "metadata": {}
  },
  "capabilities": {
    "rotation": true,
    "translation": true,
    "supports_static": true
  },
  "execution_contract": {
    "action_map": {
      "static": "STAY",
      "yaw_left": "TURN_LEFT",
      "yaw_right": "TURN_RIGHT",
      "pan_left": "MOVE_LEFT",
      "pan_right": "MOVE_RIGHT"
    }
  },
  "notes": "External discrete-action world model."
}
```

### Paradigm 3: Prompt / API Camera (`prompt_camera`)

Unifies prompt-only T2V and closed API video models as `active` records (`translation_unit: "none"`):

```json
{
  "key": "my-api-video-model",
  "status": "active",
  "input_kind": "none",
  "model_input": "T2V",
  "viewpoint_condition_type": "prompt-only",
  "adapter": "prompt_camera",
  "payload_type": "prompt_camera",
  "default_width": 1280,
  "default_height": 720,
  "default_fps": 24,
  "default_frames": 49,
  "amplitude": {
    "rotation_gain": 1.0,
    "translation_gain": 1.0,
    "max_amount": 1.0,
    "translation_unit": "none",
    "calibration_status": "not_applicable",
    "metadata": {}
  },
  "capabilities": {
    "rotation": true,
    "translation": true,
    "supports_static": true
  },
  "execution_contract": {
    "prompt_templates": {
      "static": "{prompt} Keep the camera completely static.",
      "yaw_LR": "{prompt} Smoothly pan left and then return right to the starting view.",
      "yaw_RL": "{prompt} Smoothly pan right and then return left to the starting view."
    }
  },
  "notes": "Prompt-controlled API or T2V video model."
}
```

### Validate and compile an external spec

```bash
wrbench --model-spec ./my_model.json doctor --model my-pose-model
wrbench --model-spec ./my_model.json generate \
  --model my-pose-model \
  --camera preset:yaw_LR \
  --image /path/to/first_frame.png \
  --prompt "A cat sits on the bedroom floor." \
  --out ./out.mp4
```

### Run real generation with `command_template`

Add a `models.<key>` entry in `wrbench.runtime.json` using `"backend": "command_template"`:

```json
{
  "schema_version": 1,
  "model_spec_files": ["./my_model.json"],
  "models": {
    "my-pose-model": {
      "backend": "command_template",
      "python_bin": "/path/to/venv/bin/python",
      "repo_root": "/path/to/my-model-repo",
      "model_path": "/path/to/checkpoint",
      "gpu_id": 0,
      "command_template": [
        "{python_bin}",
        "{repo_root}/infer.py",
        "--checkpoint", "{model_path}",
        "--image", "{image_path}",
        "--poses", "{target_c2w_path}",
        "--payload", "{payload_json_path}",
        "--prompt", "{prompt}",
        "--out", "{output_path}"
      ],
      "env": {}
    }
  }
}
```

Supported placeholders: `{python_bin}`, `{repo_root}`, `{model_path}`, `{gpu_id}`, `{model}`, `{prompt}`, `{prompt_to_send}`, `{output_path}`, `{image_path}`, `{source_video_path}`, `{work_dir}`, `{payload_json_path}`, `{target_c2w_path}`, `{camera_trajectory_json_path}`, `{camera_sidecar_path}`, `{pose_file}`, plus any key defined in `extra_paths`.

Then verify readiness and run:

```bash
wrbench doctor --model my-pose-model --runtime-config wrbench.runtime.json
wrbench generate \
  --model my-pose-model \
  --camera preset:yaw_LR \
  --image /path/to/first_frame.png \
  --prompt "A cat sits on the bedroom floor." \
  --out ./out.mp4 \
  --runtime-config wrbench.runtime.json \
  --no-dry-run
```

---

## 3. Python plugins

### Custom `module:callable` references and `entry_points`

For complex coordinate transforms, custom API clients, or third-party evaluators, point directly to Python symbols in `module.path:AttrName` syntax:

- **Model spec adapter**: `"adapter": "my_pkg.adapter:MyCustomAdapter"`
- **Runtime generation backend**: `"backend": "my_pkg.backend:MyBackend"`
- **Evaluation backend**: `"eval": { "backends": { "pose": "my_pkg.eval:recover_poses" } }`

To auto-register models, adapters, backends, launchers, or evaluators when a user runs `pip install my-wrbench-plugin`, declare standard Python entry points in your package's `pyproject.toml`:

```toml
[project.entry-points."wrbench.plugins"]
my_world_model = "my_pkg.plugin:register_generation"

[project.entry-points."wrbench.eval_plugins"]
my_evaluator = "my_pkg.eval_plugin:register_eval"
```

Example registration module:

```python
from wrbench.adapters.base import register as register_adapter
from wrbench.backends.registry import register_generation_backend
from wrbench.registry import register_model_record, register_translation_unit


def register_generation() -> None:
    register_translation_unit("custom:my_units")
    register_model_record({
        "key": "my-plugin-model",
        "status": "active",
        "input_kind": "image",
        "adapter": "my_pkg.adapter:MyCustomAdapter",
        "payload_type": "my_custom_payload",
        "default_width": 832,
        "default_height": 480,
        "default_fps": 16,
        "default_frames": 49,
        "amplitude": {
            "rotation_gain": 1.0,
            "translation_gain": 1.0,
            "max_amount": 1.0,
            "translation_unit": "custom:my_units",
            "calibration_status": "initial_manual",
            "metadata": {},
        },
        "capabilities": {"rotation": True, "translation": True, "supports_static": True},
    }, overwrite=True)
    register_generation_backend("my_backend", MyBackend)
```

Verify discovery with `wrbench plugins --json` and `wrbench doctor --model my-plugin-model`.
