# Natural-25 prompt suite

This directory ships the **Natural-25** scene/event prompt grid and bundled first frames used by WRBench.

| File | Description |
| --- | --- |
| `scene_events_25x4.csv` | 25 scene families × 4 event axes (`none` / `spatial` / `state` / `full`) |
| `families.jsonl` | Scene-family metadata and prompt templates |
| `variants.jsonl` | Deterministic TI2V prompt variants: 25 families × 4 event tiers × 4 camera gaps |
| `t2v_layout_anchors.jsonl` | Layout facts for text-only Natural-25 generation (subject, interactor, surface, and background anchors) |
| `t2v_event_tails.jsonl` | Text-only event/action tails keyed by `variant_id`, separate from the TI2V prompt set |
| `prompt_profiles/t2v_layout_anchor.json` | Prompt-profile policy for text-only models |
| `camera_scopes/t2v_rotation_stress_30_60.json` | T2V rotation-stress camera scope: `static`, `yaw30_LR`, `yaw30_RL`, `yaw60_LR`, `yaw60_RL` |
| `first_frames/` | Bundled Natural-25 PNG first frames, one per `family_id` |
| `first_frames_manifest.json` | First-frame paths, SHA-256 image digests, and generation-prompt hashes |
| `releases/paper_main_20260608/` | Immutable 23-model / 9,600-row paper provenance contract (prompts, first frames, camera scope, TV2V sources, HyDRA policy) |

## Usage and provenance

The bundled prompts and first frames are released under the repository's Apache-2.0 license so camera-compile and task-export examples run out of the box. You can also regenerate first frames with `wrbench firstframe` (optional extra) or supply images from your own T2I pipeline.

- **Paper reproduction:** use the frozen prompt catalogs in `releases/paper_main_20260608/` (`variants.local_ti2v_tv2v.jsonl`, `variants.api_source.jsonl`, and `first_frame_generation_families.jsonl`, bound to the PNG bytes by `first_frames_manifest.json`).
- **Toolkit compatibility:** top-level `variants.jsonl` and `families.jsonl` provide the deterministic toolkit surface used by compile and export helpers.
- **Prompt-only T2V track:** materialize the `t2v_layout_anchor` prompt profile and `t2v_rotation_stress_30_60` camera scope rather than reusing TI2V first-frame prompts.

## Hugging Face release

- Hugging Face Collection: <https://huggingface.co/collections/WRBench/wrbench-current-world-models-lack-a-persistent-state-core>
- Natural-25 dataset card and viewer: <https://huggingface.co/datasets/WRBench/wrbench-natural25>
- Benchmark videos: <https://huggingface.co/datasets/WRBench/wrbench-videos>
- WRBench leaderboard: <https://huggingface.co/spaces/WRBench/wrbench-leaderboard>
- Paper page: <https://huggingface.co/papers/2606.20545>
