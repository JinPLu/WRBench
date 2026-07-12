# Prompt generation

`wrbench.prompts` generates scene, task, and camera text prompts for benchmark runs.

## Camera text (stdlib, no extra deps)

Natural-language camera clauses and API-model assembly:

```python
from wrbench.prompts import preset_camera_text, assemble_ti2v_prompt, build_prompt_to_send

# Map wrbench preset → NL camera clause
text = preset_camera_text("yaw_LR", pronoun="she", offscreen_area="empty stone paving")

# Full TI2V prompt
prompt = assemble_ti2v_prompt(scene_start, event, "she", "empty floor", "yaw_LR")

# API model assembly (Hailuo vs copy-optimized)
api_prompt = build_prompt_to_send(base_prompt, "yaw_LR", model="hailuo-2.3")
```

```bash
wrbench prompt camera --preset yaw_LR --pronoun she --offscreen-area "empty floor"
wrbench prompt camera --model hailuo-2.3 --source-prompt "Scene. Action." --camera-motion yaw_LR
```

## Scene prompt (T2I / first-frame caption)

Requires `pip install 'wrbench[prompts]'`. Pass the provider, model, API key,
base URL, and temperature explicitly.

```python
from wrbench.prompts.scene import generate_t2i_scene

t2i_scene = generate_t2i_scene(
    family_dict,
    provider="dashscope",
    model="qwen-max",
    api_key="YOUR_API_KEY",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.2,
)
```

```bash
wrbench prompt scene \
  --family-json family.json \
  --provider dashscope \
  --model qwen-max \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --api-key YOUR_API_KEY \
  --temperature 0.2
```

System prompt templates live in `src/wrbench/prompts/templates/`.

## Task prompt (Natural-25 unified video prompt)

WRBench ships a ready-to-use Natural-25 prompt set:

```python
from wrbench.datasets import natural25_variants_path
from wrbench.prompts.task import load_jsonl

variants = list(load_jsonl(natural25_variants_path()))
print(len(variants))  # 400 = 25 families × 4 event tiers × 4 camera gaps
print(variants[0]["ti2v_prompt"])
```

Each variant's `ti2v_prompt` is a current deterministic, first-frame-anchored
toolkit prompt. It is kept for compatibility, but it is not the file of record
for the frozen paper table. Exact paper prompts are versioned under
`src/wrbench/data/natural25/releases/paper_main_20260608/`.

Text-only models have no first frame, so Natural-25 ships a separate
layout-anchored prompt profile for T2V runs:

```python
from wrbench.datasets import (
    load_jsonl,
    load_natural25_t2v_layout_anchors,
    natural25_variants_path,
    resolve_variant_prompt,
)

variant = next(load_jsonl(natural25_variants_path()))
anchors = load_natural25_t2v_layout_anchors()
prompt = resolve_variant_prompt(
    variant,
    prompt_profile="t2v_layout_anchor",
    layout_anchors=anchors,
)
```

`t2v_layout_anchor` writes the initial layout into the generation prompt:
subject on the left, interactor far right, open surface between them, and
background anchors visible. It intentionally does not copy photography, lens,
lighting, palette, or model-specific wording from T2I prompts. Camera control
is still supplied by the benchmark camera scope and backend payload, not by
natural-language camera clauses.

The frozen paper release has two distinct prompt surfaces:

- `variants.local_ti2v_tv2v.jsonl` is the exact local-generation catalog. Its
  100 `oov_gap=none` rows supplied content prompts while camera control was
  separate.
- `variants.api_source.jsonl` is the historical API source catalog. Its
  `static`, `yaw_LR`, and `yaw_RL` rows are not automatically identical to a
  provider-specific `prompt_to_send`; exact request wording requires request
  evidence.

`prompt_usage.json` maps every frozen model to exactly one catalog. See
`src/wrbench/data/results/README.md` for the corresponding 23-model scope.
The TV2V source manifest separately preserves all 100 exact Wan2.7 I2V provider
request sidecars. Seventy-five are represented by a frozen catalog; 25
`T2_div_a` requests predate both catalogs. Their task-to-static asset mapping is
not a claim of prompt-text equality.

**Deterministic path** (no LLM): rebuild Natural-25 style variants from bundled data or custom inputs.

```bash
# Omit custom paths to use the bundled Natural-25 data shipped inside the package
wrbench prompt task --deterministic --output variants.jsonl

# Or specify custom paths
wrbench prompt task --deterministic \
  --candidates-json candidates.json \
  --families-jsonl families.jsonl \
  --output variants.jsonl
```

**LLM path** (Python):

```python
from wrbench.prompts.task import generate_ti2v_variants_llm

variants = generate_ti2v_variants_llm(
    tier_variants,
    family,
    provider="dashscope",
    model="qwen-max",
    api_key="YOUR_API_KEY",
    temperature=0.2,
)
```
