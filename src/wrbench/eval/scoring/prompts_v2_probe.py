"""Runtime V2 D3-D6 yes/no probe prompts.

The scoring signal is the next-token probability of answering Yes versus No for
each compact probe; continuous dimension scores are derived after the call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


PROBE_CATALOG_VERSION = "runtime_v2_probe_catalog_v1"
DEFAULT_PROMPT_MODE = "runtime_v2_probe_logprob"
PROMPT_MODE_WRBENCH_PAPER = "wrbench_paper"
PROMPT_MODE_P22_NATIVE_D5D6 = "runtime_v2_probe_logprob_p22_native_d5d6_score_only"
PROMPT_MODE_P25_D3D4_SLOT_PARSE = "runtime_v2_probe_logprob_p25_d3d4_slot_parse"
PROMPT_MODE_ALIASES: dict[str, str] = {
    PROMPT_MODE_WRBENCH_PAPER: PROMPT_MODE_P25_D3D4_SLOT_PARSE,
}
SUPPORTED_PROMPT_MODES = {
    DEFAULT_PROMPT_MODE,
    PROMPT_MODE_WRBENCH_PAPER,
    PROMPT_MODE_P22_NATIVE_D5D6,
    PROMPT_MODE_P25_D3D4_SLOT_PARSE,
}
CAMERA_MOTION_CONTEXT_KEY = "__camera_motion__"


@dataclass(frozen=True)
class RuntimeV2Probe:
    probe_id: str
    dimension: str
    role: str
    polarity: str
    question: str
    gate_kind: str | None = None


RUNTIME_V2_PROBE_CATALOG: tuple[RuntimeV2Probe, ...] = (
    RuntimeV2Probe(
        probe_id="D3_POSITION_RELATION",
        dimension="spatial_fidelity",
        role="score",
        polarity="positive",
        question=(
            "During visible, judgeable moments, is the prompt-critical subject "
            "or target in a plausible world-space position, relation, contact, "
            "placement, and path relative to stable scene anchors and the prompt?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D3_STATIC_ANCHORS",
        dimension="spatial_fidelity",
        role="score",
        polarity="positive",
        question=(
            "Do stable anchors such as walls, floor, furniture, doors, counters, "
            "or fixed background structure support the same spatial interpretation "
            "rather than contradicting the subject or target motion?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D3_NEG_WRONG_SPATIAL",
        dimension="spatial_fidelity",
        role="score",
        polarity="negative",
        question=(
            "Is there clear visible spatial counterevidence, such as the subject "
            "going to the wrong target, wrong side, impossible placement, frame-locked "
            "sliding, or a path that contradicts the prompt?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D4_ACTION_STATE",
        dimension="state_fidelity",
        role="score",
        polarity="positive",
        question=(
            "During visible, judgeable moments, does the subject or target actually "
            "perform the prompt-required action, pose, state, contact result, or "
            "state change rather than merely being described by the prompt?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D4_TEMPORAL_PROGRESS",
        dimension="state_fidelity",
        role="score",
        polarity="positive",
        question=(
            "Does the visible action or state evolve appropriately over time, or "
            "remain stably maintained when the prompt requires a static state?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D4_NEG_ABSENT_FREEZE",
        dimension="state_fidelity",
        role="score",
        polarity="negative",
        question=(
            "Is there clear visible state counterevidence, such as absent action, "
            "wrong behavior, freezing, looping, reset, regression, or an impossible "
            "pose/state transition?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D5_GATE_HIDDEN",
        dimension="spatial_reasoning",
        role="gate",
        polarity="positive",
        gate_kind="hidden",
        question=(
            "Does the prompt-critical subject or target become fully out of view, "
            "fully occluded, or spatially unjudgeable for a relevant interval?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D5_GATE_RETURN",
        dimension="spatial_reasoning",
        role="gate",
        polarity="positive",
        gate_kind="return",
        question=(
            "After that spatially hidden interval, does a later view show the "
            "expected world region where the subject or target should be spatially "
            "judgeable again?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D5_GATE_UNIDENTIFIABLE",
        dimension="spatial_reasoning",
        role="gate",
        polarity="negative",
        gate_kind="unidentifiable",
        question=(
            "After the later view returns, is the subject or target so unidentifiable "
            "that no spatial continuity judgment can be made at all?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D5_RETURN_POSITION",
        dimension="spatial_reasoning",
        role="score",
        polarity="positive",
        question=(
            "When the expected region returns, is the subject or target position "
            "spatially plausible relative to the last judgeable pre-hidden evidence, "
            "elapsed hidden time, the prompt, and stable scene anchors?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D5_NEG_VANISH_RESET",
        dimension="spatial_reasoning",
        role="score",
        polarity="negative",
        question=(
            "When the expected region returns, is there clear spatial failure such "
            "as the target being absent, reappearing at a wrong/impossible location, "
            "resetting to an earlier place, or making an impossible jump?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D6_GATE_HIDDEN",
        dimension="state_reasoning",
        role="gate",
        polarity="positive",
        gate_kind="hidden",
        question=(
            "Does the prompt-critical action, pose, result, or state become hidden "
            "or state-unjudgeable for a relevant interval?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D6_GATE_RETURN",
        dimension="state_reasoning",
        role="gate",
        polarity="positive",
        gate_kind="return",
        question=(
            "After that state-hidden interval, does later evidence make the action, "
            "pose, result, or state judgeable again?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D6_GATE_UNIDENTIFIABLE",
        dimension="state_reasoning",
        role="gate",
        polarity="negative",
        gate_kind="unidentifiable",
        question=(
            "After later evidence returns, is the subject or target so broken or "
            "unclear that no action/state continuity judgment can be made at all?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D6_HIDDEN_PROGRESS",
        dimension="state_reasoning",
        role="score",
        polarity="positive",
        question=(
            "When later evidence returns, does the action, pose, result, or state "
            "look like a plausible continuation through hidden time rather than a "
            "paused frame?"
        ),
    ),
    RuntimeV2Probe(
        probe_id="D6_NEG_FREEZE_RESET",
        dimension="state_reasoning",
        role="score",
        polarity="negative",
        question=(
            "When later evidence returns, is there clear hidden-time state failure "
            "such as freezing, reset, regression, vanished result, or an impossible "
            "skipped action/state change?"
        ),
    ),
)

P6_QUESTION_OVERRIDES: dict[str, str] = {
    "D5_GATE_HIDDEN": (
        "Does the prompt-critical subject or target become out of view, occluded, "
        "or otherwise spatially unjudgeable for a relevant interval?"
    ),
    "D5_GATE_RETURN": (
        "After that interval, does later evidence make the same subject, target, "
        "or expected region visible enough for a spatial judgment? Do not require "
        "evidence that the location is correct."
    ),
    "D5_GATE_UNIDENTIFIABLE": (
        "After later evidence returns, is the subject or target too unclear to "
        "match for any spatial continuity judgment?"
    ),
    "D5_RETURN_POSITION": (
        "When later evidence is judgeable, does it support spatial continuity of "
        "the same prompt-critical subject or target in the expected world region "
        "relative to stable anchors and before-hidden evidence? Reappearance or "
        "visibility alone is insufficient, but exact pixel position or camera "
        "framing match is not required."
    ),
    "D5_NEG_VANISH_RESET": (
        "When later evidence is judgeable, is there visible spatial counterevidence "
        "such as absence from the expected region, wrong location, reset, impossible "
        "jump, or inconsistent reappearance?"
    ),
}

P8_QUESTION_OVERRIDES: dict[str, str] = {
    "D4_ACTION_STATE": (
        "Does the prompt-critical primary subject and any causal target actually "
        "perform or reach the required action, pose, contact result, state change, "
        "or final state?"
    ),
    "D4_TEMPORAL_PROGRESS": (
        "Is the required action/result completed and maintained over a stable "
        "ending segment, or is a required static state maintained without visible "
        "regression?"
    ),
    "D4_NEG_ABSENT_FREEZE": (
        "Is there clear visible state counterevidence, such as only starting or "
        "suggesting the action, wrong behavior, freeze, loop, reset, regression, "
        "or wrong final state?"
    ),
}

P25_D3D4_QUESTIONS: dict[str, str] = {
    "D3_POSITION_RELATION": (
        "During visible, judgeable moments, is the task-critical subject or target "
        "in a plausible world-space position, relation, contact, placement, and "
        "path relative to the parsed task slots and stable scene anchors?"
    ),
    "D3_STATIC_ANCHORS": (
        "Do stable scene anchors support the same spatial interpretation for the "
        "task-critical subject, target, support surface, destination, or region?"
    ),
    "D3_NEG_WRONG_SPATIAL": (
        "Is there clear visible spatial counterevidence, such as the subject going "
        "to the wrong target, wrong side, impossible placement, missing required "
        "contact/support/containment, frame-locked sliding, or a path that "
        "contradicts the parsed task slots?"
    ),
    "D4_ACTION_STATE": (
        "Does visible evidence show the task-critical subject or target performing "
        "or reaching the required action, pose, contact result, state change, or "
        "final state from the parsed task slots?"
    ),
    "D4_TEMPORAL_PROGRESS": (
        "Does the required action or state visibly progress, complete, or remain "
        "maintained in a way consistent with the parsed task slots?"
    ),
    "D4_NEG_ABSENT_FREEZE": (
        "Is there clear visible state counterevidence, such as absent action, wrong "
        "behavior, freeze, loop, reset, regression, wrong final result, or an "
        "impossible state transition relative to the parsed task slots?"
    ),
}


def _p25_dimension_guidance(probe: RuntimeV2Probe) -> str:
    if probe.dimension == "spatial_fidelity":
        return """Spatial judgment guidance:
- Use the parsed subject, target object, support surface, destination, and region to decide what spatial relation matters.
- Judge coarse 3D world relation, contact, support, containment, placement, and path relative to stable anchors.
- Do not require exact pixel position, identical camera framing, or a fully visible trajectory.
- Camera motion, cropping, and brief hidden intervals are not spatial failures by themselves."""
    return """Action/state judgment guidance:
- Use the parsed subject, target object, required action, contact/result, and final state to decide what state evidence matters.
- A clear final result or stable maintained state can support Yes even if the middle of the action is partly hidden.
- Do not require perfect animation of every intermediate step when the visible evidence supports the requested result.
- Camera motion, cropping, and brief hidden intervals are not state failures by themselves."""


def _p25_answer_rule(probe: RuntimeV2Probe) -> str:
    if probe.polarity == "negative":
        return """How to answer this failure question:
- Answer Yes only when clear visible video evidence shows the failure described in the question.
- Do not answer Yes from weak motion, brief artifacts, ambiguous glimpses, or the task description alone.
- If the failure is missing, unclear, or only possible but not visible, answer No."""
    return """How to answer this support question:
- Answer Yes when visible evidence reasonably supports the requested spatial relation, action, result, or maintained state.
- Answer No when the required subject/target/result is absent, too ambiguous to identify, only implied by the task description, or clearly contradicted by the video.
- Prefer moderate support from clear before/after or final-state evidence over demanding a perfect continuous trajectory."""


def _build_p25_d3d4_prompt(
    *, world_state_prompt: str, probe: RuntimeV2Probe, sampling_text: str
) -> str:
    return f"""You are judging one visual question about an AI-generated video.

Answer exactly one token: Yes or No.
Do not output JSON, explanations, markdown, punctuation, or extra words.

Watch the whole sampled video before answering. You are seeing {sampling_text}, not necessarily every source frame.

Task description:
{world_state_prompt}

Before answering, silently parse the task description into these task slots:
- primary subject or actor;
- manipulated or target object;
- support surface, container, destination, region, or other spatial anchor;
- required action, pose, contact, movement, or state change;
- expected contact result, final state, or maintained state.

Then inspect the video directly:
1. Focus on the parsed task-critical subject, target object, region, action, and result.
2. Use stable scene anchors such as the floor, walls, furniture, counters, doors, and fixed background structure.
3. Ignore incidental objects unless they directly affect the parsed task slots.
4. Judge visible evidence, not the task description alone.

{_p25_dimension_guidance(probe)}

{_p25_answer_rule(probe)}

Question:
{question_for_probe(probe, PROMPT_MODE_P25_D3D4_SLOT_PARSE)}

Answer exactly one token: Yes or No."""


def _p22_native_score_rule(probe: RuntimeV2Probe) -> str:
    if probe.dimension == "spatial_reasoning":
        return """How to judge this question:
- Compare the last clear view before the relevant person, object, or region became hard to judge with the later clear view after it becomes judgeable again.
- Use stable scene anchors to judge whether the later evidence fits the same world-region continuation.
- A later similar-looking object is not enough by itself to prove spatial continuity for the original prompt-relevant object.
- Brief duplicate-looking blur, partial silhouettes, transparent remnants, motion trails, or render echoes should be ignored unless they are clear, separate, persistent across later sampled evidence, and directly relevant to the prompt.
- Missing middle frames or a hidden interval alone is not failure.
- For questions asking about counterevidence, answer Yes only when clear visible counterevidence is actually shown."""
    if probe.dimension == "state_reasoning":
        return """How to judge this question:
- Compare the last clear action, pose, result, or state before it became hard to judge with the later clear evidence after it becomes judgeable again.
- Judge whether the later evidence looks like a plausible continuation of the same prompt-relevant action or result.
- A later similar-looking person, object, or result is not enough by itself to prove that the original action or result continued.
- Brief duplicate-looking blur, partial silhouettes, transparent remnants, motion trails, or render echoes should be ignored unless they are clear, separate, persistent across later sampled evidence, and directly relevant to the prompt.
- Missing middle frames or a hidden interval alone is not failure.
- For questions asking about counterevidence, answer Yes only when clear visible counterevidence is actually shown."""
    raise RuntimeError(f"no P22 native score rule for {probe.probe_id}")


def _build_p22_d5d6_prompt(
    *,
    world_state_prompt: str,
    probe: RuntimeV2Probe,
    task_context: dict[str, Any] | None,
    sampling_text: str,
    evidence_context: dict[str, Any] | None,
    evidence_context_mode: str | None,
) -> str:
    context_text = format_task_context(task_context)
    evidence_text = format_evidence_context(
        evidence_context, evidence_context_mode=evidence_context_mode
    )
    extra_context_block = ""
    if context_text.strip() not in {"", "{}", "- none"} or evidence_text.strip():
        extra_context_block = f"""
Additional prompt/task context, when available:
{context_text}
{evidence_text}
"""
    return f"""You are judging one visual question about an AI-generated video.

Answer exactly one token: Yes or No.
Do not output JSON, explanations, markdown, punctuation, or extra words.

Watch the whole sampled video before answering. You are seeing {sampling_text}, not necessarily every source frame.

Use the text prompt only to identify the main person, object, target, action, and result that matter for this question. Answer from the video evidence, not from the text prompt alone.

{_p22_native_score_rule(probe)}

Text prompt used to generate the video:
{world_state_prompt}
{extra_context_block}
Question:
{question_for_probe(probe, PROMPT_MODE_P22_NATIVE_D5D6)}

Answer exactly one token: Yes or No."""


def validate_prompt_mode(prompt_mode: str) -> str:
    if prompt_mode not in SUPPORTED_PROMPT_MODES:
        raise ValueError(
            f"unsupported prompt_mode: {prompt_mode}; "
            f"expected one of {sorted(SUPPORTED_PROMPT_MODES)}"
        )
    return PROMPT_MODE_ALIASES.get(prompt_mode, prompt_mode)


def question_for_probe(probe: RuntimeV2Probe, prompt_mode: str = DEFAULT_PROMPT_MODE) -> str:
    prompt_mode = validate_prompt_mode(prompt_mode)
    if prompt_mode == PROMPT_MODE_P25_D3D4_SLOT_PARSE and probe.probe_id in P25_D3D4_QUESTIONS:
        return P25_D3D4_QUESTIONS[probe.probe_id]
    if prompt_mode in {
        PROMPT_MODE_P22_NATIVE_D5D6,
        PROMPT_MODE_P25_D3D4_SLOT_PARSE,
    }:
        if probe.dimension == "state_fidelity":
            return P8_QUESTION_OVERRIDES.get(probe.probe_id, probe.question)
        if probe.dimension == "spatial_reasoning":
            return P6_QUESTION_OVERRIDES.get(probe.probe_id, probe.question)
    return probe.question


def active_probe_catalog(prompt_mode: str = DEFAULT_PROMPT_MODE) -> tuple[RuntimeV2Probe, ...]:
    validate_prompt_mode(prompt_mode)
    return RUNTIME_V2_PROBE_CATALOG


def probes_for_dimension(dimension: str) -> list[RuntimeV2Probe]:
    return [probe for probe in RUNTIME_V2_PROBE_CATALOG if probe.dimension == dimension]


def probe_by_id(probe_id: str) -> RuntimeV2Probe:
    for probe in RUNTIME_V2_PROBE_CATALOG:
        if probe.probe_id == probe_id:
            return probe
    raise KeyError(probe_id)


def format_task_context(task_context: dict[str, Any] | None) -> str:
    if not task_context:
        return "- none"
    lines: list[str] = []
    for key in sorted(task_context):
        value = task_context[key]
        if value is None:
            continue
        text = str(value).strip()
        if text:
            lines.append(f"- {key}: {text}")
    return "\n".join(lines) if lines else "- none"


def format_camera_motion_context(task_context: dict[str, Any] | None) -> str | None:
    if not task_context or CAMERA_MOTION_CONTEXT_KEY not in task_context:
        return None
    text = str(task_context.get(CAMERA_MOTION_CONTEXT_KEY) or "").strip()
    return text or None


def task_context_without_camera_motion(task_context: dict[str, Any] | None) -> dict[str, Any]:
    if not task_context:
        return {}
    return {key: value for key, value in task_context.items() if key != CAMERA_MOTION_CONTEXT_KEY}


def format_evidence_context(
    evidence_context: dict[str, Any] | None,
    *,
    evidence_context_mode: str | None = None,
) -> str:
    if not evidence_context:
        return ""
    label = (
        "Qwen3-VL structured subquestion evidence from a previous pass"
        if evidence_context_mode == "subquestion_v1"
        else "Qwen3-VL structured visibility evidence from a previous pass"
    )
    compact_json = json.dumps(evidence_context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"""
{label}:
{compact_json}

Use this evidence as a checklist, not as ground truth. Inspect the video again.
If the evidence conflicts with the video, answer from the video.
Wrong but visible/judgeable evidence is still judgeable; correctness is scored
by D5/D6 score probes, not by the gate.
"""


def build_runtime_v2_probe_prompt(
    *,
    world_state_prompt: str,
    video_id: str,
    probe: RuntimeV2Probe,
    task_context: dict[str, Any] | None = None,
    fps: str | None = None,
    frames_used: int | None = None,
    prompt_mode: str = DEFAULT_PROMPT_MODE,
    evidence_context: dict[str, Any] | None = None,
    evidence_context_mode: str | None = None,
) -> str:
    prompt_mode = validate_prompt_mode(prompt_mode)
    sampling_bits: list[str] = []
    if fps:
        sampling_bits.append(f"frames sampled at {fps} fps")
    if frames_used is not None:
        sampling_bits.append(f"approximately {frames_used} video frames")
    sampling_text = ", ".join(sampling_bits) if sampling_bits else "the frames provided by the runner"
    if (
        prompt_mode == PROMPT_MODE_P25_D3D4_SLOT_PARSE
        and probe.role == "score"
        and probe.dimension in {"spatial_fidelity", "state_fidelity"}
    ):
        if evidence_context:
            raise ValueError("P25 clean task-slot parsing does not accept evidence_context")
        return _build_p25_d3d4_prompt(
            world_state_prompt=world_state_prompt,
            probe=probe,
            sampling_text=sampling_text,
        )
    if (
        prompt_mode in {PROMPT_MODE_P22_NATIVE_D5D6, PROMPT_MODE_P25_D3D4_SLOT_PARSE}
        and probe.role == "score"
        and probe.dimension in {"spatial_reasoning", "state_reasoning"}
    ):
        return _build_p22_d5d6_prompt(
            world_state_prompt=world_state_prompt,
            probe=probe,
            task_context=task_context,
            sampling_text=sampling_text,
            evidence_context=evidence_context,
            evidence_context_mode=evidence_context_mode,
        )

    combined_modes = {
        PROMPT_MODE_P22_NATIVE_D5D6,
        PROMPT_MODE_P25_D3D4_SLOT_PARSE,
    }
    if prompt_mode in combined_modes and probe.dimension == "state_fidelity":
        scaffold = """Use this internal observation scaffold:
- Inspect the entire sampled video and maintain a simple 3D scene while watching it.
- For this D4 visible-state probe, focus on the prompt-critical primary subject and any object or region that causally determines the requested result.
- Ignore incidental moving objects unless they directly affect the prompt-required action, contact result, state change, or final state.
- Judge whether the required action/result is actually reached and then maintained when the prompt needs persistence.
- For prompts describing a static pose or state, check that the state remains stable instead of briefly appearing, resetting, looping, or regressing.
- Track the prompt-critical subject, target, result object, and expected region when they determine the required action or state.
- For negative probes, answer Yes only when visible state counterevidence is actually shown.
- Judge the video evidence, not the text expectation."""
    elif prompt_mode in combined_modes and probe.dimension == "spatial_reasoning":
        scaffold = """Use this internal observation scaffold:
- Maintain a simple 3D world state while watching the sampled video.
- For this D5 spatial-continuity probe, compare the last judgeable before-hidden evidence with later judgeable evidence.
- Track whether the same prompt-critical subject, target, result object, or expected region can be matched across the hidden interval.
- Gate probes only judge whether later evidence is inspectable enough for a spatial judgment; they do not judge whether the video is good or whether the location is correct.
- For return-position probes, judge expected world-region continuity relative to static anchors, not exact pixel position or camera framing.
- Visible or identifiable again is not enough by itself; later evidence should support spatial continuity in the expected world region.
- Hidden interval or missing middle frames alone is not failure.
- For negative probes, answer Yes only when visible spatial counterevidence is actually shown.
- Judge the video evidence, not the text expectation."""
    else:
        scaffold = """Use this internal observation scaffold:
- Maintain a simple mental 3D scene.
- Use stable scene anchors such as walls, floor, furniture, doors, counters, and fixed background structure.
- Judge position, relation, contact, placement, plausible path, action, state, and result from visible evidence.
- Do not apply a fixed camera-motion subtraction rule.
- For hidden intervals, compare before hidden evidence with after return evidence. Hidden time can imply continuation; it should not be treated as a pause unless the video shows that.
- Answer from the video evidence, not from a caption-like expectation."""

    camera_motion_context = format_camera_motion_context(task_context)
    if camera_motion_context is None:
        task_context_block = f"""Task context from manifest, when available:
{format_task_context(task_context)}"""
        video_id_block = f"""Video id:
{video_id}
"""
    else:
        extra_task_context = task_context_without_camera_motion(task_context)
        extra_task_context_text = format_task_context(extra_task_context)
        extra_task_context_block = ""
        if extra_task_context_text.strip() not in {"", "{}", "- none"}:
            extra_task_context_block = f"""

Additional prompt/task context, when available:
{extra_task_context_text}"""
        task_context_block = f"""How camera move:
{camera_motion_context}{extra_task_context_block}"""
        video_id_block = ""

    return f"""You are a strict video consistency auditor for AI-generated mobile-camera videos.

Your job is to answer one binary probe. Answer exactly one token: Yes or No.
Do not output JSON, explanations, markdown, punctuation, or extra words.

Inspect the entire provided sampled video before answering. You are seeing {sampling_text}, not necessarily every source frame.

{scaffold}

Text prompt used to generate the video:
{world_state_prompt}

{video_id_block}
{task_context_block}
{format_evidence_context(evidence_context, evidence_context_mode=evidence_context_mode)}

Probe id:
{probe.probe_id}

Probe dimension:
{probe.dimension}

Probe role:
{probe.role}

Probe polarity:
{probe.polarity}

Question:
{question_for_probe(probe, prompt_mode)}

Answer exactly one token: Yes or No."""
