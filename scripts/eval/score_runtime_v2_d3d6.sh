#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRCAM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCORING_DIR="${WRCAM_ROOT}/src/wrcam/eval/scoring"
REPO_ROOT="${WRCAM_ROOT}"
export PYTHONPATH="${WRCAM_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export WORLD_STATE_VIDEO_BACKEND="${WORLD_STATE_VIDEO_BACKEND:-decord}"

PY_SCORER="${PY_SCORER:-python}"
PY_HELPER="${PY_HELPER:-${PY_SCORER}}"

MANIFEST="${MANIFEST:-}"
OUT_DIR="${OUT_DIR:-${WRCAM_ROOT}/eval_outputs/runtime_v2_d3d6_$(date +%Y%m%d_%H%M%S)}"
QWEN35_MODEL="${QWEN35_MODEL:-}"
QWEN3VL_MODEL="${QWEN3VL_MODEL:-}"

SCORER_PROFILE="${SCORER_PROFILE:-wrbench_default}"
SOURCE_SCORES="${SOURCE_SCORES:-}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_IDS="${SHARD_IDS:-0}"
CUDA_DEVICES_CSV="${CUDA_DEVICES_CSV:-0}"
FPS="${FPS:-2}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1}"
STRICT_MANIFEST_CONTRACT="${STRICT_MANIFEST_CONTRACT:-0}"

case "$SCORER_PROFILE" in
  wrbench_default|current_benchmark_p25_p22_e14)
    RUN_TAG="${RUN_TAG:-runtime_v2_d3d6_wrbench_default}"
    PROMPT_MODE="${PROMPT_MODE:-runtime_v2_probe_logprob_p25_d3d4_slot_parse}"
    TASK_CONTEXT_MODE="${TASK_CONTEXT_MODE:-none}"
    qwen35_runner_prefix=(
      "${SCORING_DIR}/run_qwen35_p25_d3d4_slot_parse_overlay.py"
      --repo-root "$REPO_ROOT"
      --
    )
    ;;
  legacy_p9_all_manifest_metadata|ablation_manifest_metadata)
    RUN_TAG="${RUN_TAG:-runtime_v2_d3d6_legacy_p9_allmeta}"
    PROMPT_MODE="${PROMPT_MODE:-runtime_v2_probe_logprob_p9_d4_p8_d5_p6_combined}"
    TASK_CONTEXT_MODE="${TASK_CONTEXT_MODE:-all_manifest_metadata}"
    qwen35_runner_prefix=("${SCORING_DIR}/run_local_qwen35_probe_logprob_scorer.py")
    ;;
  custom)
    RUN_TAG="${RUN_TAG:-runtime_v2_d3d6_custom}"
    PROMPT_MODE="${PROMPT_MODE:?PROMPT_MODE is required when SCORER_PROFILE=custom}"
    TASK_CONTEXT_MODE="${TASK_CONTEXT_MODE:?TASK_CONTEXT_MODE is required when SCORER_PROFILE=custom}"
    qwen35_runner_prefix=("${SCORING_DIR}/run_local_qwen35_probe_logprob_scorer.py")
    ;;
  *)
    echo "[error] unsupported SCORER_PROFILE: $SCORER_PROFILE" >&2
    echo "[error] expected wrbench_default, ablation_manifest_metadata, legacy_p9_all_manifest_metadata, or custom" >&2
    exit 2
    ;;
esac

QWEN35_OUT="${OUT_DIR}/qwen35_worldstate_scores"
BINARY_GATE_OUT="${OUT_DIR}/qwen3vl_oov_binary_gate"
RESCUE_OUT="${OUT_DIR}/qwen3vl_oov_guarded_rescue"
GATE_OVERLAY_OUT="${OUT_DIR}/qwen3vl_oov_gate_overlay"
FINAL_OUT="${OUT_DIR}/final_exports"
LOG_DIR="${OUT_DIR}/logs"

usage() {
  cat <<'EOF'
Usage:
  MANIFEST=/abs/path/manifest.json OUT_DIR=/abs/path/out \
  QWEN35_MODEL=/abs/path/Qwen3.5-9B QWEN3VL_MODEL=/abs/path/Qwen3-VL-8B-Instruct \
  bash scripts/eval/score_runtime_v2_d3d6.sh <stage>

Stages:
  preflight | qwen35 | merge_qwen35 | gate_binary | merge_binary |
  build_rescue | gate_rescue | merge_rescue | overlay_gate | export | all

Manifest contract:
  JSON list. Each row must contain video_id, path or video_path, and
  world_state_prompt or prompt_text. Optional benchmark metadata fields are
  passed through when present.

Notes:
  Default SCORER_PROFILE is wrbench_default (alias: current_benchmark_p25_p22_e14):
  visible D3/D4 + returned D5/D6 probe logprob scoring with shared re-observation
  judgeability gate for returned-state metrics.
  Use SCORER_PROFILE=ablation_manifest_metadata (alias legacy_p9_all_manifest_metadata)
  only for ablation reruns.
EOF
}

stage="${1:-preflight}"
case "$stage" in
  preflight|qwen35|merge_qwen35|gate_binary|merge_binary|build_rescue|gate_rescue|merge_rescue|overlay_gate|export|all)
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "[error] unknown stage: $stage" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ -z "$MANIFEST" ]]; then
  echo "[error] MANIFEST is required" >&2
  usage >&2
  exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "[error] MANIFEST is not readable: $MANIFEST" >&2
  exit 2
fi

mkdir -p "$QWEN35_OUT" "$BINARY_GATE_OUT" "$RESCUE_OUT" "$GATE_OVERLAY_OUT" "$FINAL_OUT" "$LOG_DIR"

json_list_len() {
  "$PY_HELPER" -c 'import json, sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "$1"
}

run_sharded() {
  local name="$1"
  local py="$2"
  shift 2
  IFS=',' read -ra GPUS <<< "$CUDA_DEVICES_CSV"
  if [[ "${#GPUS[@]}" -eq 0 ]]; then
    echo "[error] CUDA_DEVICES_CSV must contain at least one device" >&2
    return 2
  fi
  local i=0
  local pids=()
  for sid in $SHARD_IDS; do
    local gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
    echo "[launch] ${name} shard=${sid}/${NUM_SHARDS} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="$gpu" "$py" "$@" \
      --num-shards "$NUM_SHARDS" \
      --shard-id "$sid" \
      --local-rank 0 \
      > "${LOG_DIR}/${name}_shard_${sid}.log" 2>&1 &
    pids+=("$!")
    i=$((i + 1))
  done
  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "$status"
}

preflight() {
  "$PY_HELPER" - "$MANIFEST" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
rows = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(rows, list):
    raise SystemExit("manifest must be a JSON list")
seen = set()
missing = []
unreadable = []
for idx, row in enumerate(rows):
    if not isinstance(row, dict):
        raise SystemExit(f"manifest row {idx} is not an object")
    video_id = str(row.get("video_id") or "").strip()
    video_path = str(row.get("path") or row.get("video_path") or "").strip()
    prompt = str(row.get("world_state_prompt") or row.get("prompt_text") or "").strip()
    if not video_id or not video_path or not prompt:
        missing.append(idx)
    if video_id in seen:
        raise SystemExit(f"duplicate video_id: {video_id}")
    seen.add(video_id)
    if video_path and not Path(video_path).exists():
        unreadable.append(video_id or f"row-{idx}")
if missing:
    raise SystemExit(f"rows missing video_id/path/world_state_prompt: {missing[:10]}")
if unreadable:
    raise SystemExit(f"unreadable video paths: {unreadable[:10]}")
print(f"[preflight] manifest_records={len(rows)}")
PY
}

qwen35_args=(
  "${qwen35_runner_prefix[@]}"
  --experiment-id "$RUN_TAG"
  --manifest-path "$MANIFEST"
  --output-dir "$QWEN35_OUT"
  --model-path "$QWEN35_MODEL"
  --fps "$FPS"
  --progress-every "$PROGRESS_EVERY"
  --prompt-mode "$PROMPT_MODE"
  --task-context-mode "$TASK_CONTEXT_MODE"
)
if [[ -n "$SOURCE_SCORES" ]]; then
  qwen35_args+=(--source-scores "$SOURCE_SCORES")
fi
if [[ "$STRICT_MANIFEST_CONTRACT" == "1" ]]; then
  qwen35_args+=(--strict-manifest-contract)
else
  qwen35_args+=(--skip-existing)
fi

run_stage() {
  local selected="$1"
  case "$selected" in
    preflight)
      preflight
      ;;
    qwen35)
      if [[ -z "$QWEN35_MODEL" || ! -d "$QWEN35_MODEL" ]]; then
        echo "QWEN35_MODEL must point to a local model directory (configure eval.scorers.qwen35_model)" >&2
        exit 1
      fi
      run_sharded qwen35 "$PY_SCORER" "${qwen35_args[@]}"
      ;;
    merge_qwen35)
      "$PY_SCORER" "${qwen35_args[@]}" --merge-only
      ;;
    gate_binary)
      run_sharded qwen3vl_binary "$PY_SCORER" \
        "${SCORING_DIR}/run_local_qwen3vl_video_evidence.py" \
        --manifest-path "$MANIFEST" \
        --output-dir "$BINARY_GATE_OUT" \
        --model-path "$QWEN3VL_MODEL" \
        --prompt-schema subject_judgeability_v1 \
        --fps "$FPS" \
        --skip-existing
      ;;
    merge_binary)
      "$PY_SCORER" "${SCORING_DIR}/run_local_qwen3vl_video_evidence.py" \
        --manifest-path "$MANIFEST" \
        --output-dir "$BINARY_GATE_OUT" \
        --model-path "$QWEN3VL_MODEL" \
        --prompt-schema subject_judgeability_v1 \
        --fps "$FPS" \
        --merge-only
      ;;
    build_rescue)
      "$PY_HELPER" "${SCORING_DIR}/build_qwen3vl_binary_na_rescue_manifest.py" \
        --manifest-path "$MANIFEST" \
        --binary-evidence-jsonl "$BINARY_GATE_OUT/evidence.jsonl" \
        --out-manifest "$RESCUE_OUT/manifest_qwen3vl_guarded_rescue.json" \
        --out-summary "$RESCUE_OUT/manifest_qwen3vl_guarded_rescue_summary.json"
      ;;
    gate_rescue)
      local rescue_records
      rescue_records="$(json_list_len "$RESCUE_OUT/manifest_qwen3vl_guarded_rescue.json")"
      if [[ "$rescue_records" == "0" ]]; then
        echo "[skip] qwen3vl_rescue: rescue manifest is empty"
      else
        run_sharded qwen3vl_rescue "$PY_SCORER" \
          "${SCORING_DIR}/run_local_qwen3vl_video_evidence.py" \
          --manifest-path "$RESCUE_OUT/manifest_qwen3vl_guarded_rescue.json" \
          --output-dir "$RESCUE_OUT" \
          --model-path "$QWEN3VL_MODEL" \
          --prompt-schema guarded_teacher_gate_v3 \
          --fps "$FPS" \
          --skip-existing
      fi
      ;;
    merge_rescue)
      local rescue_records
      rescue_records="$(json_list_len "$RESCUE_OUT/manifest_qwen3vl_guarded_rescue.json")"
      if [[ "$rescue_records" == "0" ]]; then
        : > "$RESCUE_OUT/evidence.jsonl"
        echo "[skip] qwen3vl_rescue merge: rescue manifest is empty"
      else
        "$PY_SCORER" "${SCORING_DIR}/run_local_qwen3vl_video_evidence.py" \
          --manifest-path "$RESCUE_OUT/manifest_qwen3vl_guarded_rescue.json" \
          --output-dir "$RESCUE_OUT" \
          --model-path "$QWEN3VL_MODEL" \
          --prompt-schema guarded_teacher_gate_v3 \
          --fps "$FPS" \
          --merge-only
      fi
      ;;
    overlay_gate)
      "$PY_HELPER" "${SCORING_DIR}/merge_qwen3vl_rescue_evidence.py" \
        --manifest-path "$MANIFEST" \
        --baseline-evidence-jsonl "$BINARY_GATE_OUT/evidence.jsonl" \
        --rescue-evidence-jsonl "$RESCUE_OUT/evidence.jsonl" \
        --out-evidence-jsonl "$GATE_OVERLAY_OUT/evidence_qwen3vl_guarded_overlay.jsonl" \
        --out-summary-json "$GATE_OVERLAY_OUT/evidence_qwen3vl_guarded_overlay_summary.json"
      ;;
    export)
      "$PY_HELPER" "${SCORING_DIR}/export_runtime_v2_evidence_first.py" \
        --scores-v7 "$QWEN35_OUT/scores_v7_candidate_runtime_v2_probe_score_available.json" \
        --evidence-jsonl "$GATE_OVERLAY_OUT/evidence_qwen3vl_guarded_overlay.jsonl" \
        --manifest-path "$MANIFEST" \
        --out-dir "$FINAL_OUT"
      ;;
  esac
}

if [[ "$stage" == "all" ]]; then
  for s in preflight qwen35 merge_qwen35 gate_binary merge_binary build_rescue gate_rescue merge_rescue overlay_gate export; do
    run_stage "$s"
  done
else
  run_stage "$stage"
fi

echo "[metric] out_dir=$OUT_DIR"
