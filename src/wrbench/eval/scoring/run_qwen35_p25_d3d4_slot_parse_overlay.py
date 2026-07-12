#!/usr/bin/env python3
"""Run Qwen3.5 with the declared P25 D3/D4 + P22 D5/D6 profile."""

from __future__ import annotations

import argparse
from pathlib import Path

from wrbench.eval.scoring.prompts_v2_probe import PROMPT_MODE_P25_D3D4_SLOT_PARSE


P25_PROMPT_MODE = PROMPT_MODE_P25_D3D4_SLOT_PARSE


def _ensure_p25_prompt_mode(runner_args: list[str]) -> list[str]:
    normalized = list(runner_args)
    for idx, arg in enumerate(normalized):
        if arg == "--prompt-mode":
            if idx + 1 >= len(normalized):
                raise SystemExit("--prompt-mode requires a value")
            if normalized[idx + 1] != P25_PROMPT_MODE:
                raise SystemExit(f"this P25 wrapper only supports --prompt-mode {P25_PROMPT_MODE}")
            return normalized
        if arg.startswith("--prompt-mode="):
            if arg.split("=", 1)[1] != P25_PROMPT_MODE:
                raise SystemExit(f"this P25 wrapper only supports --prompt-mode {P25_PROMPT_MODE}")
            return normalized
    return ["--prompt-mode", P25_PROMPT_MODE, *normalized]


def _ensure_no_task_context(runner_args: list[str]) -> list[str]:
    normalized = list(runner_args)
    for idx, arg in enumerate(normalized):
        if arg == "--task-context-mode":
            if idx + 1 >= len(normalized):
                raise SystemExit("--task-context-mode requires a value")
            if normalized[idx + 1] != "none":
                raise SystemExit("current P25/P22 scoring only supports --task-context-mode none")
            return normalized
        if arg.startswith("--task-context-mode="):
            if arg.split("=", 1)[1] != "none":
                raise SystemExit("current P25/P22 scoring only supports --task-context-mode none")
            return normalized
    return ["--task-context-mode", "none", *normalized]


def _reject_evidence_context_args(runner_args: list[str]) -> None:
    for idx, arg in enumerate(runner_args):
        if arg == "--evidence-jsonl":
            if idx + 1 >= len(runner_args):
                raise SystemExit("--evidence-jsonl requires a value")
            raise SystemExit("P25 clean task-slot parsing does not support --evidence-jsonl")
        if arg.startswith("--evidence-jsonl="):
            raise SystemExit("P25 clean task-slot parsing does not support --evidence-jsonl")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Qwen3.5 current P25/P22 probe scorer")
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Accepted for CLI compatibility; the scorer is now self-contained.",
    )
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    runner_args = args.runner_args
    if runner_args and runner_args[0] == "--":
        runner_args = runner_args[1:]
    if not runner_args:
        parser.error("pass runner arguments after --")

    runner_args = _ensure_p25_prompt_mode(runner_args)
    runner_args = _ensure_no_task_context(runner_args)
    _reject_evidence_context_args(runner_args)
    from wrbench.eval.scoring import run_local_qwen35_probe_logprob_scorer as runner

    return int(runner.main(runner_args))


if __name__ == "__main__":
    raise SystemExit(main())
