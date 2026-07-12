#!/usr/bin/env python3
"""Run Qwen3.5 scoring with the declared P22 D5/D6 prompt profile."""

from __future__ import annotations

import argparse
from pathlib import Path

from wrbench.eval.scoring.prompts_v2_probe import PROMPT_MODE_P22_NATIVE_D5D6


P22_PROMPT_MODE = PROMPT_MODE_P22_NATIVE_D5D6


def _ensure_p22_prompt_mode(runner_args: list[str]) -> list[str]:
    normalized = list(runner_args)
    for idx, arg in enumerate(normalized):
        if arg == "--prompt-mode":
            if idx + 1 >= len(normalized):
                raise SystemExit("--prompt-mode requires a value")
            if normalized[idx + 1] != P22_PROMPT_MODE:
                raise SystemExit(f"this P22 wrapper only supports --prompt-mode {P22_PROMPT_MODE}")
            return normalized
        if arg.startswith("--prompt-mode="):
            if arg.split("=", 1)[1] != P22_PROMPT_MODE:
                raise SystemExit(f"this P22 wrapper only supports --prompt-mode {P22_PROMPT_MODE}")
            return normalized
    return ["--prompt-mode", P22_PROMPT_MODE, *normalized]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Qwen3.5 probe scorer with P22 native D5/D6 score prompts"
    )
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

    from wrbench.eval.scoring import run_local_qwen35_probe_logprob_scorer as runner

    return int(runner.main(_ensure_p22_prompt_mode(runner_args)))


if __name__ == "__main__":
    raise SystemExit(main())
