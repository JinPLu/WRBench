#!/usr/bin/env python3
"""Compatibility alias exporting ``wrbench.eval.aggregate.build_main_table``."""

from __future__ import annotations

from wrbench.eval.aggregate.build_main_table import (
    D1_SPEC,
    D2_SPEC,
    DIMENSION_SPECS,
    EXCLUDED_MAIN_TABLE_MODELS,
    GATE_FIELD,
    _fmt,
    _gate_applicable,
    _load_records,
    _viewpoint_condition,
    build_table,
    main,
    parse_args,
    write_csv,
    write_md,
)

__all__ = [
    "D1_SPEC",
    "D2_SPEC",
    "DIMENSION_SPECS",
    "EXCLUDED_MAIN_TABLE_MODELS",
    "GATE_FIELD",
    "_fmt",
    "_gate_applicable",
    "_load_records",
    "_viewpoint_condition",
    "build_table",
    "main",
    "parse_args",
    "write_csv",
    "write_md",
]

if __name__ == "__main__":
    raise SystemExit(main())
