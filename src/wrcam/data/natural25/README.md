# Natural-25 prompt suite

This directory ships the **Natural-25** scene/event prompt grid used by WRBench.

| File | Description |
| --- | --- |
| `scene_events_25x4.csv` | 25 scene families × 4 event axes (none / spatial / state / full) |
| `families.jsonl` | Scene-family metadata and prompt variants |

First-frame PNGs are generated per family via `wrcam firstframe` (optional extra)
or supplied by your own T2I pipeline. Human annotation verdicts are released
separately from this repository.
