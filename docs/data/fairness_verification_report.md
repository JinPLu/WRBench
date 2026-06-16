# Fairness and capability verification report
Generated: 2026-06-16T19:18:57Z

Policy: **fair comparison + full capability utilization**, not perfect metrics.

- **Fairness gate**: VGGT-Omega/D1 direction-correct + frame decode clean (contact-sheet auto QC).
- **Advisory only**: D1 amplitude / mean accuracy (`NEEDS_CALIBRATION` ≠ automatic fail).
- **Capability**: native payload faithfulness via WRCam sidecar taxonomy (`certification_kind`, `target_c2w_is_model_effective`).

| model | fairness_gate | final_decision | direction_ok | frame_clean | d1_mean (advisory) | certification_kind | target_c2w_is_model_effective |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| cosmos3-nano-generator | NO_VIDEOS | NO_VIDEOS | None | None | None | compile-time in adapter | n/a |
| easyanimate-v51-camera | fair_direction_pass | ACCEPT | 1.0 | 1.0 | 0.7109300840087885 | compile-time in adapter | True |
| gen3c | advisory_amplitude_only | NEEDS_CALIBRATION | 0.6 | 1.0 | 0.8825131411840673 | compile-time in adapter | n/a |
| hunyuan-game-craft | advisory_amplitude_only | NEEDS_CALIBRATION | 0.75 | 1.0 | 0.4308538310191491 | compile-time in adapter | n/a |
| hunyuan-worldplay | fair_direction_pass | NEEDS_CALIBRATION | 0.95 | 1.0 | 0.5493237751132787 | compile-time in adapter | n/a |
| hunyuanworld-voyager | BLOCKED | BLOCKED | 0.0 | 1.0 | None | compile-time in adapter | True |
| hydra | fair_direction_pass | ACCEPT | 1.0 | 1.0 | 0.7528783221704645 | compile-time in adapter | n/a |
| hyworld-worldgen | BLOCKED | BLOCKED | None | None | None | compile-time in adapter | True |
| inspatio-world | advisory_amplitude_only | NEEDS_CALIBRATION | 0.5 | 1.0 | 0.3362165272958881 | compile-time in adapter | n/a |
| lingbot-world | advisory_amplitude_only | NEEDS_CALIBRATION | 0.5 | 1.0 | 0.27958285418986784 | compile-time in adapter | n/a |
| lingbot-world-act | fair_direction_pass | NEEDS_CALIBRATION | 0.95 | 1.0 | 0.5411589273529342 | compile-time in adapter | n/a |
| liveworld | fair_direction_pass | ACCEPT | 1.0 | 1.0 | 0.6749335969021878 | compile-time in adapter | n/a |
| magicworld | fair_direction_pass | ACCEPT | 1.0 | 1.0 | 0.7318267733924727 | compile-time in adapter | n/a |
| matrix-game-2 | NO_VIDEOS | NO_VIDEOS | None | None | None | compile-time in adapter | False |
| matrix-game-3 | fair_direction_pass | ACCEPT | None | 1.0 | None | compile-time in adapter | False |
| minwm-hy-action2v | NO_VIDEOS | NO_VIDEOS | None | None | None | compile-time in adapter | False |
| recammaster | fair_direction_pass | ACCEPT | 1.0 | 1.0 | 0.6566921355665001 | compile-time in adapter | n/a |
| sana-wm | fair_direction_pass | ACCEPT | 1.0 | 1.0 | 0.7593948903742633 | compile-time in adapter | n/a |
| spatia | advisory_amplitude_only | NEEDS_CALIBRATION | 0.75 | 1.0 | 0.512007600996119 | compile-time in adapter | n/a |
| versecrafter | fair_direction_pass | NEEDS_CALIBRATION | 0.9 | 1.0 | 0.6433810375386437 | compile-time in adapter | n/a |
| wan21-fun-14b-cam | fair_direction_pass | ACCEPT | 1.0 | 1.0 | 0.644722848738313 | compile-time in adapter | n/a |
| wan21-fun-1p3b-cam | fair_direction_pass | ACCEPT | 1.0 | 1.0 | 0.7362307337114251 | compile-time in adapter | n/a |
| wan22-fun-5b-cam | fair_direction_pass | ACCEPT | 1.0 | 1.0 | 0.6184077896394539 | compile-time in adapter | n/a |
| wan22-fun-a14b-cam | fair_direction_pass | ACCEPT | 1.0 | 1.0 | 0.6674993978958618 | compile-time in adapter | n/a |
