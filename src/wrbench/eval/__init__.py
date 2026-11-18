"""WRBench D1-D6 evaluation metrics and pluggable evaluation SPI."""

from wrbench.eval.aggregate import latest_d1_d6_metrics
from wrbench.eval.plugins import (
    DEFAULT_D1_SCORER_CONFIG,
    ConcatSegmentPolicy,
    get_pose_backend,
    register_d2_scorer,
    register_pose_backend,
    register_segment_policy,
    register_vlm_scorer,
    registered_d2_scorers,
    registered_pose_backends,
    registered_segment_policies,
    registered_vlm_scorers,
)

__all__ = [
    "ConcatSegmentPolicy",
    "DEFAULT_D1_SCORER_CONFIG",
    "get_pose_backend",
    "latest_d1_d6_metrics",
    "register_d2_scorer",
    "register_pose_backend",
    "register_segment_policy",
    "register_vlm_scorer",
    "registered_d2_scorers",
    "registered_pose_backends",
    "registered_segment_policies",
    "registered_vlm_scorers",
]
