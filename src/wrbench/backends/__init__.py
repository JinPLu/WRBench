"""Generation backends (optional real inference)."""

from wrbench.backends.base import (
    DryRunBackend,
    GenerationBackend,
    GenerationRequest,
    GenerationResult,
    default_backend,
)
from wrbench.backends.command_template import CommandTemplateBackend
from wrbench.backends.local_subprocess import LocalSubprocessBackend
from wrbench.backends.registry import (
    list_backends,
    register_generation_backend,
    registered_generation_backends,
    resolve_backend,
)

__all__ = [
    "CommandTemplateBackend",
    "DryRunBackend",
    "GenerationBackend",
    "GenerationRequest",
    "GenerationResult",
    "LocalSubprocessBackend",
    "default_backend",
    "list_backends",
    "register_generation_backend",
    "registered_generation_backends",
    "resolve_backend",
]
