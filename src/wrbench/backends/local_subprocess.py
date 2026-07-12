"""Config-driven local subprocess generation backend."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from wrbench.backends.base import GenerationRequest, GenerationResult
from wrbench.backends.launchers import launcher_spec, supported_models
from wrbench.registry import canonical_model_key
from wrbench.runtime import RuntimeConfig


class LocalSubprocessBackend:
    """Launch model-native inference using ``wrbench.runtime.json``."""

    name = "local_subprocess"

    def __init__(self, runtime: RuntimeConfig) -> None:
        self._runtime = runtime

    def available(self) -> tuple[bool, str]:
        if not self._runtime.models:
            return False, "runtime config has no model entries"
        supported = sorted(set(self._runtime.models) & supported_models())
        if not supported:
            return False, f"no supported model runtime entries (supported: {sorted(supported_models())})"
        return True, f"configured for {', '.join(supported)}"

    def available_for(self, model: str) -> tuple[bool, str]:
        key = canonical_model_key(model)
        spec = launcher_spec(key)
        if spec is None:
            return False, f"local_subprocess backend does not support {key!r} yet"
        runtime = self._runtime.model(key)
        if runtime is None:
            return False, f"no runtime entry for {key!r} in wrbench.runtime.json"
        missing = list(dict.fromkeys(spec.validate_runtime(runtime)))
        if missing:
            return False, f"missing or invalid runtime fields: {', '.join(missing)}"
        return True, f"ready for {key}"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        key = canonical_model_key(request.model)
        ok, reason = self.available_for(key)
        if not ok:
            return GenerationResult(success=False, message=reason)

        spec = launcher_spec(key)
        runtime = self._runtime.model(key)
        assert spec is not None and runtime is not None
        launch = spec.prepare(request, runtime)
        proc = subprocess.run(
            launch.cmd,
            cwd=str(launch.cwd),
            env={**os.environ, **launch.env},
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            tail = "\n".join(part for part in (proc.stdout, proc.stderr) if part)[-4096:]
            output_path = Path(request.output_path).resolve()
            return GenerationResult(
                success=False,
                output_path=output_path if output_path.is_file() else None,
                message=f"subprocess failed (exit {proc.returncode}): {tail}",
            )
        return launch.finalize()
