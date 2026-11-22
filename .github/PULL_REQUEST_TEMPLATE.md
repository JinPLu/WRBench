## Summary

<!-- Briefly describe what this PR changes and why. -->

## Type of change

- [ ] Bug fix
- [ ] New feature / plugin capability
- [ ] Model adapter or generation backend
- [ ] Evaluation pipeline / metric contract update
- [ ] Documentation / governance

## Verification checklist

- [ ] `python3 -m pytest -q` passes
- [ ] `bash scripts/oss_verify.sh` passes (leak scan + clean venv install + release contract checks)
- [ ] No private paths, internal hostnames, credentials, or untracked artifacts introduced
- [ ] Frozen paper artifacts under `src/wrbench/data/` are untouched (unless explicitly part of a reviewed release update)
- [ ] If adding/updating models: `wrbench doctor --model <model-key>` passes
- [ ] If updating metric columns or scorers: `wrbench eval contract` artifacts are regenerated and checked
