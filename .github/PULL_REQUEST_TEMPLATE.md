# Pull Request

## Summary

Describe the implementation and the research assumption or behavior it changes.

## Evidence Scope

- Is this a code-path result, simulation result, or empirical result?
- Which baseline and configuration are held constant?
- What claim should **not** be inferred from this change?

## Validation

- [ ] `ruff check src tests scripts`
- [ ] `pytest -q`
- [ ] `MPLBACKEND=Agg clarification-reward-demo --output-dir artifacts/check`
- [ ] Default changes include an intentional update to `examples/reference-trace.json`
- [ ] Figures and docs were regenerated when assumptions or outputs changed
- [ ] No result is described more strongly than its evidence supports

## Artifacts

Attach or link the relevant trace, comparison, or failure reproduction.
