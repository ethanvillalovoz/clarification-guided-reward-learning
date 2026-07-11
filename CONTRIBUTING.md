# Contributing

Contributions should make the inference easier to inspect, reproduce, or falsify. This repository is a simulation prototype, so claim discipline matters as much as code quality.

## Good Contributions

- Numerical-stability fixes with regression tests.
- Alternative observation or clarification models with explicit assumptions.
- New toy hypothesis spaces that include a committed trace and interpretation.
- Reproducibility, documentation, or accessibility improvements.
- Evaluations that compare against the same correction sequence and report uncertainty honestly.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Verification

```bash
ruff check src tests
pytest -q
MPLBACKEND=Agg clarification-reward-demo --output-dir artifacts/check
```

## Pull Requests

- State the assumption or behavior being changed.
- Distinguish deterministic simulation output from empirical evidence.
- Add a focused regression test for inference changes.
- Include regenerated figures when visualization or defaults change.
- Update `examples/reference-trace.json` only when an intentional default changes.
- Do not silently change the designated true hypothesis, baselines, or likelihood model.

## Historical Artifact

The original RISS implementation is preserved in `v1.0.0`. Do not reintroduce the archived presentation code into the main package; link to the tag when historical context is needed.
