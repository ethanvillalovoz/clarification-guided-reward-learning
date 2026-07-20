# Reproducibility

## Scope

The reproducible unit is one deterministic simulation over six reward hypotheses and three objects. It is intended to verify the inference path and expose assumptions. It is not a participant experiment.

## Environment

- Python 3.11 or newer.
- NumPy for probability vectors.
- Matplotlib and Pillow for artifacts.
- No GPU, network service, dataset, or random seed is required.

## Command

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
MPLBACKEND=Agg clarification-reward-demo --output-dir artifacts/latest
```

Regenerate the committed README overview from the same default simulation:

```bash
MPLBACKEND=Agg python scripts/render_public_figures.py
```

## Default Configuration

| Parameter | Value |
| --- | ---: |
| Hypotheses | 6 |
| Objects | 3 |
| Bradley-Terry `beta` | 2.0 |
| Feature-answer match likelihood | 0.8 |
| Clarification entropy threshold | 0.55 |
| Designated true hypothesis | color + object type |

The correction-only and clarification conditions use the same objects, true hypothesis, human actions, and correction model. Their only difference is whether the entropy-gated feature answer is applied.

## Trace Schema

Each step in `trace.json` records:

- Object and timestep.
- Robot action and human-preferred action.
- Prior and posterior after the correction.
- Whether clarification was asked.
- Feature answer and posterior after clarification.
- Normalized entropy before and after clarification.

## Interpreting The Output

The final posterior and entropy are deterministic consequences of the configured toy problem. They can catch regressions and illustrate the proposed mechanism. They cannot estimate real-world effect size, statistical significance, usability, or robustness.

## Changing Assumptions

- Edit `default_problem()` in `models.py` to change objects or hypotheses.
- Pass CLI parameters to vary `beta`, clarification accuracy, or entropy threshold.
- Use `run_simulation()` directly to supply a custom hypothesis space.
- Add a test whenever changing the observation model or normalization behavior.
