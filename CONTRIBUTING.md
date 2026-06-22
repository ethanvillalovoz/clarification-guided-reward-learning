# Contributing

Thanks for your interest in improving Clarification-Guided Reward Learning. This repository is a research prototype, so the best contributions are focused, reproducible, and careful about preserving the behavior of the experiment loop.

## Good Contribution Areas

- Documentation improvements for setup, usage, or research artifacts.
- Lightweight tests for reward lookup, belief updates, state transitions, and utility behavior.
- Visualization cleanup that preserves the generated figure intent.
- New preference trees, object configurations, or experiment settings with clear notes.
- Bug fixes that include a small reproduction or regression test when practical.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Verification

Before opening a pull request, run:

```bash
python -m py_compile src/clarification_guided_interaction.py src/multi_object_mdp.py src/utils/console.py
python -m unittest discover tests
```

If your change affects the interactive demo, also run:

```bash
python src/clarification_guided_interaction.py
```

## Pull Request Guidelines

- Keep pull requests scoped to one change.
- Explain the experiment behavior you expect to preserve.
- Include screenshots or generated plots when visualization output changes.
- Add tests for deterministic logic when possible.
- Avoid committing generated cache files, local virtual environments, or OS metadata.

## Reporting Issues

When opening an issue, please include:

- What command you ran.
- What you expected to happen.
- What happened instead.
- Your Python version and operating system.
- Any relevant screenshots, plots, or traceback output.
