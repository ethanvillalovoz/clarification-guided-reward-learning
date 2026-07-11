# Changelog

## v1.1.0 - Reproducible Reference Simulation

### Added

- Installable `clarification_reward_learning` package and command-line runner.
- Tested Bradley-Terry correction likelihood, Bayesian normalization, entropy gating, and feature clarification.
- Deterministic correction-only baseline and committed JSON reference trace.
- Generated static comparison and animated belief-update artifacts.
- Explicit reproducibility notes, evidence status, limitations, and claim boundaries.
- Ruff, pytest, Python-version matrix CI, and artifact-contract checks.

### Changed

- Replaced the presentation-oriented execution path with a compact typed implementation.
- Moved historical prototypes and generated rollout clutter to the existing `v1.0.0` tag.
- Reduced runtime dependencies to NumPy, Matplotlib, and Pillow.

## v1.0.0 - Public Research Baseline

Initial polished public baseline for the CMU RISS 2024 clarification-guided reward-learning prototype.

### Added

- Research-grade README with project overview, visuals, artifact links, setup instructions, and verification commands.
- Lightweight unit tests for belief initialization and core Gridworld behavior.
- GitHub Actions CI for source compilation and unit tests.
- Research artifact notes for the working paper, poster, final presentation, and presentation video.
- Contributor, issue, and pull request templates tailored to the research prototype.

### Changed

- Cleaned dependency list and removed duplicate package entries.
- Removed placeholder deployment workflow.
- Tidied duplicate imports and stale roadmap text from the main interaction script without changing experiment behavior.
