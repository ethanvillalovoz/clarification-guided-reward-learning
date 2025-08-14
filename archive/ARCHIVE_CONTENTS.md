# Archive Folder Contents

This document describes the contents of the `archive/` directory in the Clarification-Guided Reward Learning project. The archive folder is used to store legacy files, experiment results, and utility scripts that are not part of the main codebase but may be useful for reference or reproducibility.

---

## Directory Structure

- `rollouts/`
  - Contains saved rollout images from experiments, such as state visualizations at different timesteps.
    - `state_0.png`, `state_1.png`, ..., `state_8.png`: Images showing the environment or agent state at various points during a rollout.

- `utility_scripts/`
  - Helper scripts for data processing, visualization, or experiment management.
    - `image_border_generator.py`: Script for adding borders to images, useful for presentation or publication-quality figures.
    - `preference_learning_interaction.py`: Early or alternative implementation of the preference learning interaction loop. May contain experimental or legacy code.

---

## Usage Notes
- The archive folder is not required for running the main experiments, but provides useful resources for analysis, visualization, and reproducibility.
- Utility scripts can be run independently for data preparation or figure generation.
- Rollout images can be used in reports, presentations, or for debugging experiment results.

---

## Best Practices
- Store only non-essential or legacy files in the archive to keep the main codebase clean.
- Use descriptive filenames and keep this document updated as new files are added or removed.

---

## Contact
For questions about archived files or scripts, contact the project maintainers listed in the main `README.md`.
