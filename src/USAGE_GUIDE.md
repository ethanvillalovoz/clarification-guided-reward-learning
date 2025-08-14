# Clarification-Guided Reward Learning: Source Code Usage Guide

This document explains the purpose and usage of each file in the `src/` directory of the Clarification-Guided Reward Learning project. It is intended to help new users and contributors understand how to run experiments, extend the codebase, and integrate new features.

---

## File Overview

### 1. `clarification_guided_interaction.py`
- **Purpose:** Main entry point for running clarification-guided reward learning experiments.
- **Functionality:** Implements the interactive loop where the robot learns human preferences through demonstrations, corrections, and feature clarification questions. Maintains and updates a Bayesian belief distribution over preference models.
- **How to Run:**
  ```bash
  python src/clarification_guided_interaction.py
  ```
- **Customization:** Modify the main loop or object definitions to change the experiment setup.

### 2. `multi_object_mdp.py`
- **Purpose:** Defines the Markov Decision Process (MDP) logic, object properties, preference trees, and the `Gridworld` environment.
- **Functionality:** Encodes how objects are placed, how rewards are computed, and how preferences are represented. Used by the main interaction script.
- **Customization:** Add new object types, materials, or preference models by editing this file.

### 3. `utils/console.py`
- **Purpose:** Provides research-grade logging and console output utilities.
- **Functionality:** Enables color-coded, structured, and hierarchical terminal output for better experiment tracking and debugging.
- **Usage:** Import and use the `log` object for formatted output in your scripts.

### 4. `__init__.py`
- **Purpose:** Marks the `utils` directory as a Python package. Can be used for shared utility functions.

---

## Typical Workflow

1. **Configure Experiment:**
   - Edit `clarification_guided_interaction.py` to set up the objects, preference models, and experiment parameters.
2. **Run the Main Script:**
   - Use the command above to start an interactive session.
3. **Review Output:**
   - Belief updates and visualizations are saved in the `beliefs/` directory.
   - Console output provides step-by-step feedback and debugging information.
4. **Extend Functionality:**
   - Add new preference models or object types in `multi_object_mdp.py`.
   - Improve logging or output formatting in `utils/console.py`.

---

## Adding New Features
- To add new types of objects or preferences, update the relevant dictionaries and classes in `multi_object_mdp.py`.
- To change the interaction protocol or add new types of clarification, modify `clarification_guided_interaction.py`.
- For advanced logging or output, extend `utils/console.py`.

---

## Troubleshooting
- If you encounter import errors, ensure you are running scripts from the project root and that your Python path includes the `src/` directory.
- For missing dependencies, install required packages listed in `requirements.txt`.

---

## Contact
For questions or contributions, please contact the project maintainers listed in the `README.md`.
