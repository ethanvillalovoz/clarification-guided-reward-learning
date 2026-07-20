# System Overview Figure Contract

## Claim

This figure should allow a skeptical robotics reviewer to conclude that, in the fixed default simulation, a feature clarification disambiguates reward hypotheses left compatible by the same state correction because it shows the corrected state, posterior before and after clarification, and the next action under both conditions.

## Role and size

- Role: README teaser and research-system overview.
- Final width: 7.16 inches, suitable for a two-column paper width.
- Final height: 5.60 inches, allowing complete labels and legible type at placed size.
- Editable source: `src/clarification_reward_learning/visualization.py`.
- Exports: vector SVG/PDF and a 300 dpi PNG.

## Evidence

- Default comparison from `run_comparison()`.
- First corrected object: `yellow glass cup`, robot `Q1`, human `Q2`.
- Quadrant convention: `Q1` upper-right, `Q2` upper-left, `Q3` lower-left,
  and `Q4` lower-right, matching the original project paper and simulation utilities.
- Posterior and normalized entropy after that correction and clarification.
- Immediately following object: `red glass cup`.
- Correction-only next action `Q4`, corrected by the human to `Q1`.
- Clarification-condition next action `Q1`, matching the human action.

The first correction and immediately following object are selected by temporal order before inspecting outcomes; they are not cherry-picked examples.

## Conditions and boundary

- Six hand-authored reward hypotheses and three synthetic objects.
- Designated true hypothesis: `color + object type`.
- Bradley-Terry `beta=2.0`.
- Feature-answer match likelihood `0.8`.
- Clarification threshold `0.55` on normalized entropy.
- One deterministic trace with no seeds, trials, participants, or uncertainty estimate.

The figure does not establish real-robot performance, user-study outcomes, task-completion improvement, statistical significance, or robustness beyond the configured toy hypothesis space.
