import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multi_object_mdp import EXIT, Gridworld, f_Ada, obj_1, obj_2, obj_3  # noqa: E402


class GridworldTests(unittest.TestCase):
    def setUp(self):
        self.game = Gridworld(f_Ada, [obj_1, obj_2, obj_3])

    def test_initializes_object_tuples_and_actions(self):
        expected_objects = [
            (2, 1, 1, 1),  # yellow glass cup
            (1, 1, 1, 2),  # red glass cup
            (3, 1, 2, 3),  # purple glass bowl
        ]

        self.assertEqual(self.game.object_type_tuple, expected_objects)
        self.assertEqual(len(self.game.possible_single_actions), 13)
        self.assertEqual(self.game.possible_single_actions[-1], EXIT)

    def test_initial_state_places_each_object_once(self):
        state = self.game.get_initial_state()

        self.assertFalse(state["exit"])
        self.assertEqual(state[(2, 1, 1, 1)]["pos"], (0, 0))
        self.assertEqual(state[(1, 1, 1, 2)]["pos"], (1, 0))
        self.assertEqual(state[(3, 1, 2, 3)]["pos"], (-1, 0))
        self.assertTrue(all(not state[obj]["done"] for obj in self.game.object_type_tuple))

    def test_lookup_quadrant_reward_uses_preference_tree(self):
        reward = self.game.lookup_quadrant_reward(self.game.get_initial_state())

        self.assertEqual(reward, 6)

    def test_step_given_state_moves_object_to_requested_quadrant(self):
        state = self.game.get_initial_state()
        yellow_cup = (2, 1, 1, 1)

        next_state, reward, done = self.game.step_given_state(state, (yellow_cup, "Q3"))

        self.assertEqual(next_state[yellow_cup]["pos"], (-2, -2))
        self.assertTrue(next_state[yellow_cup]["done"])
        self.assertAlmostEqual(reward, -0.1)
        self.assertFalse(done)


if __name__ == "__main__":
    unittest.main()
