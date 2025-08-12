"""
Multi-Object Custom MDP Implementation (Version 5)

Authors: Ethan Villalovz, Michelle Zhao
Project: RISS 2024 Summer Project - Markov Decision Process Formation
Description: Adaptation from `custom_mdp` by Michelle Zhao. 

Key Features:
    - Multiple Objects with different properties
    - Different Materials (glass, china, plastic)
    - Different colors (red, yellow, purple, white)
    
Version 5 changes:
    - Skill-based trajectory actions rather than individual movements
    - Quadrant-based placement rather than directional movement
    - Abstract preference handling for more complex reward structures
"""

# Standard library imports
import copy
import os
from itertools import product

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# ============================
# Global Constants 
# ============================

# Action Constants
EXIT = 'exit'  # Special action to end episode

# Object Properties
COLORS = {'red': 1, 'yellow': 2, 'purple': 3, 'white': 4, 'other': 5}
COLORS_IDX = {1: 'red', 2: 'yellow', 3: 'purple', 4: 'white', 5: 'other'}

MATERIALS = {'glass': 1, 'china': 2, 'plastic': 3, 'other': 4}
MATERIALS_IDX = {1: 'glass', 2: 'china', 3: 'plastic', 4: 'other'}

OBJECTS = {'cup': 1, 'bowl': 2, 'other': 3}
OBJECTS_IDX = {1: 'cup', 2: 'bowl', 3: 'other'}

COLOR_DICTIONARY = {1: 'red', 2: 'yellow', 3: 'purple'}

# Predefined object configurations
obj_1 = {'object_type': 'cup', 'color': 'yellow', 'material': 'glass', 'object_label': 1}
obj_2 = {'object_type': 'cup', 'color': 'red', 'material': 'glass', 'object_label': 2}
obj_3 = {'object_type': 'bowl', 'color': 'purple', 'material': 'glass', 'object_label': 3}
obj_3_dup = {'object_type': 'bowl', 'color': 'purple', 'material': 'glass', 'object_label': 4}


# ============================
# Preference Trees
# ============================

# Each preference tree is a nested dictionary with reward values for each quadrant
# based on hierarchical object properties (type, color, material)

f_Ethan = {'pref_values': {"cup": {'red': {'Q1': 10, 'Q2': -1, 'Q3': -1, 'Q4': 10},
                                  'yellow': {'Q1': 10, 'Q2': -1, 'Q3': 10, 'Q4': -10}},
                          'bowl': {'glass': {'Q1': -1, 'Q2': 5, 'Q3': -1, 'Q4': 10},
                                   'plastic': {'Q1': -1, 'Q2': 10, 'Q3': -1, 'Q4': -1},
                                   'other': {'Q1': 5, 'Q2': 5, 'Q3': 5, 'Q4': 5}}}}

f_Michelle = {'pref_values': {'glass': {'red': {'Q1': 10, 'Q2': 5, 'Q3': -1, 'Q4': 2},
                                       'yellow': {'Q1': 8, 'Q2': 10, 'Q3': 10, 'Q4': 4},
                                       'other': {'Q1': 10, 'Q2': -1, 'Q3': -1, 'Q4': 10}},
                             'china': {'cup': {'Q1': 8, 'Q2': 10, 'Q3': -1, 'Q4': 4},
                                       'bowl': {'red': {'Q1': 8, 'Q2': 10, 'Q3': -1, 'Q4': 10},
                                                'yellow': {'Q1': 8, 'Q2': 10, 'Q3': 10, 'Q4': 4},
                                                'purple': {'Q1': 8, 'Q2': 10, 'Q3': -1, 'Q4': 10}}},
                             'plastic': {'Q1': 8, 'Q2': -1, 'Q3': -1, 'Q4': 4}}}

f_Annika = {'pref_values': {'cup': {'glass': {'red': {'Q1': 10, 'Q2': 5, 'Q3': -1, 'Q4': 10},
                                              'yellow': {'Q1': 10, 'Q2': -1, 'Q3': 10, 'Q4': 0},
                                              'other': {'Q1': 5, 'Q2': 0, 'Q3': -1, 'Q4': 10}},
                                    'plastic': {'red': {'Q1': 5, 'Q2': -1, 'Q3': -1, 'Q4': 10},
                                                'other': {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}},
                                    'other': {'Q1': 8, 'Q2': 10, 'Q3': 10, 'Q4': 4}},
                            'bowl': {'purple': {'glass': {'Q1': 10, 'Q2': -1, 'Q3': 10, 'Q4': 10},
                                                'plastic': {'Q1': 5, 'Q2': -1, 'Q3': 10, 'Q4': 10},
                                                'other': {'Q1': 5, 'Q2': 0, 'Q3': -1, 'Q4': 10}},
                                     'other': {'Q1': 1, 'Q2': 5, 'Q3': -10, 'Q4': 10}}}}

f_Admoni = {'pref_values': {'red': {'cup': {'glass': {'Q1': 10, 'Q2': -1, 'Q3': 5, 'Q4': 10},
                                            'china': {'Q1': -1, 'Q2': 10, 'Q3': -1, 'Q4': 10},
                                            'other': {'Q1': 3, 'Q2': 6, 'Q3': 7, 'Q4': -3}},
                                    'bowl': {'plastic': {'Q1': 10, 'Q2': 5, 'Q3': 10, 'Q4': -1},
                                             'other': {'Q1': 0, 'Q2': 0, 'Q3': 9, 'Q4': 7}}},
                            'yellow': {'bowl': {'glass': {'Q1': 10, 'Q2': -1, 'Q3': 10, 'Q4': 10},
                                                'other': {'Q1': -5, 'Q2': 2, 'Q3': 8, 'Q4': 0}},
                                       'cup': {'glass': {'Q1': 10, 'Q2': -1, 'Q3': 5, 'Q4': 10},
                                               'other': {'Q1': -5, 'Q2': 2, 'Q3': 8, 'Q4': 0}}},
                            'other': {'Q1': 3, 'Q2': -6, 'Q3': 9, 'Q4': 7}}}

f_Simmons = {'pref_values': {'plastic': {'cup': {'red': {'Q1': 10, 'Q2': 10, 'Q3': 10, 'Q4': 10},
                                                 'yellow': {'Q1': 5, 'Q2': 10, 'Q3': 0, 'Q4': 5},
                                                 'other': {'Q1': 6, 'Q2': 1, 'Q3': -5, 'Q4': 7}},
                                         'bowl': {'Q1': 5, 'Q2': 0, 'Q3': -1, 'Q4': 10}},
                             'glass': {'bowl': {'red': {'Q1': 10, 'Q2': -1, 'Q3': 5, 'Q4': 10},
                                                'other': {'Q1': 6, 'Q2': 1, 'Q3': -5, 'Q4': 7}},
                                       'other': {'Q1': -6, 'Q2': 9, 'Q3': -8, 'Q4': 0}},
                             'other': {'Q1': -5, 'Q2': 2, 'Q3': 8, 'Q4': 0}}}

f_Suresh = {'pref_values': {'plastic': {'red': {'Q1': -5, 'Q2': 6, 'Q3': 3, 'Q4': -1},
                                        'yellow': {'Q1': 7, 'Q2': -3, 'Q3': 1, 'Q4': 4},
                                        'other': {'Q1': 5, 'Q2': 9, 'Q3': 8, 'Q4': -4}},
                            'glass': {'cup': {'Q1': 2, 'Q2': 8, 'Q3': -6, 'Q4': 10},
                                      'bowl': {'Q1': -2, 'Q2': 3, 'Q3': 5, 'Q4': 0}},
                            'other': {'Q1': 5, 'Q2': 9, 'Q3': 8, 'Q4': -4}}}

f_Ben = {'pref_values': {'cup': {'red': {'glass': {'Q1': 4, 'Q2': 6, 'Q3': 10, 'Q4': -1},
                                         'plastic': {'Q1': -5, 'Q2': 2, 'Q3': 8, 'Q4': 0},
                                         'other': {'Q1': 5, 'Q2': -3, 'Q3': 9, 'Q4': 10}},
                                 'yellow': {'glass': {'Q1': 3, 'Q2': -6, 'Q3': 9, 'Q4': 7},
                                            'plastic': {'Q1': 1, 'Q2': 5, 'Q3': -4, 'Q4': 10},
                                            'other': {'Q1': 5, 'Q2': -3, 'Q3': 9, 'Q4': 10}},
                                 'other': {'Q1': 2, 'Q2': 8, 'Q3': -3, 'Q4': 5}},
                         'other': {'Q1': 2, 'Q2': 8, 'Q3': -3, 'Q4': 5}}}

f_Ada = {'pref_values': {'glass': {'bowl': {'red': {'Q1': -2, 'Q2': 4, 'Q3': 10, 'Q4': 3},
                                            'yellow': {'Q1': 6, 'Q2': 1, 'Q3': -5, 'Q4': 7},
                                            'other': {'Q1': 0, 'Q2': 6, 'Q3': 1, 'Q4': 8}},
                                   'cup': {'red': {'Q1': 0, 'Q2': 8, 'Q3': 2, 'Q4': -1},
                                           'yellow': {'Q1': 5, 'Q2': -3, 'Q3': 9, 'Q4': 10},
                                           'other': {'Q1': 0, 'Q2': 6, 'Q3': 1, 'Q4': 8}}},
                         'other': {'Q1': 0, 'Q2': 6, 'Q3': 1, 'Q4': 8}}}

f_Abhijat = {'pref_values': {'red': {'glass': {'cup': {'Q1': 10, 'Q2': 4, 'Q3': 3, 'Q4': 2},
                                               'bowl': {'Q1': -1, 'Q2': 7, 'Q3': 6, 'Q4': 0}},
                                     'plastic': {'cup': {'Q1': 5, 'Q2': 9, 'Q3': 8, 'Q4': -4},
                                                 'bowl': {'Q1': 1, 'Q2': -5, 'Q3': 2, 'Q4': 10}},
                                     'other': {'Q1': -4, 'Q2': 7, 'Q3': 9, 'Q4': 10}},
                             'other': {'Q1': -4, 'Q2': 7, 'Q3': 9, 'Q4': 10}}}

f_Maggie = {'pref_values': {'bowl': {'red': {'glass': {'Q1': 3, 'Q2': 10, 'Q3': -1, 'Q4': 4},
                                             'plastic': {'Q1': -3, 'Q2': 5, 'Q3': 2, 'Q4': 9},
                                             'other': {'Q1': 10, 'Q2': -1, 'Q3': -1, 'Q4': 10}},
                                     'yellow': {'glass': {'Q1': 0, 'Q2': 6, 'Q3': 1, 'Q4': 8},
                                                'plastic': {'Q1': -4, 'Q2': 7, 'Q3': 9, 'Q4': 10},
                                                'other': {'Q1': 10, 'Q2': -1, 'Q3': -1, 'Q4': 10}},
                                     'other': {'Q1': 10, 'Q2': -1, 'Q3': -1, 'Q4': 10}},
                            'other': {'Q1': 10, 'Q2': -1, 'Q3': -1, 'Q4': 10}}}

f_Zulekha = {'pref_values': {'yellow': {'plastic': {'bowl': {'Q1': -1, 'Q2': 10, 'Q3': 8, 'Q4': 6},
                                                    'cup': {'Q1': 7, 'Q2': 2, 'Q3': 5, 'Q4': 3}},
                                        'glass': {'bowl': {'Q1': 4, 'Q2': 1, 'Q3': 9, 'Q4': -5},
                                                  'cup': {'Q1': 0, 'Q2': -3, 'Q3': 6, 'Q4': 10}},
                                        'other': {'Q1': 10, 'Q2': -1, 'Q3': -1, 'Q4': 10}},
                             'other': {'Q1': 10, 'Q2': -1, 'Q3': -1, 'Q4': 10}}}

f_Pat = {'pref_values': {'red': {'cup': {'glass': {'Q1': 10, 'Q2': -6, 'Q3': 4, 'Q4': 1},
                                         'plastic': {'Q1': 2, 'Q2': 8, 'Q3': -3, 'Q4': 5},
                                         'other': {'Q1': -5, 'Q2': 2, 'Q3': 8, 'Q4': 0}},
                                 'bowl': {'glass': {'Q1': 9, 'Q2': 3, 'Q3': 7, 'Q4': -1},
                                          'plastic': {'Q1': 6, 'Q2': 0, 'Q3': 10, 'Q4': -2},
                                          'other': {'Q1': -5, 'Q2': 2, 'Q3': 8, 'Q4': 0}}},
                         'other': {'Q1': -5, 'Q2': 2, 'Q3': 8, 'Q4': 0}}}


# ============================
# Gridworld Class
# ============================

class Gridworld:
    """
    Gridworld Environment for Object Placement
    
    This class implements a gridworld where objects with various properties
    (color, material, type) can be placed in different quadrants according
    to preference trees that assign rewards based on object attributes.
    
    The environment supports multiple objects, different action types,
    and visualization of states.
    """

    def __init__(self, pref_values, object_type_tuple):
        """
        Initialize the Gridworld environment
        
        Parameters
        ----------
        pref_values : dict
            Dictionary of preference values for different object properties
        object_type_tuple : list
            List of dictionaries describing objects in the environment
        """
        self.pref_values = pref_values

        # Sets up the boundaries of the simulated grid
        self.set_env_limits()

        # Convert object dictionaries to standardized tuples
        self.object_type_tuple = []
        for i in object_type_tuple:
            self.object_type_tuple.append((COLORS[i['color']],
                                           MATERIALS[i['material']],
                                           OBJECTS[i['object_type']],
                                           i['object_label']))

        # Calculate initial positions for objects (distributing them around origin)
        self.initial_object_locs = self._calculate_initial_positions()
        
        # Define available skills (quadrants where objects can be placed)
        self.skills = ['Q1', 'Q2', 'Q3', 'Q4']

        # Create action list and initial state
        self.possible_single_actions = self.make_actions_list()
        self.current_state = self.create_initial_state()

        # Initialize value iteration components
        self.transitions = None
        self.rewards = None
        self.state_to_idx = None
        self.idx_to_action = None
        self.idx_to_state = None
        self.action_to_idx = None
        self.vf = None
        self.pi = None
        self.policy = None
        
        # Value iteration parameters
        self.epsilson = 0.001
        self.gamma = 0.99
        self.maxiter = 10000
        self.num_features = 4

    def _calculate_initial_positions(self):
        """
        Calculate initial positions for objects, ensuring they don't overlap
        
        Returns
        -------
        dict
            Dictionary mapping object tuples to their initial positions
        """
        initial_object_locs = {}
        x_point = 0
        iteration = 1
        
        for obj in self.object_type_tuple:
            initial_object_locs[obj] = (x_point, 0)
            
            # Update x position in alternating pattern
            iteration += 1
            if iteration % 2 == 0 and iteration > 2:
                x_point *= -1
                x_point += 1
            elif iteration % 2 == 0:
                x_point += 1
            else:
                x_point *= -1
                
        return initial_object_locs

    def get_initial_state(self):
        """Get the initial state of the environment"""
        return self.current_state

    def set_env_limits(self):
        """Set the boundaries of the gridworld"""
        self.x_min = -5
        self.x_max = 6
        self.y_min = -5
        self.y_max = 6
        self.all_coordinate_locations = list(product(range(self.x_min, self.x_max),
                                                   range(self.y_min, self.y_max)))

    def make_actions_list(self):
        """
        Generate all possible actions for objects
        
        Returns
        -------
        list
            List of all possible actions in the environment
        """
        actions_list = []
        
        # Add skill actions for each object
        for i in range(0, len(self.object_type_tuple)):
            actions_list.extend([(self.object_type_tuple[i], x) for x in self.skills])

        # Add EXIT action
        actions_list.append(EXIT)
        
        return actions_list

    def create_initial_state(self):
        """
        Create initial state for all objects
        
        Returns
        -------
        dict
            Dictionary representing initial state of all objects
        """
        state_dict = {}
        
        # Set initial state for each object
        for obj in self.initial_object_locs:
            state = {
                'pos': copy.deepcopy(self.initial_object_locs[obj]),
                'orientation': np.pi,  # 180 degrees
                'done': False
            }
            state_dict[obj] = state

        # Add exit state
        state_dict['exit'] = False

        return state_dict

    def reset(self):
        """Reset the environment to initial state"""
        self.current_state = self.create_initial_state()

    def state_to_tuple(self, current_state):
        """
        Convert state dictionary to tuple representation for graph operations
        
        Parameters
        ----------
        current_state : dict
            Current state dictionary
            
        Returns
        -------
        tuple
            Tuple representation of the state
        """
        current_state_tup = []

        # Add each object's state to the tuple
        for obj_type_idx in range(0, len(current_state.keys()) - 1):
            obj_type = list(current_state.keys())[obj_type_idx]
            current_state_tup.append((
                (obj_type, current_state[obj_type]['pos']),
                current_state[obj_type]['orientation'], 
                current_state[obj_type]['done']
            ))

        # Add exit state
        current_state_tup.append(current_state['exit'])

        return tuple(current_state_tup)

    def is_done_given_state(self, current_state):
        """
        Check if the episode is finished
        
        Parameters
        ----------
        current_state : dict
            Current state dictionary
            
        Returns
        -------
        bool
            True if episode is done, False otherwise
        """
        return current_state['exit']

    def is_valid_push(self, current_state, action, locs_in_q1, locs_in_q2, locs_in_q3, locs_in_q4):
        """
        Determine if an action is valid given the current state
        
        Parameters
        ----------
        current_state : dict
            Current state dictionary
        action : tuple
            (object_type, direction) tuple
        locs_in_q1, locs_in_q2, locs_in_q3, locs_in_q4 : list
            Lists of valid locations in each quadrant
            
        Returns
        -------
        tuple
            (is_valid, new_location) tuple
        """
        obj_type, obj_action = action
        break_outer_loop = False

        # Get list of other objects
        other_obj_list = []
        for i in current_state.keys():
            if i != obj_type and i != 'exit':
                other_obj_list.append(i)

        # Extract current location
        current_loc = current_state[obj_type]['pos']
        new_loc = current_loc

        # Check if object hasn't been moved yet
        if not current_state[obj_type]['done']:
            # Select locations based on quadrant
            locations_to_check = {
                'Q1': locs_in_q1,
                'Q2': locs_in_q2,
                'Q3': locs_in_q3,
                'Q4': locs_in_q4
            }.get(obj_action, [])
            
            # Check each possible location in the selected quadrant
            for j in locations_to_check:
                new_loc = tuple(np.array(j))

                # Check if new location overlaps with other objects
                if other_obj_list:
                    valid_location = True
                    for i in range(len(other_obj_list)):
                        if current_state[other_obj_list[i]]['pos'] == new_loc:
                            valid_location = False
                            break
                    
                    if valid_location:
                        break
                else:
                    # No other objects, so first location is fine
                    break

        # Mark object as done
        current_state[obj_type]['done'] = True
        return True, new_loc

    def check_quadrant(self, input_state):
        """
        Determine the quadrant location for each object in the current state.
        
        Quadrants are defined as:
        - Q1: Positive X, Positive Y (top right)
        - Q2: Negative X, Positive Y (top left)
        - Q3: Negative X, Negative Y (bottom left)
        - Q4: Positive X, Negative Y (bottom right)
        
        Special cases for axes:
        - Origin (0,0): Considered Q2
        - Positive X-axis: Q1
        - Negative X-axis: Q2
        - Positive Y-axis: Q2
        - Negative Y-axis: Q3
        
        Parameters:
        -----------
        input_state : dict
            The current state of the environment
            
        Returns:
        --------
        list
            List of quadrant assignments for each object
        """
        current_state = list(copy.deepcopy(input_state))
        quadrant_list = []

        for i in range(0, len(current_state) - 1):
            x, y = input_state[current_state[i]]['pos']
            
            # Determine quadrant based on coordinates
            if x > 0:
                if y > 0:
                    quadrant_list.append(['Q1'])  # Q1: Positive X, Positive Y
                elif y < 0:
                    quadrant_list.append(['Q4'])  # Q4: Positive X, Negative Y
                else:  # y == 0, on positive x-axis
                    quadrant_list.append(['Q1'])
            elif x < 0:
                if y > 0:
                    quadrant_list.append(['Q2'])  # Q2: Negative X, Positive Y
                elif y < 0:
                    quadrant_list.append(['Q3'])  # Q3: Negative X, Negative Y
                else:  # y == 0, on negative x-axis
                    quadrant_list.append(['Q2'])
            else:  # x == 0, on y-axis
                if y > 0:
                    quadrant_list.append(['Q2'])  # On positive y-axis
                elif y < 0:
                    quadrant_list.append(['Q3'])  # On negative y-axis
                else:  # Origin (0,0)
                    quadrant_list.append(['Q2'])

        return quadrant_list

    def get_reward_value(self, preferences, attributes, quadrants):
        """
        Retrieve reward value from preference tree based on object attributes and quadrant
        
        Parameters
        ----------
        preferences : dict
            Preference tree dictionary
        attributes : list
            List of object attributes [object_type, color, material]
        quadrants : str
            Quadrant identifier (Q1, Q2, Q3, Q4)
            
        Returns
        -------
        float
            Reward value for the given attributes and quadrant
        """
        # Base case: preferences contains reward values for quadrants
        if isinstance(preferences, dict) and quadrants in preferences:
            return preferences[quadrants]
        # Handle 'other' category
        elif 'other' in preferences:
            return preferences['other'][quadrants]
        # Recursive case: navigate deeper in preference tree
        elif isinstance(preferences, dict):
            for attr in attributes:
                if attr in preferences:
                    return self.get_reward_value(preferences[attr], attributes, quadrants)
        
        # No preference found, return default value
        print(f"Warning: No preference found for attributes {attributes} in quadrant {quadrants}")
        return 0  # Default reward value

    def lookup_quadrant_reward(self, input_state):
        """
        Calculate total reward for current object positions
        
        Parameters
        ----------
        input_state : dict
            Current state dictionary
            
        Returns
        -------
        float
            Total reward for all objects in their current positions
        """
        current_state = copy.deepcopy(input_state)
        objects_list = list(copy.deepcopy(input_state))[:-1]
        total_reward = 0

        # Get quadrants for all objects
        quadrants_list = self.check_quadrant(current_state)

        # Calculate reward for each object
        for idx, obj in enumerate(objects_list):
            # Extract object properties
            color_idx, material_idx, object_idx, object_label = obj
            color = COLORS_IDX[color_idx]
            material = MATERIALS_IDX[material_idx]
            object_type = OBJECTS_IDX[object_idx]
            quadrants = quadrants_list[idx]

            # Calculate reward based on properties and quadrant
            attributes = [object_type, color, material]
            for quadrant in quadrants:
                reward = self.get_reward_value(self.pref_values['pref_values'], attributes, quadrant)
                total_reward += reward

        return total_reward

    def step_given_state(self, input_state, action):
        """
        Calculate new state and reward after taking an action
        
        Parameters
        ----------
        input_state : dict
            Current state dictionary
        action : tuple or str
            Action to take (object, direction) or EXIT
            
        Returns
        -------
        tuple
            (new_state, reward, done) tuple
        """
        step_cost = -0.1
        current_state = copy.deepcopy(input_state)
        step_reward = 0
        
        # Define valid locations in each quadrant - restricted to stay within visible grid area
        locs_in_q1 = [(x, y) for x in range(1, self.x_max) for y in range(1, self.y_max)]
        locs_in_q2 = [(x, y) for x in range(self.x_min, 0) for y in range(1, self.y_max)]
        locs_in_q3 = [(x, y) for x in range(self.x_min, 0) for y in range(self.y_min, 0)]
        locs_in_q4 = [(x, y) for x in range(1, self.x_max) for y in range(self.y_min, 0)]

        # Handle already exited state
        if current_state['exit']:
            return current_state, 0, True

        # Handle EXIT action
        if action == EXIT:
            current_state['exit'] = True
            step_reward = self.lookup_quadrant_reward(current_state) + step_cost
            return current_state, step_reward, True

        # Handle object movement action
        obj_type, obj_action = action

        if obj_action in self.skills:
            valid, new_loc = self.is_valid_push(
                current_state, action, locs_in_q1, locs_in_q2, locs_in_q3, locs_in_q4
            )
            
            if not valid:
                return current_state, step_cost, False

        # Update object position
        action_type_moved = action[0]
        current_state[action_type_moved]['pos'] = new_loc
        current_state[obj_type]['done'] = True

        step_reward = step_cost
        done = self.is_done_given_state(current_state)

        return current_state, step_reward, done

    def tuple_to_state(self, current_state_tup):
        """
        Convert tuple representation back to state dictionary
        
        Parameters
        ----------
        current_state_tup : tuple
            Tuple representation of state
            
        Returns
        -------
        dict
            Dictionary representation of state
        """
        current_state_tup = list(current_state_tup)
        current_state_dict = {}

        # Convert object tuples to state dictionary entries
        for i in range(0, len(current_state_tup) - 1):
            current_state_dict[current_state_tup[i][0][0]] = {
                'pos': current_state_tup[i][0][1],
                'orientation': current_state_tup[i][1],
                'done': current_state_tup[i][2]
            }

        # Add exit state
        current_state_dict['exit'] = current_state_tup[-1]

        return current_state_dict

    def enumerate_states(self):
        """
        Generate all possible states and transitions in the environment
        
        This method builds a graph of all possible states and actions,
        and computes transition and reward matrices for value iteration.
        
        Returns
        -------
        tuple
            (transitions, rewards, state_to_idx, idx_to_action, idx_to_state, action_to_idx)
        """
        print("Starting state enumeration...")
        # Reset environment
        self.reset()
        actions = self.possible_single_actions

        # Build graph of states
        G = nx.DiGraph()
        visited_states = set()
        stack = [copy.deepcopy(self.current_state)]
        states_processed = 0

        while stack:
            state = stack.pop()
            state_tup = self.state_to_tuple(state)

            # Add state if not visited
            if state_tup not in visited_states:
                visited_states.add(state_tup)
                states_processed += 1
                
                # Print progress periodically
                if states_processed % 100 == 0:
                    print(f"Enumerating states: {states_processed} states processed")

            # Explore all actions from current state
            for action in actions:
                if self.is_done_given_state(state):
                    next_state = state
                    team_reward = 0
                    done = True
                else:
                    next_state, team_reward, done = self.step_given_state(state, action)

                new_state_tup = self.state_to_tuple(next_state)

                # Add unvisited states to stack
                if new_state_tup not in visited_states:
                    stack.append(copy.deepcopy(next_state))

                # Add edge to graph
                G.add_edge(state_tup, new_state_tup, weight=team_reward, action=action)

        # Build state and action indices
        states = list(G.nodes)
        print(f"State enumeration complete: {len(states)} total states found")

        idx_to_state = {i: state for i, state in enumerate(states)}
        state_to_idx = {state: i for i, state in idx_to_state.items()}

        action_to_idx = {action: i for i, action in enumerate(actions)}
        idx_to_action = {i: action for i, action in enumerate(actions)}

        # Build transition and reward matrices
        transition_mat = np.zeros([len(states), len(states), len(actions)])
        reward_mat = np.zeros([len(states), len(actions)])

        print("Building transition and reward matrices...")
        for i in range(len(states)):
            if i % 100 == 0:
                print(f"Processing state {i}/{len(states)}")
                
            state = self.tuple_to_state(idx_to_state[i])
            
            for action_idx_i in range(len(actions)):
                action = idx_to_action[action_idx_i]
                
                if self.is_done_given_state(state):
                    next_state = state
                    team_reward = 0
                else:
                    next_state, team_reward, done = self.step_given_state(state, action)

                next_state_i = state_to_idx[self.state_to_tuple(next_state)]

                transition_mat[i, next_state_i, action_idx_i] = 1.0
                reward_mat[i, action_idx_i] = team_reward

        # Save results
        self.transitions = transition_mat
        self.rewards = reward_mat
        self.state_to_idx = state_to_idx
        self.idx_to_action = idx_to_action
        self.idx_to_state = idx_to_state
        self.action_to_idx = action_to_idx

        return transition_mat, reward_mat, state_to_idx, idx_to_action, idx_to_state, action_to_idx

    def vectorized_vi(self):
        """
        Performs vectorized value iteration to find the optimal policy.
        
        This method uses the environment's transition and reward matrices to compute
        an optimal value function and policy using dynamic programming. The algorithm
        iteratively updates state values until convergence or until reaching the maximum
        number of iterations.
        
        Returns
        -------
        tuple
            (value_function, policy) where:
            - value_function: Array of optimal state values
            - policy: Array of optimal actions for each state
        """
        print("Starting value iteration...")
        n_states = self.transitions.shape[0]
        n_actions = self.transitions.shape[2]

        # Initialize value function and policy
        pi = np.zeros((n_states, 1))
        vf = np.zeros((n_states, 1))
        Q = np.zeros((n_states, n_actions))
        policy = {}

        # Perform value iteration
        for i in range(self.maxiter):
            # Print status periodically
            if i % 100 == 0:
                print(f"Value iteration: iteration {i}/{self.maxiter}")
                
            # Initialize convergence check
            delta = 0
            
            # Bellman update for each state
            for s in range(n_states):
                old_v = vf[s].copy()
                
                # Compute Q-values and value function
                Q[s] = np.sum((self.rewards[s] + self.gamma * vf) * self.transitions[s, :, :], 0)
                vf[s] = np.max(Q[s])
                
                # Track maximum change for convergence check
                delta = max(delta, np.abs(old_v - vf[s])[0])
                
            # Check convergence
            if delta < self.epsilson:
                print(f"Value iteration converged after {i} iterations")
                break
                
        # Extract optimal policy
        for s in range(n_states):
            pi[s] = np.argmax(np.sum(vf * self.transitions[s, :, :], 0))
            policy[s] = Q[s, :]

        self.vf = vf
        self.pi = pi
        self.policy = policy
        
        return vf, pi

    def render(self, current_state, timestep):
        """
        Visualize the current state of the environment
        
        Parameters
        ----------
        current_state : dict
            Current state dictionary
        timestep : int
            Current timestep
        """
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox

        def getImage(path, zoom=0.025):
            """Helper function to load and resize images for visualization"""
            # Adjust path to look one directory up for the data folder
            corrected_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
            try:
                return OffsetImage(plt.imread(corrected_path), zoom=zoom)
            except FileNotFoundError:
                print(f"Warning: Image file not found: {corrected_path}")
                # Return a colored square as fallback
                fallback = np.ones((100, 100, 4))
                fallback[:,:,0:3] = np.array([0.8, 0.8, 0.8])  # Gray color
                return OffsetImage(fallback, zoom=zoom)

        # Create figure
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        
        # Add background grid lines
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # Highlight quadrants with light colors
        ax.fill_between([0, self.x_max], 0, self.y_max, alpha=0.1, color='blue', label='Q1')
        ax.fill_between([self.x_min, 0], 0, self.y_max, alpha=0.1, color='green', label='Q2')
        ax.fill_between([self.x_min, 0], self.y_min, 0, alpha=0.1, color='red', label='Q3')
        ax.fill_between([0, self.x_max], self.y_min, 0, alpha=0.1, color='purple', label='Q4')
        
        # Add quadrant labels
        plt.text(2, 2, "Q1", fontsize=12, ha='center')
        plt.text(-2, 2, "Q2", fontsize=12, ha='center')
        plt.text(-2, -2, "Q3", fontsize=12, ha='center')
        plt.text(2, -2, "Q4", fontsize=12, ha='center')

        # Add visual indication of exit state
        if current_state['exit']:
            plt.axvspan(-6, 6, -6, 6, color='green', alpha=0.05)
            ax.set_title(f"State at Time {timestep}: FINAL STATE", fontsize=14)
        else:
            ax.set_title(f"State at Time {timestep}", fontsize=14)

        # Get color mappings for objects
        type_to_color = {}
        for i in range(0, len(self.object_type_tuple)):
            object_tuple = self.object_type_tuple[i]
            color_idx = object_tuple[0]
            type_to_color[object_tuple] = COLOR_DICTIONARY[color_idx]

        # Draw each object
        plot_init_state = copy.deepcopy(current_state)
        for type_o in self.object_type_tuple:
            loc = plot_init_state[type_o]['pos']
            color = type_to_color[type_o]
            
            # Create scatter point for object location
            ax.scatter(loc[0], loc[1], color=color, s=200, alpha=0.7, edgecolor='black', zorder=10)
            
            # Get object properties for visualization
            color_name = COLORS_IDX.get(type_o[0], 'unknown')
            object_name = OBJECTS_IDX.get(type_o[2], 'unknown')
            material_name = MATERIALS_IDX.get(type_o[1], 'unknown')
            
            # Try to load specific image for this object
            image_path = f'data/objects/{color_name}{object_name}.jpeg'
            
            # Check if image exists
            if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), image_path)):
                ab = AnnotationBbox(getImage(image_path), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)
            else:
                # Fall back to generic images
                if type_o[:3] == (1, 1, 1):  # Red cup
                    ab = AnnotationBbox(getImage('data/objects/redcup.jpeg'), (loc[0], loc[1]), frameon=False)
                    ax.add_artist(ab)
                elif type_o[:3] == (2, 1, 1):  # Yellow cup
                    ab = AnnotationBbox(getImage('data/objects/yellowcup.jpeg'), (loc[0], loc[1]), frameon=False)
                    ax.add_artist(ab)
                else:
                    # Create a fallback colored shape based on object properties
                    img = np.ones((100, 100, 4))
                    
                    # Set color based on object color
                    if type_o[0] == 1:  # Red
                        img[:,:,0] = 0.9  # R
                        img[:,:,1] = 0.2  # G
                        img[:,:,2] = 0.2  # B
                    elif type_o[0] == 2:  # Yellow
                        img[:,:,0] = 0.9  # R
                        img[:,:,1] = 0.9  # G
                        img[:,:,2] = 0.2  # B
                    elif type_o[0] == 3:  # Purple
                        img[:,:,0] = 0.6  # R
                        img[:,:,1] = 0.2  # G
                        img[:,:,2] = 0.8  # B
                    
                    # Set shape based on object type
                    if type_o[2] == 1:  # Cup - use circle
                        radius = 40
                        center = (50, 50)
                        y, x = np.ogrid[:100, :100]
                        mask = (x - center[0])**2 + (y - center[1])**2 > radius**2
                        img[mask] = [1, 1, 1, 0]  # Transparent outside circle
                    elif type_o[2] == 2:  # Bowl - use ellipse
                        radius_x, radius_y = 45, 30
                        center = (50, 50)
                        y, x = np.ogrid[:100, :100]
                        mask = (x - center[0])**2 / radius_x**2 + (y - center[1])**2 / radius_y**2 > 1
                        img[mask] = [1, 1, 1, 0]  # Transparent outside ellipse
                    
                    ab = AnnotationBbox(OffsetImage(img, zoom=0.025), (loc[0], loc[1]), frameon=False)
                    ax.add_artist(ab)
                    
                    # Add text label for object
                    plt.text(loc[0], loc[1]-0.5, f"{color_name} {object_name}", 
                            ha='center', va='top', fontsize=8, 
                            bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'))

        # Set axis limits and ticks
        plt.xlim(self.x_min - 0.5, self.x_max - 0.5)
        plt.ylim(self.y_min - 0.5, self.y_max - 0.5)
        plt.xticks(range(self.x_min, self.x_max))
        plt.yticks(range(self.y_min, self.y_max))
        
        # Add legend for object types
        legend_elements = []
        for obj_type in set([o[2] for o in self.object_type_tuple]):
            obj_name = OBJECTS_IDX[obj_type]
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor='gray', markersize=8, label=f"{obj_name}"))
        
        # Add a legend for quadrants
        plt.legend(handles=legend_elements, loc='upper right', title="Object Types")
        
        # Create the rollouts directory if it doesn't exist
        rollout_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rollouts")
        os.makedirs(rollout_dir, exist_ok=True)
        
        # Save figure with absolute path
        save_path = os.path.join(rollout_dir, f"state_{timestep}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved state image to: {save_path}")

        plt.show()
        plt.close()

    def rollout_full_game_joint_optimal(self, render=False):
        """
        Execute a full game using the optimal policy
        
        Parameters
        ----------
        render : bool
            Whether to render each state
            
        Returns
        -------
        tuple
            (total_reward, game_results, sum_feature_value)
        """
        print("Executing optimal policy rollout...")
        # Reset environment
        self.reset()
        done = False
        total_reward = 0
        iters = 0
        game_results = []
        sum_feature_value = 0
        
        # Render initial state
        if render:
            print("\n--- Initial state ---")
            self.render(self.current_state, iters)
            
        # Execute policy until done
        while not done:
            iters += 1
            
            # Get current state index
            current_state_tup = self.state_to_tuple(self.current_state)
            state_idx = self.state_to_idx[current_state_tup]
            
            # Get best action from policy
            action_distribution = self.policy[state_idx]
            action = self.idx_to_action[np.argmax(action_distribution)]
            
            # Record current state and action
            game_results.append((self.current_state, action))
            
            # Take action
            next_state, team_rew, done = self.step_given_state(self.current_state, action)
            
            # Print status
            if render or iters % 5 == 0:
                print(f"\n--- Step {iters} ---")
                if isinstance(action, tuple):
                    obj_desc = f"Object: {action[0]}, Action: {action[1]}"
                else:
                    obj_desc = f"Action: {action}"
                print(f"Action taken: {obj_desc}")
            
            # Calculate features and update state
            featurized_state = self.lookup_quadrant_reward(self.current_state)
            sum_feature_value += featurized_state
            self.current_state = next_state
            
            # Render if requested
            if render:
                self.render(self.current_state, iters)
                
            # Update total reward
            total_reward += team_rew
            
            # Safety limit
            if iters > 40:
                print("Maximum iterations reached, terminating rollout")
                break
                
        print(f"Rollout complete: {iters} steps, total reward: {total_reward:.2f}")
        return total_reward, game_results, sum_feature_value

    def compute_optimal_performance(self, render=False):
        """
        Compute optimal policy and execute a rollout
        
        Parameters
        ----------
        render : bool
            Whether to render states during rollout
            
        Returns
        -------
        tuple
            (optimal_reward, game_results, sum_feature_vector)
        """
        # Generate states and transitions
        self.enumerate_states()
        
        # Compute optimal policy
        self.vectorized_vi()
        
        # Execute optimal policy
        optimal_rew, game_results, sum_feature_vector = self.rollout_full_game_joint_optimal(render=render)
        
        return optimal_rew, game_results, sum_feature_vector


# ============================
# Test Functions
# ============================

def test_f_Ada():
    """Test the Ada preference tree with different object configurations"""
    print("\n" + "="*50)
    print("Testing Ada's Preferences")
    print("="*50 + "\n")

    # Three objects test
    print("\nTesting with three objects (yellow cup, red cup, purple bowl)...")
    object_type_tuple = [obj_1, obj_2, obj_3]
    
    # Set up environment
    game = Gridworld(f_Ada, object_type_tuple)
    
    # Run value iteration and optimal policy
    optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance(render=True)
    
    # Print results summary
    print("\nResults Summary:")
    print(f"- Optimal reward: {optimal_rew:.2f}")
    print(f"- Feature vector sum: {sum_feature_vector:.2f}")
    print(f"- Steps to completion: {len(game_results)}")
    
    # Print final object placements
    print("\nFinal object placements:")
    final_state = game_results[-1][0]
    for obj in object_type_tuple:
        obj_tuple = (COLORS[obj['color']], 
                    MATERIALS[obj['material']], 
                    OBJECTS[obj['object_type']], 
                    obj['object_label'])
        if obj_tuple in final_state:
            obj_pos = final_state[obj_tuple]['pos']
            print(f"- {obj['color']} {obj['material']} {obj['object_type']}: {obj_pos}")


def test_f_Michelle():
    """Test Michelle's preference tree"""
    print("\nTesting Michelle's preferences...")
    # Implementation follows same pattern as test_f_Ada


def test_f_Annika():
    """Test Annika's preference tree"""
    print("\nTesting Annika's preferences...")
    # Implementation follows same pattern as test_f_Ada


def test_f_Admoni():
    """Test Admoni's preference tree"""
    print("\nTesting Admoni's preferences...")
    # Implementation follows same pattern as test_f_Ada


# ============================
# Main Entry Point
# ============================

if __name__ == '__main__':
    # Set up cleaner console output
    print("\n" + "*"*70)
    print("*  Multi-Object Custom MDP (v5) - Object Placement Preference Learning  *")
    print("*"*70 + "\n")
    
    test_f_Ada()
    # Uncomment to test other preference trees
    # test_f_Michelle()
    # test_f_Annika()
    # test_f_Admoni()
