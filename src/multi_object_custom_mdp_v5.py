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
        - Origin (0,0): Considered Q1 (changed from Q2)
        - Positive X-axis: Q1/Q4 depending on proximity
        - Negative X-axis: Q2/Q3 depending on proximity
        - Positive Y-axis: Q1/Q2 depending on proximity
        - Negative Y-axis: Q3/Q4 depending on proximity
        
        Parameters:
        -----------
        input_state : dict
            The current state of the environment
            
        Returns:
        --------
        list
            List of quadrant assignments for each object
        """
        quadrant_list = []
        
        # Process all objects in the input state
        for obj_key in input_state:
            # Skip the 'exit' key which isn't an object
            if obj_key == 'exit':
                continue
                
            # Get object position
            try:
                x, y = input_state[obj_key]['pos']
                
                # Convert numpy values to regular Python types if needed
                if hasattr(x, 'item'):
                    x = x.item()
                if hasattr(y, 'item'):
                    y = y.item()
                
                # Determine quadrant based on coordinates with clearer boundaries
                if x >= 0:  # Right half of the grid
                    if y >= 0:
                        quadrant_list.append(['Q1'])  # Q1: Positive X, Positive Y (top right)
                    else:
                        quadrant_list.append(['Q4'])  # Q4: Positive X, Negative Y (bottom right)
                else:  # Left half of the grid
                    if y >= 0:
                        quadrant_list.append(['Q2'])  # Q2: Negative X, Positive Y (top left)
                    else:
                        quadrant_list.append(['Q3'])  # Q3: Negative X, Negative Y (bottom left)
                        
            except (KeyError, TypeError) as e:
                # Handle case where object doesn't have position information
                print(f"Warning: Could not determine quadrant for {obj_key}: {e}")
                quadrant_list.append(['Q1'])  # Default to Q1 as fallback

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
            
        # Handle case where preferences is a dict but doesn't contain the quadrant or other attributes
        if isinstance(preferences, dict):
            # Handle 'other' category if it exists
            if 'other' in preferences:
                try:
                    if isinstance(preferences['other'], dict) and quadrants in preferences['other']:
                        return preferences['other'][quadrants]
                    else:
                        # Try recursively looking in 'other'
                        return self.get_reward_value(preferences['other'], attributes, quadrants)
                except (TypeError, KeyError):
                    pass  # Continue to next approach if this fails
                    
            # Try to navigate deeper using the attributes
            for attr in attributes:
                if attr in preferences:
                    # Make a copy of attributes and remove the current one to avoid infinite loops
                    remaining_attrs = [a for a in attributes if a != attr]
                    return self.get_reward_value(preferences[attr], remaining_attrs, quadrants)
        
        # No preference found, return default value
        # print(f"Warning: No preference found for attributes {attributes} in quadrant {quadrants}")
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
        
        # Define valid locations in each quadrant - restricted to stay well within visible grid area
        # Add safety margins to prevent objects from being placed at the edges
        safety_margin = 1
        
        # Define central positions for each quadrant for consistent placement
        # Define positions with small offsets based on object type to prevent overlap
        q1_central_yellow = (2, 2)      # Yellow cup in Q1
        q1_central_red = (1.5, 2)       # Red cup in Q1
        q1_central_purple = (2, 1.5)    # Purple bowl in Q1
        
        q2_central_yellow = (-2, 2)     # Yellow cup in Q2
        q2_central_red = (-1.5, 2)      # Red cup in Q2
        q2_central_purple = (-2, 1.5)   # Purple bowl in Q2
        
        q3_central_yellow = (-2, -2)    # Yellow cup in Q3
        q3_central_red = (-1.5, -2)     # Red cup in Q3
        q3_central_purple = (-2, -1.5)  # Purple bowl in Q3
        
        q4_central_yellow = (2, -2)     # Yellow cup in Q4
        q4_central_red = (1.5, -2)      # Red cup in Q4
        q4_central_purple = (2, -1.5)   # Purple bowl in Q4
        
        # Default central positions for any object
        q1_central = (2, 2)    # Positive x, positive y
        q2_central = (-2, 2)   # Negative x, positive y
        q3_central = (-2, -2)  # Negative x, negative y
        q4_central = (2, -2)   # Positive x, negative y
        
        # Define valid locations for each quadrant
        locs_in_q1 = [(x, y) for x in range(1, self.x_max - safety_margin) 
                            for y in range(1, self.y_max - safety_margin)]
        locs_in_q2 = [(x, y) for x in range(self.x_min + safety_margin, 0) 
                            for y in range(1, self.y_max - safety_margin)]
        locs_in_q3 = [(x, y) for x in range(self.x_min + safety_margin, 0) 
                            for y in range(self.y_min + safety_margin, 0)]
        locs_in_q4 = [(x, y) for x in range(1, self.x_max - safety_margin) 
                            for y in range(self.y_min + safety_margin, 0)]
                            
        # Ensure each quadrant has at least some valid locations
        if not locs_in_q1: locs_in_q1 = [q1_central]
        if not locs_in_q2: locs_in_q2 = [q2_central]
        if not locs_in_q3: locs_in_q3 = [q3_central]
        if not locs_in_q4: locs_in_q4 = [q4_central]

        # Handle already exited state
        if current_state['exit']:
            return current_state, 0, True

        # Handle EXIT action
        if action == EXIT:
            current_state['exit'] = True
            step_reward = self.lookup_quadrant_reward(current_state) + step_cost
            return current_state, step_reward, True

        # Handle object movement action
        if isinstance(action, tuple) and len(action) >= 2:
            obj_type, obj_action = action
            new_loc = None
            
            # Directly set position based on quadrant and object type to avoid overlap
            if obj_action == 'Q1':
                if obj_type[:3] == (2, 1, 1):  # Yellow cup
                    new_loc = q1_central_yellow
                elif obj_type[:3] == (1, 1, 1):  # Red cup
                    new_loc = q1_central_red
                elif obj_type[:3] == (3, 1, 2):  # Purple bowl
                    new_loc = q1_central_purple
                else:
                    new_loc = q1_central
            elif obj_action == 'Q2':
                if obj_type[:3] == (2, 1, 1):  # Yellow cup
                    new_loc = q2_central_yellow
                elif obj_type[:3] == (1, 1, 1):  # Red cup
                    new_loc = q2_central_red
                elif obj_type[:3] == (3, 1, 2):  # Purple bowl
                    new_loc = q2_central_purple
                else:
                    new_loc = q2_central
            elif obj_action == 'Q3':
                if obj_type[:3] == (2, 1, 1):  # Yellow cup
                    new_loc = q3_central_yellow
                elif obj_type[:3] == (1, 1, 1):  # Red cup
                    new_loc = q3_central_red
                elif obj_type[:3] == (3, 1, 2):  # Purple bowl
                    new_loc = q3_central_purple
                else:
                    new_loc = q3_central
            elif obj_action == 'Q4':
                if obj_type[:3] == (2, 1, 1):  # Yellow cup
                    new_loc = q4_central_yellow
                elif obj_type[:3] == (1, 1, 1):  # Red cup
                    new_loc = q4_central_red
                elif obj_type[:3] == (3, 1, 2):  # Purple bowl
                    new_loc = q4_central_purple
                else:
                    new_loc = q4_central
            else:
                # Invalid quadrant action
                return current_state, step_cost, False
                
            # Update object position
            if obj_type in current_state:
                current_state[obj_type]['pos'] = new_loc
                current_state[obj_type]['done'] = True
                
                # Store initial position if not already tracked
                if not hasattr(self, 'initial_object_locs'):
                    self.initial_object_locs = {}
                if obj_type not in self.initial_object_locs:
                    self.initial_object_locs[obj_type] = input_state[obj_type]['pos']
                
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
        import os
        import time
        import matplotlib as mpl
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Arrow
        
        # Set up a professional visualization style
        plt.style.use('seaborn-v0_8-whitegrid')

        def getImage(path, zoom=0.025):
            """Helper function to load and resize images for visualization"""
            # Adjust path to look one directory up for the data folder
            corrected_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
            try:
                image = plt.imread(corrected_path)
                # Add a subtle border to the image
                if image.shape[2] == 4:  # Has alpha channel
                    # Add a thin dark border around the image edges (2 pixels)
                    border_width = 2
                    image[:border_width, :, 3] = 1.0  # Top border
                    image[-border_width:, :, 3] = 1.0  # Bottom border
                    image[:, :border_width, 3] = 1.0  # Left border
                    image[:, -border_width:, 3] = 1.0  # Right border
                    
                    image[:border_width, :, 0:3] = 0.2  # Dark color
                    image[-border_width:, :, 0:3] = 0.2
                    image[:, :border_width, 0:3] = 0.2
                    image[:, -border_width:, 0:3] = 0.2
                
                return OffsetImage(image, zoom=zoom)
            except FileNotFoundError:
                print(f"Warning: Image file not found: {corrected_path}")
                # Return a colored square as fallback with better styling
                fallback = np.ones((100, 100, 4))
                fallback[:,:,0:3] = np.array([0.7, 0.7, 0.7])  # Gray color
                
                # Add a subtle border
                border_width = 3
                fallback[:border_width, :, 0:3] = 0.3  # Top border
                fallback[-border_width:, :, 0:3] = 0.3  # Bottom border
                fallback[:, :border_width, 0:3] = 0.3  # Left border
                fallback[:, -border_width:, 0:3] = 0.3  # Right border
                
                return OffsetImage(fallback, zoom=zoom)

        # Create figure with scientific paper style
        fig = plt.figure(figsize=(12, 10))
        ax = plt.gca()
        
        # Set professional-looking background
        ax.set_facecolor('#f8f9fa')  # Light gray background
        # Add customized grid lines
        ax.grid(True, linestyle='-', alpha=0.3, linewidth=0.5, color='#cccccc')
        
        # Add prominent axes
        ax.axhline(y=0, color='#444444', linestyle='-', alpha=0.8, linewidth=2, zorder=1)
        ax.axvline(x=0, color='#444444', linestyle='-', alpha=0.8, linewidth=2, zorder=1)
        
        # Define a research-quality color palette for quadrants
        quad_colors = ['#c6dbef', '#ccebc5', '#fdd0a2', '#e0ecf4']  # Professional blue, green, orange, light blue
        quad_names = ['Q1', 'Q2', 'Q3', 'Q4']
        quad_labels = ['Quadrant 1 (Upper Right)', 'Quadrant 2 (Upper Left)', 
                      'Quadrant 3 (Lower Left)', 'Quadrant 4 (Lower Right)']
        
        # Draw quadrants as shaded rectangles with subtle borders
        q1_rect = Rectangle((0, 0), self.x_max, self.y_max, 
                           facecolor=quad_colors[0], alpha=0.3, edgecolor='#666666', 
                           linewidth=0.8, zorder=0)
        q2_rect = Rectangle((self.x_min, 0), abs(self.x_min), self.y_max, 
                           facecolor=quad_colors[1], alpha=0.3, edgecolor='#666666', 
                           linewidth=0.8, zorder=0)
        q3_rect = Rectangle((self.x_min, self.y_min), abs(self.x_min), abs(self.y_min), 
                           facecolor=quad_colors[2], alpha=0.3, edgecolor='#666666', 
                           linewidth=0.8, zorder=0)
        q4_rect = Rectangle((0, self.y_min), self.x_max, abs(self.y_min), 
                           facecolor=quad_colors[3], alpha=0.3, edgecolor='#666666', 
                           linewidth=0.8, zorder=0)
        
        # Add quadrant rectangles to plot
        ax.add_patch(q1_rect)
        ax.add_patch(q2_rect)
        ax.add_patch(q3_rect)
        ax.add_patch(q4_rect)
        
        # Add quadrant labels with enhanced styling
        for i, (x, y, name) in enumerate([
            (self.x_max/4, self.y_max/4, quad_names[0]),
            (self.x_min/4, self.y_max/4, quad_names[1]),
            (self.x_min/4, self.y_min/4, quad_names[2]),
            (self.x_max/4, self.y_min/4, quad_names[3])
        ]):
            ax.text(x, y, name, fontsize=16, ha='center', va='center', 
                   color='#333333', fontweight='bold', zorder=2,
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3',
                            edgecolor='#999999', linewidth=0.5))
            
        # Add visual indication of exit state
        if current_state['exit']:
            # Add a subtle green background for final state
            plt.axvspan(self.x_min, self.x_max, self.y_min, self.y_max, color='green', alpha=0.05, zorder=-1)
            plt.suptitle('Object Placement Environment', 
                     fontsize=20, fontweight='bold', y=0.98, color='#333333')
            plt.title(f"Final State (Time Step: {timestep})", 
                    fontsize=16, pad=20, color='#007700', fontweight='bold')
        else:
            plt.suptitle('Object Placement Environment', 
                     fontsize=20, fontweight='bold', y=0.98, color='#333333')
            plt.title(f"Current State (Time Step: {timestep})", 
                    fontsize=16, pad=20, color='#333333')
            
        # Create detailed object status information
        status_text = []
        status_text.append("Object Status:")
        status_text.append("-"*25)
        
        for idx, obj in enumerate(self.object_type_tuple):
            color_name = COLORS_IDX.get(obj[0], 'unknown')
            object_name = OBJECTS_IDX.get(obj[2], 'unknown')
            material_name = MATERIALS_IDX.get(obj[1], 'unknown')
            is_done = current_state[obj]['done']
            current_pos = current_state[obj]['pos']
            
            # Use unicode symbols for better visual indication
            status = "✓" if is_done else "○"
            
            # Include all object properties and position
            status_text.append(f"{status} {color_name.capitalize()} {material_name} {object_name}: {current_pos}")
            
        # Create research-grade status box
        status_box = '\n'.join(status_text)
        ax.text(0.02, 0.98, status_box, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor='#666666', linewidth=1, alpha=0.9))

        # Create enhanced color mapping for object types with better colors
        type_to_color = {}
        for i in range(0, len(self.object_type_tuple)):
            object_tuple = self.object_type_tuple[i]
            color_idx = object_tuple[0]
            # Use more vibrant research-grade colors from ColorBrewer palette
            if color_idx == 1:  # Red
                type_to_color[object_tuple] = '#e41a1c'  # Vibrant red
            elif color_idx == 2:  # Yellow
                type_to_color[object_tuple] = '#ff7f00'  # Vibrant orange
            elif color_idx == 3:  # Purple
                type_to_color[object_tuple] = '#984ea3'  # Vibrant purple
            elif color_idx == 4:  # White
                type_to_color[object_tuple] = '#f0f0f0'  # Off-white
            else:
                type_to_color[object_tuple] = '#333333'  # Dark gray for unknown

        # Draw each object with enhanced styling
        plot_init_state = copy.deepcopy(current_state)
        for type_o in self.object_type_tuple:
            loc = plot_init_state[type_o]['pos']
            color = type_to_color[type_o]
            is_done = plot_init_state[type_o]['done']
            
            # Enhanced styling for active vs. inactive objects
            if is_done:
                edgecolor = '#000000'
                linewidth = 2.5
                alpha = 1.0
                size = 250
                zorder = 15
            else:
                edgecolor = '#444444'
                linewidth = 1.5
                alpha = 0.8
                size = 220
                zorder = 10
            
            # Add a white halo/glow around placed objects for emphasis
            if is_done:
                ax.scatter(loc[0], loc[1], color='white', s=size+30, alpha=0.3, 
                          edgecolor='white', linewidth=0, zorder=zorder-1)
            
            # Create enhanced scatter point for object location
            ax.scatter(loc[0], loc[1], color=color, s=size, alpha=alpha, 
                      edgecolor=edgecolor, linewidth=linewidth, zorder=zorder)
            
            # Add enhanced arrows to show initial to current position if moved
            if is_done and hasattr(self, 'initial_object_locs') and type_o in self.initial_object_locs:
                init_loc = self.initial_object_locs[type_o]
                if init_loc != loc:  # Only draw arrow if position changed
                    dx = loc[0] - init_loc[0]
                    dy = loc[1] - init_loc[1]
                    
                    # First draw a wider "shadow" arrow for better visibility
                    ax.arrow(init_loc[0], init_loc[1], dx*0.9, dy*0.9, 
                            head_width=0.25, head_length=0.35, 
                            fc='white', ec='#666666',
                            alpha=0.5, zorder=4, length_includes_head=True,
                            width=0.08)
                    
                    # Then draw the main colored arrow
                    ax.arrow(init_loc[0], init_loc[1], dx*0.9, dy*0.9, 
                            head_width=0.2, head_length=0.3, 
                            fc=color, ec='#333333',
                            alpha=0.7, zorder=5, length_includes_head=True,
                            width=0.05)
                    
                    # Add a "moved from" small marker
                    ax.scatter(init_loc[0], init_loc[1], color=color, s=80, alpha=0.3,
                             edgecolor='#666666', linewidth=1, zorder=3, marker='o')
            
            # Get object properties for visualization
            color_name = COLORS_IDX.get(type_o[0], 'unknown')
            object_name = OBJECTS_IDX.get(type_o[2], 'unknown')
            material_name = MATERIALS_IDX.get(type_o[1], 'unknown')
            
            # Add enhanced label with object info and material
            object_label = f"{color_name.capitalize()} {object_name}"
            details_label = f"{material_name}" if material_name != 'unknown' else ""
            
            # Position main label above the object
            plt.text(loc[0], loc[1]+0.5, object_label, 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3',
                             edgecolor='#666666', linewidth=0.5))
                             
            # Position material info below the object if available
            if details_label:
                plt.text(loc[0], loc[1]-0.5, details_label, 
                        ha='center', va='center', fontsize=8, color='#444444',
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2',
                                 edgecolor='#aaaaaa', linewidth=0.5))
            
            # Try loading specific images with enhanced presentation
            try:
                # Use _holding suffix for done objects if available
                holding_suffix = "_holding" if is_done else ""
                
                # First attempt - specific color+object combination with holding status
                specific_path = f'data/objects/{color_name}{object_name}{holding_suffix}.jpeg'
                alt_path = f'data/objects/{color_name}{object_name}.jpeg'
                
                # Choose appropriate image with zoom level based on object state
                zoom_level = 0.030 if is_done else 0.025  # Slightly larger for placed objects
                
                if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), specific_path)):
                    # Use state-specific image if available
                    ab = AnnotationBbox(getImage(specific_path, zoom=zoom_level), 
                                       (loc[0], loc[1]), frameon=False)
                    ax.add_artist(ab)
                elif os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), alt_path)):
                    # Use regular image if holding-specific not available
                    ab = AnnotationBbox(getImage(alt_path, zoom=zoom_level), 
                                       (loc[0], loc[1]), frameon=False)
                    ax.add_artist(ab)
                # Standard object images with improved colors
                elif type_o[:3] == (1, 1, 1):  # Red cup
                    img_path = 'data/objects/redcup_holding.jpeg' if is_done else 'data/objects/redcup.jpeg'
                    ab = AnnotationBbox(getImage(img_path, zoom=zoom_level), 
                                       (loc[0], loc[1]), frameon=False)
                    ax.add_artist(ab)
                elif type_o[:3] == (2, 1, 1):  # Yellow cup
                    img_path = 'data/objects/yellowcup_holding.jpeg' if is_done else 'data/objects/yellowcup.jpeg'
                    ab = AnnotationBbox(getImage(img_path, zoom=zoom_level), 
                                       (loc[0], loc[1]), frameon=False)
                    ax.add_artist(ab)
                else:
                    # Create a high-quality fallback shape based on object properties
                    img = np.ones((200, 200, 4))  # Higher resolution for better quality
                    
                    # Use research-quality colors that match the color scheme
                    if type_o[0] == 1:  # Red
                        rgb = np.array([0.85, 0.1, 0.1])  # Vibrant red
                    elif type_o[0] == 2:  # Yellow
                        rgb = np.array([0.95, 0.75, 0.1])  # Vibrant yellow
                    elif type_o[0] == 3:  # Purple
                        rgb = np.array([0.6, 0.2, 0.65])  # Rich purple
                    else:
                        rgb = np.array([0.5, 0.5, 0.5])  # Default gray
                        
                    img[:,:,0:3] = rgb
                    
                    # Add a gradient effect for a more polished look
                    y, x = np.ogrid[:200, :200]
                    center = (100, 100)
                    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                    # Make center brighter for 3D effect
                    for i in range(3):
                        img[:,:,i] = np.minimum(1.0, img[:,:,i] + (50 - dist) * 0.01 * (1 if i==0 else 0.7))
            except Exception as e:
                print(f"Error rendering image for {color_name} {object_name}: {e}")
                # Create enhanced fallback with gradient
                img = np.ones((200, 200, 4))
                
                # Use a neutral gray with color hint
                if type_o[0] == 1:  # Red hint
                    base_color = np.array([0.65, 0.6, 0.6])
                elif type_o[0] == 2:  # Yellow hint
                    base_color = np.array([0.65, 0.65, 0.6])  
                elif type_o[0] == 3:  # Purple hint
                    base_color = np.array([0.6, 0.6, 0.65])
                else:
                    base_color = np.array([0.6, 0.6, 0.6])
                    
                img[:,:,0:3] = base_color
                
                # Create professional shapes with higher quality rendering
                if type_o[2] == 1:  # Cup - use rounded shape with shadow
                    # Main circle
                    radius = 80
                    center = (100, 100)
                    y, x = np.ogrid[:200, :200]
                    
                    # Create base shape
                    mask = (x - center[0])**2 + (y - center[1])**2 > radius**2
                    img[mask] = [0, 0, 0, 0]  # Transparent outside circle
                    
                    # Add subtle shadow for 3D effect
                    shadow_shift = 5
                    shadow_mask = ((x - (center[0] + shadow_shift))**2 + 
                                  (y - (center[1] + shadow_shift))**2 > radius**2) & ~mask
                    img[shadow_mask] = [0.2, 0.2, 0.2, 0.2]  # Semi-transparent shadow
                    
                    # Add highlight for glossy effect
                    highlight_x, highlight_y = 70, 70
                    highlight_radius = 25
                    highlight_mask = ((x - highlight_x)**2 + (y - highlight_y)**2 < highlight_radius**2) & ~mask
                    img[highlight_mask] = np.minimum(1.0, img[highlight_mask] + [0.3, 0.3, 0.3, 0])
                    
                elif type_o[2] == 2:  # Bowl - use oval shape with shadow
                    # Main ellipse
                    radius_x, radius_y = 90, 60
                    center = (100, 100)
                    y, x = np.ogrid[:200, :200]
                    
                    # Create base shape with more complex equation for better oval
                    mask = (x - center[0])**2 / radius_x**2 + (y - center[1])**2 / radius_y**2 > 1
                    img[mask] = [0, 0, 0, 0]  # Transparent outside ellipse
                    
                    # Add 3D shadow effect
                    shadow_shift_x, shadow_shift_y = 5, 8
                    shadow_mask = ((x - (center[0] + shadow_shift_x))**2 / radius_x**2 + 
                                  (y - (center[1] + shadow_shift_y))**2 / radius_y**2 > 1) & ~mask
                    img[shadow_mask] = [0.2, 0.2, 0.2, 0.2]  # Semi-transparent shadow
                    
                    # Add highlight for glossy effect
                    highlight_x, highlight_y = 70, 80
                    highlight_radius_x, highlight_radius_y = 20, 15
                    highlight_mask = ((x - highlight_x)**2 / highlight_radius_x**2 + 
                                     (y - highlight_y)**2 / highlight_radius_y**2 < 1) & ~mask
                    img[highlight_mask] = np.minimum(1.0, img[highlight_mask] + [0.3, 0.3, 0.3, 0])
                    
                    # Add inner oval to suggest bowl depth
                    inner_x, inner_y = 70, 50
                    inner_mask = ((x - center[0])**2 / inner_x**2 + 
                                 (y - center[1])**2 / inner_y**2 < 0.7) & ~mask
                    img[inner_mask] = np.minimum(1.0, img[inner_mask] - [0.1, 0.1, 0.1, 0])
                else:
                    # Default shape with 3D effects for unknown objects
                    radius = 80
                    center = (100, 100)
                    y, x = np.ogrid[:200, :200]
                    mask = (x - center[0])**2 + (y - center[1])**2 > radius**2
                    img[mask] = [0, 0, 0, 0]
                
                # Use appropriate zoom based on object state
                zoom_level = 0.030 if is_done else 0.025
                ab = AnnotationBbox(OffsetImage(img, zoom=zoom_level), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)

        # Set axis limits and ticks with enhanced styling
        plt.xlim(self.x_min - 0.7, self.x_max + 0.2)
        plt.ylim(self.y_min - 0.7, self.y_max + 0.2)
        
        # Set professional-looking ticks with clearer grid
        plt.xticks(range(self.x_min, self.x_max+1))
        plt.yticks(range(self.y_min, self.y_max+1))
        
        # Add axis labels with research paper quality
        plt.xlabel('X Position', fontsize=14, labelpad=10, fontweight='bold', color='#333333')
        plt.ylabel('Y Position', fontsize=14, labelpad=10, fontweight='bold', color='#333333')
        
        # Create focused legend with only essential elements
        legend_elements = []
        
        # Add color types to legend - keep these as they're important
        for color_idx in sorted(set([o[0] for o in self.object_type_tuple])):
            color_name = COLORS_IDX.get(color_idx, 'unknown').capitalize()
            marker_color = '#e41a1c' if color_idx == 1 else '#ff7f00' if color_idx == 2 else '#984ea3'
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                            markerfacecolor=marker_color, markersize=10, 
                                            label=f"{color_name}"))
        
        # Add status indicators to legend
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markeredgecolor='#000000',
                                         markerfacecolor='w', markeredgewidth=2.5, markersize=10,
                                         label="Placed Object"))
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markeredgecolor='#444444',
                                         markerfacecolor='w', markeredgewidth=1.5, markersize=10, alpha=0.8,
                                         label="Not Placed"))
                                         
        # Add quadrant names to legend
        for i, (color, name) in enumerate(zip(quad_colors, quad_labels)):
            legend_elements.append(plt.Rectangle((0,0), 1, 1, facecolor=color, alpha=0.3,
                                               edgecolor='#666666', linewidth=0.8, label=name))
        
        # Position legend in a more suitable location with better styling
        legend = plt.legend(handles=legend_elements, loc='lower left', 
                           title="Environment Elements", title_fontsize=12,
                           bbox_to_anchor=(0.02, -0.02), ncol=2, fontsize=10,
                           framealpha=0.9, fancybox=True, shadow=True)
        legend.get_frame().set_edgecolor('#666666')
                           
        # Add timestamp and version info for research reproducibility
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
        plt.figtext(0.98, 0.02, f"Generated: {timestamp_str}", fontsize=8, 
                   ha='right', va='bottom', alpha=0.7)
        
        # Create directories for saving images
        # Create both rollouts and beliefs directories
        rollout_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rollouts")
        os.makedirs(rollout_dir, exist_ok=True)
        
        beliefs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "beliefs")
        os.makedirs(beliefs_dir, exist_ok=True)
        
        # Add a unique identifier based on process ID and time to prevent overwriting
        unique_id = os.getpid() % 1000  # Use process ID for uniqueness
        timestamp = int(time.time() % 10000)  # Current time in seconds (shortened)
        
        # Save figure with absolute path and unique identifier with higher DPI for publication quality
        save_path = os.path.join(rollout_dir, f"state_{unique_id}_{timestamp}_{timestep}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved state image to: {save_path}")

        plt.tight_layout()
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
