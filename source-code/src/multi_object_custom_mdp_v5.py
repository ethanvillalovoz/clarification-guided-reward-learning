# Authors: Ethan Villalovz, Michelle Zhao
# Project: RISS 2024 Summer Project - Markov Decision Process Formation
# Description: Adaptation from `custom_mdp` by Michelle Zhao. Adding on other feature capabilities:
#     - Multiple Objects
#     - Different Materials
#     - Different colors
# Version 5 changes:
#     - Changing our action to be a skills trajectory
#       - i.e. the objects movement is not single movement
#     - Directions are now just the quadrant locations and the availability if the object ca be placed in that location
#     - Handles preferences in which are more abstract

# imports
import copy
import os

import numpy as np
import pickle
import networkx as nx
from itertools import product
import matplotlib.pyplot as plt

# Global Variables

# Directions
EXIT = 'exit'

# Colors
COLORS = {'red': 1, 'yellow': 2, 'purple': 3, 'white': 4, 'other': 5}
COLORS_IDX = {1: 'red', 2: 'yellow', 3: 'purple', 4: 'white', 5: 'other'}

# Materials
MATERIALS = {'glass': 1, 'china': 2, 'plastic': 3, 'other': 4}
MATERIALS_IDX = {1: 'glass', 2: 'china', 3: 'plastic', 4: 'other'}

# Objects
OBJECTS = {'cup': 1, 'bowl': 2, 'other': 3}
OBJECTS_IDX = {1: 'cup', 2: 'bowl', 3: 'other'}

# Dictionary for `def render(self, current_state, timestep):`
COLOR_DICTIONARY = {1: 'red', 2: 'yellow', 3: 'purple'}

# objects
obj_1 = {'object_type': 'cup', 'color': 'yellow', 'material': 'glass', 'object_label': 1}
obj_2 = {'object_type': 'cup', 'color': 'red', 'material': 'glass', 'object_label': 2}
obj_3 = {'object_type': 'bowl', 'color': 'purple', 'material': 'glass', 'object_label': 3}
obj_3_dup = {'object_type': 'bowl', 'color': 'purple', 'material': 'glass', 'object_label': 4}

# preference trees
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


# Class Gridworld
#   Description: contains all the data set up to run simulation of robot moving actions based on human reward
class Gridworld():

    # Initializing function
    # Sets up all of the self. variables for the class
    def __init__(self, pref_values, object_type_tuple):
        self.pref_values = pref_values

        # Sets up the boundaries of the simulated grid
        self.set_env_limits()

        self.object_type_tuple = []

        for i in object_type_tuple:
            self.object_type_tuple.append((COLORS[i['color']],
                                           MATERIALS[i['material']],
                                           OBJECTS[i['object_type']],
                                           i['object_label']))

        # Uses a dictionary to initialize the locations of each object
        self.initial_object_locs = {}
        # For loop to handle multiple objects and not have them set (initialized) on the same point
        # i.e. we do not want to have objects started by being "stacked" on top of each other
        x_point = 0
        iteration = 1
        for obj in self.object_type_tuple:

            self.initial_object_locs[obj] = (x_point, 0)

            # changes the x-coordinate based on the number of objects we are using (flips back and forth and
            # increases further away from the origin
            iteration += 1
            if iteration % 2 == 0 and iteration > 2:
                x_point *= -1
                x_point += 1
            elif iteration % 2 == 0:
                x_point += 1
            else:
                x_point *= -1

        self.skills = ['Q1', 'Q2', 'Q3', 'Q4']

        # get possible joint actions and actions
        self.possible_single_actions = self.make_actions_list()
        self.current_state = self.create_initial_state()

        # set value iteration components
        self.transitions, self.rewards, self.state_to_idx, self.idx_to_action, \
            self.idx_to_state, self.action_to_idx = None, None, None, None, None, None
        self.vf = None
        self.pi = None
        self.policy = None
        self.epsilson = 0.001
        self.gamma = 0.99
        self.maxiter = 10000

        self.num_features = 4

    # getter function to retrieve the initial state of the environment
    def get_initial_state(self):
        return self.current_state

    # Sets the boundaries of the simulated grid
    def set_env_limits(self):
        # set environment limits
        self.x_min = -5
        self.x_max = 6
        self.y_min = -5
        self.y_max = 6
        self.all_coordinate_locations = list(product(range(self.x_min, self.x_max),
                                                     range(self.y_min, self.y_max)))

    # Generates all possible actions for each object that can occur
    # Implemented with EXIT being an action not connected with an object, used as to stop the entire episode
    def make_actions_list(self):
        actions_list = []
        for i in range(0, len(self.object_type_tuple)):
            actions_list.extend([(self.object_type_tuple[i], x) for x in self.skills])

        actions_list.append(EXIT)
        return actions_list

    # Takes each object and creates a dictionary to give each object attributes of the actions undergone
    # Also has exit as its own key to check whether all of objects are done within the episode
    def create_initial_state(self):
        # create dictionary of object location to object type and picked up state
        state_dict = {}
        for obj in self.initial_object_locs:
            state = {}
            state['pos'] = copy.deepcopy(self.initial_object_locs[obj])  # type tuple (color, type) to location
            state['orientation'] = np.pi  # The orientation options are 0 or pi, which correspond to 0 or
            # 180 degrees for the cup
            state['done'] = False
            state_dict[obj] = state

        state_dict['exit'] = False

        return state_dict

    # Resets the all objects with default attributes
    def reset(self):
        self.current_state = self.create_initial_state()

    # Takes an in a givens state, which has the objects and attributes as well as the 'exit' key, and converts the
    # dictionary format into a tuple
    def state_to_tuple(self, current_state):
        # List to store the data conversion
        current_state_tup = []

        # Iterates over all of the keys (objects), but not the 'exit' key
        for obj_type_idx in range(0, len(current_state.keys()) - 1):
            obj_type = list(current_state.keys())[obj_type_idx]
            # Specific format for tuple
            current_state_tup.append(((obj_type, current_state[obj_type]['pos']),
                                      current_state[obj_type]['orientation'], current_state[obj_type]['done']))

        # Appends the 'exit' key at the very end of the list
        current_state_tup.append(current_state['exit'])

        # Type casts the outer list data structure to a tuple
        return tuple(current_state_tup)

    # Checker to determine whether the episode has finished
    def is_done_given_state(self, current_state):
        # check if player at exit location
        if current_state['exit']:
            return True

        return False

    # Determines if the movement action, direction (i.e. up, down, left, right on the grid) is valid
    # Prevents if objects are going out of bounds and that objects are not on the same place on the grid
    def is_valid_push(self, current_state, action, locs_in_q1, locs_in_q2, locs_in_q3, locs_in_q4):
        # Splits the action (obj_type, obj_action)
        obj_type, obj_action = action
        break_outer_loop = False

        # creates a list with all of the other objects (iterating over the keys) and appending them to a list
        # this will also have 'exit' within this list
        other_obj_list = []
        for i in current_state.keys():
            if i != obj_type and i != 'exit':
                other_obj_list.append(i)

        # extracts location of the current object
        current_loc = current_state[obj_type]['pos']
        new_loc = current_loc

        # Checks if we already have moved the object before. Shouldn't be able to move it again
        if current_state[obj_type]['done'] == False:

            if obj_action == 'Q1':

                for j in locs_in_q1:

                    # Calculates new location from the given action from passing the previous condition
                    new_loc = tuple(np.array(j))

                    # Checks if the location is within the bounds of the grid
                    # if new_loc[0] < self.x_min or new_loc[0] >= self.x_max or new_loc[1] < self.y_min or new_loc[1] >= self.y_max:
                    #     return False, None

                    # Checks if the object location is not on top of another object
                    if len(other_obj_list) > 0:
                        for i in range(0, len(other_obj_list)):
                            if current_state[other_obj_list[i]]['pos'] == new_loc:
                                break_outer_loop = False
                                break
                            else:
                                break_outer_loop = True

                    if break_outer_loop or len(other_obj_list) == 0:
                        break_outer_loop = False
                        break

            elif obj_action == 'Q2':

                for j in locs_in_q2:

                    # Calculates new location from the given action from passing the previous condition
                    new_loc = tuple(np.array(j))

                    # Checks if the location is within the bounds of the grid
                    # if new_loc[0] < self.x_min or new_loc[0] >= self.x_max or new_loc[1] < self.y_min or new_loc[
                    #     1] >= self.y_max:
                    #     return False, None

                    # Checks if the object location is not on top of another object
                    if len(other_obj_list) > 0:
                        for i in range(0, len(other_obj_list)):
                            if current_state[other_obj_list[i]]['pos'] == new_loc:
                                break_outer_loop = False
                                break
                            else:
                                break_outer_loop = True

                    if break_outer_loop or len(other_obj_list) == 0:
                        break_outer_loop = False
                        break

            elif obj_action == 'Q3':

                for j in locs_in_q3:

                    # Calculates new location from the given action from passing the previous condition
                    new_loc = tuple(np.array(j))

                    # Checks if the location is within the bounds of the grid
                    # if new_loc[0] < self.x_min or new_loc[0] >= self.x_max or new_loc[1] < self.y_min or new_loc[
                    #     1] >= self.y_max:
                    #     return False, None

                    # Checks if the object location is not on top of another object
                    if len(other_obj_list) > 0:
                        for i in range(0, len(other_obj_list)):
                            if current_state[other_obj_list[i]]['pos'] == new_loc:
                                break_outer_loop = False
                                break
                            else:
                                break_outer_loop = True

                    if break_outer_loop or len(other_obj_list) == 0:
                        break_outer_loop = False
                        break

            elif obj_action == 'Q4':

                for j in locs_in_q4:

                    # Calculates new location from the given action from passing the previous condition
                    new_loc = tuple(np.array(j))

                    # Checks if the location is within the bounds of the grid
                    # if new_loc[0] < self.x_min or new_loc[0] >= self.x_max or new_loc[1] < self.y_min or new_loc[
                    #     1] >= self.y_max:
                    #     return False, None


                    # Checks if the object location is not on top of another object
                    if len(other_obj_list) > 0:
                        for i in range(0, len(other_obj_list)):
                            if current_state[other_obj_list[i]]['pos'] == new_loc:
                                break_outer_loop = False
                                break
                            else:
                                break_outer_loop = True

                    if break_outer_loop or len(other_obj_list) == 0:
                        break_outer_loop = False
                        break



        current_state[obj_type]['done'] = True
        return True, new_loc

    # grabs the quadrant location of the objects within the current_state
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
            
            # More concise quadrant determination logic
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
                # Removed incorrect Q4 append that was causing a bug

        return quadrant_list

    def get_reward_value(self, preferences, attributes, quadrants):
        if isinstance(preferences, dict) and quadrants in preferences:  # Base case: preferences is a reward value
            return preferences[quadrants]

        elif 'other' in preferences:
            return preferences['other'][quadrants]

        elif isinstance(preferences, dict):
            for attr in attributes:
                if attr in preferences:
                    return self.get_reward_value(preferences[attr], attributes, quadrants)
        
        # Add default return value if no match is found
        print(f"Warning: No preference found for attributes {attributes} in quadrant {quadrants}")
        return 0  # Default reward value

    # retrieves the reward value that should be used to assign to the step reward for the current location of the object
    def lookup_quadrant_reward(self, input_state):
        current_state = copy.deepcopy(input_state)
        objects_list = list(copy.deepcopy(input_state))[:-1]
        total_reward = 0

        # extracts the quadrant location of the current object
        quadrants_list = self.check_quadrant(current_state)

        # iterates over each object to calculate the reward value for where they are positioned on the grid
        for idx, obj in enumerate(objects_list):
            color_idx, material_idx, object_idx, object_label = obj
            color = COLORS_IDX[color_idx]
            material = MATERIALS_IDX[material_idx]
            object_type = OBJECTS_IDX[object_idx]
            quadrants = quadrants_list[idx]

            attributes = [object_type, color, material]
            for quadrant in quadrants:
                reward = self.get_reward_value(self.pref_values['pref_values'], attributes, quadrant)
                total_reward += reward

        return total_reward

    # calculates the new state and step reward cost for each action that the object can undergo based on each
    # possible action
    def step_given_state(self, input_state, action):
        step_cost = -0.1
        current_state = copy.deepcopy(input_state)
        step_reward = 0
        locs_in_q1 = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
                      (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2),
                      (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3),
                      (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4),
                      (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5),
                      (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)]

        locs_in_q2 = [(-1, 1), (-2, 1), (-3, 1), (-4, 1), (-5, 1), (-6, 1),
                      (-1, 2), (-2, 2), (-3, 2), (-4, 2), (-5, 2), (-6, 2),
                      (-1, 3), (-2, 3), (-3, 3), (-4, 3), (-5, 3), (-6, 3),
                      (-1, 4), (-2, 4), (-3, 4), (-4, 4), (-5, 4), (-6, 4),
                      (-1, 5), (-2, 5), (-3, 5), (-4, 5), (-5, 5), (-6, 5),
                      (-1, 6), (-2, 6), (-3, 6), (-4, 6), (-5, 6), (-6, 6)]

        locs_in_q3 = [(-1, -1), (-2, -1), (-3, -1), (-4, -1), (-5, -1), (-6, -1),
                      (-1, -2), (-2, -2), (-3, -2), (-4, -2), (-5, -2), (-6, -2),
                      (-1, -3), (-2, -3), (-3, -3), (-4, -3), (-5, -3), (-6, -3),
                      (-1, -4), (-2, -4), (-3, -4), (-4, -4), (-5, -4), (-6, -4),
                      (-1, -5), (-2, -5), (-3, -5), (-4, -5), (-5, -5), (-6, -5),
                      (-1, -6), (-2, -6), (-3, -6), (-4, -6), (-5, -6), (-6, -6)]

        locs_in_q4 = [(1, -1), (2, -1), (3, -1), (4, -1), (5, -1), (6, -1),
                      (1, -2), (2, -2), (3, -2), (4, -2), (5, -2), (6, -2),
                      (1, -3), (2, -3), (3, -3), (4, -3), (5, -3), (6, -3),
                      (1, -4), (2, -4), (3, -4), (4, -4), (5, -4), (6, -4),
                      (1, -5), (2, -5), (3, -5), (4, -5), (5, -5), (6, -5),
                      (1, -6), (2, -6), (3, -6), (4, -6), (5, -6), (6, -6)]

        new_loc = None

        if current_state['exit']:
            step_reward = 0

            return current_state, step_reward, True

        if action == EXIT:
            current_state['exit'] = True
            step_reward = self.lookup_quadrant_reward(current_state)
            step_reward += step_cost
            return current_state, step_reward, True

        # Splits the object type and the action
        obj_type, obj_action = action

        if action[1] in self.skills:
            valid, new_loc = self.is_valid_push(current_state, action, locs_in_q1, locs_in_q2, locs_in_q3, locs_in_q4)
            if valid is False:
                step_reward = step_cost
                return current_state, step_reward, False

        # Calculates the new location for the object
        # Only used for actions [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # The condition above for these directions is to determine if we can do this
        action_type_moved = action[0]
        current_loc = current_state[action_type_moved]['pos']

        current_state[action_type_moved]['pos'] = new_loc
        # Marks that the process of moving the object once is done
        current_state[obj_type]['done'] = True

        step_reward = step_cost

        # Determines if the given state is done
        done = self.is_done_given_state(current_state)

        return current_state, step_reward, done

    # Takes an in a givens state, which has the objects and attributes as well as the 'exit' key, and converts the
    # dictionary format into a tuple
    def tuple_to_state(self, current_state_tup):
        # convert current_state to tuple
        current_state_tup = list(current_state_tup)
        current_state_dict = {}

        # Iterates over all of the objects and formats them with respective format
        for i in range(0, len(current_state_tup) - 1):
            current_state_dict[current_state_tup[i][0][0]] = {'pos': current_state_tup[i][0][1],
                                                              'orientation': current_state_tup[i][1],
                                                              'done': current_state_tup[i][2]}

        # Adds the 'exit' state
        current_state_dict['exit'] = current_state_tup[-1]

        return current_state_dict

    # Generates all of the combinations the object can undergo based on the possible actions, positions, and states on
    # the simulated grid
    def enumerate_states(self):

        # Sets the enviroment
        self.reset()

        actions = self.possible_single_actions

        # All states will be stored in a graph
        G = nx.DiGraph()

        visited_states = set()

        stack = [copy.deepcopy(self.current_state)]

        while stack:

            state = stack.pop()

            # convert old state to tuple
            state_tup = self.state_to_tuple(state)

            # if state has not been visited, add it to the set of visited states
            if state_tup not in visited_states:
                visited_states.add(state_tup)
                # Only print status every 100 states to reduce console output
                if len(visited_states) % 100 == 0:
                    print(f"Enumerating states: {len(visited_states)} visited so far...")

            # get the neighbors of this state by looping through possible actions
            for idx, action in enumerate(actions):
                if self.is_done_given_state(state):
                    team_reward = 0
                    next_state = state
                    done = True

                else:
                    next_state, team_reward, done = self.step_given_state(state, action)

                new_state_tup = self.state_to_tuple(next_state)

                if new_state_tup not in visited_states:
                    stack.append(copy.deepcopy(next_state))

                # add edge to graph from current state to new state with weight equal to reward
                G.add_edge(state_tup, new_state_tup, weight=team_reward, action=action)

        states = list(G.nodes)
        print("Found states: ", len(states))

        # Converts all of the states in a dictionary for ease of access
        idx_to_state = {i: state for i, state in enumerate(states)}
        state_to_idx = {state: i for i, state in idx_to_state.items()}

        # Converts all of the actions in a dictionary for ease of access
        action_to_idx = {action: i for i, action in enumerate(actions)}
        idx_to_action = {i: action for i, action in enumerate(actions)}

        # construct transition matrix and reward matrix of shape [# states, # states, # actions] based on graph
        transition_mat = np.zeros([len(states), len(states), len(actions)])
        reward_mat = np.zeros([len(states), len(actions)])

        # iterates through all of the states and each action to compute and store the values of the transition matrix
        # and reward matrix
        for i in range(len(states)):
            state = self.tuple_to_state(idx_to_state[i])
            for action_idx_i in range(len(actions)):
                action = idx_to_action[action_idx_i]
                if self.is_done_given_state(state):
                    team_reward = 0
                    next_state = state

                else:
                    next_state, team_reward, done = self.step_given_state(state, action)

                next_state_i = state_to_idx[self.state_to_tuple(next_state)]

                transition_mat[i, next_state_i, action_idx_i] = 1.0
                reward_mat[i, action_idx_i] = team_reward

        # check that for each state and action pair, the sum of the transition probabilities is 1 (or 0 for terminal states)
        self.transitions, self.rewards, self.state_to_idx, \
            self.idx_to_action, self.idx_to_state, self.action_to_idx = transition_mat, reward_mat, state_to_idx, \
            idx_to_action, idx_to_state, action_to_idx

        return transition_mat, reward_mat, state_to_idx, idx_to_action, idx_to_state, action_to_idx

    def vectorized_vi(self):
        """
        Performs vectorized value iteration to find the optimal policy.
        
        This method uses the environment's transition and reward matrices to compute
        an optimal value function and policy using dynamic programming. The algorithm
        iteratively updates state values until convergence or until reaching the maximum
        number of iterations.
        
        The method uses instance variables:
        - transitions: Transition probability matrix of size (# states, # states, # actions)
        - rewards: Reward matrix of size (# states, # actions)
        - epsilon: Convergence threshold (default: 0.0001)
        - gamma: Discount factor (default: 0.99) 
        - maxiter: Maximum number of iterations (default: 10000)
        
        Returns
        -------
        tuple
            (value_function, Q_values, policy) where:
            - value_function: Array of optimal state values
            - Q_values: State-action values 
            - policy: Dictionary mapping states to optimal actions
        
        Notes
        -----
        The implementation uses efficient vectorized operations for speed but
        may be memory-intensive for environments with large state spaces.
        
                The value function. Of size (# states, 1).
            pi : array_like
                The optimal policy. Of size (# states, 1).
        """
        n_states = self.transitions.shape[0]
        n_actions = self.transitions.shape[2]

        # initialize value function
        pi = np.zeros((n_states, 1))
        vf = np.zeros((n_states, 1))
        Q = np.zeros((n_states, n_actions))
        policy = {}

        for i in range(self.maxiter):
            # Only print status occasionally to reduce output spam
            if i % 100 == 0:
                print(f"Value iteration progress: {i}/{self.maxiter} iterations")
            # initalize delta
            delta = 0
            # perform Bellman update
            for s in range(n_states):
                # store old value function
                old_v = vf[s].copy()
                # compute new value function
                Q[s] = np.sum((self.rewards[s] + self.gamma * vf) * self.transitions[s, :, :], 0)
                vf[s] = np.max(np.sum((self.rewards[s] + self.gamma * vf) * self.transitions[s, :, :], 0))
                # compute delta
                delta = np.max((delta, np.abs(old_v - vf[s])[0]))
            # check for convergence
            if delta < self.epsilson:
                break
        # compute optimal policy
        for s in range(n_states):
            pi[s] = np.argmax(np.sum(vf * self.transitions[s, :, :], 0))
            policy[s] = Q[s, :]

        self.vf = vf
        self.pi = pi
        self.policy = policy
        return vf, pi

    def render(self, current_state, timestep):
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        import os

        def getImage(path, zoom=1):
            zoom = 0.025
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

        plot_init_state = copy.deepcopy(current_state)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        (ax1) = ax

        if current_state['exit'] is True:
            ax1.axvline(x=0.5, color='red', linewidth=10, alpha=0.1)
            ax1.axhline(y=-0.5, color='red', linewidth=10, alpha=0.1)
        else:
            ax1.axvline(x=0.5, color='black', linewidth=7, alpha=0.1)
            ax1.axhline(y=-0.5, color='black', linewidth=7, alpha=0.1)

        type_to_color = {}
        for i in range(0, len(self.object_type_tuple)):
            object = self.object_type_tuple[i]
            color = object[0]
            type_to_color[object] = COLOR_DICTIONARY[color]
        type_to_loc_init = {}

        for type_o in self.object_type_tuple:
            loc = plot_init_state[type_o]['pos']
            color = type_to_color[type_o]
            type_to_loc_init[type_o] = loc

            ax1.scatter(loc[0], loc[1], color=color, s=500, alpha=0.99)
            # Get color, object type, and material names for better error messages
            color_name = COLORS_IDX.get(type_o[0], 'unknown')
            object_name = OBJECTS_IDX.get(type_o[1], 'unknown')
            material_name = MATERIALS_IDX.get(type_o[2], 'unknown')
            
            # Try to load object-specific image based on color and type
            image_path = f'data/objects/{color_name}{object_name}.jpeg'
            
            # Check if this specific combination exists
            if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), image_path)):
                ab = AnnotationBbox(getImage(image_path), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)
            else:
                # Fall back to type-specific images
                if type_o[:3] == (1, 1, 1):  # Red cup
                    ab = AnnotationBbox(getImage('data/objects/redcup.jpeg'), (loc[0], loc[1]), frameon=False)
                    ax.add_artist(ab)
                elif type_o[:3] == (2, 1, 1):  # Yellow cup
                    ab = AnnotationBbox(getImage('data/objects/yellowcup.jpeg'), (loc[0], loc[1]), frameon=False)
                    ax.add_artist(ab)
                else:
                    # Create a fallback colored shape based on object properties
                    print(f"Creating fallback visualization for: {color_name} {material_name} {object_name}")
                    
                    # Create a color-coded fallback image
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
                    if type_o[1] == 1:  # Cup - use circle
                        radius = 40
                        center = (50, 50)
                        y, x = np.ogrid[:100, :100]
                        mask = (x - center[0])**2 + (y - center[1])**2 > radius**2
                        img[mask] = [1, 1, 1, 0]  # Transparent outside circle
                    elif type_o[1] == 2:  # Bowl - use ellipse
                        radius_x, radius_y = 45, 30
                        center = (50, 50)
                        y, x = np.ogrid[:100, :100]
                        mask = (x - center[0])**2 / radius_x**2 + (y - center[1])**2 / radius_y**2 > 1
                        img[mask] = [1, 1, 1, 0]  # Transparent outside ellipse
                    
                    ab = AnnotationBbox(OffsetImage(img, zoom=0.025), (loc[0], loc[1]), frameon=False)
                    ax.add_artist(ab)

        offset = 0.1
        top_offset = -0.9
        ax1.set_xlim(self.x_min - offset, self.x_max + top_offset)
        ax1.set_ylim(self.y_min - offset, self.y_max + top_offset)

        ax1.set_xticks(np.arange(self.x_min - 1, self.x_max + 1, 1))
        ax1.set_yticks(np.arange(self.y_min - 1, self.y_max + 1, 1))
        ax1.grid()
        if current_state['exit'] is True:
            ax1.set_title(f"State at Time {timestep}: FINAL STATE")
        else:
            ax1.set_title(f"State at Time {timestep}")
        
        # Create the rollouts directory if it doesn't exist
        import os
        # Use relative path from the current script location
        rollout_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rollouts")
        os.makedirs(rollout_dir, exist_ok=True)
        
        # Save figure with absolute path
        save_path = os.path.join(rollout_dir, f"state_{timestep}.png")
        plt.savefig(save_path)
        print(f"Saved state image to: {save_path}")

        plt.show()
        plt.close()

    def rollout_full_game_joint_optimal(self, render=False):
        self.reset()
        done = False
        total_reward = 0

        iters = 0
        game_results = []
        number_of_objects = len(self.object_type_tuple)
        sum_feature_value = 0
        if render:
            self.render(self.current_state, iters)
        while not done:
            iters += 1
            current_state_tup = self.state_to_tuple(self.current_state)
            state_idx = self.state_to_idx[current_state_tup]

            action_distribution = self.policy[state_idx]
            action = np.argmax(action_distribution)
            action = self.idx_to_action[action]

            # print q values
            action_to_q = {}
            action_to_reward = {}
            for i in range(len(action_distribution)):
                action_to_q[self.idx_to_action[i]] = action_distribution[i]
                action_to_reward[self.idx_to_action[i]] = self.rewards[state_idx, i]

            game_results.append((self.current_state, action))
            next_state, team_rew, done = self.step_given_state(self.current_state, action)

            # render the current state
            print("\niteration: ", iters)
            print("current_state:", self.current_state)
            print("Action: ", action)
            print("next_state", next_state)

            featurized_state = self.lookup_quadrant_reward(self.current_state)
            sum_feature_value += featurized_state
            self.current_state = next_state
            if render:
                self.render(self.current_state, iters)

            total_reward += team_rew

            if iters > 40:
                break

        return total_reward, game_results, sum_feature_value

    def compute_optimal_performance(self, render=False):
        self.enumerate_states()

        self.vectorized_vi()

        optimal_rew, game_results, sum_feature_vector = self.rollout_full_game_joint_optimal(render=False)
        return optimal_rew, game_results, sum_feature_vector


def test_f_Ada():

    print("Ada Preferences:\n\n")

    # # # One object
    # object_type_tuple = [obj_1]  # just a global variable for the cup
    #
    # # now we set up the environment
    # game = Gridworld(f_Ada, object_type_tuple)
    #
    # # Run value iteration to get the optimal policy
    # optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance(render=True)
    # print("optimal_rew", optimal_rew)
    # print("game_results", game_results)
    #
    # # Two objects
    # object_type_tuple = [obj_1, obj_2]  # just a global variable for the cup
    #
    # # now we set up the environment
    # game = Gridworld(f_Ada, object_type_tuple)
    #
    # # Run value iteration to get the optimal policy
    # optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance()
    # print("optimal_rew", optimal_rew)
    # print("game_results", game_results)
    #
    # # Two of the same object
    # object_type_tuple = [obj_3, obj_3_dup]  # just a global variable for the cup
    #
    # # now we set up the environment
    # game = Gridworld(f_Ada, object_type_tuple)
    #
    # # Run value iteration to get the optimal policy
    # optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance()
    # print("optimal_rew", optimal_rew)
    # print("game_results", game_results)
    #
    # Three objects
    object_type_tuple = [obj_1, obj_2, obj_3]  # just a global variable for the cup

    # now we set up the environment
    game = Gridworld(f_Ada, object_type_tuple)

    # Run value iteration to get the optimal policy
    optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance()
    print("optimal_rew", optimal_rew)
    print("game_results", game_results)
    for i in game_results:
        print(i)


def test_f_Michelle():

    print("Michelle Preferences:\n\n")

    # One object
    object_type_tuple = [obj_1]  # just a global variable for the cup

    # now we set up the environment
    game = Gridworld(f_Michelle, object_type_tuple)

    # Run value iteration to get the optimal policy
    optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance()
    print("optimal_rew", optimal_rew)
    print("game_results", game_results)

    # Two objects
    object_type_tuple = [obj_1, obj_2]  # just a global variable for the cup

    # now we set up the environment
    game = Gridworld(f_Michelle, object_type_tuple)

    # Run value iteration to get the optimal policy
    optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance()
    print("optimal_rew", optimal_rew)
    print("game_results", game_results)

    # Two of the same object
    object_type_tuple = [obj_3, obj_3_dup]  # just a global variable for the cup

    # now we set up the environment
    game = Gridworld(f_Michelle, object_type_tuple)

    # Run value iteration to get the optimal policy
    optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance()
    print("optimal_rew", optimal_rew)
    print("game_results", game_results)

    # Three objects
    object_type_tuple = [obj_1, obj_2, obj_3]  # just a global variable for the cup

    # now we set up the environment
    game = Gridworld(f_Michelle, object_type_tuple)

    # Run value iteration to get the optimal policy
    optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance()
    print("optimal_rew", optimal_rew)
    print("game_results", game_results)


def test_f_Annika():

    print("Annika Preferences:\n\n")

    # One object
    object_type_tuple = [obj_1]  # just a global variable for the cup

    # now we set up the environment
    game = Gridworld(f_Annika, object_type_tuple)

    # Run value iteration to get the optimal policy
    optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance()
    print("optimal_rew", optimal_rew)
    print("game_results", game_results)

    # Two objects
    object_type_tuple = [obj_1, obj_2]  # just a global variable for the cup

    # now we set up the environment
    game = Gridworld(f_Annika, object_type_tuple)

    # Run value iteration to get the optimal policy
    optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance()
    print("optimal_rew", optimal_rew)
    print("game_results", game_results)

    # Two of the same object
    object_type_tuple = [obj_3, obj_3_dup]  # just a global variable for the cup

    # now we set up the environment
    game = Gridworld(f_Annika, object_type_tuple)

    # Run value iteration to get the optimal policy
    optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance()
    print("optimal_rew", optimal_rew)
    print("game_results", game_results)

    # Three objects
    object_type_tuple = [obj_1, obj_2, obj_3]  # just a global variable for the cup

    # now we set up the environment
    game = Gridworld(f_Annika, object_type_tuple)

    # Run value iteration to get the optimal policy
    optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance()
    print("optimal_rew", optimal_rew)
    print("game_results", game_results)

def test_f_Admoni():

    print("Admoni Preferences:\n\n")

    # One object
    object_type_tuple = [obj_1]  # just a global variable for the cup

    # now we set up the environment
    game = Gridworld(f_Admoni, object_type_tuple)

    # Run value iteration to get the optimal policy
    optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance()
    print("optimal_rew", optimal_rew)
    print("game_results", game_results)

    # Two objects
    object_type_tuple = [obj_1, obj_2]  # just a global variable for the cup

    # now we set up the environment
    game = Gridworld(f_Admoni, object_type_tuple)

    # Run value iteration to get the optimal policy
    optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance()
    print("optimal_rew", optimal_rew)
    print("game_results", game_results)

    # Two of the same object
    object_type_tuple = [obj_3, obj_3_dup]  # just a global variable for the cup

    # now we set up the environment
    game = Gridworld(f_Admoni, object_type_tuple)

    # Run value iteration to get the optimal policy
    optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance()
    print("optimal_rew", optimal_rew)
    print("game_results", game_results)

    # Three objects
    object_type_tuple = [obj_1, obj_2, obj_3]  # just a global variable for the cup

    # now we set up the environment
    game = Gridworld(f_Admoni, object_type_tuple)

    # Run value iteration to get the optimal policy
    optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance()
    print("optimal_rew", optimal_rew)
    print("game_results", game_results)


# Start of program
if __name__ == '__main__':
    test_f_Ada()
    # test_f_Michelle()
    # test_f_Annika()
    # test_f_Admoni()
