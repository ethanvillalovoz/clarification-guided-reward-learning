# Authors: Ethan Villalovz, Michelle Zhao
# Project: RISS 2024 Summer Project
# Description: Adaptation from `custom_mdp` by Michelle Zhao. Adding on other feature capabilities:
#     - Multiple Objects
#     - Different Materials
#     - Different colors
# Version 4 changes:
#     - Adding condition that once an object is placed down it cannot be picked up again
#     - Creating exit condition in step given state to handle more diverse human preferences
#     - Adding unit tests (multiple object combinations + different preference trees)

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
PICK_UP = 'pick_up'
PLACE_DOWN = 'place_down'

# Colors
COLORS = {'red': 1, 'yellow': 2, 'purple': 3, 'white': 4}
COLORS_IDX = {1: 'red', 2: 'yellow', 3: 'purple', 4: 'white'}

# Materials
MATERIALS = {'glass': 1, 'china': 2, 'plastic': 3}
MATERIALS_IDX = {1: 'glass', 2: 'china', 3: 'plastic'}

# Objects
OBJECTS = {'cup': 1, 'bowl': 2}
OBJECTS_IDX = {1: 'cup', 2: 'bowl'}

# Dictionary for `def render(self, current_state, timestep):`
COLOR_DICTIONARY = {1: 'red', 2: 'yellow', 3: 'purple'}


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

        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

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

    # Sets the boundaries of the simulated grid
    def set_env_limits(self):
        # set environment limits
        self.x_min = -2
        self.x_max = 2
        self.y_min = -2
        self.y_max = 2
        # self.x_min = -2
        # self.x_max = 3
        # self.y_min = -2
        # self.y_max = 3
        # self.x_min = -3
        # self.x_max = 4
        # self.y_min = -3
        # self.y_max = 4
        self.all_coordinate_locations = list(product(range(self.x_min, self.x_max),
                                                     range(self.y_min, self.y_max)))

    # Generates all possible actions for each object that can occur
    # Implemented with EXIT being an action not connected with an object, used as to stop the entire episode
    def make_actions_list(self):
        actions_list = []
        for i in range(0, len(self.object_type_tuple)):
            actions_list.extend([(self.object_type_tuple[i], x) for x in self.directions])
            actions_list.append((self.object_type_tuple[i], PICK_UP))
            actions_list.append((self.object_type_tuple[i], PLACE_DOWN))

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
            state['holding'] = False
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
            current_state_tup.append(((obj_type, current_state[obj_type]['pos']), current_state[obj_type]['holding'],
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
    def is_valid_push(self, current_state, action):
        # Splits the action (obj_type, obj_action)
        obj_type, obj_action = action

        # creates a list with all of the other objects (iterating over the keys) and appending them to a list
        # this will also have 'exit' within this list
        other_obj_list = []
        for i in current_state.keys():
            if i != obj_type:
                other_obj_list.append(i)

        # extracts location of the current object
        current_loc = current_state[obj_type]['pos']

        # checks if we are holding the object
        # we only want to be able to move the object if we are holding the object
        if current_state[obj_type]['holding'] is False:
            return False

        # Calculates new location from the given action from passing the previous condition
        new_loc = tuple(np.array(current_loc) + np.array(obj_action))

        # Checks if the location is within the bounds of the grid
        if new_loc[0] < self.x_min or new_loc[0] >= self.x_max or new_loc[1] < self.y_min or new_loc[1] >= self.y_max:
            return False

        # Checks if the object location is not on top of another object
        for i in range(0, len(other_obj_list) - 1):
            if current_state[other_obj_list[i]]['pos'] == new_loc:
                return False

        return True

    # grabs the quadrant location of the objects within the current_state
    def check_quadrant(self, input_state):
        current_state = list(copy.deepcopy(input_state))
        quadrant_list = []

        for i in range(0, len(current_state) - 1):
            # the position is on the origin
            if input_state[current_state[i]]['pos'] == (0, 0):
                quadrant_list.append(['Q1'])
            # the position is between the Q1 and Q4 on the x-axis
            elif input_state[current_state[i]]['pos'][0] > 0 and input_state[current_state[i]]['pos'][1] == 0:
                quadrant_list.append(['Q1'])
            # the position is between the Q2 and Q3 on the x-axis
            elif input_state[current_state[i]]['pos'][0] < 0 and input_state[current_state[i]]['pos'][1] == 0:
                quadrant_list.append(['Q2'])
            # the position is between the Q1 and Q2 on the y-axis
            elif input_state[current_state[i]]['pos'][0] == 0 and input_state[current_state[i]]['pos'][1] > 0:
                quadrant_list.append(['Q2'])
            # the position is between the Q3 and Q4 on the y-axis
            elif input_state[current_state[i]]['pos'][0] == 0 and input_state[current_state[i]]['pos'][1] < 0:
                quadrant_list.append(['Q3'])
            # the position is within Q1
            elif input_state[current_state[i]]['pos'][0] > 0 and input_state[current_state[i]]['pos'][1] > 0:
                quadrant_list.append(['Q1'])
            # the position is within Q2
            elif input_state[current_state[i]]['pos'][0] < 0 and input_state[current_state[i]]['pos'][1] > 0:
                quadrant_list.append(['Q2'])
            # the position is within Q3
            elif input_state[current_state[i]]['pos'][0] < 0 and input_state[current_state[i]]['pos'][1] < 0:
                quadrant_list.append(['Q3'])
            # the position is within Q4
            elif input_state[current_state[i]]['pos'][0] > 0 and input_state[current_state[i]]['pos'][1] < 0:
                quadrant_list.append(['Q4'])

        return quadrant_list

    def get_reward_value(self, preferences, attributes, quadrants):
        if isinstance(preferences, dict) and quadrants in preferences:  # Base case: preferences is a reward value
            return preferences[quadrants]

        elif isinstance(preferences, dict):
            for attr in attributes:
                if attr in preferences:
                    return self.get_reward_value(preferences[attr], attributes, quadrants)

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

        if current_state['exit']:
            step_reward = 0

            return current_state, step_reward, True

        if action == EXIT:
            current_state['exit'] = True
            step_reward = self.lookup_quadrant_reward(current_state)
            step_reward += step_cost
            return current_state, step_reward, True

        if action[1] in self.directions:
            if self.is_valid_push(current_state, action) is False:
                step_reward = step_cost
                return current_state, step_reward, False

        if action[1] == PICK_UP:
            # changes the status of holding for each object

            # Splits the object type and the action
            obj_type, obj_action = action
            other_obj_list = []
            obj_done_list = []

            # Checks if we already have moved the object before. Shouldn't be able to move it again
            if current_state[obj_type]['done'] == False:

                # Creates a list of all of the remaining object
                for i in current_state.keys():
                    if i != obj_type:
                        other_obj_list.append(i)

                for i in current_state.keys():
                    if i != obj_type:
                        obj_done_list.append(i)
                    elif i == obj_type:
                        break

                # Checks if the robot is currently not holding the object and not holding any of the other objects
                if current_state[obj_type]['holding'] == False and all(
                        current_state[other_obj_list[i]]['holding'] for i in range(0, len(other_obj_list) - 1)) == False and len(other_obj_list) > 1\
                        and all(
                        current_state[obj_done_list[i]]['done'] for i in range(0, len(obj_done_list) - 1)) == True:
                    current_state[obj_type]['holding'] = True
                    step_reward = step_cost

                # There is only one object in the simulation
                if current_state[obj_type]['holding'] == False and len(other_obj_list) == 1:
                    current_state[obj_type]['holding'] = True
                    step_reward = step_cost

            return current_state, step_reward, False

        if action[1] == PLACE_DOWN:
            # changes the status of holding for each object

            # Splits the object type and the action
            obj_type, obj_action = action
            other_obj_list = []
            obj_done_list = []

            # Checks if we already have moved the object before. Shouldn't be able to move it again
            if current_state[obj_type]['done'] == False:

                # Creates a list of all of the remaining object
                for i in current_state.keys():
                    if i != obj_type:
                        other_obj_list.append(i)

                for i in current_state.keys():
                    if i != obj_type:
                        obj_done_list.append(i)
                    elif i == obj_type:
                        break

                # Checks if the robot is currently holding the object and not holding any of the other objects
                if current_state[obj_type]['holding'] == True and all(
                        current_state[other_obj_list[i]]['holding'] for i in range(0, len(other_obj_list) - 1)) == False and len(other_obj_list) > 1\
                        and all(
                        current_state[obj_done_list[i]]['done'] for i in range(0, len(obj_done_list) - 1)) == True:
                    current_state[obj_type]['holding'] = False
                    step_reward = step_cost

                # There is only one object in the simulation
                if current_state[obj_type]['holding'] == True and len(other_obj_list) == 1:
                    current_state[obj_type]['holding'] = False
                    step_reward = step_cost

                # Marks that the process of moving the object once is done
                current_state[obj_type]['done'] = True

            return current_state, step_reward, False

        # Calculates the new location for the object
        # Only used for actions [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # The condition above for these directions is to determine if we can do this
        action_type_moved = action[0]
        current_loc = current_state[action_type_moved]['pos']

        new_loc = tuple(np.array(current_loc) + np.array(action[1]))
        current_state[action_type_moved]['pos'] = new_loc

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
                                                              'holding': current_state_tup[i][1],
                                                              'orientation': current_state_tup[i][2],
                                                              'done': current_state_tup[i][3]}

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
                print("Total visited States:", len(visited_states))

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
        # def spatial_environment(transitions, rewards, epsilson=0.0001, gamma=0.99, maxiter=10000):
        """
        Parameters
        ----------
            transitions : array_like
                Transition probability matrix. Of size (# states, # states, # actions).
            rewards : array_like
                Reward matrix. Of size (# states, # actions).
            epsilson : float, optional
                The convergence threshold. The default is 0.0001.
            gamma : float, optional
                The discount factor. The default is 0.99.
            maxiter : int, optional
                The maximum number of iterations. The default is 10000.
        Returns
        -------
            value_function : array_like
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
            print("vi iteration: ", i)
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
            zoom = 0.05
            # Adjust path to look in the source-code/data/images folder
            if 'images' in path:
                corrected_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "images", os.path.basename(path))
            else:
                corrected_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "images", os.path.basename(path))
            print(f"Looking for image at: {corrected_path}")
            return OffsetImage(plt.imread(corrected_path), zoom=zoom)

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

        path_red = 'data/images/redcup.jpeg'
        path180_red = 'data/images/redcup_180.jpeg'
        orientation_red = plot_init_state[(1, 1, 1, 2)]['orientation']
        path_yellow = 'data/images/yellowcup.jpeg'
        path180_yellow = 'data/images/yellowcup_180.jpeg'
        orientation_yellow = plot_init_state[(2, 1, 1, 1)]['orientation']

        for type_o in self.object_type_tuple:
            loc = plot_init_state[type_o]['pos']
            color = type_to_color[type_o]
            type_to_loc_init[type_o] = loc

            ax1.scatter(loc[0], loc[1], color=color, s=500, alpha=0.99)
            if orientation_red == 0 and type_o == (1, 1, 1, 2) and plot_init_state[type_o]['holding']:
                ab = AnnotationBbox(getImage('data/images/redcup_holding.jpeg'), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)
            elif orientation_red == 0 and type_o == (1, 1, 1, 2) and plot_init_state[type_o]['holding'] == False:
                ab = AnnotationBbox(getImage(path_red), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)
            elif orientation_red == np.pi and type_o == (1, 1, 1, 2) and plot_init_state[type_o]['holding']:
                ab = AnnotationBbox(getImage('data/images/redcup_180_holding.jpeg'), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)
            elif orientation_red == np.pi and type_o == (1, 1, 1, 2) and plot_init_state[type_o]['holding'] == False:
                ab = AnnotationBbox(getImage(path180_red), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)
            if orientation_yellow == 0 and type_o == (2, 1, 1, 1) and plot_init_state[type_o]['holding']:
                ab = AnnotationBbox(getImage('data/images/yellowcup_holding.jpeg'), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)
            elif orientation_yellow == 0 and type_o == (2, 1, 1, 1) and plot_init_state[type_o]['holding'] == False:
                ab = AnnotationBbox(getImage(path_yellow), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)
            elif orientation_yellow == np.pi and type_o == (2, 1, 1, 1) and plot_init_state[type_o]['holding']:
                ab = AnnotationBbox(getImage('data/images/yellowcup_180_holding.jpeg'), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)
            elif orientation_yellow == np.pi and type_o == (2, 1, 1, 1) and plot_init_state[type_o]['holding'] == False:
                ab = AnnotationBbox(getImage(path180_yellow), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)
            if orientation_yellow == 0 and type_o == (2, 1, 1, 3) and plot_init_state[type_o]['holding']:
                ab = AnnotationBbox(getImage('data/images/yellowcup_holding.jpeg'), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)
            elif orientation_yellow == 0 and type_o == (2, 1, 1, 3) and plot_init_state[type_o]['holding'] == False:
                ab = AnnotationBbox(getImage(path_yellow), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)
            elif orientation_yellow == np.pi and type_o == (2, 1, 1, 3) and plot_init_state[type_o]['holding']:
                ab = AnnotationBbox(getImage('data/images/yellowcup_180_holding.jpeg'), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)
            elif orientation_yellow == np.pi and type_o == (2, 1, 1, 3) and plot_init_state[type_o]['holding'] == False:
                ab = AnnotationBbox(getImage(path180_yellow), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)
            if type_o == (3, 2, 2, 3) and plot_init_state[type_o]['holding']:
                ab = AnnotationBbox(getImage('data/images/purplebowl_holding.jpeg'), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)
            elif type_o == (3, 2, 2, 3) and plot_init_state[type_o]['holding'] == False:
                ab = AnnotationBbox(getImage('data/images/purplebowl.jpeg'), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)
            elif type_o == (3, 2, 2, 3) and plot_init_state[type_o]['holding']:
                ab = AnnotationBbox(getImage('data/images/purplebowl_holding.jpeg'), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)
            elif type_o == (3, 2, 2, 3) and plot_init_state[type_o]['holding'] == False:
                ab = AnnotationBbox(getImage('data/images/purplebowl.jpeg'), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)
            if type_o == (3, 2, 2, 4) and plot_init_state[type_o]['holding']:
                ab = AnnotationBbox(getImage('data/images/purplebowl_holding.jpeg'), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)
            elif type_o == (3, 2, 2, 4) and plot_init_state[type_o]['holding'] == False:
                ab = AnnotationBbox(getImage('data/images/purplebowl.jpeg'), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)
            elif type_o == (3, 2, 2, 4) and plot_init_state[type_o]['holding']:
                ab = AnnotationBbox(getImage('data/images/purplebowl_holding.jpeg'), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)
            elif type_o == (3, 2, 2, 4) and plot_init_state[type_o]['holding'] == False:
                ab = AnnotationBbox(getImage('data/images/purplebowl.jpeg'), (loc[0], loc[1]), frameon=False)
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
        rollouts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rollouts")
        os.makedirs(rollouts_dir, exist_ok=True)
        plt.savefig(os.path.join(rollouts_dir, f"state_{timestep}.png"))

        plt.show()
        plt.close()

    def rollout_full_game_joint_optimal(self):
        self.reset()
        done = False
        total_reward = 0

        iters = 0
        game_results = []
        number_of_objects = len(self.object_type_tuple)
        sum_feature_value = 0

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

            self.render(self.current_state, iters)

            total_reward += team_rew

            if iters > 40:
                break

        return total_reward, game_results, sum_feature_value

    def compute_optimal_performance(self):
        self.enumerate_states()

        self.vectorized_vi()

        optimal_rew, game_results, sum_feature_vector = self.rollout_full_game_joint_optimal()
        return optimal_rew, game_results, sum_feature_vector


# Start of program
if __name__ == '__main__':
    # Define objects in our environment formatted to have the following attributes ['object', 'material', 'color'] as a
    # dictionary format
    obj_1 = {'object_type': 'cup', 'color': 'yellow', 'material': 'glass', 'object_label': 1}
    obj_2 = {'object_type': 'cup', 'color': 'red', 'material': 'glass', 'object_label': 2}
    # obj_3 = {'object_type': 'cup', 'color': 'yellow', 'material': 'glass', 'object_label': 3}

    # obj_3 = {'object_type': 'cup', 'color': 'yellow', 'material': 'glass'}
    obj_3 = {'object_type': 'bowl', 'color': 'purple', 'material': 'china', 'object_label': 3}
    obj_4 = {'object_type': 'bowl', 'color': 'purple', 'material': 'china', 'object_label': 4}

    # obj_4 = {'object_type': 'bowl', 'color': 'white', 'material': 'plastic'}

    # Tree preference structure
    f_Ethan = {'pref_values': {"cup": {'red': {'Q1': 10, 'Q2': -1, 'Q3': -1, 'Q4': 41},
                                       'yellow': {'Q1': 45, 'Q2': -1, 'Q3': 100, 'Q4': -1}},
                               'bowl': {'glass': {'Q1': -1, 'Q2': 5, 'Q3': -1, 'Q4': 10},
                                        'plastic': {'Q1': -1, 'Q2': 10, 'Q3': -1, 'Q4': -1},
                                        'china': {'Q1': -1, 'Q2': 5, 'Q3': 90, 'Q4': 5}}}}

    f_Michelle = {'pref_values': {'glass': {'red': {'Q1': 80, 'Q2': 1100, 'Q3': -1, 'Q4': 41},
                                            'yellow': {'Q1': 80, 'Q2': 100, 'Q3': 1111, 'Q4': 41}},
                                  'china': {'cup': {'Q1': 80, 'Q2': 100, 'Q3': -1, 'Q4': 41},
                                            'bowl': {'red': {'Q1': 80, 'Q2': 100, 'Q3': -1, 'Q4': 410},
                                                     'yellow': {'Q1': 80, 'Q2': 100, 'Q3': 221, 'Q4': 41},
                                                     'purple': {'Q1': 80, 'Q2': 100, 'Q3': 221, 'Q4': 41}}},
                                  'plastic': {'Q1': 80, 'Q2': -1, 'Q3': -1, 'Q4': 41}}}

    object_type_tuple = [obj_1, obj_2]  # just a global variable for the cup

    # now we set up the environment
    game = Gridworld(f_Michelle, object_type_tuple)

    # Run value iteration to get the optimal policy
    optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance()
    print("optimal_rew", optimal_rew)
    print("game_results", game_results)
