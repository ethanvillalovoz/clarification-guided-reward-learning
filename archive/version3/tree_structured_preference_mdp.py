"""
Tree-structured Preference MDP for Multi-object Gridworld

This module implements a Markov Decision Process (MDP) for a gridworld environment
where an agent can manipulate multiple objects with different properties (color, material, type).
Version 3 extends previous implementations with a hierarchical preference structure.

Features:
- Tree-structured preference representation (object type → color/material → quadrant)
- Multiple object types, colors, and materials
- Grid-based movement in cardinal directions
- Object pickup and placement capabilities
- Quadrant-based reward system
- Value iteration for policy computation
- Visualization of environment states

Authors: Ethan Villalovoz, Michelle Zhao
Project: RISS 2024 Summer Project
"""

# imports
import copy
import numpy as np
import pickle
import networkx as nx
from itertools import product
import matplotlib.pyplot as plt
import os

# Constants
# =========

# Actions
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

# Dictionary for rendering
COLOR_DICTIONARY = {1: 'red', 2: 'yellow', 3: 'purple'}


class Gridworld:
    """
    Gridworld environment for multi-object placement with tree-structured preferences.
    
    This class implements a grid-based environment where an agent can manipulate multiple
    objects with different properties based on hierarchical reward functions capturing
    user preferences for object placement in different quadrants.
    """
    
    def __init__(self, pref_values, object_type_tuple):
        """
        Initialize the Gridworld environment.
        
        Args:
            pref_values (dict): Tree-structured preferences for object placement
            object_type_tuple (list): List of object definitions with color, material, and type
        """
        self.pref_values = pref_values

        # Set environment boundaries
        self.set_env_limits()

        # Convert object definitions to internal representation
        self.object_type_tuple = []
        for obj in object_type_tuple:
            self.object_type_tuple.append((
                COLORS[obj['color']],
                MATERIALS[obj['material']],
                OBJECTS[obj['object_type']],
                obj['object_label']
            ))

        # Initialize object locations with appropriate spacing
        self.initial_object_locs = {}
        x_point = 0
        iteration = 1
        
        for obj in self.object_type_tuple:
            self.initial_object_locs[obj] = (x_point, 0)

            # Change x-coordinate based on number of objects
            # Alternating sides, increasing distance from origin
            iteration += 1
            if iteration % 2 == 0 and iteration > 2:
                x_point *= -1
                x_point += 1
            elif iteration % 2 == 0:
                x_point += 1
            else:
                x_point *= -1

        # Define movement directions (Up, Down, Right, Left)
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        # Generate possible actions and initial state
        self.possible_single_actions = self.make_actions_list()
        self.current_state = self.create_initial_state()

        # Value iteration components
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
        self.epsilson = 0.001  # Convergence threshold
        self.gamma = 0.99     # Discount factor
        self.maxiter = 10000  # Maximum iterations

        self.num_features = 4

    def set_env_limits(self):
        """Set the boundaries of the environment grid."""
        self.x_min = -2
        self.x_max = 3
        self.y_min = -2
        self.y_max = 3
        self.all_coordinate_locations = list(product(
            range(self.x_min, self.x_max),
            range(self.y_min, self.y_max)
        ))

    def make_actions_list(self):
        """
        Create list of possible actions in the environment.
        
        Returns:
            list: All possible actions (object-specific movements, pickups, placements, and exit)
        """
        actions_list = []
        
        # For each object, add directional movements and object-specific actions
        for i in range(len(self.object_type_tuple)):
            # Add directional movements for this object
            actions_list.extend([(self.object_type_tuple[i], x) for x in self.directions])
            
            # Add object-specific actions
            actions_list.append((self.object_type_tuple[i], PICK_UP))
            actions_list.append((self.object_type_tuple[i], PLACE_DOWN))

        # Add global exit action
        actions_list.append(EXIT)
        return actions_list

    def create_initial_state(self):
        """
        Create the initial state of the environment with all objects.
        
        Returns:
            dict: State dictionary with objects and their attributes
        """
        state_dict = {}
        
        # Initialize each object with position, orientation, and holding status
        for obj in self.initial_object_locs:
            state = {
                'pos': copy.deepcopy(self.initial_object_locs[obj]),
                'orientation': np.pi,  # Initial orientation (180 degrees)
                'holding': False       # Initially not being held
            }
            state_dict[obj] = state

        # Add exit status to state dictionary
        state_dict['exit'] = False
        return state_dict

    def reset(self):
        """Reset the environment to initial state."""
        self.current_state = self.create_initial_state()

    def state_to_tuple(self, current_state):
        """
        Convert state dict to hashable tuple for graph representation.
        
        Args:
            current_state (dict): State to convert
            
        Returns:
            tuple: Hashable representation of state
        """
        current_state_tup = []
        
        # Get all keys except 'exit'
        for obj_type_idx in range(len(current_state.keys()) - 1):
            obj_type = list(current_state.keys())[obj_type_idx]
            current_state_tup.append((
                (obj_type, current_state[obj_type]['pos']), 
                current_state[obj_type]['holding'],
                current_state[obj_type]['orientation']
            ))
        
        # Add exit status at the end
        current_state_tup.append(current_state['exit'])
        
        return tuple(current_state_tup)

    def tuple_to_state(self, current_state_tup):
        """
        Convert tuple representation back to state dict.
        
        Args:
            current_state_tup (tuple): Tuple representation of state
            
        Returns:
            dict: State dictionary
        """
        current_state_tup = list(current_state_tup)
        current_state_dict = {}
        
        # Extract object properties
        for i in range(len(current_state_tup) - 1):
            current_state_dict[current_state_tup[i][0][0]] = {
                'pos': current_state_tup[i][0][1],
                'holding': current_state_tup[i][1],
                'orientation': current_state_tup[i][2]
            }
        
        # Extract exit status
        current_state_dict['exit'] = current_state_tup[-1]
        
        return current_state_dict

    def is_done_given_state(self, current_state):
        """
        Check if the episode would be done in the given state.
        
        Args:
            current_state (dict): State to check
            
        Returns:
            bool: True if exit action was taken
        """
        return current_state['exit']

    def is_valid_push(self, current_state, action):
        """
        Check if a push action is valid from the current state.
        
        Args:
            current_state (dict): Current state
            action (tuple): (object_type, direction) to push
            
        Returns:
            bool: True if action is valid
        """
        # Split the action into object type and direction
        obj_type, obj_action = action
        
        # Find all other objects in the environment
        other_obj_list = []
        for i in current_state.keys():
            if i != obj_type:
                other_obj_list.append(i)

        # Get current object location
        current_loc = current_state[obj_type]['pos']

        # Can only move object if we're holding it
        if not current_state[obj_type]['holding']:
            return False

        # Calculate new location after movement
        new_loc = tuple(np.array(current_loc) + np.array(obj_action))

        # Check if new location is within grid boundaries
        if (new_loc[0] < self.x_min or new_loc[0] >= self.x_max or 
            new_loc[1] < self.y_min or new_loc[1] >= self.y_max):
            return False

        # Check if new location overlaps with any other object
        for i in range(len(other_obj_list) - 1):  # -1 to exclude 'exit'
            if current_state[other_obj_list[i]]['pos'] == new_loc:
                return False

        return True

    def check_quadrant(self, input_state):
        """
        Determine which quadrant each object is located in.
        
        Args:
            input_state (dict): Current state
            
        Returns:
            list: List of quadrants for each object
        """
        current_state = list(copy.deepcopy(input_state))
        quadrant_list = []

        # Determine quadrant for each object
        for i in range(len(current_state) - 1):  # Exclude 'exit'
            obj_key = current_state[i]
            pos_x, pos_y = input_state[obj_key]['pos']
            
            # Determine quadrant based on position
            if pos_x == 0 and pos_y == 0:
                # Origin
                quadrant_list.append(['Q1'])
            elif pos_x > 0 and pos_y == 0:
                # Positive x-axis
                quadrant_list.append(['Q1'])
            elif pos_x < 0 and pos_y == 0:
                # Negative x-axis
                quadrant_list.append(['Q2'])
            elif pos_x == 0 and pos_y > 0:
                # Positive y-axis
                quadrant_list.append(['Q2'])
            elif pos_x == 0 and pos_y < 0:
                # Negative y-axis
                quadrant_list.append(['Q3'])
            elif pos_x > 0 and pos_y > 0:
                # First quadrant
                quadrant_list.append(['Q1'])
            elif pos_x < 0 and pos_y > 0:
                # Second quadrant
                quadrant_list.append(['Q2'])
            elif pos_x < 0 and pos_y < 0:
                # Third quadrant
                quadrant_list.append(['Q3'])
            elif pos_x > 0 and pos_y < 0:
                # Fourth quadrant
                quadrant_list.append(['Q4'])

        return quadrant_list

    def lookup_quadrant_reward(self, input_state):
        """
        Calculate reward based on object locations and user preferences.
        
        Args:
            input_state (dict): Current state
            
        Returns:
            float: Total reward from all objects
        """
        current_state = copy.deepcopy(input_state)
        objects_list = list(copy.deepcopy(input_state))[:-1]  # Exclude 'exit'
        total_reward = 0

        # Get quadrant for each object
        quadrants_list = self.check_quadrant(current_state)

        # Calculate reward for each object based on preferences
        for idx, obj in enumerate(objects_list):
            color_idx, material_idx, object_idx, object_label_idx = obj
            color = COLORS_IDX[color_idx]
            material = MATERIALS_IDX[material_idx]
            object_type = OBJECTS_IDX[object_idx]
            quadrants = quadrants_list[idx]

            # Get preferences for this object type
            if object_type in self.pref_values['pref_values']:
                object_pref = self.pref_values['pref_values'][object_type]

                # Cups consider color preferences
                if object_type == 'cup' and color in object_pref:
                    color_pref = object_pref[color]
                    for quadrant in quadrants:
                        if quadrant in color_pref:
                            total_reward += color_pref[quadrant]
                
                # Bowls consider material preferences
                elif object_type == 'bowl' and material in object_pref:
                    material_pref = object_pref[material]
                    for quadrant in quadrants:
                        if quadrant in material_pref:
                            total_reward += material_pref[quadrant]

        return total_reward

    def step_given_state(self, input_state, action):
        """
        Execute action from given state and return next state, reward, and done flag.
        
        Args:
            input_state (dict): Starting state
            action: Action to execute
            
        Returns:
            tuple: (next_state, reward, done)
        """
        step_cost = -0.1  # Small penalty for each step
        current_state = copy.deepcopy(input_state)
        step_reward = 0

        # If already in terminal state
        if current_state['exit']:
            return current_state, 0, True

        # Handle exit action
        if action == EXIT:
            current_state['exit'] = True
            step_reward = self.lookup_quadrant_reward(current_state)
            step_reward += step_cost
            return current_state, step_reward, True

        # Handle directional movement
        if isinstance(action, tuple) and action[1] in self.directions:
            if not self.is_valid_push(current_state, action):
                return current_state, step_cost, False
                
            # Move object
            obj_type = action[0]
            current_loc = current_state[obj_type]['pos']
            new_loc = tuple(np.array(current_loc) + np.array(action[1]))
            current_state[obj_type]['pos'] = new_loc
            
            step_reward = step_cost
            done = self.is_done_given_state(current_state)
            return current_state, step_reward, done

        # Handle pickup action
        if isinstance(action, tuple) and action[1] == PICK_UP:
            obj_type = action[0]
            other_obj_list = [i for i in current_state.keys() if i != obj_type]
            
            # Can only pick up if we're not holding this or any other object
            if (not current_state[obj_type]['holding'] and
                all(not current_state[other_obj_list[i]]['holding'] 
                    for i in range(len(other_obj_list) - 1))):  # -1 to exclude 'exit'
                
                current_state[obj_type]['holding'] = True
                step_reward = step_cost
            
            return current_state, step_reward, False

        # Handle place down action
        if isinstance(action, tuple) and action[1] == PLACE_DOWN:
            obj_type = action[0]
            
            # Can only place down if we're holding this object
            if current_state[obj_type]['holding']:
                current_state[obj_type]['holding'] = False
                step_reward = step_cost
            
            return current_state, step_reward, False

        # Shouldn't reach here, but just in case
        return current_state, step_cost, False

    def enumerate_states(self):
        """
        Enumerate all possible states and transitions for value iteration.
        
        Returns:
            tuple: (transition_matrix, reward_matrix, state_to_idx, idx_to_action, idx_to_state, action_to_idx)
        """
        # Reset environment
        self.reset()
        actions = self.possible_single_actions

        # Create directional graph to represent all states
        G = nx.DiGraph()
        visited_states = set()
        stack = [copy.deepcopy(self.current_state)]

        # Perform DFS to discover all states
        while stack:
            state = stack.pop()
            state_tup = self.state_to_tuple(state)

            # Add state if not visited
            if state_tup not in visited_states:
                visited_states.add(state_tup)
                print(f"Total visited States: {len(visited_states)}")

            # Explore all actions from this state
            for action in actions:
                if self.is_done_given_state(state):
                    team_reward = 0
                    next_state = state
                    done = True
                else:
                    next_state, team_reward, done = self.step_given_state(state, action)

                # Add next state to stack if not visited
                new_state_tup = self.state_to_tuple(next_state)
                if new_state_tup not in visited_states:
                    stack.append(copy.deepcopy(next_state))

                # Add edge to graph
                G.add_edge(state_tup, new_state_tup, weight=team_reward, action=action)

        # Create mappings between states/actions and indices
        states = list(G.nodes)
        print(f"Found states: {len(states)}")

        idx_to_state = {i: state for i, state in enumerate(states)}
        state_to_idx = {state: i for i, state in idx_to_state.items()}

        action_to_idx = {action: i for i, action in enumerate(actions)}
        idx_to_action = {i: action for i, action in enumerate(actions)}

        # Construct transition and reward matrices
        transition_mat = np.zeros([len(states), len(states), len(actions)])
        reward_mat = np.zeros([len(states), len(actions)])

        # Fill transition and reward matrices
        for i in range(len(states)):
            state = self.tuple_to_state(idx_to_state[i])
            for action_idx_i in range(len(actions)):
                action = idx_to_action[action_idx_i]
                
                if self.is_done_given_state(state):
                    team_reward = 0
                    next_state = state
                else:
                    next_state, team_reward, done = self.step_given_state(state, action)

                # Add to matrices
                next_state_i = state_to_idx[self.state_to_tuple(next_state)]
                transition_mat[i, next_state_i, action_idx_i] = 1.0
                reward_mat[i, action_idx_i] = team_reward

        # Store matrices and mappings
        self.transitions = transition_mat
        self.rewards = reward_mat
        self.state_to_idx = state_to_idx
        self.idx_to_action = idx_to_action
        self.idx_to_state = idx_to_state
        self.action_to_idx = action_to_idx

        return transition_mat, reward_mat, state_to_idx, idx_to_action, idx_to_state, action_to_idx

    def vectorized_vi(self):
        """
        Perform vectorized value iteration to find optimal policy.
        
        Returns:
            tuple: (value_function, policy)
        """
        n_states = self.transitions.shape[0]
        n_actions = self.transitions.shape[2]

        # Initialize value function and policy
        pi = np.zeros((n_states, 1))
        vf = np.zeros((n_states, 1))
        Q = np.zeros((n_states, n_actions))
        policy = {}

        # Perform value iteration
        for i in range(self.maxiter):
            print(f"VI iteration: {i}")
            delta = 0  # Convergence measure

            # Perform Bellman update for each state
            for s in range(n_states):
                # Store old value function
                old_v = vf[s].copy()
                
                # Compute Q-values and update value function
                Q[s] = np.sum((self.rewards[s] + self.gamma * vf) * self.transitions[s, :, :], 0)
                vf[s] = np.max(Q[s])
                
                # Update delta for convergence check
                delta = max(delta, np.abs(old_v - vf[s])[0])
                
            # Check for convergence
            if delta < self.epsilson:
                break
                
        # Compute optimal policy
        for s in range(n_states):
            pi[s] = np.argmax(Q[s])
            policy[s] = Q[s]
        
        # Store results
        self.vf = vf
        self.pi = pi
        self.policy = policy
        
        return vf, pi

    def render(self, current_state, timestep):
        """
        Render the current state of the environment.
        
        Args:
            current_state (dict): State to render
            timestep (int): Current timestep for display
        """
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        
        def getImage(path, zoom=0.05):
            """Load and resize an image for display."""
            # Ensure data directory exists
            if not os.path.exists('data'):
                os.makedirs('data')
            return OffsetImage(plt.imread(path), zoom=zoom)

        # Create a copy of the state to avoid modifying it
        plot_init_state = copy.deepcopy(current_state)

        # Create figure and axis
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax1 = ax

        # Draw grid centerlines with color based on exit state
        if current_state['exit']:
            ax1.axvline(x=0.5, color='red', linewidth=10, alpha=0.1)
            ax1.axhline(y=-0.5, color='red', linewidth=10, alpha=0.1)
        else:
            ax1.axvline(x=0.5, color='black', linewidth=7, alpha=0.1)
            ax1.axhline(y=-0.5, color='black', linewidth=7, alpha=0.1)

        # Set up color mapping for objects
        type_to_color = {}
        for i in range(len(self.object_type_tuple)):
            obj = self.object_type_tuple[i]
            color = obj[0]
            type_to_color[obj] = COLOR_DICTIONARY[color]
        type_to_loc_init = {}

        # Define image paths
        path_yellow = 'data/yellowcup.jpeg'
        path180_yellow = 'data/yellowcup_180.jpeg'
        
        # Draw each object
        for type_o in self.object_type_tuple:
            loc = plot_init_state[type_o]['pos']
            color = type_to_color[type_o]
            is_holding = plot_init_state[type_o]['holding']
            orientation = plot_init_state[type_o]['orientation']
            type_to_loc_init[type_o] = loc

            # Draw marker for object location
            ax1.scatter(loc[0], loc[1], color=color.lower(), s=500, alpha=0.99)
            
            # Draw appropriate image based on object properties
            if type_o[0] == 2:  # Yellow objects
                if orientation == 0:  # 0 degrees
                    if is_holding:
                        ab = AnnotationBbox(getImage('data/yellowcup_holding.jpeg'), (loc[0], loc[1]), frameon=False)
                    else:
                        ab = AnnotationBbox(getImage(path_yellow), (loc[0], loc[1]), frameon=False)
                else:  # 180 degrees
                    if is_holding:
                        ab = AnnotationBbox(getImage('data/yellowcup_180_holding.jpeg'), (loc[0], loc[1]), frameon=False)
                    else:
                        ab = AnnotationBbox(getImage(path180_yellow), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)
            elif type_o[0] == 3:  # Purple objects
                if is_holding:
                    ab = AnnotationBbox(getImage('data/purplebowl_holding.jpeg'), (loc[0], loc[1]), frameon=False)
                else:
                    ab = AnnotationBbox(getImage('data/purplebowl.jpeg'), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)

        # Set plot limits and grid
        offset = 0.1
        top_offset = -0.9
        ax1.set_xlim(self.x_min - offset, self.x_max + top_offset)
        ax1.set_ylim(self.y_min - offset, self.y_max + top_offset)
        ax1.set_xticks(np.arange(self.x_min - 1, self.x_max + 1, 1))
        ax1.set_yticks(np.arange(self.y_min - 1, self.y_max + 1, 1))
        ax1.grid()
        
        # Set title based on state
        if current_state['exit']:
            ax1.set_title(f"State at Time {timestep}: FINAL STATE")
        else:
            ax1.set_title(f"State at Time {timestep}")
        
        # Ensure rollouts directory exists
        if not os.path.exists('rollouts'):
            os.makedirs('rollouts')
            
        # Save image and display
        plt.savefig(f"rollouts/state_{timestep}.png")
        plt.show()
        plt.close()

    def rollout_full_game_joint_optimal(self):
        """
        Execute a full rollout using the optimal policy.
        
        Returns:
            tuple: (total_reward, game_results, sum_feature_value)
        """
        self.reset()
        done = False
        total_reward = 0
        iters = 0
        game_results = []
        sum_feature_value = 0

        # Render initial state
        self.render(self.current_state, iters)
        
        # Run until done or max iterations
        while not done:
            iters += 1
            
            # Get current state and optimal action
            current_state_tup = self.state_to_tuple(self.current_state)
            state_idx = self.state_to_idx[current_state_tup]
            
            action_distribution = self.policy[state_idx]
            action = np.argmax(action_distribution)
            action = self.idx_to_action[action]

            # Store Q-values and rewards for debugging
            action_to_q = {}
            action_to_reward = {}
            for i in range(len(action_distribution)):
                action_to_q[self.idx_to_action[i]] = action_distribution[i]
                action_to_reward[self.idx_to_action[i]] = self.rewards[state_idx, i]

            # Store current state and action
            game_results.append((self.current_state, action))
            
            # Take step
            next_state, team_rew, done = self.step_given_state(self.current_state, action)
            
            # Debug information
            print(f"\nIteration: {iters}")
            print(f"Current state: {self.current_state}")
            print(f"Action: {action}")
            print(f"Next state: {next_state}")

            # Update feature sum and state
            featurized_state = self.lookup_quadrant_reward(self.current_state)
            sum_feature_value += featurized_state
            self.current_state = next_state
            
            # Render current state
            self.render(self.current_state, iters)
            
            # Update total reward
            total_reward += team_rew
            
            # Avoid infinite loops
            if iters > 40:
                break

        return total_reward, game_results, sum_feature_value

    def compute_optimal_performance(self):
        """
        Compute and execute optimal policy.
        
        Returns:
            tuple: (optimal_reward, game_results, sum_feature_vector)
        """
        # Enumerate states and run value iteration
        self.enumerate_states()
        self.vectorized_vi()
        
        # Execute optimal policy
        optimal_rew, game_results, sum_feature_vector = self.rollout_full_game_joint_optimal()
        return optimal_rew, game_results, sum_feature_vector


# Start of program
if __name__ == '__main__':
    """
    Main execution block to demonstrate the tree-structured preference MDP.
    """
    # Define objects with properties
    obj_1 = {'object_type': 'cup', 'color': 'yellow', 'material': 'glass', 'object_label': 1}
    # obj_2 = {'object_type': 'cup', 'color': 'red', 'material': 'glass', 'object_label': 2}
    # obj_3 = {'object_type': 'bowl', 'color': 'purple', 'material': 'china', 'object_label': 3}
    # obj_4 = {'object_type': 'bowl', 'color': 'purple', 'material': 'china', 'object_label': 4}

    # Tree-structured preferences
    f_Ethan = {
        'pref_values': {
            "cup": {
                'red': {'Q1': 10, 'Q2': -1, 'Q3': -1, 'Q4': 41},
                'yellow': {'Q1': 45, 'Q2': -1, 'Q3': 100, 'Q4': -1}
            },
            'bowl': {
                'glass': {'Q1': -1, 'Q2': 5, 'Q3': -1, 'Q4': 10},
                'plastic': {'Q1': -1, 'Q2': 10, 'Q3': -1, 'Q4': -1},
                'china': {'Q1': -1, 'Q2': 5, 'Q3': 90, 'Q4': 5}
            }
        }
    }

    # Create environment with yellow cup object
    object_type_tuple = [obj_1]
    game = Gridworld(f_Ethan, object_type_tuple)

    # Run value iteration to find optimal policy
    optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance()
    print(f"Optimal reward: {optimal_rew}")
    print(f"Feature vector sum: {sum_feature_vector}")
    print(f"Number of actions taken: {len(game_results)}")
