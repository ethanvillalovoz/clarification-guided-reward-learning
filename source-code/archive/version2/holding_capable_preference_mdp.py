"""
Multi-Object Preference Learning in Gridworld with Object Holding

This module implements a Markov Decision Process (MDP) for a gridworld environment
where an agent can manipulate multiple objects with different properties. Version 2
extends the original implementation with object holding capability and improved
computational pipeline.

Features:
- Support for multiple objects with different properties (color, type)
- Grid-based movement in cardinal directions
- Object orientation changes (0° or 180°)
- Object picking up and placing down
- Value iteration for policy computation
- Visualization with holding state indicators

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
ROTATE180 = 'rotate180'
PICK_UP = 'pick_up'
PLACE_DOWN = 'place_down'

# Colors
RED = 1
YELLOW = 2
COLOR_DICTIONARY = {1: 'red', 2: 'yellow'}

# Objects
CUP = 1

# Objects Combinations
RED_CUP = (RED, CUP)
YELLOW_CUP = (YELLOW, CUP)


class Gridworld:
    """
    Gridworld environment for multi-object placement preference learning with holding capability.
    
    This class implements a grid-based environment where an agent can move multiple
    objects to different positions, pick them up, place them down, and rotate them
    based on reward functions capturing user preferences.
    """
    
    def __init__(self, reward_weights, true_f_indices, object_type_tuple, red_centroid, blue_centroid):
        """
        Initialize the Gridworld environment.
        
        Args:
            reward_weights (list): Weights for different reward components per object
            true_f_indices (list): Binary indicators for which features matter to the user per object
            object_type_tuple (list): List of object types to place
            red_centroid (tuple): Coordinates (x, y) of the red centroid
            blue_centroid (tuple): Coordinates (x, y) of the blue centroid
        """
        self.true_f_indices = true_f_indices
        self.reward_weights = reward_weights

        # Set environment boundaries and parameters
        self.set_env_limits()
        
        # Set up object properties
        self.object_type_tuple = object_type_tuple
        
        # Uses a dictionary to initialize the locations of each object
        # Ensures objects aren't initialized on top of each other
        self.initial_object_locs = {}
        x_point = 0
        iteration = 1
        
        for obj in object_type_tuple:
            self.initial_object_locs[obj] = (x_point, 0)

            # Changes x-coordinate based on number of objects (alternating sides, increasing distance)
            iteration += 1
            if iteration % 2 == 0:
                x_point += 1
            else:
                x_point *= -1

        # Define possible movement directions (Up, Down, Right, Left)
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        # Set up possible actions and initial state
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

        # Feature parameters
        self.num_features = 4
        self.red_centroid = red_centroid
        self.blue_centroid = blue_centroid

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
            list: All possible actions (object-specific movements, rotations, pickups, placements, and exit)
        """
        actions_list = []
        
        # For each object, add directional movements and object-specific actions
        for i in range(len(self.object_type_tuple)):
            # Add directional movements for this object
            actions_list.extend([(self.object_type_tuple[i], x) for x in self.directions])
            
            # Add object-specific actions
            actions_list.append((self.object_type_tuple[i], ROTATE180))
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
        
        # Add each object's properties to the tuple
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

    def featurize_state(self, current_state):
        """
        Convert state to feature vector for reward calculation.
        
        Args:
            current_state (dict): State to featurize
            
        Returns:
            list: List of feature vectors, one per object
        """
        state_feature = []
        obj_list = list(current_state.keys())
        
        # Calculate features for each object
        for i in range(len(current_state) - 1):  # Exclude 'exit'
            current_loc = current_state[obj_list[i]]['pos']

            # Calculate distances to centroids
            dist_to_red_centroid = np.linalg.norm(np.array(current_loc) - np.array(self.red_centroid))
            dist_to_blue_centroid = np.linalg.norm(np.array(current_loc) - np.array(self.blue_centroid))

            # Get orientation and y-position
            orientation = current_state[obj_list[i]]['orientation']
            pos_y = current_loc[1]

            # Apply feature importance mask
            state_feature.append(
                np.multiply(
                    np.array([orientation, dist_to_red_centroid, dist_to_blue_centroid, pos_y]),
                    self.true_f_indices[i]
                )
            )

        return state_feature

    def step_given_state(self, input_state, action):
        """
        Execute action from given state and return next state, reward, and done flag.
        
        Args:
            input_state (dict): Starting state
            action: Action to execute (either EXIT or (object_type, action_type))
            
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
            featurized_state = self.featurize_state(current_state)

            # Flatten feature vectors and reward weights for dot product
            featurized_state_vector = []
            for i in range(len(featurized_state)):
                for j in range(len(featurized_state[i])):
                    featurized_state_vector.append(featurized_state[i][j])

            reward_weights_vector = []
            for i in range(len(self.reward_weights)):
                for j in range(len(self.reward_weights[i])):
                    reward_weights_vector.append(self.reward_weights[i][j])

            step_reward = np.dot(featurized_state_vector, reward_weights_vector) + step_cost
            return current_state, step_reward, True

        # Handle directional movement
        if action[1] in self.directions:
            if not self.is_valid_push(current_state, action):
                return current_state, step_cost, False
                
            # Move object
            action_type_moved = action[0]
            current_loc = current_state[action_type_moved]['pos']
            new_loc = tuple(np.array(current_loc) + np.array(action[1]))
            current_state[action_type_moved]['pos'] = new_loc
            
            step_reward = step_cost
            done = self.is_done_given_state(current_state)
            return current_state, step_reward, done

        # Handle rotation action
        if action[1] == ROTATE180:
            obj_type, _ = action
            other_obj_list = []
            
            # Create list of all other objects
            for i in current_state.keys():
                if i != obj_type:
                    other_obj_list.append(i)

            # Can only rotate if holding this object and not holding any others
            if (current_state[obj_type]['holding'] and
                all(not current_state[other_obj_list[i]]['holding'] 
                    for i in range(len(other_obj_list) - 1))):  # -1 to exclude 'exit'
                
                # Rotate 180 degrees (π radians)
                current_state[obj_type]['orientation'] = (
                    current_state[obj_type]['orientation'] + np.pi
                ) % (2 * np.pi)
                
                step_reward = step_cost
            
            return current_state, step_reward, False

        # Handle pickup action
        if action[1] == PICK_UP:
            obj_type, _ = action
            other_obj_list = []
            
            # Create list of all other objects
            for i in current_state.keys():
                if i != obj_type:
                    other_obj_list.append(i)

            # Can only pick up if not holding this or any other object
            if (not current_state[obj_type]['holding'] and
                all(not current_state[other_obj_list[i]]['holding'] 
                    for i in range(len(other_obj_list) - 1))):  # -1 to exclude 'exit'
                
                current_state[obj_type]['holding'] = True
                step_reward = step_cost
            
            return current_state, step_reward, False

        # Handle place down action
        if action[1] == PLACE_DOWN:
            obj_type, _ = action
            other_obj_list = []
            
            # Create list of all other objects
            for i in current_state.keys():
                if i != obj_type:
                    other_obj_list.append(i)

            # Can only place down if holding this object and not holding any others
            if (current_state[obj_type]['holding'] and
                all(not current_state[other_obj_list[i]]['holding'] 
                    for i in range(len(other_obj_list) - 1))):  # -1 to exclude 'exit'
                
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
        
        def getImage(path, zoom=1):
            """Load and resize an image for display."""
            zoom = 0.05
            # Make sure the path includes 'data' directory
            if not path.startswith('data/'):
                path = 'data/' + os.path.basename(path)
            return OffsetImage(plt.imread(path), zoom=zoom)

        # Create a copy of the state to avoid modifying it
        plot_init_state = copy.deepcopy(current_state)

        # Create figure and axis
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax1 = ax

        # Draw grid centerlines with color based on exit state
        if current_state['exit']:
            ax1.axvline(x=0, color='red', linewidth=10, alpha=0.1)
            ax1.axhline(y=0, color='red', linewidth=10, alpha=0.1)
        else:
            ax1.axvline(x=0, color='black', linewidth=7, alpha=0.1)
            ax1.axhline(y=0, color='black', linewidth=7, alpha=0.1)

        # Set up color mapping for objects
        type_to_color = {}
        for i in range(len(self.object_type_tuple)):
            obj = self.object_type_tuple[i]
            color = obj[0]
            type_to_color[obj] = COLOR_DICTIONARY[color]
        type_to_loc_init = {}

        # Draw centroid regions
        ax1.scatter(self.red_centroid[0], self.red_centroid[1], color='red', s=800, alpha=0.1)
        ax1.scatter(self.blue_centroid[0], self.blue_centroid[1], color='blue', s=800, alpha=0.1)

        # Define image paths for different objects and orientations
        path_red = 'data/redcup.jpeg'
        path180_red = 'data/redcup_180.jpeg'
        path_red_holding = 'data/redcup_holding.jpeg'
        path180_red_holding = 'data/redcup_180_holding.jpeg'
        
        path_yellow = 'data/yellowcup.jpeg'
        path180_yellow = 'data/yellowcup_180.jpeg'
        path_yellow_holding = 'data/yellowcup_holding.jpeg'
        path180_yellow_holding = 'data/yellowcup_180_holding.jpeg'

        # Get current orientations
        orientation_red = plot_init_state.get((1, 1), {}).get('orientation', 0)
        orientation_yellow = plot_init_state.get((2, 1), {}).get('orientation', 0)

        # Draw objects
        for type_o in self.object_type_tuple:
            # Get object properties
            loc = plot_init_state[type_o]['pos']
            color = type_to_color[type_o]
            is_holding = plot_init_state[type_o]['holding']
            type_to_loc_init[type_o] = loc

            # Draw marker for object location
            ax1.scatter(loc[0], loc[1], color=color, s=500, alpha=0.99)
            
            # Add appropriate image based on object type, orientation, and holding state
            if type_o == (1, 1):  # Red cup
                if orientation_red == 0:  # 0 degrees
                    if is_holding:
                        ab = AnnotationBbox(getImage(path_red_holding), (loc[0], loc[1]), frameon=False)
                    else:
                        ab = AnnotationBbox(getImage(path_red), (loc[0], loc[1]), frameon=False)
                else:  # 180 degrees
                    if is_holding:
                        ab = AnnotationBbox(getImage(path180_red_holding), (loc[0], loc[1]), frameon=False)
                    else:
                        ab = AnnotationBbox(getImage(path180_red), (loc[0], loc[1]), frameon=False)
            elif type_o == (2, 1):  # Yellow cup
                if orientation_yellow == 0:  # 0 degrees
                    if is_holding:
                        ab = AnnotationBbox(getImage(path_yellow_holding), (loc[0], loc[1]), frameon=False)
                    else:
                        ab = AnnotationBbox(getImage(path_yellow), (loc[0], loc[1]), frameon=False)
                else:  # 180 degrees
                    if is_holding:
                        ab = AnnotationBbox(getImage(path180_yellow_holding), (loc[0], loc[1]), frameon=False)
                    else:
                        ab = AnnotationBbox(getImage(path180_yellow), (loc[0], loc[1]), frameon=False)
            
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
        os.makedirs("rollouts", exist_ok=True)
        
        # Save image and display
        plt.savefig(f"rollouts/state_{timestep}.png")
        plt.show()
        plt.close()

    def rollout_full_game_joint_optimal(self):
        """
        Execute a full rollout using the optimal policy.
        
        Returns:
            tuple: (total_reward, game_results, sum_feature_vector)
        """
        self.reset()
        done = False
        total_reward = 0
        iters = 0
        game_results = []
        
        # Initialize feature vector sum
        number_of_objects = len(self.object_type_tuple)
        sum_feature_vector = np.zeros(4 * number_of_objects)

        # Render initial state
        self.render(self.current_state, iters)
        
        # Run until done or max iterations
        while not done:
            iters += 1
            
            # Get current state and optimal action
            current_state_tup = self.state_to_tuple(self.current_state)
            state_idx = self.state_to_idx[current_state_tup]
            
            action_distribution = self.policy[state_idx]
            action_idx = np.argmax(action_distribution)
            action = self.idx_to_action[action_idx]

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
            featurized_state = self.featurize_state(self.current_state)
            featurized_state_vector = []
            for i in range(len(featurized_state)):
                for j in range(len(featurized_state[i])):
                    featurized_state_vector.append(featurized_state[i][j])
                    
            sum_feature_vector += np.array(featurized_state_vector)
            self.current_state = next_state
            
            # Render current state
            self.render(self.current_state, iters)
            
            # Update total reward
            total_reward += team_rew
            
            # Avoid infinite loops
            if iters > 40:
                break

        return total_reward, game_results, sum_feature_vector

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


if __name__ == '__main__':
    """
    Main execution block to demonstrate the multi-object Gridworld environment.
    """
    # Define reward weights for each object [orientation, red_proximity, blue_proximity, y_position]
    reward_weights = [
        [-10, -2, -2, -2],  # Red cup reward weights
        [-2, -2, 3, 4]      # Yellow cup reward weights
    ]
    
    # Define centroid positions
    red_centroid, blue_centroid = (1, 1), (-1, -1)  # The grid is 5x5
    
    # Feature importance indices (1=matters to user, 0=doesn't matter) for each object
    true_f_idx = [
        [1, 1, 1, 1],  # All features matter for red cup
        [1, 1, 1, 1]   # All features matter for yellow cup
    ]
    
    # Define object types
    object_type_tuple = [RED_CUP, YELLOW_CUP]
    
    # Create environment with multiple objects
    game = Gridworld(reward_weights, true_f_idx, object_type_tuple, red_centroid, blue_centroid)
    
    # Run value iteration to find optimal policy
    optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance()
    
    print(f"Optimal reward: {optimal_rew}")
    print(f"Feature vector sum: {sum_feature_vector}")
    print(f"Number of actions taken: {len(game_results)}")
