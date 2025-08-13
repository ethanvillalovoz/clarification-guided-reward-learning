"""
Reduced Custom MDP Implementation for Object Placement Learning

A simplified version of the original custom_mdp implementation that models
object placement preferences in a gridworld environment. This version focuses on
core functionality while maintaining the MDP structure for policy computation.

Features:
- Grid-based environment with a single object
- Object orientation and movement in four directions
- Centroid-based feature representation
- Value iteration for policy computation
- State visualization

Authors: Ethan Villalovoz, Michelle Zhao
Project: RISS 2024 Summer Project
"""

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

# Object properties
RED = 1
YELLOW = 2
CUP = 1
RED_CUP = (RED, CUP)
YELLOW_CUP = (YELLOW, CUP)


class Gridworld:
    """
    Gridworld environment for object placement preference learning.
    
    This class implements a simplified grid-based environment where an agent can
    move a single object to different positions based on reward functions
    capturing user preferences.
    """
    
    def __init__(self, reward_weights, true_f_indices, object_type_tuple, red_centroid, blue_centroid):
        """
        Initialize the Gridworld environment.
        
        Args:
            reward_weights (list): Weights for different reward components
            true_f_indices (list): Binary indicators for which features matter to the user
            object_type_tuple (tuple): Type of object to place, e.g., RED_CUP
            red_centroid (tuple): Coordinates (x, y) of the red centroid
            blue_centroid (tuple): Coordinates (x, y) of the blue centroid
        """
        self.true_f_indices = true_f_indices
        self.reward_weights = reward_weights
        
        # Environment setup
        self.set_env_limits()
        self.object_type_tuple = object_type_tuple
        self.initial_object_locs = {object_type_tuple: (0, 0)}
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Up, Down, Right, Left
        
        # Action space and initial state
        self.possible_single_actions = self.make_actions_list()
        self.current_state = self.create_initial_state()
        
        # Value iteration parameters
        self.transitions = None
        self.rewards = None
        self.state_to_idx = None
        self.idx_to_action = None
        self.idx_to_state = None
        self.action_to_idx = None
        self.vf = None
        self.pi = None
        self.policy = None
        self.epsilson = 0.001  # Convergence threshold
        self.gamma = 0.99  # Discount factor
        self.maxiter = 10000  # Maximum iterations
        
        # Feature parameters
        self.num_features = 4
        self.red_centroid = red_centroid
        self.blue_centroid = blue_centroid

    def make_actions_list(self):
        """
        Create list of possible actions in the environment.
        
        Returns:
            list: All possible actions (directions, exit, rotate)
        """
        actions_list = []
        actions_list.extend(self.directions)
        actions_list.append(EXIT)
        actions_list.append(ROTATE180)
        return actions_list

    def set_env_limits(self):
        """Set the boundaries of the environment grid."""
        self.x_min = -3
        self.x_max = 4
        self.y_min = -3
        self.y_max = 4
        self.all_coordinate_locations = list(product(
            range(self.x_min, self.x_max),
            range(self.y_min, self.y_max)
        ))

    def reset(self):
        """Reset the environment to initial state."""
        self.current_state = self.create_initial_state()

    def create_initial_state(self):
        """
        Create the initial state of the environment.
        
        Returns:
            dict: Initial state with object locations, exit flag, and orientation
        """
        state = {
            'grid': copy.deepcopy(self.initial_object_locs),  # Object locations
            'exit': False,  # Whether exit action was taken
            'orientation': np.pi  # Object orientation (0 or π radians)
        }
        return state

    def is_done(self):
        """
        Check if the current episode is done.
        
        Returns:
            bool: True if episode is done (exit action was taken)
        """
        return self.current_state['exit']

    def is_done_given_state(self, current_state):
        """
        Check if the episode would be done in the given state.
        
        Args:
            current_state (dict): State to check
            
        Returns:
            bool: True if episode would be done (exit action was taken)
        """
        return current_state['exit']

    def is_valid_push(self, current_state, action):
        """
        Check if a push action is valid from the current state.
        
        Args:
            current_state (dict): Current state
            action (tuple): Direction to push
            
        Returns:
            bool: True if action keeps object within grid boundaries
        """
        current_loc = current_state['grid'][self.object_type_tuple]
        new_loc = tuple(np.array(current_loc) + np.array(action))
        
        # Check if new location is within grid boundaries
        if (new_loc[0] < self.x_min or new_loc[0] >= self.x_max or 
            new_loc[1] < self.y_min or new_loc[1] >= self.y_max):
            return False
            
        return True

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

        # If already in terminal state
        if current_state['exit']:
            return current_state, 0, True

        # Handle exit action
        if action == EXIT:
            current_state['exit'] = True
            featurized_state = self.featurize_state(current_state)
            step_reward = np.dot(self.reward_weights, featurized_state) + step_cost
            return current_state, step_reward, True

        # Handle movement actions
        if action in self.directions:
            if not self.is_valid_push(current_state, action):
                return current_state, step_cost, False
                
            # Move object
            action_type_moved = self.object_type_tuple
            current_loc = current_state['grid'][action_type_moved]
            new_loc = tuple(np.array(current_loc) + np.array(action))
            current_state['grid'][action_type_moved] = new_loc

        # Handle rotation action
        elif action == ROTATE180:
            current_state['orientation'] = (current_state['orientation'] + np.pi) % (2 * np.pi)
            return current_state, step_cost, False

        # Default reward for non-exit actions
        step_reward = step_cost
        done = self.is_done_given_state(current_state)
        
        return current_state, step_reward, done

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
            # Make sure the path includes 'images' directory
            if not path.startswith('data/images/'):
                path = 'data/images/' + os.path.basename(path)
            return OffsetImage(plt.imread(path), zoom=zoom)

        plot_init_state = copy.deepcopy(current_state)

        # Create figure and axis
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax1 = ax
        
        # Draw grid centerlines
        if current_state['exit']:
            ax1.axvline(x=0, color='red', linewidth=10, alpha=0.1)
            ax1.axhline(y=0, color='red', linewidth=10, alpha=0.1)
        else:
            ax1.axvline(x=0, color='black', linewidth=7, alpha=0.1)
            ax1.axhline(y=0, color='black', linewidth=7, alpha=0.1)

        # Set up display properties
        type_to_color = {self.object_type_tuple: 'red'}
        type_to_loc_init = {}
        
        # Draw centroid regions
        ax1.scatter(self.red_centroid[0], self.red_centroid[1], color='red', s=800, alpha=0.1)
        ax1.scatter(self.blue_centroid[0], self.blue_centroid[1], color='blue', s=800, alpha=0.1)

        # Set image paths based on orientation
        path = 'data/images/redcup.jpeg'
        path180 = 'data/images/redcup_180.jpeg'
        orientation = plot_init_state['orientation']
        
        # Draw objects
        for type_o in plot_init_state['grid']:
            loc = plot_init_state['grid'][type_o]
            color = type_to_color[type_o]
            type_to_loc_init[type_o] = loc

            # Draw dot for object location
            ax1.scatter(loc[0], loc[1], color=color, s=500, alpha=0.99)
            
            # Draw cup image based on orientation
            if orientation == 0:
                ab = AnnotationBbox(getImage(path), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)
            else:
                ab = AnnotationBbox(getImage(path180), (loc[0], loc[1]), frameon=False)
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

    def featurize_state(self, current_state):
        """
        Convert state to feature vector for reward calculation.
        
        Args:
            current_state (dict): State to featurize
            
        Returns:
            numpy.ndarray: Feature vector [orientation, dist_red, dist_blue, y_pos]
        """
        current_loc = current_state['grid'][self.object_type_tuple]

        # Calculate distances to centroids (using Euclidean distance)
        dist_to_red_centroid = np.linalg.norm(np.array(current_loc) - np.array(self.red_centroid))
        dist_to_blue_centroid = np.linalg.norm(np.array(current_loc) - np.array(self.blue_centroid))

        # Get orientation and y-position
        orientation = current_state['orientation']
        pos_y = current_loc[1]
        
        # Create feature vector
        state_feature = np.array([orientation, dist_to_red_centroid, dist_to_blue_centroid, pos_y])
        
        # Apply feature importance mask
        state_feature = np.multiply(state_feature, self.true_f_indices)
        
        return state_feature

    def state_to_tuple(self, current_state):
        """
        Convert state dict to hashable tuple for graph representation.
        
        Args:
            current_state (dict): State to convert
            
        Returns:
            tuple: Hashable representation of state
        """
        current_state_tup = []
        
        # Add object locations
        for obj_type in current_state['grid']:
            loc = current_state['grid'][obj_type]
            current_state_tup.append((obj_type, loc))
        current_state_tup = list(sorted(current_state_tup, key=lambda x: x[1]))
        
        # Add exit status and orientation
        current_state_tup.append(current_state['exit'])
        current_state_tup.append(current_state['orientation'])
        
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
        current_state = {'grid': {}, 'orientation': 0, 'exit': False}
        
        # Extract object locations
        for i in range(len(current_state_tup) - 2):
            (obj_type, loc) = current_state_tup[i]
            current_state['grid'][obj_type] = loc
        
        # Extract exit status and orientation
        current_state['exit'] = current_state_tup[-2]
        current_state['orientation'] = current_state_tup[-1]
        
        return current_state

    def enumerate_states(self):
        """
        Enumerate all possible states and transitions for value iteration.
        
        Returns:
            tuple: (transition_matrix, reward_matrix, state_to_idx, idx_to_action, idx_to_state, action_to_idx)
        """
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
            
            # Explore all actions from this state
            for idx, action in enumerate(actions):
                if self.is_done_given_state(state):
                    team_reward = 0
                    next_state = state
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
        idx_to_state = {i: state for i, state in enumerate(states)}
        state_to_idx = {state: i for i, state in idx_to_state.items()}
        
        action_to_idx = {action: i for i, action in enumerate(actions)}
        idx_to_action = {i: action for i, action in enumerate(actions)}
        
        # Construct transition and reward matrices
        transition_mat = np.zeros([len(states), len(states), len(actions)])
        reward_mat = np.zeros([len(states), len(actions)])
        
        for i in range(len(states)):
            state = self.tuple_to_state(idx_to_state[i])
            for action_idx_i in range(len(actions)):
                action = idx_to_action[action_idx_i]
                
                # Get next state and reward
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
            delta = 0  # Convergence measure
            
            # Perform Bellman update for each state
            for s in range(n_states):
                # Store old value function
                old_v = vf[s].copy()
                
                # Compute Q-values and update value function
                Q[s] = np.sum((self.rewards[s] + self.gamma * vf) * self.transitions[s, :, :], 0)
                vf[s] = np.max(Q[s])
                
                # Update delta for convergence check
                delta = np.max((delta, np.abs(old_v - vf[s])[0]))
                
            # Check for convergence
            if delta < self.epsilson:
                break
                
        # Compute optimal policy
        for s in range(n_states):
            pi[s] = np.argmax(np.sum(vf * self.transitions[s, :, :], 0))
            policy[s] = Q[s, :]
        
        # Store results
        self.vf = vf
        self.pi = pi
        self.policy = policy
        
        return vf, pi

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
        sum_feature_vector = np.zeros(4)
        
        # Display initial state
        self.render(self.current_state, iters)
        
        # Run until done or max iterations
        while not done and iters < 40:
            iters += 1
            
            # Get current state and optimal action
            current_state_tup = self.state_to_tuple(self.current_state)
            state_idx = self.state_to_idx[current_state_tup]
            
            action_distribution = self.policy[state_idx]
            action = np.argmax(action_distribution)
            action = self.idx_to_action[action]
            
            # Store Q-values and rewards for debugging (if needed)
            action_to_q = {}
            action_to_reward = {}
            for i in range(len(action_distribution)):
                action_to_q[self.idx_to_action[i]] = action_distribution[i]
                action_to_reward[self.idx_to_action[i]] = self.rewards[state_idx, i]
            
            # Store current state and action
            game_results.append((self.current_state, action))
            
            # Take step
            next_state, team_rew, done = self.step_given_state(self.current_state, action)
            
            # Update feature sum and state
            featurized_state = self.featurize_state(self.current_state)
            sum_feature_vector += np.array(featurized_state)
            self.current_state = next_state
            
            # Render current state
            self.render(self.current_state, iters)
            
            # Update total reward
            total_reward += team_rew
        
        return total_reward, game_results, sum_feature_vector

    def save_rollouts_to_video(self):
        """Convert saved rollout images to video."""
        # Ensure ffmpeg is installed on the system
        os.system(f"ffmpeg -r 1 -i rollouts/state_%01d.png -vcodec mpeg4 -y rollout_video.mp4")
        self.clear_rollouts()

    def clear_rollouts(self):
        """Clear rollout images."""
        os.system("rm -rf rollouts")
        os.makedirs("rollouts", exist_ok=True)

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
    Main execution block to demonstrate the Gridworld environment with a red cup object.
    
    This shows how to run the environment with different reward weights and feature indices.
    """
    # Define reward weights and feature importance
    reward_weights = [-10, -2, -2, -2]  # [orientation, red_proximity, blue_proximity, y_position]
    red_centroid, blue_centroid = (2, 2), (-3, -3)  # Centroid positions in 7x7 grid
    
    # Feature importance indices (1=matters to user, 0=doesn't matter)
    true_f_idx = [1, 1, 1, 1]  # All features matter to the user
    
    # Examples of different feature combinations:
    # - [1, 1, 1, 0]: User cares about orientation, red proximity, and blue proximity
    # - [1, 1, 0, 1]: User cares about orientation, red proximity, and y position
    # - [0, 1, 1, 1]: User only cares about distances and y position, not orientation
    
    # Create environment with red cup object
    object_type_tuple = RED_CUP
    game = Gridworld(reward_weights, true_f_idx, object_type_tuple, red_centroid, blue_centroid)
    
    # Run value iteration to find optimal policy
    optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance()
    print(f"Optimal reward: {optimal_rew:.2f}")
    print(f"Feature vector sum: {sum_feature_vector}")
    print(f"Number of actions taken: {len(game_results)}")