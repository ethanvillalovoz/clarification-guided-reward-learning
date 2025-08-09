"""
Multi-Object Custom MDP for Preference Learning

This module extends the original custom_mdp implementation to support multiple objects
with different properties (colors, materials) in a gridworld environment. The environment
models user preferences for object placement through reward functions.

Features:
- Support for multiple objects with different properties (color, type)
- Grid-based movement in four cardinal directions
- Object orientation and holding capability
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
ROTATE180 = 'rotate180'
PICK_UP = 'pick_up'
PLACE = 'place_down'

# Object properties
RED = 1
YELLOW = 2

COLORS_LIST = {1: "Red", 2: "Yellow"}

CUP = 1
RED_CUP = (RED, CUP)
YELLOW_CUP = (YELLOW, CUP)


class Gridworld:
    """
    Gridworld environment for multi-object placement preference learning.
    
    This class implements a grid-based environment where an agent can move multiple
    objects to different positions based on reward functions capturing user preferences.
    """
    
    def __init__(self, reward_weights, true_f_indices, object_type_tuple, red_centroid, blue_centroid):
        """
        Initialize the Gridworld environment.
        
        Args:
            reward_weights (list): Weights for different reward components per object
            true_f_indices (list): Binary indicators for which features matter to the user per object
            object_type_tuple (tuple): Types of objects to place
            red_centroid (tuple): Coordinates (x, y) of the red centroid
            blue_centroid (tuple): Coordinates (x, y) of the blue centroid
        """
        self.true_f_indices = true_f_indices
        self.reward_weights = reward_weights
        self.object_reward_index = 0

        # Set environment boundaries and parameters
        self.set_env_limits()
        
        # Set up object properties
        self.object_type_tuple = object_type_tuple
        self.initial_object_locs = {}

        # Place objects in initial positions with appropriate spacing
        x_point = 0
        iteration = 1
        for obj in object_type_tuple:
            self.initial_object_locs[obj] = (x_point, 0)

            # Changes the x-coordinate based on the number of objects we are using
            iteration += 1
            if iteration % 2 == 0:
                x_point += 1
            else:
                x_point *= -1

        # Define possible movement directions
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Up, Down, Right, Left
        
        # Set up possible actions and initial state
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
        self.epsilson = 0.001  # Convergence threshold
        self.gamma = 0.99     # Discount factor
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
        Create the initial state of the environment with all objects.
        
        Returns:
            list: List of state dictionaries, one per object
        """
        state_list = []
        for obj in self.initial_object_locs:
            state = {}
            state['grid'] = copy.deepcopy({obj: self.initial_object_locs[obj]})
            state['exit'] = False
            # The orientation options are 0 or pi (0 or 180 degrees for the cup)
            state['orientation'] = np.pi
            state_list.append(state)

        return state_list

    def is_done(self):
        """
        Check if the current episode is done.
        
        Returns:
            bool: True if exit action was taken
        """
        if self.current_state['exit']:
            return True
        return False

    def is_done_given_state(self, current_state):
        """
        Check if the episode would be done in the given state.
        
        Args:
            current_state (dict): State to check
            
        Returns:
            bool: True if exit action was taken
        """
        if current_state['exit']:
            return True
        return False

    def is_valid_push(self, current_state, action):
        """
        Check if a push action is valid from the current state.
        
        Args:
            current_state (dict): Current state
            action (tuple): Direction to push
            
        Returns:
            bool: True if action keeps object within grid boundaries
        """
        # Get current location of the object
        current_loc = current_state['grid'][list(current_state['grid'].keys())[0]]
        
        # Calculate new location after push
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
        if current_state['exit'] == True:
            step_reward = 0
            return current_state, step_reward, True

        # Handle exit action
        if action == EXIT:
            current_state['exit'] = True
            featurized_state = self.featurize_state(current_state)
            step_reward = np.dot(featurized_state, self.reward_weights[self.object_reward_index])
            step_reward += step_cost
            return current_state, step_reward, True

        # Handle movement actions
        if action in self.directions:
            if self.is_valid_push(current_state, action) is False:
                step_reward = step_cost
                return current_state, step_reward, False
                
            # Move the object
            action_type_moved = list(current_state['grid'].keys())[0]
            current_loc = current_state['grid'][action_type_moved]
            new_loc = tuple(np.array(current_loc) + np.array(action))
            current_state['grid'][action_type_moved] = new_loc

        # Handle rotation action
        if action == ROTATE180:
            # Add 180 to orientation and normalize to [0, 2π]
            current_state['orientation'] = (current_state['orientation'] + np.pi) % (2 * np.pi)
            step_reward = step_cost
            return current_state, step_reward, False

        # Calculate reward and check if done
        featurized_state = self.featurize_state(current_state)
        step_reward = step_cost
        done = self.is_done_given_state(current_state)
        
        return current_state, step_reward, done

    def render(self, current_state, timestep):
        """
        Render the current state of the environment.
        
        Args:
            current_state (list or dict): State(s) to render
            timestep (int): Current timestep for display
        """
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        
        def getImage(path, zoom=1):
            """Load and resize an image for display."""
            zoom = 0.03
            # Make sure the path includes 'images' directory
            if not path.startswith('data/images/'):
                path = 'data/images/' + os.path.basename(path)
            return OffsetImage(plt.imread(path), zoom=zoom)

        plot_init_state = copy.deepcopy(current_state)

        # Create figure and axis
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax1 = ax
        
        # Determine if all states have exited or current_state is a single state
        if isinstance(current_state, list) and all(state['exit'] for state in current_state):
            ax1.axvline(x=0, color='red', linewidth=10, alpha=0.1)
            ax1.axhline(y=0, color='red', linewidth=10, alpha=0.1)
        elif isinstance(current_state, dict) and current_state['exit']:
            ax1.axvline(x=0, color='red', linewidth=10, alpha=0.1)
            ax1.axhline(y=0, color='red', linewidth=10, alpha=0.1)
        else:
            ax1.axvline(x=0, color='black', linewidth=7, alpha=0.1)
            ax1.axhline(y=0, color='black', linewidth=7, alpha=0.1)

        # Set up color mapping
        type_to_color = {}
        for i in range(0, len(self.object_type_tuple)):
            type_to_color[self.object_type_tuple[i]] = COLORS_LIST[self.object_type_tuple[i][0]]
        type_to_loc_init = {}

        # Draw centroid regions
        ax1.scatter(self.red_centroid[0], self.red_centroid[1], color='red', s=800, alpha=0.1)
        ax1.scatter(self.blue_centroid[0], self.blue_centroid[1], color='blue', s=800, alpha=0.1)

        # Define image paths
        path_red = 'data/images/redcup.jpeg'
        path180_red = 'data/images/redcup_180.jpeg'
        path_yellow = 'data/images/yellowcup.jpeg'
        path180_yellow = 'data/images/yellowcup_180.jpeg'

        # Handle both single state and list of states
        if isinstance(current_state, dict):
            states_to_render = [current_state]
        else:
            states_to_render = current_state
            
        # Draw each object
        for i in range(0, len(states_to_render)):
            orientation = states_to_render[i]['orientation']
            for type_o in states_to_render[i]['grid']:
                loc = states_to_render[i]['grid'][type_o]
                color = type_to_color[type_o]
                type_to_loc_init[type_o] = loc

                # Draw object marker
                ax1.scatter(loc[0], loc[1], color=color.lower(), s=500, alpha=0.99)
                
                # Draw appropriate cup image based on color and orientation
                if color == 'Red':
                    if orientation == 0:
                        ab = AnnotationBbox(getImage(path_red), (loc[0], loc[1]), frameon=False)
                        ax.add_artist(ab)
                    else:
                        ab = AnnotationBbox(getImage(path180_red), (loc[0], loc[1]), frameon=False)
                        ax.add_artist(ab)
                else:  # Yellow
                    if orientation == 0:
                        ab = AnnotationBbox(getImage(path_yellow), (loc[0], loc[1]), frameon=False)
                        ax.add_artist(ab)
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
        if isinstance(current_state, list) and all(state['exit'] for state in current_state):
            ax1.set_title(f"State at Time {timestep}: FINAL STATE")
        elif isinstance(current_state, dict) and current_state['exit']:
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
        # Get current location of the object
        current_loc = current_state['grid'][list(current_state['grid'].keys())[0]]

        # Calculate distances to centroids (using Euclidean distance)
        dist_to_red_centroid = np.linalg.norm(np.array(current_loc) - np.array(self.red_centroid))
        dist_to_blue_centroid = np.linalg.norm(np.array(current_loc) - np.array(self.blue_centroid))

        # Get orientation and y-position
        orientation = current_state['orientation']
        pos_y = current_loc[1]
        
        # Create feature vector
        state_feature = np.array([orientation, dist_to_red_centroid, dist_to_blue_centroid, pos_y])
        
        # Apply feature importance mask
        state_feature = np.multiply(state_feature, self.true_f_indices[self.object_reward_index])
        
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
        stack = []
        
        # Initialize stack with each object's state
        for i in self.current_state:
            stack.append(copy.deepcopy([i]))

        number_of_objects = len(self.current_state)

        # Explore state space for each object
        for i in range(0, len(self.current_state)):
            while stack[i]:
                state = stack[i].pop()
                state_tup = self.state_to_tuple(state)

                # Add state if not visited
                if state_tup not in visited_states:
                    visited_states.add(state_tup)
                
                # Explore all actions from this state
                for idx, action in enumerate(actions):
                    if self.is_done_given_state(state):
                        team_reward = 0
                        next_state = state
                        done = True
                    else:
                        next_state, team_reward, done = self.step_given_state(state, action)
                    
                    # Add next state to stack if not visited
                    new_state_tup = self.state_to_tuple(next_state)
                    if new_state_tup not in visited_states:
                        stack[i].append(copy.deepcopy(next_state))
                    
                    # Add edge to graph
                    G.add_edge(state_tup, new_state_tup, weight=team_reward, action=action)
            
            # Move to next object's reward function
            self.object_reward_index += 1

        # Create mappings between states/actions and indices
        states = list(G.nodes)
        idx_to_state = {i: state for i, state in enumerate(states)}
        state_to_idx = {state: i for i, state in idx_to_state.items()}
        
        action_to_idx = {action: i for i, action in enumerate(actions)}
        idx_to_action = {i: action for i, action in enumerate(actions)}
        
        # Construct transition and reward matrices
        transition_mat = np.zeros([len(states), len(states), len(actions)])
        reward_mat = np.zeros([len(states), len(actions)])
        
        # Reset object reward index for computing rewards
        self.object_reward_index = 0
        splits = len(states) / number_of_objects
        
        # Fill transition and reward matrices
        for i in range(len(states)):
            state = self.tuple_to_state(idx_to_state[i])
            
            # Update reward index when switching to next object's states
            if i % splits == 0 and i != 0:
                self.object_reward_index += 1
            
            # Compute transitions and rewards for each action
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
                
                # Handle both scalar and array rewards
                if isinstance(team_reward, (np.ndarray, list)):
                    reward_mat[i, action_idx_i] = team_reward[0]
                else:
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
                vf[s] = np.max(np.sum((self.rewards[s] + self.gamma * vf) * self.transitions[s, :, :], 0))
                
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
        Execute a full rollout using the optimal policy for all objects.
        
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
        
        # Execute policy for each object
        for i in range(0, len(self.current_state)):
            done = False  # Reset done flag for each object
            while not done:
                iters += 1
                
                # Get current state and optimal action
                current_state_tup = self.state_to_tuple(self.current_state[i])
                state_idx = self.state_to_idx[current_state_tup]
                
                action_distribution = self.policy[state_idx]
                action = np.argmax(action_distribution)
                action = self.idx_to_action[action]
                
                # Store Q-values and rewards for debugging
                action_to_q = {}
                action_to_reward = {}
                for j in range(len(action_distribution)):
                    action_to_q[self.idx_to_action[j]] = action_distribution[j]
                    action_to_reward[self.idx_to_action[j]] = self.rewards[state_idx, j]
                
                # Store state and action
                game_results.append((self.current_state[i], action))
                
                # Take step
                next_state, team_rew, done = self.step_given_state(self.current_state[i], action)
                
                # Update feature sum and state
                featurized_state = self.featurize_state(self.current_state[i])
                sum_feature_vector += np.array(featurized_state)
                self.current_state[i] = next_state
                
                # Render current state
                self.render(self.current_state, iters)
                
                # Update total reward
                total_reward += team_rew
                
                # Avoid infinite loops
                if iters > 40:
                    break
        
        return total_reward, game_results, sum_feature_vector

    def save_rollouts_to_video(self):
        """Convert saved rollout images to video using ffmpeg."""
        import os
        os.system(f"ffmpeg -r 1 -i rollouts/state_%01d.png -vcodec mpeg4 -y rollout_video.mp4")
        self.clear_rollouts()

    def clear_rollouts(self):
        """Clear rollout images and recreate rollouts directory."""
        import os
        os.system("rm -rf rollouts")
        os.makedirs("rollouts", exist_ok=True)

    def compute_optimal_performance(self):
        """
        Compute and execute optimal policy for all objects.
        
        Returns:
            tuple: (optimal_reward, game_results, sum_feature_vector)
        """
        # Enumerate states and run value iteration
        self.enumerate_states()
        self.vectorized_vi()
        
        # Execute optimal policy
        optimal_rew, game_results, sum_feature_vector = self.rollout_full_game_joint_optimal()
        return optimal_rew, game_results, sum_feature_vector


def compute_new_policy(reward_weights, true_f_idx, object_type_tuple, red_centroid, blue_centroid):
    """
    Compute optimal policy for given reward weights and feature importance.
    
    Args:
        reward_weights (list): Weights for reward components
        true_f_idx (list): Feature importance indicators
        object_type_tuple (tuple): Object types
        red_centroid (tuple): Position of red centroid
        blue_centroid (tuple): Position of blue centroid
        
    Returns:
        tuple: (policy, state_to_idx, idx_to_state, idx_to_action, action_to_idx, transitions, rewards)
    """
    game = Gridworld(reward_weights, true_f_idx, object_type_tuple, red_centroid, blue_centroid)
    game.enumerate_states()
    game.vectorized_vi()
    return (game.policy, game.state_to_idx, game.idx_to_state, 
            game.idx_to_action, game.action_to_idx, game.transitions, game.rewards)


if __name__ == '__main__':
    """
    Main execution block to demonstrate the multi-object Gridworld environment.
    """
    # Define reward weights for each object [orientation, red_proximity, blue_proximity, y_position]
    reward_weights = [
        [-10, -2, -2, -2],  # Red cup reward weights
        [-2, 1, -1, 1]      # Yellow cup reward weights
    ]
    
    # Define centroid positions
    red_centroid, blue_centroid = (2, 2), (-3, -3)  # The grid is 7x7
    
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
