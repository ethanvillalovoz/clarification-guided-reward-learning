# Authors: Ethan Villalovz, Michelle Zhao
# Project: RISS 2024 Summer Project
# Description: Adaptation from `custom_mdp` by Michelle Zhao. Code reduction for simplification.
# Last Updated:

# imports
import copy

import numpy as np
import pickle
import networkx as nx
from itertools import product
import matplotlib.pyplot as plt

# Global Variables
EXIT = 'exit'
ROTATE180 = 'rotate180'

RED = 1
YELLOW = 2

CUP = 1
RED_CUP = (RED, CUP)
YELLOW_CUP = (YELLOW, CUP)

# Class Gridworld
#   Description: contains all the data set up to run simulation of robot moving actions based on human reward
class Gridworld():
    def __init__(self, reward_weights, true_f_indices, object_type_tuple, red_centroid, blue_centroid):
        self.true_f_indices = true_f_indices
        self.reward_weights = reward_weights

        # # define savefilename = merge true f indices and reward weights
        # self.savefilename = "videos/task_obj" + str(object_type_tuple[0]) + str(object_type_tuple[1])  + \
        #                     "_reward_weights_" + "".join([str(i) for i in reward_weights]) + \
        #                      "_true_f_indices_" + "".join([str(i) for i in true_f_indices])

        self.set_env_limits()

        self.object_type_tuple = object_type_tuple
        self.initial_object_locs = {object_type_tuple: (0, 0)}

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
        self.red_centroid = red_centroid
        self.blue_centroid = blue_centroid

    def make_actions_list(self):
        actions_list = []
        actions_list.extend(self.directions)
        actions_list.append(EXIT)
        actions_list.append(ROTATE180)
        return actions_list

    def set_env_limits(self):
        # set environment limits
        self.x_min = -3
        self.x_max = 4
        self.y_min = -3
        self.y_max = 4
        self.all_coordinate_locations = list(product(range(self.x_min,self.x_max),
                                                     range(self.y_min, self.y_max)))

    def reset(self):
        self.current_state = self.create_initial_state()

    def create_initial_state(self):
        # create dictionary of object location to object type and picked up state
        state = {}
        state['grid'] = copy.deepcopy(self.initial_object_locs) # type tuple (color, type) to location
        state['exit'] = False
        state['orientation'] = np.pi  # The orientation options are 0 or pi, which correspond to 0 or 180 degrees for the cup

        return state

    def is_done(self):
        # check if player at exit
        if self.current_state['exit']:
            return True

        return False

    def is_done_given_state(self, current_state):
        # check if player at exit location
        if current_state['exit']:
            return True

        return False

    def is_valid_push(self, current_state, action):

        current_loc = current_state['grid'][self.object_type_tuple]

        new_loc = tuple(np.array(current_loc) + np.array(action))
        if new_loc[0] < self.x_min or new_loc[0] >= self.x_max or new_loc[1] < self.y_min or new_loc[1] >= self.y_max:
            return False

        return True

    def step_given_state(self, input_state, action):
        step_cost = -0.1
        current_state = copy.deepcopy(input_state)

        if current_state['exit'] == True:
            # current_state['exit'] = True

            step_reward = 0

            return current_state, step_reward, True

        if action == EXIT:
            current_state['exit'] = True
            featurized_state = self.featurize_state(current_state)
            step_reward = np.dot(self.reward_weights, featurized_state)
            step_reward += step_cost
            return current_state, step_reward, True

        if action in self.directions:
            if self.is_valid_push(current_state, action) is False:
                step_reward = step_cost
                return current_state, step_reward, False

        if action == ROTATE180:
            # add 180 to orientation and make between 0 and 360`
            # convert to radians
            current_state['orientation'] = (current_state['orientation'] + np.pi) % (2 * np.pi)

            # featurized_state = self.featurize_state(current_state)
            step_reward = step_cost
            return current_state, step_reward, False

        action_type_moved = self.object_type_tuple
        current_loc = current_state['grid'][action_type_moved]

        new_loc = tuple(np.array(current_loc) + np.array(action))
        current_state['grid'][action_type_moved] = new_loc

        # featurized_state = self.featurize_state(current_state)
        step_reward = step_cost

        done = self.is_done_given_state(current_state)

        return current_state, step_reward, done

    def render(self, current_state, timestep):
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox

        def getImage(path, zoom=1):
            zoom = 0.05
            return OffsetImage(plt.imread(path), zoom=zoom)

        plot_init_state = copy.deepcopy(current_state)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        (ax1) = ax

        if current_state['exit'] is True:
            ax1.axvline(x=0, color='red', linewidth=10, alpha=0.1)
            ax1.axhline(y=0, color='red', linewidth=10, alpha=0.1)
        else:
            ax1.axvline(x=0, color='black', linewidth=7, alpha=0.1)
            ax1.axhline(y=0, color='black', linewidth=7, alpha=0.1)

        type_to_color = {self.object_type_tuple: 'red'}
        type_to_loc_init = {}

        ax1.scatter(self.red_centroid[0], self.red_centroid[1], color='red', s=800, alpha=0.1)
        ax1.scatter(self.blue_centroid[0], self.blue_centroid[1], color='blue', s=800, alpha=0.1)

        path = 'data/redcup.jpeg'
        path180 = 'data/redcup_180.jpeg'
        orientation = plot_init_state['orientation']
        for type_o in plot_init_state['grid']:
            loc = plot_init_state['grid'][type_o]
            color = type_to_color[type_o]
            type_to_loc_init[type_o] = loc

            ax1.scatter(loc[0], loc[1], color=color, s=500, alpha=0.99)
            if orientation == 0:
                ab = AnnotationBbox(getImage(path), (loc[0], loc[1]), frameon=False)
                ax.add_artist(ab)
            else:
                ab = AnnotationBbox(getImage(path180), (loc[0], loc[1]), frameon=False)
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
        plt.savefig(f"rollouts/state_{timestep}.png")

        plt.show()
        plt.close()

    def featurize_state(self, current_state):

        current_loc = current_state['grid'][self.object_type_tuple]

        dist_to_red_centroid = np.linalg.norm(np.array(current_loc) - np.array(self.red_centroid))
        dist_to_blue_centroid = np.linalg.norm(np.array(current_loc) - np.array(self.blue_centroid))

        # compute dist_to_red_centroid as manhattan distance
        orientation = current_state['orientation']

        pos_y = current_loc[1]
        state_feature = np.array([orientation, dist_to_red_centroid, dist_to_blue_centroid, pos_y])

        # elementwise multiply by true_f_idx
        state_feature = np.multiply(state_feature, self.true_f_indices)

        return state_feature

    def state_to_tuple(self, current_state):
        # convert current_state to tuple
        current_state_tup = []
        for obj_type in current_state['grid']:
            loc = current_state['grid'][obj_type]
            current_state_tup.append((obj_type, loc))
        current_state_tup = list(sorted(current_state_tup, key=lambda x: x[1]))

        current_state_tup.append(current_state['exit'])
        current_state_tup.append(current_state['orientation'])

        return tuple(current_state_tup)

    def tuple_to_state(self, current_state_tup):
        # convert current_state to tuple
        current_state_tup = list(current_state_tup)
        current_state = {'grid': {}, 'orientation': 0, 'exit': False}
        for i in range(len(current_state_tup)-2):
            (obj_type, loc) =  current_state_tup[i]
            current_state['grid'][obj_type] = loc

        current_state['exit'] = current_state_tup[-2]
        current_state['orientation'] = current_state_tup[-1]

        return current_state

    def enumerate_states(self):
        self.reset()

        actions = self.possible_single_actions

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
        idx_to_state = {i: state for i, state in enumerate(states)}
        state_to_idx = {state: i for i, state in idx_to_state.items()}

        action_to_idx = {action: i for i, action in enumerate(actions)}
        idx_to_action = {i: action for i, action in enumerate(actions)}

        # construct transition matrix and reward matrix of shape [# states, # states, # actions] based on graph
        transition_mat = np.zeros([len(states), len(states), len(actions)])
        reward_mat = np.zeros([len(states), len(actions)])

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

    def rollout_full_game_joint_optimal(self):
        self.reset()
        done = False
        total_reward = 0

        iters = 0
        game_results = []
        sum_feature_vector = np.zeros(4)

        self.render(self.current_state, iters)
        while not done:
            iters += 1
            current_state_tup = self.state_to_tuple(self.current_state)
            state_idx = self.state_to_idx[current_state_tup]

            action_distribution = self.policy[state_idx]
            action = np.argmax(action_distribution)
            action = self.idx_to_action[action]

            # print q values
            action_to_q  = {}
            action_to_reward = {}
            for i in range(len(action_distribution)):
                action_to_q[self.idx_to_action[i]] = action_distribution[i]
                action_to_reward[self.idx_to_action[i]] = self.rewards[state_idx, i]

            game_results.append((self.current_state, action))
            next_state, team_rew, done = self.step_given_state(self.current_state, action)

            featurized_state = self.featurize_state(self.current_state)
            sum_feature_vector += np.array(featurized_state)
            self.current_state = next_state

            # render the current state
            self.render(self.current_state, iters)

            total_reward += team_rew

            if iters > 40:
                break

        # self.save_rollouts_to_video()

        return total_reward, game_results, sum_feature_vector



    def save_rollouts_to_video(self):
        # for all images in the rollouts direction, convert to a video and delete the images
        import os

        os.system(f"ffmpeg -r 1 -i rollouts/state_%01d.png -vcodec mpeg4 -y {self.savefilename}.mp4")
        self.clear_rollouts()

    def clear_rollouts(self):
        import os
        os.system("rm -rf rollouts")
        os.system("mkdir rollouts")

    def compute_optimal_performance(self):
        self.enumerate_states()

        self.vectorized_vi()

        optimal_rew, game_results, sum_feature_vector = self.rollout_full_game_joint_optimal()
        return optimal_rew, game_results, sum_feature_vector

if __name__ == '__main__':
    # reward_weights = [0,10,-10,0,10,0,-10,-10,-10] # first should be negative, second can be pos or neg, last should be negative
    # reward_weights = [np.random.uniform(-10, 10) for _ in range(9)]
    reward_weights = [-10, -2, -2, -2]  # orientation, red prox, blue prox, pos y

    # reward_weights = [(float(i) / np.linalg.norm(reward_weights)) for i in reward_weights]
    # print("reward_weights", reward_weights)
    red_centroid, blue_centroid = (2, 2), (-3, -3)  # The grid is 7x7
    true_f_idx = [1, 1, 1, 1]

    # here, is the set of features that matter to the user.
    # If the user cares about the orientation, red proximity, and blue proximity, then the true_f_idx would be [1, 1, 1, 0]
    # If the user cares about the orientation, red proximity, and y position, then the true_f_idx would be [1, 1, 0, 1]
    # etc.

    object_type_tuple = RED_CUP  # just a global variable for the cup

    # now we set up the environment
    game = Gridworld(reward_weights, true_f_idx, object_type_tuple, red_centroid, blue_centroid)

    # Run value iteration to get the optimal policy
    optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance()
    print("optimal_rew", optimal_rew)
    print("game_results", game_results)