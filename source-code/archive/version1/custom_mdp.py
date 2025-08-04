import copy
import math
import pdb

import numpy as np
import pickle
import sys
import networkx as nx
from itertools import product
import itertools
import matplotlib.pyplot as plt

TYPE_1 = 1 # plastic
TYPE_2 = 2 # glass
TYPE_3 = 3 # ceramic
TYPE_4 = 4 # metal
TYPE_TO_NAME = {TYPE_1: 'plastic', TYPE_2: 'glass', TYPE_3: 'ceramic', TYPE_4: 'metal'}
EXIT = 'exit'

PUSH_1 = 'push_1'
PUSH_2 = 'push_2'
PUSH_3 = 'push_3'
PUSH_4 = 'push_4'
SWITCH = 'switch'

RED = 1
ROTATE180 = 'rotate180'
CUP = 1
RED_CUP = (RED, CUP)

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
        self.initial_object_locs = {object_type_tuple:(0, 0)}



        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]


        # get possible joint actions and actions
        self.possible_single_actions = self.make_actions_list()
        # print("possible single actions", self.possible_single_actions)


        self.current_state = self.create_initial_state()
        # self.reset()


        # set value iteration components
        self.transitions, self.rewards, self.state_to_idx, self.idx_to_action, \
        self.idx_to_state, self.action_to_idx = None, None, None, None, None, None
        self.vf = None
        self.pi = None
        self.policy = None
        self.epsilson = 0.001
        self.gamma = 0.99
        self.maxiter = 10000

        # self.step_cost = -0.01
        # self.push_switch_cost = -0.05

        # self.step_cost = reward_weights[-2]
        # self.push_switch_cost = reward_weights[-1]

        self.num_features = 4
        self.red_centroid = red_centroid
        self.blue_centroid = blue_centroid



    def make_actions_list(self):
        actions_list = []
        # for type in [TYPE_1, TYPE_2, TYPE_3, TYPE_4]:
        #     for dir in self.directions:
        #         actions_list.append((type, dir))
        actions_list.extend(self.directions)
        # actions_list.extend(self.push_types)
        actions_list.append(EXIT)
        actions_list.append(ROTATE180)
        # actions_list.append(EXIT)
        return actions_list


    def set_env_limits(self):
        # set environment limits
        self.x_min = -3
        self.x_max = 4
        self.y_min = -3
        self.y_max = 4

        self.all_coordinate_locations = list(product(range(self.x_min,self.x_max),
                                                     range(self.y_min, self.y_max)))
        # print("all coordinate locations", self.all_coordinate_locations)
        # pdb.set_trace()
        # self.red_centroid = (1,1)
        # self.blue_centroid = (1,1)
        # self.red_centroid = (2, 0)
        # self.blue_centroid = (-2, 0)

    def reset(self):
        self.current_state = self.create_initial_state()




    def create_initial_state(self):
        # create dictionary of object location to object type and picked up state
        state = {}
        state['grid'] = copy.deepcopy(self.initial_object_locs) # type tuple (color, type) to location
        state['exit'] = False
        state['orientation'] =  np.pi # The orientation options are 0 or pi, which correspond to 0 or 180 degrees for the cup
        # state['currently_pushing'] = None
        # state['completed'] = []

        return state


    def is_done(self):
        # check if player at exit
        if self.current_state['exit']:
            return True
        # if len(self.current_state['completed']) == 4:
        #     return True
        return False

    def is_done_given_state(self, current_state):
        # check if player at exit location
        # print("current state", current_state)
        if current_state['exit']:
            return True
        # if len(current_state['completed']) == 4:
        #     return True
        return False

    def is_valid_push(self, current_state, action):
        # action_type_moved = current_state['currently_pushing']
        # if action_type_moved is None:
        #     return False
        # print("action type moved", action_type_moved)
        current_loc = current_state['grid'][self.object_type_tuple]

        new_loc = tuple(np.array(current_loc) + np.array(action))
        if new_loc[0] < self.x_min or new_loc[0] >= self.x_max or new_loc[1] < self.y_min or new_loc[1] >= self.y_max:
            return False

        # if new_loc in current_state['grid'].values() and new_loc != current_loc:
        #     return False

        return True


    def step_given_state_old(self, input_state, action):
        # step_cost = self.step_cost
        current_state = copy.deepcopy(input_state)

        # print("action", action)
        # check if action is exit
        # print("action in step", action)

        if current_state['exit'] == True:
            current_state['exit'] = True
            # featurized_state = self.featurize_state(current_state)
            featurized_state = self.featurize_state(current_state)
            step_reward = np.dot(self.reward_weights, featurized_state)
            # step_reward = 0
            # step_reward = -1000
            return current_state, step_reward, True


        if action == EXIT:
            current_state['exit'] = True
            # featurized_state = self.featurize_state(current_state)
            featurized_state = self.featurize_state(current_state)
            step_reward = np.dot(self.reward_weights, featurized_state)
            # step_reward = 0
            # step_reward = -1000
            return current_state, step_reward, True

        # pdb.set_trace()
        # if action in self.push_types:
        #     push_obj_type = self.push_to_type_idx[action]
        #     current_push_type = current_state['currently_pushing']
        #     if current_push_type is None:
        #         current_push_type = 0
        #     if push_obj_type in current_state['completed'] or push_obj_type != current_push_type+1:
        #         # featurized_state = self.featurize_state(current_state)
        #         step_reward = step_cost
        #         return current_state, step_reward, False
        #
        #     current_state['currently_pushing'] = self.push_to_type_idx[action]
        #     step_reward = self.push_switch_cost + step_cost
        #     return current_state, step_reward, False
        # if action == SWITCH:
        #     current_push_type = current_state['currently_pushing']
        #     if current_push_type is None:
        #         current_push_type = 0
        #     push_obj_type = current_push_type + 1
        #     if push_obj_type > max(self.obj_types_present):
        #         # featurized_state = self.featurize_state(current_state)
        #         current_state['exit'] = True
        #         step_reward = self.push_switch_cost + step_cost
        #         # featurized_state = self.featurize_state(current_state)
        #         # step_reward = np.dot(self.reward_weights, featurized_state)
        #         return current_state, step_reward, False
        #
        #     current_state['currently_pushing'] = push_obj_type
        #     step_reward = self.push_switch_cost + step_cost
        #     return current_state, step_reward, False

        if action in self.directions:
            if self.is_valid_push(current_state, action) is False:
                # featurized_state = self.featurize_state(current_state)
                # step_reward = step_cost
                featurized_state = self.featurize_state(current_state)
                step_reward = np.dot(self.reward_weights, featurized_state)
                return current_state, step_reward, False

        if action == ROTATE180:
            # add 180 to orientation and make between 0 and 360`
            # current_state['orientation'] = (current_state['orientation'] + 180) % 360
            # convert to radians
            current_state['orientation'] = (current_state['orientation'] + np.pi) % (2 * np.pi)



            # current_state['orientation'] = (current_state['orientation'] + 180) % 4
            # featurized_state = self.featurize_state(current_state)
            featurized_state = self.featurize_state(current_state)
            step_reward = np.dot(self.reward_weights, featurized_state)
            # step_reward = 0
            return current_state, step_reward, False

        action_type_moved = self.object_type_tuple
        # print("action_type_moved", action_type_moved)
        current_loc = current_state['grid'][action_type_moved]

        new_loc = tuple(np.array(current_loc) + np.array(action))
        current_state['grid'][action_type_moved] = new_loc


        featurized_state = self.featurize_state(current_state)
        # step_reward = np.dot(self.reward_weights, featurized_state) + step_cost
        step_reward = np.dot(self.reward_weights, featurized_state)
        # print("step reward", step_reward)

        done = self.is_done_given_state(current_state)


        return current_state, step_reward, done

    def step_given_state(self, input_state, action):
        step_cost = -0.1
        current_state = copy.deepcopy(input_state)

        # print("action", action)
        # check if action is exit
        # print("action in step", action)

        if current_state['exit'] == True:
            current_state['exit'] = True
            # featurized_state = self.featurize_state(current_state)
            # featurized_state = self.featurize_state(current_state)
            # step_reward = np.dot(self.reward_weights, featurized_state)
            step_reward = 0
            # step_reward = -1000
            return current_state, step_reward, True


        if action == EXIT:
            current_state['exit'] = True
            # featurized_state = self.featurize_state(current_state)
            featurized_state = self.featurize_state(current_state)
            step_reward = np.dot(self.reward_weights, featurized_state)
            step_reward += step_cost
            # step_reward = -1000
            return current_state, step_reward, True

        # pdb.set_trace()
        # if action in self.push_types:
        #     push_obj_type = self.push_to_type_idx[action]
        #     current_push_type = current_state['currently_pushing']
        #     if current_push_type is None:
        #         current_push_type = 0
        #     if push_obj_type in current_state['completed'] or push_obj_type != current_push_type+1:
        #         # featurized_state = self.featurize_state(current_state)
        #         step_reward = step_cost
        #         return current_state, step_reward, False
        #
        #     current_state['currently_pushing'] = self.push_to_type_idx[action]
        #     step_reward = self.push_switch_cost + step_cost
        #     return current_state, step_reward, False
        # if action == SWITCH:
        #     current_push_type = current_state['currently_pushing']
        #     if current_push_type is None:
        #         current_push_type = 0
        #     push_obj_type = current_push_type + 1
        #     if push_obj_type > max(self.obj_types_present):
        #         # featurized_state = self.featurize_state(current_state)
        #         current_state['exit'] = True
        #         step_reward = self.push_switch_cost + step_cost
        #         # featurized_state = self.featurize_state(current_state)
        #         # step_reward = np.dot(self.reward_weights, featurized_state)
        #         return current_state, step_reward, False
        #
        #     current_state['currently_pushing'] = push_obj_type
        #     step_reward = self.push_switch_cost + step_cost
        #     return current_state, step_reward, False

        if action in self.directions:
            if self.is_valid_push(current_state, action) is False:
                # featurized_state = self.featurize_state(current_state)
                # step_reward = step_cost
                # featurized_state = self.featurize_state(current_state)
                # step_reward = np.dot(self.reward_weights, featurized_state)
                step_reward = step_cost
                # current_state['exit'] = True
                return current_state, step_reward, False

        if action == ROTATE180:
            # add 180 to orientation and make between 0 and 360`
            # current_state['orientation'] = (current_state['orientation'] + 180) % 360
            # convert to radians
            og_orientation = current_state['orientation']
            current_state['orientation'] = (current_state['orientation'] + np.pi) % (2 * np.pi)
            # if og_orientation == np.pi/2:
            #     current_state['orientation'] = -np.pi / 2
            # else:
            #     current_state['orientation'] = np.pi / 2



            # current_state['orientation'] = (current_state['orientation'] + 180) % 4
            # featurized_state = self.featurize_state(current_state)
            featurized_state = self.featurize_state(current_state)
            step_reward = np.dot(self.reward_weights, featurized_state)
            step_reward = step_cost
            # step_reward = 0
            return current_state, step_reward, False

        action_type_moved = self.object_type_tuple
        # print("action_type_moved", action_type_moved)
        current_loc = current_state['grid'][action_type_moved]

        new_loc = tuple(np.array(current_loc) + np.array(action))
        current_state['grid'][action_type_moved] = new_loc


        featurized_state = self.featurize_state(current_state)
        # step_reward = np.dot(self.reward_weights, featurized_state) + step_cost
        step_reward = np.dot(self.reward_weights, featurized_state)
        step_reward = step_cost
        # print("step reward", step_reward)

        done = self.is_done_given_state(current_state)


        return current_state, step_reward, done

    def render(self, current_state, timestep):
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        def getImage(path, zoom=1):
            # pdb.set_trace()
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

        # current_push = plot_init_state['currently_pushing']
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
            # ab = AnnotationBbox(getImage(path), (loc[0], loc[1]), frameon=False)
            # ax.add_artist(ab)

        #         plt.text(loc[0], loc[1], str(type), fontsize=12)
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

        # featurized_state = np.zeros(self.num_features)
        current_loc = current_state['grid'][self.object_type_tuple]

        dist_to_red_centroid = np.linalg.norm(np.array(current_loc) - np.array(self.red_centroid))
        dist_to_blue_centroid = np.linalg.norm(np.array(current_loc) - np.array(self.blue_centroid))

        # compute dist_to_red_centroid as manhattan distance
        # dist_to_red_centroid = np.abs(current_loc[0] - self.red_centroid[0]) + np.abs(current_loc[1] - self.red_centroid[1])
        # dist_to_blue_centroid = np.abs(current_loc[0] - self.blue_centroid[0]) + np.abs(current_loc[1] - self.blue_centroid[1])
        # dist_to_red_centroid = np.abs(current_loc[0] - self.red_centroid[0])
        # dist_to_blue_centroid = np.abs(current_loc[0] - self.blue_centroid[0])

        orientation = current_state['orientation']

        # set orientation to pi/2 or -pi/2
        # if orientation == np.pi:
        #     orientation = np.pi/2
        # else:
        #     orientation = -np.pi/2

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
        # print("actions", actions)
        # pdb.set_trace()
        # create directional graph to represent all states
        G = nx.DiGraph()

        visited_states = set()

        stack = [copy.deepcopy(self.current_state)]

        while stack:
            # print("len visited_states", len(visited_states))
            # print("len stack", len(stack))
            # print("visited_states", visited_states)
            state = stack.pop()

            # convert old state to tuple
            state_tup = self.state_to_tuple(state)
            # print("new_state_tup", state_tup)

            # if state has not been visited, add it to the set of visited states
            if state_tup not in visited_states:
                visited_states.add(state_tup)

            # get the neighbors of this state by looping through possible actions
            # actions = self.get_possible_actions_in_state(state)
            # print("POSSIBLE actions", actions)
            for idx, action in enumerate(actions):
                # print("action", action)
                if self.is_done_given_state(state):
                    team_reward = 0
                    next_state = state
                    done = True

                else:
                    next_state, team_reward, done = self.step_given_state(state, action)
                # print("state", state)
                # print("action", action)
                # print("team_reward", team_reward)
                # print("done", done)
                # print("next_state", next_state)


                # if done:
                #     print("DONE")
                #     print("team_reward", team_reward)
                #
                #     print("state", state)
                #     print("next_state", next_state)
                #     print("action", action)
                #     print()
                #     team_reward += 10

                new_state_tup = self.state_to_tuple(next_state)
                # print("new_state_tup", new_state_tup)
                # print("new_state_tup in visited_states = ", new_state_tup in visited_states)
                # print()
                # pdb.set_trace()

                if new_state_tup not in visited_states:
                    stack.append(copy.deepcopy(next_state))

                # add edge to graph from current state to new state with weight equal to reward
                # if state_tup == new_state_tup:
                G.add_edge(state_tup, new_state_tup, weight=team_reward, action=action)
                # if state == {'grid': {(1, 1): (3, 3)}, 'exit': False, 'orientation': 0}:
                #     el = G.out_edges(state_tup, data=True)
                #     print("len el", len(el))
                #     if action == (1,0):
                #         pdb.set_trace()
                #     G.add_edge(state_tup, new_state_tup, weight=team_reward, action=str(action))
                #
                #     el = G.out_edges(state_tup, data=True)
                #     print("new len el", len(el))
                #     print("el", el)
                #     print("action", action)
                #     print()

                # if state_tup == new_state_tup:
                #     pdb.set_trace()
                # if state_tup != new_state_tup:
                #     G.add_edge(state_tup, new_state_tup, weight=team_reward, action=action)
                # if state_tup == new_state_tup:
                #     if self.is_done_given_state(state) is False:
                #         G.add_edge(state_tup, new_state_tup, weight=-200, action=action)
                #     else:
                #         G.add_edge(state_tup, new_state_tup, weight=0, action=action)
                        # pdb.set_trace()
        # pdb.set_trace()
        states = list(G.nodes)
        # print("NUMBER OF STATES", len(state
        idx_to_state = {i: state for i, state in enumerate(states)}
        state_to_idx = {state: i for i, state in idx_to_state.items()}

        # pdb.set_trace()
        action_to_idx = {action: i for i, action in enumerate(actions)}
        idx_to_action = {i: action for i, action in enumerate(actions)}

        # construct transition matrix and reward matrix of shape [# states, # states, # actions] based on graph
        transition_mat = np.zeros([len(states), len(states), len(actions)])
        reward_mat = np.zeros([len(states), len(actions)])

        for i in range(len(states)):
            # get all outgoing edges from current state
            # edges = G.out_edges(states[i], data=True)
            # if self.tuple_to_state(idx_to_state[i]) == {'grid': {(1, 1): (3, 3)}, 'exit': False, 'orientation': 0}:
            #     edges = G.out_edges(states[i], data=True)
            #     print("edges= ", edges)
            #     pdb.set_trace()
            state = self.tuple_to_state(idx_to_state[i])
            for action_idx_i in range(len(actions)):
                action = idx_to_action[action_idx_i]
                if self.is_done_given_state(state):
                    team_reward = 0
                    next_state = state
                    done = True

                else:
                    next_state, team_reward, done = self.step_given_state(state, action)
            # for edge in edges:
                # get index of action in action_idx
                # pdb.set_trace()
                # action_idx_i = action_to_idx[edge[2]['action']]
                # get index of next state in node list
                # next_state_i = states.index(edge[1])
                next_state_i = state_to_idx[self.state_to_tuple(next_state)]
                # add edge to transition matrix
                # if i == next_state_i:
                #     reward_mat[i, action_idx_i] = -200
                # else:
                #     reward_mat[i, action_idx_i] = edge[2]['weight']
                #     transition_mat[i, next_state_i, action_idx_i] = 0.0
                #
                # else:
                transition_mat[i, next_state_i, action_idx_i] = 1.0
                # reward_mat[i, action_idx_i] = edge[2]['weight']
                reward_mat[i, action_idx_i] = team_reward
                # pdb.set_trace()
                # if idx_to_action[action_idx_i] == (0, 1) and self.tuple_to_state(idx_to_state[i]) == {'grid': {(1, 1): (3, 3)}, 'exit': False, 'orientation': 0}:
                #     # reward_mat[i, action_idx_i] = 0.0
                #     pdb.set_trace()
                # if self.tuple_to_state(idx_to_state[i]) == {'grid': {(1, 1): (3, 3)}, 'exit': False, 'orientation': 0}:
                #     edges = G.out_edges(states[i], data=True)
                #     print("edges= ", edges)
                #     print("action", idx_to_action[action_idx_i])
                    # pdb.set_trace()

        # check that for each state and action pair, the sum of the transition probabilities is 1 (or 0 for terminal states)
        # for i in range(len(states)):
        #     for j in range(len(actions)):
        #         print("np.sum(transition_mat[i, :, j])", np.sum(transition_mat[i, :, j]))
        #         print("np.sum(transition_mat[i, :, j]", np.sum(transition_mat[i, :, j]))
        # assert np.isclose(np.sum(transition_mat[i, :, j]), 1.0) or np.isclose(np.sum(transition_mat[i, :, j]),
        #                                                                       0.0)
        self.transitions, self.rewards, self.state_to_idx, \
        self.idx_to_action, self.idx_to_state, self.action_to_idx = transition_mat, reward_mat, state_to_idx, \
                                                                    idx_to_action, idx_to_state, action_to_idx

        # print("number of states", len(states))
        # print("number of actions", len(actions))
        # print("transition matrix shape", transition_mat.shape)
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
        human_only_reward = 0
        robot_only_reward = 0

        human_trace = []
        robot_trace = []
        human_greedy_alt = []
        robot_greedy_alt = []
        iters = 0
        game_results = []
        # self.setup_grid()
        sum_feature_vector = np.zeros(4)

        # self.render(self.current_state, iters)
        while not done:
            iters += 1
            current_state_tup = self.state_to_tuple(self.current_state)
            state_idx = self.state_to_idx[current_state_tup]

            action_distribution = self.policy[state_idx]
            action = np.argmax(action_distribution)
            action = self.idx_to_action[action]

            if action in self.directions:
                isvalid = self.is_valid_push(self.current_state, action)

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
        # import cv2
        import os
        #
        # image_folder = 'rollouts'
        # video_name = 'video.mp4'
        #
        # images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        # frame = cv2.imread(os.path.join(image_folder, images[0]))
        # height, width, layers = frame.shape
        #
        # video = cv2.VideoWriter(video_name, 0, 1, (width, height))
        #
        # for image in images:
        #     video.write(cv2.imread(os.path.join(image_folder, image)))
        #
        # cv2.destroyAllWindows()
        # video.release()
        os.system(f"ffmpeg -r 1 -i rollouts/state_%01d.png -vcodec mpeg4 -y {self.savefilename}.mp4")
        self.clear_rollouts()

    def clear_rollouts(self):
        import os
        os.system("rm -rf rollouts")
        os.system("mkdir rollouts")


    def compute_optimal_performance(self):
        # print("start enumerating states")
        self.enumerate_states()
        # print("done enumerating states")
        # print("start vi")
        self.vectorized_vi()
        # print("done vi")

        optimal_rew, game_results, sum_feature_vector = self.rollout_full_game_joint_optimal()
        return optimal_rew, game_results, sum_feature_vector


def generate_preference_dataset(red_centroid, blue_centroid, to_save=False):
    trajectories = {}
    rankings_on_scale = []

    true_reward_weights = [-2, -2, 2, -1]  # orientation, red prox, blue prox, pos y

    true_f_idx = [1, 1, 1, 1]
    # true_reward_weights = [(float(i) / np.linalg.norm(true_reward_weights)) for i in true_reward_weights]
    # true_reward_weights = [float(i) / sum(true_reward_weights) for i in true_reward_weights]
    # true_f_idx = [0, 5, 1, 3]
    # f_options = [[1,1,1,1]]
    f_options = [[1,1,1,1], [1,1,1,0], [1,1,0,1], [1,0,1,1], [0,1,1,1]]

    # seen = [true_reward_weights]
    object_type = RED_CUP

    counter = 0
    all_reward_weight_possilbilities = []
    all_reward_weight_possilbilities = [[-1,-1,-1,-1],
                                        [-1,-1,-1,1],
                                        [-1,-1,1,-1],
                                        [-1,1,-1,-1],
                                        [1,-1,-1,-1],
                                        [-1,-1,1,1],
                                        [-1,1,-1,1],
                                        [-1,1,1,-1],
                                        [1,-1,-1,1],
                                        [1,-1,1,-1],
                                        [1,1,-1,-1],
                                        [-1,1,1,1],
                                        [1,-1,1,1],
                                        [1,1,-1,1],
                                        [1,1,1,-1],
                                        [1,1,1,1]]
    for i in range(len(all_reward_weight_possilbilities)):
        all_reward_weight_possilbilities[i][0] *= 1
        all_reward_weight_possilbilities[i][1] *= 2
        all_reward_weight_possilbilities[i][2] *= 2
        all_reward_weight_possilbilities[i][3] *= 2
        # rw[1] = rw[1] * 2
    # print(len(all_reward_weight_possilbilities))


    for i in range(len(all_reward_weight_possilbilities)):
        reward_weights = all_reward_weight_possilbilities[i] # first should be negative, second can be pos or neg, last should be negative
        # normalize reward weights
        # if reward_weights in seen:
        #     continue
        # reward_weights.extend([-0.1, -0.5])
        # reward_weights = [(float(i) / np.linalg.norm(reward_weights)) for i in reward_weights]
        # reward_weights = [float(i) / sum(reward_weights) for i in reward_weights]
        # rand_f_idx = [np.random.randint(0, 7) for _ in range(4)]
        # permutations_of_true_f = list(itertools.permutations(true_f_idx))
        for perm_idx in range(0, len(f_options)):
            rand_f_idx = f_options[perm_idx]

            game = Gridworld(reward_weights, rand_f_idx, object_type, red_centroid, blue_centroid)
            optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance()

            trajectories[counter] = {}
            trajectories[counter]['f'] = rand_f_idx
            trajectories[counter]['w_idx'] = i
            trajectories[counter]['w'] = reward_weights
            trajectories[counter]['rew_wrt_optimal'] = np.dot(true_reward_weights, sum_feature_vector)
            trajectories[counter]['sum_feature_vector'] = sum_feature_vector
            trajectories[counter]['final_state'] = game_results[-1]
            # save optimal policy, state_to_idx, idx_to_state, idx_to_action, action_to_idx, transitions, rewards to trajectories
            trajectories[counter]['optimal_policy'] = game.policy
            trajectories[counter]['state_to_idx'] = game.state_to_idx
            trajectories[counter]['idx_to_state'] = game.idx_to_state
            trajectories[counter]['idx_to_action'] = game.idx_to_action
            trajectories[counter]['action_to_idx'] = game.action_to_idx
            trajectories[counter]['transitions'] = game.transitions
            trajectories[counter]['rewards'] = game.rewards
            counter += 1

    if to_save:
        pickle.dump(trajectories, open("data/movemdp_1obj.pkl", "wb"))
    return trajectories


def compute_new_policy(reward_weights, true_f_idx, object_type_tuple, red_centroid, blue_centroid):
    game = Gridworld(reward_weights, true_f_idx, object_type_tuple, red_centroid, blue_centroid)
    optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance()
    return game.policy, game.state_to_idx, game.idx_to_state, game.idx_to_action, game.action_to_idx, game.transitions, game.rewards



if __name__ == '__main__':
    # reward_weights = [0,10,-10,0,10,0,-10,-10,-10] # first should be negative, second can be pos or neg, last should be negative
    # reward_weights = [np.random.uniform(-10, 10) for _ in range(9)]
    reward_weights = [-10, -2, -2, -2]  # orientation, red prox, blue prox, pos y

    # reward_weights = [(float(i) / np.linalg.norm(reward_weights)) for i in reward_weights]
    # print("reward_weights", reward_weights)
    red_centroid, blue_centroid = (2,2), (-3,-3) # The grid is 7x7
    true_f_idx = [1, 1, 1, 1]

    # here, is the set of features that matter to the user.
    # If the user cares about the orientation, red proximity, and blue proximity, then the true_f_idx would be [1, 1, 1, 0]
    # If the user cares about the orientation, red proximity, and y position, then the true_f_idx would be [1, 1, 0, 1]
    # etc.


    object_type_tuple = RED_CUP # just a global variable for the cup

    # now we set up the environment
    game = Gridworld(reward_weights, true_f_idx, object_type_tuple, red_centroid, blue_centroid)

    # Run value iteration to get the optimal policy
    optimal_rew, game_results, sum_feature_vector = game.compute_optimal_performance()
    print("optimal_rew", optimal_rew)
    print("game_results", game_results)

    # pdb.set_trace()
    # trajs = generate_preference_dataset(red_centroid, blue_centroid, to_save=False)


