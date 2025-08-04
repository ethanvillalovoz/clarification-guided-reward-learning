# Authors: Ethan Villalovz, Michelle Zhao
# Project: RISS 2024 Summer Project - Bayesian Learning Interaction
# Description: Incorporating the entire interaction between the human and robot object simulation. Updates the human
# preference through bayesian inference after each time step
import pdb

# imports
from multi_object_custom_mdp_v5 import *


# Initialize robot beliefs
def initialize_robot_beliefs(hypothesis_reward_space):
    # set to uniform
    return np.ones(len(hypothesis_reward_space)) / len(hypothesis_reward_space)


# Get weighted robot action
# def get_weighted_robot_action(state, timestep, robot_beliefs, hypothesis_reward_space, object_type_tuple):
#     tree_idx = np.random.choice(np.arange(len(robot_beliefs)), p=robot_beliefs)
#     print("tree index:", tree_idx)
#     tree = hypothesis_reward_space[tree_idx]
#     tree_policy = Gridworld(tree, object_type_tuple)
#     optimal_rew, game_results, sum_feature_vector = tree_policy.compute_optimal_performance(render=False)
#     print("game_results", game_results)
#     # if timestep < len(game_results):
#     new_state = game_results[timestep + 1][0]
#     objects = list(game_results[timestep + 1][0].keys())
#     action = game_results[timestep + 1][0][objects[timestep]]['pos']
#     # elif len(game_results) == 1:
#     #     new_state = game_results[0][0]
#     #     objects = list(game_results[0][0].keys())
#     #     action = game_results[0][0][objects[timestep]]['pos']
#     # else:
#     #     new_state = game_results[timestep][0]
#     #     objects = list(game_results[timestep][0].keys())
#     #     action = game_results[timestep][0][objects[timestep - 1]]['pos']
#
#     print(type(action[0]))
#     if int(action[0]) > 0 and int(action[1]) > 0:
#         action = 'Q1'
#     elif int(action[0]) < 0 and int(action[1]) > 0:
#         action = 'Q2'
#     elif int(action[0]) < 0 and int(action[1]) < 0:
#         action = 'Q3'
#     elif int(action[0]) > 0 and int(action[1]) < 0:
#         action = 'Q4'
#
#     if timestep > 0:
#         new_state[objects[timestep - 1]]['pos'] = state[objects[timestep - 1]]['pos']
#
#     # # actual_new_State
#     # actual_new_state = copy.deepcopy(state)
#     # # change only the object that action
#     # actual_new_state[object at timestep] set to action # <-- this is pseudocode
#
#     print(action)
#
#     return action, new_state

def get_weighted_robot_action(state, timestep, robot_beliefs, hypothesis_reward_space, object_type_tuple):
    tree_idx = np.random.choice(np.arange(len(robot_beliefs)), p=robot_beliefs)
    print("tree index:", tree_idx)
    tree = hypothesis_reward_space[tree_idx]
    
    tree_policy = Gridworld(tree, object_type_tuple)
    optimal_rew, game_results, sum_feature_vector = tree_policy.compute_optimal_performance()
    
    # Get the next action from game_results based on the timestep
    if timestep + 1 < len(game_results):
        next_state = game_results[timestep + 1][0]
        action = game_results[timestep + 1][1]  # This should be the action
        print("robot given state", state)
        print("robot action", action)
        print("robot next", next_state)
        
        if isinstance(action, tuple) and len(action) > 1:
            action = action[1]  # Extract the actual action part
        
        print("action", action)
        return action, next_state
    else:
        # Handle case where timestep+1 is out of range
        print("Timestep out of range in game_results")
        # Return last state and EXIT action as fallback
        return EXIT, state
# Get correction from human
# def get_correction_from_human(new_state, timestep, robot_action, true_reward_tree, object_type_tuple):
#     tree_policy = Gridworld(true_reward_tree, object_type_tuple)
#     optimal_rew, game_results, sum_feature_vector = tree_policy.compute_optimal_performance()
#     print(game_results)
#     # if timestep < len(game_results):
#     corrected_state = game_results[timestep + 1][0]
#     objects = list(game_results[timestep + 1][0].keys())
#     action = game_results[timestep + 1][0][objects[timestep]]['pos']
#     # elif len(game_results) == 1:
#     #     corrected_state = game_results[0][0]
#     #     objects = list(game_results[0][0].keys())
#     #     action = game_results[0][0][objects[timestep]]['pos']
#     # else:
#     #     corrected_state = game_results[timestep][0]
#     #     objects = list(game_results[timestep][0].keys())
#     #     action = game_results[timestep][0][objects[timestep - 1]]['pos']
#
#     print(type(action[0]))
#     if int(action[0]) > 0 and int(action[1]) > 0:
#         action = 'Q1'
#     elif int(action[0]) < 0 and int(action[1]) > 0:
#         action = 'Q2'
#     elif int(action[0]) < 0 and int(action[1]) < 0:
#         action = 'Q3'
#     elif int(action[0]) > 0 and int(action[1]) < 0:
#         action = 'Q4'
#
#     print(action)
#     return action, corrected_state

def get_correction_from_human(new_state, timestep, robot_action, true_reward_tree, object_type_tuple):
    tree_policy = Gridworld(true_reward_tree, object_type_tuple)
    optimal_rew, game_results, sum_feature_vector = tree_policy.compute_optimal_performance()
    
    # Get the next action from game_results based on the timestep
    if timestep + 1 < len(game_results):
        next_state = game_results[timestep + 1][0]
        action = game_results[timestep + 1][1]  # This should be the action
        print("new_state", new_state)
        print("corrected action", action)
        print("next_state", next_state)
        
        if isinstance(action, tuple) and len(action) > 1:
            action = action[1]  # Extract the actual action part
        
        print("action", action)
        return action, next_state
    else:
        # Handle case where timestep+1 is out of range
        print("Timestep out of range in game_results for correction")
        # Return last state and EXIT action as fallback
        return EXIT, new_state


def get_correction_from_human_keyboard_input(new_state, timestep, robot_action, object_type_tuple, object_tuples_list):
    current_object = object_tuples_list[timestep] # (2,1,1)
    # current_object_description = get_description(current_object) # TODO
    selected_quadrant = input(f"for the object that just moved {current_object}, which quadrant should it be in ([Q1, Q2, Q3, Q4]?")
    # selected_quadrant is going to be string like 'Q1'
    print("selected_quadrant", selected_quadrant)

    empty_reward_tree = {}

    tree_policy = Gridworld(empty_reward_tree, object_type_tuple, new_state)
    action = (current_object, selected_quadrant)
    next_state, team_rew, done = tree_policy.step_given_state(new_state,  action)
    print("new_state", new_state)
    print("corrected action", selected_quadrant)
    print("next_state", next_state)
    action = selected_quadrant

    # pdb.set_trace()

    print("action", action)
    return action, next_state


# Update robot beliefs using Bayesian inference
def update_robot_beliefs(s0_starting_state, sr_state, sh_state, robot_beliefs,
                         hypothesis_reward_space, object_type_tuple):
    # check condition
    cond_1, cond_2, cond_3 = False, False, False

    if sh_state != sr_state:
        cond_1 = True
    if sr_state == sh_state and s0_starting_state != sr_state:
        cond_2 = True
    if sh_state != s0_starting_state:
        cond_3 = True

    # Placeholder for likelihood computation and Bayes update
    # likelihoods = []
    beta = 1
    new_beliefs = []
    for tree_idx in range(len(hypothesis_reward_space)):
        print("tree_idx", tree_idx)
        prior_belief_of_theta_i = robot_beliefs[tree_idx]  # P(theta_i)

        tree = hypothesis_reward_space[tree_idx]
        tree_policy = Gridworld(tree, object_type_tuple)
        # tree_policy.compute_optimal_performance(render=False)
        s0_reward = tree_policy.lookup_quadrant_reward(s0_starting_state)
        sr_reward = tree_policy.lookup_quadrant_reward(sr_state)
        sh_reward = tree_policy.lookup_quadrant_reward(sh_state)

        print("s0_reward", s0_reward)
        print("sr_reward", sr_reward)
        print("sh_reward", sh_reward)

        # Compute the likelihood of human correction given robot action and the hypothesis tree
        # likelihood = tree_policy.compute_likelihood(state, robot_action, human_correction)
        # likelihoods.append(likelihood)
        aggregated_likelihood = 1  # P(all d| tree theta_i)

        # print conditions
        print("cond_1", cond_1)
        print("cond_2", cond_2)
        print("cond_3", cond_3)

        if cond_1:
            beta_cond_1 = 5
            prob_sh_greater_than_sr = np.exp(beta_cond_1 * sh_reward) / (np.exp(beta_cond_1 * sr_reward) + np.exp(beta_cond_1 * sh_reward))
            print("prob_sh_greater_than_sr", prob_sh_greater_than_sr)

            aggregated_likelihood *= prob_sh_greater_than_sr

        if cond_2:
            beta_cond_2 = 0.5
            prob_sr_greater_than_s0 = np.exp(beta_cond_2 * sr_reward) / (np.exp(beta_cond_2 * sr_reward) + np.exp(beta_cond_2 * s0_reward))
            print("prob_sr_greater_than_s0", prob_sr_greater_than_s0)
            aggregated_likelihood *= prob_sr_greater_than_s0

        if cond_3:
            beta_cond_3 = 0.5
            prob_sh_greater_than_s0 = np.exp(beta_cond_3 * sh_reward) / (np.exp(beta_cond_3 * sh_reward) + np.exp(beta_cond_3 * s0_reward))
            print("prob_sh_greater_than_s0", prob_sh_greater_than_s0)
            aggregated_likelihood *= prob_sh_greater_than_s0

        prob_theta_i_given_data = aggregated_likelihood * prior_belief_of_theta_i
        new_beliefs.append(prob_theta_i_given_data)

    print("new_beliefs", new_beliefs)
    # Bayes' update
    new_beliefs = np.array(new_beliefs)
    # robot_beliefs = robot_beliefs * likelihoods
    new_beliefs /= np.sum(new_beliefs)
    return new_beliefs


def ask_clarification_questions(new_state, corrected_state, object_type_tuple):
    # Placeholder for dialogue system
    questions = [
        "Why did you prefer this new position over the one I chose?",
        "Can you explain why this location is better?",
        "Which attribute of the object influenced your correction the most?",
        "Was the position of the object the main reason for your correction, or was it something else?",
        "How would you like me to position similar objects in the future?",
        "Are there specific rules I should follow when placing objects like this?",
        "I think you prefer objects to be placed closer to the center. Is that correct?",
        "It seems like you prefer objects to be in quadrant Q1. Is that true for all objects?"
    ]

    for question in questions:
        print(question)
        # In a real implementation, this would be where the robot receives and processes the human's response


def get_relevant_features(preferences, attributes, list_of_relevant_features):
    if isinstance(preferences, dict) and 'Q1' in preferences:  # Base case: preferences is a reward value
        return preferences, list_of_relevant_features

    if isinstance(preferences, dict):
        for attr in attributes:
            if attr in preferences:
                list_of_relevant_features.append(attr)
                # print(f"got reward {self.get_reward_value(preferences[attr], attributes, quadrants)} for attributes {attributes} at quadrant {quadrants}")
                return get_relevant_features(preferences[attr], attributes, list_of_relevant_features)

    if 'other' in preferences:
        # print("found other:", preferences['other'])
        return preferences['other'], list_of_relevant_features
def ask_feature_clarification_question(robot_beliefs, hypothesis_reward_space,
                                       s0_starting_state, sr_starting_state, sh_starting_state, timestep,
                                       current_object):
    color_idx, material_idx, object_idx, object_label = current_object
    color = COLORS_IDX[color_idx]
    material = MATERIALS_IDX[material_idx]
    object_type = OBJECTS_IDX[object_idx]
    attributes = [object_type, color, material]


    hyp_idx_to_relevant_features = {}
    for hyp_idx in range(len(hypothesis_reward_space)):
        print('hyp_idx', hyp_idx)
        hypothesis_tree = hypothesis_reward_space[hyp_idx]
        pref_tree = hypothesis_tree['pref_values']
        quadrant_preference_tree, relevant_features = get_relevant_features(pref_tree, attributes, [])
        print("relevant_features", relevant_features)
        hyp_idx_to_relevant_features[hyp_idx] = relevant_features

    # ask question
    # ideally, we have question = get_llm_query(hyp_idx_to_relevant_features), but for now, we will do this naively
    true_relevant_features = []
    human_response = input(f"for the recent {attributes} object, which features of [color, type, and material] were relevant to the location it should be in? (give comma separated responses)")
    if 'type' in human_response:
        true_relevant_features.append(object_type)
    if 'color' in human_response:
        true_relevant_features.append(color)
    if 'material' in human_response:
        true_relevant_features.append(material)

    print("true_relevant_features", true_relevant_features)

    # update based on human response
    likelihood_of_tree_given_correct_response = 0.8
    likelihood_of_tree_given_incorrect_response = 1 - likelihood_of_tree_given_correct_response

    new_robot_beliefs = copy.deepcopy(robot_beliefs)
    for hyp_idx in range(len(hypothesis_reward_space)):
        prior_belief_of_theta_i = new_robot_beliefs[hyp_idx]  # P(theta_i)
        relevant_features_for_tree_idx = hyp_idx_to_relevant_features[hyp_idx]
        if relevant_features_for_tree_idx == true_relevant_features:
            prob_of_hyp_idx_given_relevant_features = likelihood_of_tree_given_correct_response # P(data | theta_i)
        else:
            prob_of_hyp_idx_given_relevant_features = likelihood_of_tree_given_incorrect_response  # P(data | theta_i)

        new_robot_beliefs[hyp_idx] = prob_of_hyp_idx_given_relevant_features * prior_belief_of_theta_i

    # normalize
    new_beliefs = np.array(new_robot_beliefs)
    # robot_beliefs = robot_beliefs * likelihoods
    new_beliefs /= np.sum(new_beliefs)
    return new_beliefs

# Run the interaction loop
def run_interaction():
    hypothesis_reward_space = [f_Ethan, f_Michelle, f_Annika, f_Admoni, f_Simmons, f_Suresh, f_Ben, f_Ada, f_Abhijat,
                               f_Maggie, f_Zulekha, f_Pat]
    labels = ['Ethan', 'Michelle', 'Annika', 'Admoni', 'Simmons', 'Suresh', 'Ben', 'Ada', 'Abhijat', 'Maggie',
              'Zulekha', 'Pat']
    true_reward_tree = f_Ethan
    list_of_present_object_tuples = [obj_1, obj_2, obj_3]  # Define your object type tuple as per your requirements
    render_game = Gridworld(f_Ethan, list_of_present_object_tuples)
    initial_state = Gridworld(f_Ethan, list_of_present_object_tuples).get_initial_state()

    # normalize hypotheses
    for tree_i in range(len(hypothesis_reward_space)):
        tree = hypothesis_reward_space[tree_i]
        new_tree = copy.deepcopy(tree)
        # print("tree", tree)
        for feat1 in tree['pref_values']:
            if list(tree['pref_values'][feat1].keys()) == ['Q1', 'Q2', 'Q3', 'Q4']:
                # add the minimum value of the tree to all values
                min_val = min([tree['pref_values'][feat1][Q] for Q in tree['pref_values'][feat1]])
                for Q in tree['pref_values'][feat1]:
                    new_tree['pref_values'][feat1][Q] -= min_val

                # normalize the values of the Q dict
                sum_Q = sum([tree['pref_values'][feat1][Q] for Q in tree['pref_values'][feat1]])
                for Q in tree['pref_values'][feat1]:
                    new_tree['pref_values'][feat1][Q] /= sum_Q
            else:
                for feat2 in tree['pref_values'][feat1]:
                    if list(tree['pref_values'][feat1][feat2].keys()) == ['Q1', 'Q2', 'Q3', 'Q4']:
                        # add the minimum value of the tree to all values
                        min_val = min([tree['pref_values'][feat1][feat2][Q] for Q in tree['pref_values'][feat1][feat2]])
                        for Q in tree['pref_values'][feat1][feat2]:
                            new_tree['pref_values'][feat1][feat2][Q] -= min_val

                        # normalize the values of the Q dict
                        # print("tree['pref_values'][feat1][feat2]", tree['pref_values'][feat1][feat2])

                        sum_Q = sum([tree['pref_values'][feat1][feat2][Q] for Q in tree['pref_values'][feat1][feat2]])
                        # print("sum_Q", sum_Q)
                        for Q in tree['pref_values'][feat1][feat2]:
                            new_tree['pref_values'][feat1][feat2][Q] /= sum_Q
                    else:
                        for feat3 in tree['pref_values'][feat1][feat2]:
                            # add the minimum value of the tree to all values
                            min_val = min([tree['pref_values'][feat1][feat2][feat3][Q] for Q in
                                           tree['pref_values'][feat1][feat2][feat3]])
                            for Q in tree['pref_values'][feat1][feat2][feat3]:
                                new_tree['pref_values'][feat1][feat2][feat3][Q] -= min_val

                            # normalize the values of the Q dict
                            sum_Q = sum([tree['pref_values'][feat1][feat2][feat3][Q] for Q in
                                         tree['pref_values'][feat1][feat2][feat3]])
                            # print("sum_Q", sum_Q)
                            
                            # Avoid division by zero
                            if sum_Q > 0:
                                for Q in tree['pref_values'][feat1][feat2][feat3]:
                                    new_tree['pref_values'][feat1][feat2][feat3][Q] /= sum_Q
                            else:
                                # If sum is zero, set all values to equal (uniform) probabilities
                                uniform_value = 0.25  # 1/4 for 4 quadrants
                                for Q in tree['pref_values'][feat1][feat2][feat3]:
                                    new_tree['pref_values'][feat1][feat2][feat3][Q] = uniform_value

        hypothesis_reward_space[tree_i] = new_tree

    # print("hypothesis_reward_space", hypothesis_reward_space)
    robot_beliefs = initialize_robot_beliefs(hypothesis_reward_space)

    plt.bar(np.arange(len(robot_beliefs)), height=robot_beliefs, tick_label=labels)
    plt.title('Robot Beliefs')
    plt.xlabel('Model Preferences')
    plt.ylabel('Certainty')
    plt.xticks(rotation=90)
    plt.show()

    state = initial_state
    object_tuples_list = [keyname for keyname in initial_state.keys() if keyname != 'exit']
    for t in range(len(list_of_present_object_tuples)):
        print("--- Initial state ---")
        render_game.render(state, t)
        robot_action, new_state = get_weighted_robot_action(state, t, robot_beliefs, hypothesis_reward_space,
                                                            list_of_present_object_tuples)  # take one of the objects and move it to a quadrant
        print("new_state", new_state)
        # pdb.set_trace()
        print("--- Robot moved to new state ---")
        render_game.render(new_state, t)
        # set the objects back to not done
        new_state_reset_obj = copy.deepcopy(new_state)
        current_object = object_tuples_list[t]
        new_state_reset_obj[current_object]['done'] = False

        # User Input
        # Asks where do you want yellow cup to go
        # Input: 'Q1'
        # human_correction, new_corrected_state = get_correction_from_human_keyboard_input(new_state_reset_obj, t, robot_action, list_of_present_object_tuples,
        #                                                                   object_tuples_list) <-- this is not pretty, should be drag and drop

        # Get correction from human
        human_correction, new_corrected_state = get_correction_from_human(new_state_reset_obj, t, robot_action, true_reward_tree,
                                                                          list_of_present_object_tuples)

        print("--- Human corrected to new state ---")
        render_game.render(new_corrected_state, t)
        # pdb.set_trace()
        print("updating beliefs")
        # render_game.render(state, t, "beliefs: state")
        # render_game.render(new_state, t, "beliefs: new_state")
        # render_game.render(new_corrected_state, t, "beliefs: new_corrected_state")

        # Render, color human actions differently
        s0_starting_state = copy.deepcopy(state)  # starting current state
        sr_starting_state = copy.deepcopy(new_state)  # robot moved state
        sh_starting_state = copy.deepcopy(new_corrected_state)  # human corrected state
        robot_beliefs = update_robot_beliefs(s0_starting_state, sr_starting_state, sh_starting_state, robot_beliefs,
                                             hypothesis_reward_space, list_of_present_object_tuples)

        # state[] = new_corrected_state
        plt.bar(np.arange(len(robot_beliefs)), height=robot_beliefs, tick_label=labels)
        plt.title('Robot Beliefs (prior to the question)')
        plt.xlabel('Model Preferences')
        plt.ylabel('Certainty')
        plt.xticks(rotation=90)
        plt.show()

        # Clarification question step can be added here
        # Clarification question step
        # ask_clarification_questions(new_state, new_corrected_state, object_type_tuple)
        new_robot_beliefs = ask_feature_clarification_question(robot_beliefs, hypothesis_reward_space,
                                                               s0_starting_state, sr_starting_state, sh_starting_state, t,
                                                               current_object)
        robot_beliefs = new_robot_beliefs

        plt.bar(np.arange(len(robot_beliefs)), height=robot_beliefs, tick_label=labels)
        plt.title('Robot Beliefs (after clarification)')
        plt.xlabel('Model Preferences')
        plt.ylabel('Certainty')
        plt.xticks(rotation=90)
        plt.show()

        # Break if task is done (Define your task completion condition)
        state = new_corrected_state


# Start of program
if __name__ == '__main__':
    run_interaction()

# more diverse tree, fix the "other in tree" issue (characteristics), reduce the size of all tree values (-10, and 10), push.
# DONE

# big task - handle multiple objects. Code needs modifications.
# big task - handle multiple objects. Code needs modifications.
# for multiple objects, you need to force/check human correction should be on the object the robot just moved.


# brainstorm types of questions you would ask (imagine you are the robot and the human just gave you a correction),

# Clarification Questions
#
# Preference Clarification:
# "Why did you prefer this new position over the one I chose?"
# "Can you explain why this location is better?"
# Attribute Focus:
# "Which attribute of the object influenced your correction the most?"
# "Was the position of the object the main reason for your correction, or was it something else?"
# Contextual Understanding:
# "Is there a specific context or scenario in which this placement is better?"
# "How does this placement fit into your overall plan or goal?"
#
# Preference Questions
#
# Future Preferences:
# "How would you like me to position similar objects in the future?"
# "Are there specific rules I should follow when placing objects like this?"
# Comparative Questions:
# "Between these two positions, which one do you prefer and why?"
# "If I had placed the object here instead, would that have been acceptable?"
#
# Hypothesis Testing Questions
#
# Hypothesis Validation:
# "I think you prefer objects to be placed closer to the center. Is that correct?"
# "It seems like you prefer objects to be in quadrant Q1. Is that true for all objects?"
# Scenario Simulation:
# "If this object were larger, would you still prefer this position?"
# "What if there were more objects in the environment? How would that change your preference?"


# brainstorm how to bootstrap the learning via language.
# 1. Natural Language Instructions
#
# Allow the human to provide detailed feedback using natural language. For example:
# "Place the object near the top left corner but not too close to the edge."
# "I prefer objects to be placed in well-lit areas."
# 2. Active Learning
#
# Implement active learning where the robot actively asks questions to refine its understanding:
# "Do you prefer the object to be placed closer to the center or the edge?"
# "Is it more important for the object to be accessible or out of the way?"
# 3. Semantic Parsing
#
# Use semantic parsing to convert natural language instructions into a formal representation that the robot can understand and act upon. For example, converting "place the object near the top left corner" into coordinates or specific actions.
# 4. Reinforcement Learning with Natural Language
#
# Integrate natural language feedback into a reinforcement learning framework where the robot updates its policies based on human feedback:
# Positive Feedback: "Good job, this is the correct position."
# Negative Feedback: "No, this is not where I want it."
# 5. Knowledge Graphs
#
# Build knowledge graphs that capture human preferences and rules about object placement. Update these graphs with new information obtained through natural language interactions.
# 6. Interactive Dialogue Systems
#
# Develop an interactive dialogue system where the robot and human can have a back-and-forth conversation to clarify and refine preferences:
# Robot: "Do you want the object in quadrant Q1?"
# Human: "Yes, but closer to the center."
