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
def get_weighted_robot_action(state, robot_beliefs, hypothesis_reward_space, object_type_tuple):
    tree_idx = np.random.choice(np.arange(len(robot_beliefs)), p=robot_beliefs)
    print("tree index:", tree_idx)
    tree = hypothesis_reward_space[tree_idx]
    tree_policy = Gridworld(tree, object_type_tuple)
    optimal_rew, game_results, sum_feature_vector = tree_policy.compute_optimal_performance(render=False)
    print("game_results", game_results)
    new_state = game_results[-1][0]
    objects = list(game_results[-1][0].keys())
    action = game_results[-1][0][objects[0]]['pos']

    print(type(action[0]))
    if int(action[0]) > 0 and int(action[1]) > 0:
        action = 'Q1'
    elif int(action[0]) < 0 and int(action[1]) > 0:
        action = 'Q2'
    elif int(action[0]) < 0 and int(action[1]) < 0:
        action = 'Q3'
    elif int(action[0]) > 0 and int(action[1]) < 0:
        action = 'Q4'

    print(action)

    return action, new_state


# Get correction from human
def get_correction_from_human(new_state, robot_action, true_reward_tree, object_type_tuple):
    tree_policy = Gridworld(true_reward_tree, object_type_tuple)
    optimal_rew, game_results, sum_feature_vector = tree_policy.compute_optimal_performance()
    print(game_results)
    corrected_state = game_results[-1][0]
    objects = list(game_results[-1][0].keys())
    action = game_results[-1][0][objects[0]]['pos']

    print(type(action[0]))
    if int(action[0]) > 0 and int(action[1]) > 0:
        action = 'Q1'
    elif int(action[0]) < 0 and int(action[1]) > 0:
        action = 'Q2'
    elif int(action[0]) < 0 and int(action[1]) < 0:
        action = 'Q3'
    elif int(action[0]) > 0 and int(action[1]) < 0:
        action = 'Q4'

    print(action)
    return action, corrected_state


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
    beta = 5
    new_beliefs = []
    for tree_idx in range(len(hypothesis_reward_space)):
        prior_belief_of_theta_i = robot_beliefs[tree_idx]  # P(theta_i)

        tree = hypothesis_reward_space[tree_idx]
        tree_policy = Gridworld(tree, object_type_tuple)
        # tree_policy.compute_optimal_performance(render=False)
        s0_reward = tree_policy.lookup_quadrant_reward(s0_starting_state)
        sr_reward = tree_policy.lookup_quadrant_reward(sr_state)
        sh_reward = tree_policy.lookup_quadrant_reward(sh_state)


        # Compute the likelihood of human correction given robot action and the hypothesis tree
        # likelihood = tree_policy.compute_likelihood(state, robot_action, human_correction)
        # likelihoods.append(likelihood)
        aggregated_likelihood = 1 # P(all d| tree theta_i)
        if cond_1:
            prob_sh_greater_than_sr = np.exp(beta * sr_reward) / (np.exp(beta * sr_reward) + np.exp(beta * sh_reward))
            aggregated_likelihood *= prob_sh_greater_than_sr

        if cond_2:
            prob_sr_greater_than_s0 = np.exp(beta * sr_reward) / (np.exp(beta * sr_reward) + np.exp(beta * s0_reward))
            aggregated_likelihood *= prob_sr_greater_than_s0

        if cond_3:
            prob_sh_greater_than_s0 = np.exp(beta * sh_reward) / (np.exp(beta * sh_reward) + np.exp(beta * s0_reward))
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


# Run the interaction loop
def run_interaction():
    hypothesis_reward_space = [f_Ethan, f_Michelle, f_Annika, f_Admoni, f_Simmons, f_Suresh, f_Ben, f_Ada, f_Abhijat, f_Maggie, f_Zulekha, f_Pat]

    # normalize hypotheses
    for tree_i in range(len(hypothesis_reward_space)):
        tree = hypothesis_reward_space[tree_i]
        all_key_combinations = []
        for key_1 in hypothesis_reward_space


    labels = ['Ethan', 'Michelle', 'Annika', 'Admoni', 'Simmons', 'Suresh', 'Ben', 'Ada', 'Abhijat', 'Maggie', 'Zulekha', 'Pat']
    true_reward_tree = f_Ethan
    object_type_tuple = [obj_1, obj_2]  # Define your object type tuple as per your requirements
    render_game = Gridworld(f_Ethan, object_type_tuple)
    initial_state = Gridworld(f_Ethan, object_type_tuple).get_initial_state()
    robot_beliefs = initialize_robot_beliefs(hypothesis_reward_space)

    plt.bar(np.arange(len(robot_beliefs)), height=robot_beliefs, tick_label=labels)
    plt.xticks(rotation=90)
    plt.show()

    state = initial_state
    render_game.render(state, -1)
    for t in range(len(object_type_tuple)):
        robot_action, new_state = get_weighted_robot_action(state, robot_beliefs, hypothesis_reward_space,
                                                            object_type_tuple)  # take one of the objects and move it to a quadrant
        print("new_state", new_state)
        render_game.render(new_state, t)
        # Render the state
        human_correction, new_corrected_state = get_correction_from_human(new_state, robot_action, true_reward_tree,
                                                                          object_type_tuple)
        render_game.render(new_corrected_state, t)
        # Render, color human actions differently
        s0_starting_state = copy.deepcopy(state) # starting current state
        sr_starting_state = copy.deepcopy(new_state) # robot moved state
        sh_starting_state = copy.deepcopy(new_corrected_state) # human corrected state
        robot_beliefs = update_robot_beliefs(s0_starting_state, sr_starting_state, sh_starting_state, robot_beliefs,
                                             hypothesis_reward_space, object_type_tuple)

        # state[] = new_corrected_state

        # Clarification question step can be added here
        # Clarification question step
        # ask_clarification_questions(new_state, new_corrected_state, object_type_tuple)

        plt.bar(np.arange(len(robot_beliefs)), height=robot_beliefs, tick_label=labels)
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