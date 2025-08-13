"""
Human-Robot Preference Learning Interaction Module

This module implements the interaction loop between a human and robot in a preference learning scenario.
The robot learns human preferences through Bayesian inference after observing human corrections to the
robot's object placements in a gridworld environment.

Authors: Ethan Villalovoz, Michelle Zhao
Project: RISS 2024 Summer Project - Clarification-Guided Reward Learning
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
from multi_object_custom_mdp_v5 import (
    Gridworld, f_Ethan, f_Michelle, f_Annika, f_Admoni, 
    f_Simmons, f_Suresh, f_Ben, f_Ada, f_Abhijat,
    f_Maggie, f_Zulekha, f_Pat, obj_1, obj_2
)


def initialize_robot_beliefs(hypothesis_reward_space):
    """
    Initialize robot beliefs over reward hypotheses with a uniform distribution.
    
    Args:
        hypothesis_reward_space (list): List of reward function hypotheses
        
    Returns:
        np.ndarray: Initial uniform belief distribution
    """
    return np.ones(len(hypothesis_reward_space)) / len(hypothesis_reward_space)


def get_weighted_robot_action(state, robot_beliefs, hypothesis_reward_space, object_type_tuple):
    """
    Select an action for the robot based on current beliefs over reward functions.
    
    Args:
        state (dict): Current state of the environment
        robot_beliefs (np.ndarray): Current belief distribution over reward hypotheses
        hypothesis_reward_space (list): List of reward function hypotheses
        object_type_tuple (list): List of objects in the environment
        
    Returns:
        tuple: (action as quadrant label, new state after action)
    """
    # Sample a reward function from the belief distribution
    tree_idx = np.random.choice(np.arange(len(robot_beliefs)), p=robot_beliefs)
    print("tree index:", tree_idx)
    tree = hypothesis_reward_space[tree_idx]
    
    # Create gridworld with sampled reward function
    tree_policy = Gridworld(tree, object_type_tuple)
    
    # Compute optimal policy and get results
    optimal_rew, game_results, sum_feature_vector = tree_policy.compute_optimal_performance(render=False)
    print("game_results", game_results)
    
    # Extract new state and action from results
    new_state = game_results[-1][0]
    objects = list(game_results[-1][0].keys())
    action = game_results[-1][0][objects[0]]['pos']

    # Convert position coordinates to quadrant labels
    if int(action[0]) > 0 and int(action[1]) > 0:
        action = 'Q1'  # Top-right quadrant
    elif int(action[0]) < 0 and int(action[1]) > 0:
        action = 'Q2'  # Top-left quadrant
    elif int(action[0]) < 0 and int(action[1]) < 0:
        action = 'Q3'  # Bottom-left quadrant
    elif int(action[0]) > 0 and int(action[1]) < 0:
        action = 'Q4'  # Bottom-right quadrant

    print(f"Robot selected quadrant: {action}")
    return action, new_state


def get_correction_from_human(new_state, robot_action, true_reward_tree, object_type_tuple):
    """
    Simulate human correction based on the true reward function.
    
    Args:
        new_state (dict): State after robot action
        robot_action (str): Action taken by the robot
        true_reward_tree (dict): True reward function of the human
        object_type_tuple (list): List of objects in the environment
        
    Returns:
        tuple: (human correction as quadrant label, corrected state)
    """
    # Create gridworld with true reward function
    tree_policy = Gridworld(true_reward_tree, object_type_tuple)
    
    # Compute optimal policy and get results
    optimal_rew, game_results, sum_feature_vector = tree_policy.compute_optimal_performance()
    print(f"Human optimal policy results: {game_results}")
    
    # Extract corrected state and action from results
    corrected_state = game_results[-1][0]
    objects = list(game_results[-1][0].keys())
    action = game_results[-1][0][objects[0]]['pos']

    # Convert position coordinates to quadrant labels
    if int(action[0]) > 0 and int(action[1]) > 0:
        action = 'Q1'
    elif int(action[0]) < 0 and int(action[1]) > 0:
        action = 'Q2'
    elif int(action[0]) < 0 and int(action[1]) < 0:
        action = 'Q3'
    elif int(action[0]) > 0 and int(action[1]) < 0:
        action = 'Q4'

    print(f"Human preferred quadrant: {action}")
    return action, corrected_state


def update_robot_beliefs(s0_starting_state, sr_state, sh_state, robot_beliefs,
                         hypothesis_reward_space, object_type_tuple):
    """
    Update robot beliefs using Bayesian inference based on human corrections.
    
    Args:
        s0_starting_state (dict): Initial state before robot action
        sr_state (dict): State after robot action
        sh_state (dict): State after human correction
        robot_beliefs (np.ndarray): Current belief distribution
        hypothesis_reward_space (list): List of reward function hypotheses
        object_type_tuple (list): List of objects in the environment
        
    Returns:
        np.ndarray: Updated belief distribution
    """
    # Check which conditions apply based on state comparisons
    cond_1 = sh_state != sr_state  # Human corrected robot's action
    cond_2 = sr_state == sh_state and s0_starting_state != sr_state  # Human agreed with robot's action
    cond_3 = sh_state != s0_starting_state  # Human preferred a different state than initial

    # Rationality parameter (higher beta means more rational/optimal decisions)
    beta = 5
    new_beliefs = []
    
    for tree_idx in range(len(hypothesis_reward_space)):
        # Get prior belief for this hypothesis
        prior_belief_of_theta_i = robot_beliefs[tree_idx]
        
        # Get reward function and policy
        tree = hypothesis_reward_space[tree_idx]
        tree_policy = Gridworld(tree, object_type_tuple)
        
        # Look up rewards for each state under this hypothesis
        s0_reward = tree_policy.lookup_quadrant_reward(s0_starting_state)
        sr_reward = tree_policy.lookup_quadrant_reward(sr_state)
        sh_reward = tree_policy.lookup_quadrant_reward(sh_state)

        # Calculate likelihood using softmax model of human preferences
        aggregated_likelihood = 1  # P(all d | tree theta_i)
        
        if cond_1:  # Human corrected robot's action
            prob_sh_greater_than_sr = np.exp(beta * sh_reward) / (np.exp(beta * sr_reward) + np.exp(beta * sh_reward))
            aggregated_likelihood *= prob_sh_greater_than_sr

        if cond_2:  # Human agreed with robot's action
            prob_sr_greater_than_s0 = np.exp(beta * sr_reward) / (np.exp(beta * sr_reward) + np.exp(beta * s0_reward))
            aggregated_likelihood *= prob_sr_greater_than_s0

        if cond_3:  # Human preferred different state than initial
            prob_sh_greater_than_s0 = np.exp(beta * sh_reward) / (np.exp(beta * sh_reward) + np.exp(beta * s0_reward))
            aggregated_likelihood *= prob_sh_greater_than_s0

        # Calculate posterior probability: P(θ|d) ∝ P(d|θ) * P(θ)
        prob_theta_i_given_data = aggregated_likelihood * prior_belief_of_theta_i
        new_beliefs.append(prob_theta_i_given_data)

    # Normalize beliefs to get proper probability distribution
    new_beliefs = np.array(new_beliefs)
    new_beliefs /= np.sum(new_beliefs)
    print(f"Updated beliefs: {new_beliefs}")
    
    return new_beliefs


def ask_clarification_questions(new_state, corrected_state, object_type_tuple):
    """
    Generate and ask clarification questions to improve understanding of human preferences.
    
    Args:
        new_state (dict): State after robot action
        corrected_state (dict): State after human correction
        object_type_tuple (list): List of objects in the environment
    """
    # Define categories of clarification questions
    questions = {
        "preference": [
            "Why did you prefer this new position over the one I chose?",
            "Can you explain why this location is better?"
        ],
        "attribute": [
            "Which attribute of the object influenced your correction the most?",
            "Was the position of the object the main reason for your correction, or was it something else?"
        ],
        "future": [
            "How would you like me to position similar objects in the future?",
            "Are there specific rules I should follow when placing objects like this?"
        ],
        "hypothesis": [
            "I think you prefer objects to be placed closer to the center. Is that correct?",
            "It seems like you prefer objects to be in quadrant Q1. Is that true for all objects?"
        ]
    }

    # Print a question from each category
    print("\n--- Clarification Questions ---")
    for category, category_questions in questions.items():
        print(f"\n{category.capitalize()} question:")
        print(f"  {category_questions[0]}")
        # In a real implementation, we would wait for and process human responses


def normalize_reward_hypotheses(hypothesis_reward_space):
    """
    Normalize reward values in all hypotheses to improve numerical stability.
    
    Args:
        hypothesis_reward_space (list): List of reward function hypotheses
        
    Returns:
        list: Normalized reward hypotheses
    """
    # This is a placeholder for the normalization logic
    # The actual implementation would recursively traverse the hypothesis trees
    # and normalize the reward values
    return hypothesis_reward_space


def run_interaction():
    """
    Run the complete interaction loop between human and robot for preference learning.
    """
    # Define the hypothesis space (different possible reward functions)
    hypothesis_reward_space = [
        f_Ethan, f_Michelle, f_Annika, f_Admoni, f_Simmons, 
        f_Suresh, f_Ben, f_Ada, f_Abhijat, f_Maggie, f_Zulekha, f_Pat
    ]
    
    # Labels for visualization
    labels = [
        'Ethan', 'Michelle', 'Annika', 'Admoni', 'Simmons', 
        'Suresh', 'Ben', 'Ada', 'Abhijat', 'Maggie', 'Zulekha', 'Pat'
    ]
    
    # Set the true reward function (in real deployment, this would be unknown)
    true_reward_tree = f_Ethan
    
    # Define the objects in the environment
    object_type_tuple = [obj_1, obj_2]
    
    # Initialize gridworld and state
    render_game = Gridworld(f_Ethan, object_type_tuple)
    initial_state = render_game.get_initial_state()
    
    # Initialize robot's beliefs
    robot_beliefs = initialize_robot_beliefs(hypothesis_reward_space)

    # Visualize initial belief distribution
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(robot_beliefs)), height=robot_beliefs, tick_label=labels)
    plt.xticks(rotation=90)
    plt.title("Initial Robot Beliefs")
    plt.tight_layout()
    plt.show()

    # Initialize state
    state = initial_state
    print("--- Initial state ---")
    render_game.render(state, -1)
    
    # Main interaction loop (one iteration per object)
    for t in range(len(object_type_tuple)):
        print(f"\n--- Interaction round {t+1} ---")
        
        # 1. Robot takes action based on current beliefs
        print("\n1. Robot selects action based on current beliefs:")
        robot_action, new_state = get_weighted_robot_action(
            state, robot_beliefs, hypothesis_reward_space, object_type_tuple
        )
        render_game.render(new_state, t)
        
        # 2. Human provides correction
        print("\n2. Human provides correction:")
        human_correction, new_corrected_state = get_correction_from_human(
            new_state, robot_action, true_reward_tree, object_type_tuple
        )
        render_game.render(new_corrected_state, t)
        
        # 3. Robot updates beliefs based on correction
        print("\n3. Robot updates beliefs:")
        s0_starting_state = copy.deepcopy(state)  # starting state
        sr_starting_state = copy.deepcopy(new_state)  # robot moved state
        sh_starting_state = copy.deepcopy(new_corrected_state)  # human corrected state
        
        robot_beliefs = update_robot_beliefs(
            s0_starting_state, sr_starting_state, sh_starting_state,
            robot_beliefs, hypothesis_reward_space, object_type_tuple
        )
        
        # 4. Optional: Ask clarification questions
        # ask_clarification_questions(new_state, new_corrected_state, object_type_tuple)
        
        # 5. Visualize updated beliefs
        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(len(robot_beliefs)), height=robot_beliefs, tick_label=labels)
        plt.xticks(rotation=90)
        plt.title(f"Robot Beliefs After Interaction {t+1}")
        plt.tight_layout()
        plt.show()
        
        # Update state for next iteration
        state = new_corrected_state


if __name__ == '__main__':
    """
    Entry point for the interaction system.
    """
    run_interaction()