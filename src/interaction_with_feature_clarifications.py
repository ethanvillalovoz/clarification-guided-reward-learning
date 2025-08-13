"""
Clarification-Guided Reward Learning with Feature Explanations

Authors: Ethan Villalovz, Michelle Zhao
Project: RISS 2024 Summer Project - Bayesian Learning Interaction
Description: Incorporating the entire interaction between the human and robot object simulation. 
Updates the human preference through Bayesian inference after each time step.
"""

# Standard library imports
import os
import sys
import copy
import pdb
from pathlib import Path

# Third-party imports
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# Local imports
from multi_object_custom_mdp_v5 import *
try:
    from utils.console import log
except ImportError:
    # If utils module not found, use the logger from multi_object_custom_mdp_v5
    pass

def plot_robot_beliefs(beliefs, labels, title, filename=None, highlight_index=None):
    """
    Creates a visually appealing, research-grade plot of robot beliefs.
    
    Parameters:
    - beliefs: numpy array of belief probabilities
    - labels: list of labels for each belief
    - title: title of the plot
    - filename: if provided, save the figure to this file
    - highlight_index: index of belief to highlight (e.g., true model)
    """
    # Set styling for research-grade plots
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    
    # Create color palette - use professional blues with highlight
    if highlight_index is not None:
        colors = ['#1f77b4'] * len(beliefs)  # Default blue
        colors[highlight_index] = '#ff7f0e'  # Orange for highlight
    else:
        colors = sns.color_palette("Blues_d", len(beliefs))
    
    # Create the figure with appropriate size
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create barplot with enhanced styling
    x = np.arange(len(beliefs))
    bars = ax.bar(
        x, beliefs, 
        width=0.7,
        color=colors,
        edgecolor='black',
        linewidth=0.8,
        alpha=0.85
    )
    
    # Add belief values on top of bars (only for significant values)
    for i, v in enumerate(beliefs):
        if v >= 0.05:  # Only show labels for significant beliefs
            ax.text(
                i, v + 0.01,
                f'{v:.3f}',
                ha='center',
                fontsize=10,
                fontweight='bold' if highlight_index == i else 'normal'
            )
    
    # Add labels and title with better formatting
    ax.set_xlabel('Preference Models', fontweight='bold')
    ax.set_ylabel('Belief Probability', fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    
    # Set x-ticks and rotate labels for better readability
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Set y-axis to start at 0 and have a bit of padding at the top
    y_max = max(beliefs) * 1.15
    ax.set_ylim(0, y_max)
    
    # Add light horizontal grid lines
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Annotate the true model if highlighted
    if highlight_index is not None:
        ax.text(
            highlight_index, 
            beliefs[highlight_index] / 2,
            'True Model',
            ha='center',
            va='center',
            color='black',  # Changed from white to black for better visibility
            fontweight='bold',
            fontsize=11     # Slightly larger font size
        )
        
    # Add a subtle box around the plot
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save if filename is provided
    if filename:
        # Save in the source-code/beliefs/ directory
        import os
        # Get the source-code directory path
        source_code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        beliefs_dir = os.path.join(source_code_dir, 'beliefs')
        
        # Create directory if it doesn't exist
        if not os.path.exists(beliefs_dir):
            os.makedirs(beliefs_dir)
            
        # Save to the beliefs directory with the provided filename
        save_path = os.path.join(beliefs_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log.info(f"📊 Saved belief visualization to {save_path}", color="cyan")
        
    return fig, ax


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
    tree = hypothesis_reward_space[tree_idx]
    log.debug(f"Robot using preference model: {tree_idx}")
    
    # Create a fresh policy using the current state
    tree_policy = Gridworld(tree, object_type_tuple)
    tree_policy.current_state = copy.deepcopy(state)
    
    # Compute optimal policy
    optimal_rew, game_results, sum_feature_vector = tree_policy.compute_optimal_performance()
    
    # Create a list of tuples in the correct order based on object_type_tuple
    ordered_object_tuples = []
    for obj in object_type_tuple:
        obj_tuple = (
            COLORS[obj['color']],
            MATERIALS[obj['material']],
            OBJECTS[obj['object_type']],
            obj['object_label']
        )
        ordered_object_tuples.append(obj_tuple)
    
    if timestep >= len(ordered_object_tuples):
        log.success("✓ All objects have been processed", bold=True)
        return EXIT, state
        
    # Get the current object to move based on timestep
    current_object = ordered_object_tuples[timestep]
    log.section(f"TIMESTEP {timestep}", color="blue", bold=True)
    log.info(f"Current object: {current_object}", bold=True)
    
    # Get the next action from game_results
    if len(game_results) > 0:
        # Extract the action from game results
        next_state = game_results[0][0]  # First step's resulting state
        action = game_results[0][1]      # First step's action
        
        log.debug(f"Robot initial state: {state}", color="gray")
        
        # Get the object type, color, and material details
        color_idx, material_idx, object_idx, object_label = current_object
        object_type = OBJECTS_IDX[object_idx]
        color = COLORS_IDX[color_idx]
        material = MATERIALS_IDX[material_idx]
        
        log.subsection("ROBOT ACTION", color="green")
        log.info(f"Moving: {color} {material} {object_type}", bold=True)
        
        # Get best quadrant from the selected tree's preferences
        best_quadrant = 'Q1'  # Default
        best_reward = float('-inf')
        
        # Navigate the preference tree to find the best quadrant
        try:
            if object_type in tree['pref_values']:
                if color in tree['pref_values'][object_type]:
                    # Case: tree has object_type -> color -> quadrants
                    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                        if tree['pref_values'][object_type][color][q] > best_reward:
                            best_reward = tree['pref_values'][object_type][color][q]
                            best_quadrant = q
                elif material in tree['pref_values'][object_type]:
                    # Case: tree has object_type -> material -> quadrants
                    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                        if tree['pref_values'][object_type][material][q] > best_reward:
                            best_reward = tree['pref_values'][object_type][material][q]
                            best_quadrant = q
            elif material in tree['pref_values']:
                if color in tree['pref_values'][material]:
                    # Case: tree has material -> color -> quadrants
                    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                        if tree['pref_values'][material][color][q] > best_reward:
                            best_reward = tree['pref_values'][material][color][q]
                            best_quadrant = q
        except (KeyError, TypeError):
            # If there's any error navigating the tree, use default
            log.warning(f"Could not find exact preference for {color} {material} {object_type}, using default")
        
        action_quadrant = best_quadrant
        log.info(f"Robot chose {action_quadrant} based on preferences", indent=2)
        
        # Create the action tuple
        action = (current_object, action_quadrant)
        
        # Handle other cases
        if isinstance(action, tuple) and len(action) == 2:
            _, action_quadrant = action
        
        log.debug(f"Robot action detail: {action}", indent=2)
        log.debug(f"Robot next state: {next_state}", indent=2, color="gray")
        
        # Create a new state with the action applied
        action_obj, action_quadrant = None, None
        if isinstance(action, tuple) and len(action) == 2:
            action_obj, action_quadrant = action
        else:
            # Handle case where action is just a string (like EXIT)
            return action, state
            
        # Apply the action to create a new state
        actual_new_state = copy.deepcopy(state)
        # Set the object as done to indicate it has been moved
        if action_obj in actual_new_state:
            # Use the object's index to generate a unique position within the quadrant
            object_index = action_obj[3]  # Use object_label for positioning
            
            # Calculate position based on quadrant and object index
            # Ensure positions are integers within the quadrant, not on axes
            if action_quadrant == 'Q1':  # Positive x, positive y
                if object_index == 1:  # First object (yellow cup)
                    target_pos = (2, 3)
                elif object_index == 2:  # Second object (red cup)
                    target_pos = (3, 2)
                elif object_index == 3:  # Third object (purple bowl)
                    target_pos = (4, 4)
                else:
                    target_pos = (3, 3)
            elif action_quadrant == 'Q2':  # Negative x, positive y
                if object_index == 1:  # First object
                    target_pos = (-2, 3)
                elif object_index == 2:  # Second object
                    target_pos = (-3, 2)
                elif object_index == 3:  # Third object
                    target_pos = (-4, 4)
                else:
                    target_pos = (-3, 3)
            elif action_quadrant == 'Q3':  # Negative x, negative y
                if object_index == 1:  # First object
                    target_pos = (-2, -3)
                elif object_index == 2:  # Second object
                    target_pos = (-3, -2)
                elif object_index == 3:  # Third object
                    target_pos = (-4, -4)
                else:
                    target_pos = (-3, -3)
            elif action_quadrant == 'Q4':  # Positive x, negative y
                if object_index == 1:  # First object
                    target_pos = (2, -3)
                elif object_index == 2:  # Second object
                    target_pos = (3, -2)
                elif object_index == 3:  # Third object
                    target_pos = (4, -4)
                else:
                    target_pos = (3, -3)
                
            actual_new_state[action_obj]['pos'] = target_pos
            actual_new_state[action_obj]['done'] = True
            
        log.debug(f"Action quadrant: {action_quadrant}", indent=2)
        return action_quadrant, actual_new_state
    else:
        # Handle case where no actions were found
        log.error("No actions found in game_results")
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
    # Create a list of tuples in the correct order based on object_type_tuple
    ordered_object_tuples = []
    for obj in object_type_tuple:
        obj_tuple = (
            COLORS[obj['color']],
            MATERIALS[obj['material']],
            OBJECTS[obj['object_type']],
            obj['object_label']
        )
        ordered_object_tuples.append(obj_tuple)
    
    if timestep >= len(ordered_object_tuples):
        log.success("✓ All objects have been processed by human", bold=True)
        return EXIT, new_state
        
    current_object = ordered_object_tuples[timestep]
    log.subsection("HUMAN CORRECTION", color="red")
    log.info(f"Correcting: {current_object}", bold=True)
    
    # Create a fresh policy using the human's preferred tree
    tree_policy = Gridworld(true_reward_tree, object_type_tuple)
    tree_policy.current_state = copy.deepcopy(new_state)
    
    # Compute optimal policy
    optimal_rew, game_results, sum_feature_vector = tree_policy.compute_optimal_performance()
    
    # Get the object type, color, and material details
    color_idx, material_idx, object_idx, object_label = current_object
    object_type = OBJECTS_IDX[object_idx]
    color = COLORS_IDX[color_idx]
    material = MATERIALS_IDX[material_idx]
    
    log.info(f"Object details: {color} {material} {object_type}", indent=2)
    
    # Get best quadrant from the human's true reward tree
    best_quadrant = 'Q1'  # Default
    best_reward = float('-inf')
    
    # Navigate the preference tree to find the best quadrant
    try:
        if object_type in true_reward_tree['pref_values']:
            if color in true_reward_tree['pref_values'][object_type]:
                # Case: tree has object_type -> color -> quadrants
                for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                    if true_reward_tree['pref_values'][object_type][color][q] > best_reward:
                        best_reward = true_reward_tree['pref_values'][object_type][color][q]
                        best_quadrant = q
            elif material in true_reward_tree['pref_values'][object_type]:
                # Case: tree has object_type -> material -> quadrants
                for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                    if true_reward_tree['pref_values'][object_type][material][q] > best_reward:
                        best_reward = true_reward_tree['pref_values'][object_type][material][q]
                        best_quadrant = q
        elif material in true_reward_tree['pref_values']:
            if color in true_reward_tree['pref_values'][material]:
                # Case: tree has material -> color -> quadrants
                for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                    if true_reward_tree['pref_values'][material][color][q] > best_reward:
                        best_reward = true_reward_tree['pref_values'][material][color][q]
                        best_quadrant = q
    except (KeyError, TypeError):
        # If there's any error navigating the tree, use default
        log.warning(f"Could not find exact preference for {color} {material} {object_type}, using default")
    
    action_quadrant = best_quadrant
    log.info(f"Human chose {action_quadrant} based on true preferences", indent=2)
    
    log.debug(f"Current state: {new_state}", indent=2, color="gray")
    log.debug(f"Corrected action: ({current_object}, {action_quadrant})", indent=2)
    
    # Apply the correction to create a new state
    actual_corrected_state = copy.deepcopy(new_state)
    
    # Use the object's index to generate a unique position within the quadrant
    object_index = current_object[3]  # Use object_label for positioning
    
    # Calculate position based on quadrant and object index
    # Ensure positions are integers within the quadrant, not on axes
    if action_quadrant == 'Q1':  # Positive x, positive y
        if object_index == 1:  # First object (yellow cup)
            target_pos = (2, 3)
        elif object_index == 2:  # Second object (red cup)
            target_pos = (3, 2) 
        elif object_index == 3:  # Third object (purple bowl)
            target_pos = (4, 4)
        else:
            target_pos = (3, 3)
    elif action_quadrant == 'Q2':  # Negative x, positive y
        if object_index == 1:  # First object
            target_pos = (-2, 3)
        elif object_index == 2:  # Second object
            target_pos = (-3, 2)
        elif object_index == 3:  # Third object
            target_pos = (-4, 4)
        else:
            target_pos = (-3, 3)
    elif action_quadrant == 'Q3':  # Negative x, negative y
        if object_index == 1:  # First object
            target_pos = (-2, -3)
        elif object_index == 2:  # Second object
            target_pos = (-3, -2)
        elif object_index == 3:  # Third object
            target_pos = (-4, -4)
        else:
            target_pos = (-3, -3)
    elif action_quadrant == 'Q4':  # Positive x, negative y
        if object_index == 1:  # First object
            target_pos = (2, -3)
        elif object_index == 2:  # Second object
            target_pos = (3, -2)
        elif object_index == 3:  # Third object
            target_pos = (4, -4)
        else:
            target_pos = (3, -3)
        
    # Apply the position change
    actual_corrected_state[current_object]['pos'] = target_pos
    actual_corrected_state[current_object]['done'] = True
    
    log.debug(f"Action quadrant: {action_quadrant}", indent=2)
    log.debug(f"Next state with correction: {actual_corrected_state}", indent=2, color="gray")
    return action_quadrant, actual_corrected_state


def get_correction_from_human_keyboard_input(new_state, timestep, robot_action, object_type_tuple, object_tuples):
    current_object = object_tuples[timestep] # (2,1,1)
    # current_object_description = get_description(current_object) # TODO
    selected_quadrant = input(f"for the object that just moved {current_object}, which quadrant should it be in ([Q1, Q2, Q3, Q4]?")
    # selected_quadrant is going to be string like 'Q1'
    log.info(f"Human selected quadrant: {selected_quadrant}", indent=2)

    empty_reward_tree = {}

    tree_policy = Gridworld(empty_reward_tree, object_type_tuple, new_state)
    action = (current_object, selected_quadrant)
    next_state, team_rew, done = tree_policy.step_given_state(new_state,  action)
    log.debug(f"Current state: {new_state}", indent=2, color="gray")
    log.debug(f"Corrected action: {selected_quadrant}", indent=2)
    log.debug(f"Next state: {next_state}", indent=2, color="gray")
    action = selected_quadrant

    # pdb.set_trace()

    log.debug(f"Action detail: {action}", indent=2)
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
        log.debug(f"Processing belief for tree index: {tree_idx}", color="blue")
        prior_belief_of_theta_i = robot_beliefs[tree_idx]  # P(theta_i)

        tree = hypothesis_reward_space[tree_idx]
        tree_policy = Gridworld(tree, object_type_tuple)
        # tree_policy.compute_optimal_performance(render=False)
        s0_reward = tree_policy.lookup_quadrant_reward(s0_starting_state)
        sr_reward = tree_policy.lookup_quadrant_reward(sr_state)
        sh_reward = tree_policy.lookup_quadrant_reward(sh_state)

        log.debug(f"Initial state reward: {s0_reward:.4f}", indent=2, color="gray")
        log.debug(f"Robot state reward: {sr_reward:.4f}", indent=2, color="green")
        log.debug(f"Human corrected state reward: {sh_reward:.4f}", indent=2, color="red")

        # Compute the likelihood of human correction given robot action and the hypothesis tree
        # likelihood = tree_policy.compute_likelihood(state, robot_action, human_correction)
        # likelihoods.append(likelihood)
        aggregated_likelihood = 1  # P(all d| tree theta_i)

        # print conditions
        log.debug("Bayesian update conditions:", indent=2)
        log.debug(f"• Human > Robot: {cond_1}", indent=4, color=("green" if cond_1 else "red"))
        log.debug(f"• Robot > Initial: {cond_2}", indent=4, color=("green" if cond_2 else "red"))
        log.debug(f"• Human > Initial: {cond_3}", indent=4, color=("green" if cond_3 else "red"))

        if cond_1:
            beta_cond_1 = 5
            prob_sh_greater_than_sr = np.exp(beta_cond_1 * sh_reward) / (np.exp(beta_cond_1 * sr_reward) + np.exp(beta_cond_1 * sh_reward))
            log.debug(f"P(Human > Robot): {prob_sh_greater_than_sr:.4f}", indent=4, color="blue")

            aggregated_likelihood *= prob_sh_greater_than_sr

        if cond_2:
            beta_cond_2 = 0.5
            prob_sr_greater_than_s0 = np.exp(beta_cond_2 * sr_reward) / (np.exp(beta_cond_2 * sr_reward) + np.exp(beta_cond_2 * s0_reward))
            log.debug(f"P(Robot > Initial): {prob_sr_greater_than_s0:.4f}", indent=4, color="blue")
            aggregated_likelihood *= prob_sr_greater_than_s0

        if cond_3:
            beta_cond_3 = 0.5
            prob_sh_greater_than_s0 = np.exp(beta_cond_3 * sh_reward) / (np.exp(beta_cond_3 * sh_reward) + np.exp(beta_cond_3 * s0_reward))
            log.debug(f"P(Human > Initial): {prob_sh_greater_than_s0:.4f}", indent=4, color="blue")
            aggregated_likelihood *= prob_sh_greater_than_s0

        prob_theta_i_given_data = aggregated_likelihood * prior_belief_of_theta_i
        new_beliefs.append(prob_theta_i_given_data)

    log.subsection("BELIEF UPDATE", color="magenta")
    # Display updated beliefs in a formatted table
    log.beliefs_table(new_beliefs, [f"Model {i}" for i in range(len(new_beliefs))], 
                     indent=2, title="Updated Robot Beliefs:")
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
        log.subsection("CLARIFICATION QUESTION", color="yellow", bold=True)
        log.info(question, indent=2, color="yellow")
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
        log.debug(f"Examining hypothesis index: {hyp_idx}", indent=2, color="blue")
        hypothesis_tree = hypothesis_reward_space[hyp_idx]
        pref_tree = hypothesis_tree['pref_values']
        quadrant_preference_tree, relevant_features = get_relevant_features(pref_tree, attributes, [])
        log.debug(f"Identified relevant features: {relevant_features}", indent=2)
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

    log.info(f"True relevant features: {true_relevant_features}", indent=2, color="green", bold=True)

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
    """
    Main interaction loop for the clarification-guided reward learning system.
    This function simulates the interaction between a robot and human where the robot
    learns human preferences through observations, corrections, and asking clarifying questions.
    """
    log.section("INITIALIZING CLARIFICATION-GUIDED REWARD LEARNING", color="blue", bold=True, 
                top_line=True, bottom_line=True)
    
    # Initialize the hypothesis space and belief system
    hypothesis_reward_space = [f_Ethan, f_Michelle, f_Annika, f_Admoni, f_Simmons, f_Suresh, f_Ben, f_Ada, f_Abhijat,
                               f_Maggie, f_Zulekha, f_Pat]
    labels = ['Ethan', 'Michelle', 'Annika', 'Admoni', 'Simmons', 'Suresh', 'Ben', 'Ada', 'Abhijat', 'Maggie',
              'Zulekha', 'Pat']
    true_reward_tree = f_Ethan  # Ground truth preference model
    
    # Define objects to be placed
    list_of_present_object_tuples = [obj_1, obj_2, obj_3]  # Define your object type tuple as per your requirements
    
    log.info(f"Loaded {len(hypothesis_reward_space)} preference models")
    log.info(f"True preference model: {labels[hypothesis_reward_space.index(true_reward_tree)]}")
    log.info(f"Objects to place: {len(list_of_present_object_tuples)}")
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

    # Initialize robot belief system
    log.section("INITIALIZING ROBOT BELIEFS", color="magenta")
    robot_beliefs = initialize_robot_beliefs(hypothesis_reward_space)
    
    # Display belief distribution as a nicely formatted table
    log.beliefs_table(robot_beliefs, labels, indent=1, title="Initial Belief Distribution:")
    
    # Use enhanced visualization for robot beliefs
    log.info("Generating belief visualization...", color="cyan")
    plot_robot_beliefs(robot_beliefs, labels, 'Initial Robot Beliefs', 
                      filename='initial_beliefs.png',
                      highlight_index=0)  # Assuming Ethan (index 0) is the true model
    plt.show()

    state = initial_state
    # Create a list of tuples in the correct order based on the object definition order
    # This ensures we process objects in the order they were defined, not based on their state keys
    object_tuples = []
    for obj in list_of_present_object_tuples:
        obj_tuple = (
            COLORS[obj['color']],
            MATERIALS[obj['material']],
            OBJECTS[obj['object_type']],
            obj['object_label']
        )
        object_tuples.append(obj_tuple)
        
    log.info(f"Processing objects in order: {object_tuples}")
    
    # Main interaction loop
    for t in range(len(list_of_present_object_tuples)):
        log.section(f"TIME STEP {t}: {COLORS_IDX.get(object_tuples[t][0], '').upper()} {OBJECTS_IDX.get(object_tuples[t][2], '').upper()}")
        log.info("Initial state - awaiting robot action")
        render_game.render(state, t, "initial", object_tuples[t])
        
        # Robot takes action
        robot_action, new_state = get_weighted_robot_action(state, t, robot_beliefs, hypothesis_reward_space,
                                                            list_of_present_object_tuples)
        
        if robot_action == EXIT:
            log.info("Robot chose to exit - no more actions to take", color="blue", bold=True)
            break
            
        log.debug(f"Current state: {new_state}", indent=2, color="gray")
        log.subsection("Robot action completed")
        object_desc = f"{COLORS_IDX.get(object_tuples[t][0], '')} {MATERIALS_IDX.get(object_tuples[t][1], '')} {OBJECTS_IDX.get(object_tuples[t][2], '')}"
        # Get the quadrant from the new position
        quadrant = "unknown"
        pos = new_state[object_tuples[t]]['pos']
        if pos[0] > 0 and pos[1] > 0:
            quadrant = "Q1"
        elif pos[0] < 0 and pos[1] > 0:
            quadrant = "Q2"
        elif pos[0] < 0 and pos[1] < 0:
            quadrant = "Q3"
        elif pos[0] > 0 and pos[1] < 0:
            quadrant = "Q4"
        log.info(f"Robot placed {object_desc} in quadrant {quadrant}")
        render_game.render(new_state, t, "robot_moved", object_tuples[t])
        
        # Prepare state for human correction
        # We need to preserve the current object's "done" status for visualization
        # but mark it as not done for action taking purposes
        new_state_reset_obj = copy.deepcopy(new_state)
        current_object = object_tuples[t]
        # Only reset the done flag for the current object being moved
        if current_object in new_state_reset_obj:
            new_state_reset_obj[current_object]['done'] = False

        # Get correction from human - either automated or keyboard input
        # Uncomment this for keyboard input:
        # human_correction, new_corrected_state = get_correction_from_human_keyboard_input(
        #     new_state_reset_obj, t, robot_action, list_of_present_object_tuples, object_tuples)
        
        # Get automated correction from the true reward tree
        human_correction, new_corrected_state = get_correction_from_human(
            new_state_reset_obj, t, robot_action, true_reward_tree, list_of_present_object_tuples)

        if human_correction == EXIT:
            log.info("Human chose to exit - no more actions to take", color="red", bold=True)
            break
            
        log.subsection("Human correction applied")
        # Get the quadrant from the new corrected position
        quadrant = "unknown"
        pos = new_corrected_state[object_tuples[t]]['pos']
        if pos[0] > 0 and pos[1] > 0:
            quadrant = "Q1"
        elif pos[0] < 0 and pos[1] > 0:
            quadrant = "Q2"
        elif pos[0] < 0 and pos[1] < 0:
            quadrant = "Q3"
        elif pos[0] > 0 and pos[1] < 0:
            quadrant = "Q4"
        log.info(f"Human placed {object_desc} in quadrant {quadrant}")
        render_game.render(new_corrected_state, t, "human_corrected", object_tuples[t])
        log.info("Updating robot beliefs based on human feedback...")

        # Update robot beliefs based on the states
        s0_starting_state = copy.deepcopy(state)          # starting current state
        sr_starting_state = copy.deepcopy(new_state)      # robot moved state
        sh_starting_state = copy.deepcopy(new_corrected_state)  # human corrected state
        robot_beliefs = update_robot_beliefs(s0_starting_state, sr_starting_state, sh_starting_state, robot_beliefs,
                                             hypothesis_reward_space, list_of_present_object_tuples)
                                             
        # Update the state for next iteration
        state = copy.deepcopy(new_corrected_state)

        # state[] = new_corrected_state
        # Enhanced visualization prior to clarification question
        plot_robot_beliefs(robot_beliefs, labels, 
                          f'Robot Beliefs (prior to clarification, timestep {t})', 
                          filename=f'beliefs_prior_clarification_t{t}.png',
                          highlight_index=0)  # Assuming Ethan (index 0) is the true model
        plt.show()

        # Clarification question step can be added here
        # Clarification question step
        # ask_clarification_questions(new_state, new_corrected_state, object_type_tuple)
        new_robot_beliefs = ask_feature_clarification_question(robot_beliefs, hypothesis_reward_space,
                                                               s0_starting_state, sr_starting_state, sh_starting_state, t,
                                                               current_object)
        robot_beliefs = new_robot_beliefs

        # Enhanced visualization for after clarification
        plot_robot_beliefs(robot_beliefs, labels, 
                          f'Robot Beliefs (after clarification, timestep {t})', 
                          filename=f'beliefs_after_clarification_t{t}.png',
                          highlight_index=0)  # Assuming Ethan (index 0) is the true model
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
