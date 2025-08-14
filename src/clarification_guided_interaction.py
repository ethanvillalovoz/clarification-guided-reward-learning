"""
Clarification-Guided Reward Learning with Feature Explanations

This module implements an interactive robot learning system that learns human preferences
through a combination of demonstrations, corrections, and feature clarification questions.
The system maintains and updates a Bayesian belief distribution over potential preference models.

Authors: Ethan Villalovoz, Michelle Zhao
Project: RISS 2024 Summer Project - Bayesian Learning Interaction
License: MIT License
Version: 1.0.0
"""

# Standard library imports
import os
import copy
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from multi_object_mdp import (
    Gridworld, COLORS, COLORS_IDX, MATERIALS, MATERIALS_IDX, OBJECTS, OBJECTS_IDX,
    f_Ethan, f_Michelle, f_Annika, f_Admoni, f_Simmons, f_Suresh, f_Ben, f_Ada,
    f_Abhijat, f_Maggie, f_Zulekha, f_Pat, obj_1, obj_2, obj_3, EXIT
)

try:
    from utils.console import log
except ImportError:
    # If utils module not found, use the logger from multi_object_custom_mdp_v5
    from multi_object_mdp import log

# Type aliases for better code readability
State = Dict[Tuple, Dict[str, Any]]
ObjectTuple = Tuple[int, int, int, int]  # (color_idx, material_idx, object_idx, object_label)
ObjectConfig = Dict[str, Union[str, int]]
PreferenceTree = Dict[str, Any]
Beliefs = np.ndarray

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
from multi_object_mdp import *
try:
    from utils.console import log
except ImportError:
    # If utils module not found, use the logger from multi_object_custom_mdp_v5
    pass

# ============================
# Visualization Functions
# ============================

def plot_robot_beliefs(beliefs: np.ndarray, 
                      labels: List[str], 
                      title: str, 
                      filename: Optional[str] = None, 
                      highlight_index: Optional[int] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Creates a visually appealing, research-grade plot of robot beliefs.
    
    This function generates a bar chart visualization of the robot's belief distribution
    over different preference models, with optional highlighting of the true model.
    
    Parameters:
    -----------
    beliefs : np.ndarray
        Numpy array of belief probabilities, should sum to 1.0
    labels : List[str]
        List of labels for each belief (model names)
    title : str
        Title of the plot
    filename : Optional[str]
        If provided, save the figure to this file path
    highlight_index : Optional[int]
        Index of belief to highlight (e.g., true model)
        
    Returns:
    --------
    Tuple[plt.Figure, plt.Axes]
        The matplotlib figure and axes objects for further customization if needed
    """
    # Validate inputs
    if len(beliefs) != len(labels):
        raise ValueError(f"Length mismatch: beliefs ({len(beliefs)}) and labels ({len(labels)})")
    
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


# ============================
# Belief Initialization and Update
# ============================

def initialize_robot_beliefs(hypothesis_reward_space: List[PreferenceTree]) -> Beliefs:
    """
    Initialize the robot's belief distribution over hypothesis space.
    
    Parameters:
    -----------
    hypothesis_reward_space : List[PreferenceTree]
        List of preference models to consider
        
    Returns:
    --------
    Beliefs
        Uniform distribution over the hypothesis space
    """
    # Initialize with uniform prior (principle of maximum entropy)
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

# ============================
# Robot Action Functions
# ============================

def get_weighted_robot_action(state: State, 
                             timestep: int, 
                             robot_beliefs: Beliefs, 
                             hypothesis_reward_space: List[PreferenceTree], 
                             object_type_tuple: List[ObjectConfig]) -> Tuple[str, State]:
    """
    Determine the robot's next action based on its current belief distribution.
    
    This function samples a preference model from the robot's belief distribution,
    then uses that model to determine the best quadrant for placing the current object.
    
    Parameters:
    -----------
    state : State
        Current state of the environment
    timestep : int
        Current timestep, determines which object to place
    robot_beliefs : Beliefs
        Robot's current belief distribution over preference models
    hypothesis_reward_space : List[PreferenceTree]
        List of preference models to consider
    object_type_tuple : List[ObjectConfig]
        List of object configurations
        
    Returns:
    --------
    Tuple[str, State]
        The selected action (quadrant or EXIT) and resulting state
    """
    # Sample a preference model based on current beliefs
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
            log.warn(f"Could not find exact preference for {color} {material} {object_type}, using default")
        
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

# ============================
# Human Interaction Functions
# ============================

def get_correction_from_human(new_state: State, 
                             timestep: int, 
                             robot_action: str, 
                             true_reward_tree: PreferenceTree, 
                             object_type_tuple: List[ObjectConfig]) -> Tuple[str, State]:
    """
    Simulate human correction based on true preference model.
    
    This function uses the true preference model to determine how a human would
    correct the robot's action, moving the object to its optimal position.
    
    Parameters:
    -----------
    new_state : State
        State after robot action
    timestep : int
        Current timestep, determines which object to correct
    robot_action : str
        Action chosen by robot
    true_reward_tree : PreferenceTree
        The ground truth preference model representing human preferences
    object_type_tuple : List[ObjectConfig]
        List of object configurations
        
    Returns:
    --------
    Tuple[str, State]
        The correction action (quadrant or EXIT) and resulting corrected state
    """
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
        log.warn(f"Could not find exact preference for {color} {material} {object_type}, using default")
    
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


def update_robot_beliefs(s0_starting_state: State, 
                      sr_state: State, 
                      sh_state: State, 
                      robot_beliefs: Beliefs,
                      hypothesis_reward_space: List[PreferenceTree], 
                      object_type_tuple: List[ObjectConfig]) -> Beliefs:
    """
    Update robot beliefs using Bayesian inference based on human corrections.
    
    This function implements Bayesian belief updates based on the observed human correction,
    calculating likelihoods of the correction under different preference models.
    
    Parameters:
    -----------
    s0_starting_state : State
        Initial state before robot action
    sr_state : State
        State after robot action
    sh_state : State
        State after human correction
    robot_beliefs : Beliefs
        Prior belief distribution
    hypothesis_reward_space : List[PreferenceTree]
        List of preference models
    object_type_tuple : List[ObjectConfig]
        List of object configurations
        
    Returns:
    --------
    Beliefs
        Updated belief distribution after incorporating human feedback
    """
    # Determine which conditions apply for this interaction
    # cond_1: Human corrected the robot's action (disagreement)
    # cond_2: Human agreed with robot's action, which changed the state
    # cond_3: Human's correction differed from initial state
    cond_1 = (sh_state != sr_state)
    cond_2 = (sr_state == sh_state and s0_starting_state != sr_state)
    cond_3 = (sh_state != s0_starting_state)

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
    
    # Bayes' update - normalize beliefs first
    new_beliefs = np.array(new_beliefs)
    new_beliefs /= np.sum(new_beliefs)
    
    # Get the model names - use the same names as in the main function
    model_names = ['Ethan', 'Michelle', 'Annika', 'Admoni', 'Simmons', 'Suresh', 
                   'Ben', 'Ada', 'Abhijat', 'Maggie', 'Zulekha', 'Pat']
    
    # Display updated beliefs in a formatted table (using normalized values)
    log.beliefs_table(new_beliefs, model_names[:len(new_beliefs)], 
                     indent=2, title="Updated Robot Beliefs:")
    return new_beliefs


# ============================
# Feature Clarification Functions
# ============================

def ask_clarification_questions(new_state: State, 
                               corrected_state: State, 
                               object_type_tuple: List[ObjectConfig]) -> None:
    """
    Present a series of clarification questions to better understand human preferences.
    
    Note: This is a placeholder function for demonstrating possible clarification questions.
    In a production environment, this would interface with a dialogue system.
    
    Parameters:
    -----------
    new_state : State
        State after robot action
    corrected_state : State
        State after human correction
    object_type_tuple : List[ObjectConfig]
        List of object configurations
    """
    # Categorized clarification questions for preference elicitation
    questions = {
        "Preference Understanding": [
            "Why did you prefer this new position over the one I chose?",
            "Can you explain why this location is better?"
        ],
        "Attribute Focus": [
            "Which attribute of the object influenced your correction the most?",
            "Was the position of the object the main reason for your correction, or was it something else?"
        ],
        "Future Guidance": [
            "How would you like me to position similar objects in the future?",
            "Are there specific rules I should follow when placing objects like this?"
        ],
        "Hypothesis Testing": [
            "I think you prefer objects to be placed closer to the center. Is that correct?",
            "It seems like you prefer objects to be in quadrant Q1. Is that true for all objects?"
        ]
    }

    # Display sample questions from each category
    for category, category_questions in questions.items():
        log.subsection(f"CLARIFICATION: {category}", color="yellow", bold=True)
        for question in category_questions:
            log.info(question, indent=2, color="yellow")
        log.info("", indent=1)  # Add spacing between categories
        
    log.info("Note: In a production system, these questions would be asked interactively", 
            indent=1, color="cyan", bold=True)


def get_relevant_features(preferences: Dict[str, Any], 
                         attributes: List[str], 
                         list_of_relevant_features: List[str]) -> Tuple[Any, List[str]]:
    """
    Recursively navigate the preference tree to identify relevant features.
    
    This function traverses the hierarchical preference structure to determine
    which object features (color, type, material) are relevant to the placement decision.
    
    Parameters:
    -----------
    preferences : Dict[str, Any]
        Current node in the preference tree
    attributes : List[str]
        List of attributes to search for in the preference tree
    list_of_relevant_features : List[str]
        Accumulator for relevant features found so far
        
    Returns:
    --------
    Tuple[Any, List[str]]
        The final preference value and list of relevant features
    """
    # Base case: we've reached a leaf node with quadrant rewards
    if isinstance(preferences, dict) and 'Q1' in preferences:
        return preferences, list_of_relevant_features

    # Recursive case: traverse the preference tree
    if isinstance(preferences, dict):
        for attr in attributes:
            if attr in preferences:
                # This attribute is relevant to the decision
                list_of_relevant_features.append(attr)
                return get_relevant_features(preferences[attr], attributes, list_of_relevant_features)

    # Fallback to "other" if no specific attribute match is found
    if 'other' in preferences:
        return preferences['other'], list_of_relevant_features
    
    # Return empty dict as default if structure doesn't match expected format
    return {}, list_of_relevant_features
def ask_feature_clarification_question(robot_beliefs: Beliefs, 
                                  hypothesis_reward_space: List[PreferenceTree],
                                  s0_starting_state: State, 
                                  sr_starting_state: State, 
                                  sh_starting_state: State, 
                                  timestep: int,
                                  current_object: ObjectTuple) -> Beliefs:
    """
    Ask targeted questions about which features are relevant to the human's preference.
    
    This function presents a question to determine which object features (color, type, material)
    influenced the human's placement decision, then updates beliefs based on the response.
    
    Parameters:
    -----------
    robot_beliefs : Beliefs
        Current belief distribution
    hypothesis_reward_space : List[PreferenceTree]
        List of preference models
    s0_starting_state : State
        Initial state
    sr_starting_state : State
        State after robot action
    sh_starting_state : State
        State after human correction
    timestep : int
        Current timestep
    current_object : ObjectTuple
        Current object being placed
        
    Returns:
    --------
    Beliefs
        Updated belief distribution after incorporating feature clarification
    """
    # Extract object properties
    color_idx, material_idx, object_idx, object_label = current_object
    color = COLORS_IDX[color_idx]
    material = MATERIALS_IDX[material_idx]
    object_type = OBJECTS_IDX[object_idx]
    attributes = [object_type, color, material]

    # Analyze which features are relevant in each hypothesis model
    hyp_idx_to_relevant_features = {}
    for hyp_idx in range(len(hypothesis_reward_space)):
        log.debug(f"Examining hypothesis index: {hyp_idx}", indent=2, color="blue")
        hypothesis_tree = hypothesis_reward_space[hyp_idx]
        pref_tree = hypothesis_tree['pref_values']
        quadrant_preference_tree, relevant_features = get_relevant_features(pref_tree, attributes, [])
        log.debug(f"Identified relevant features: {relevant_features}", indent=2)
        hyp_idx_to_relevant_features[hyp_idx] = relevant_features

    # Display an enhanced clarification question
    log.subsection("FEATURE CLARIFICATION QUESTION", color="yellow", bold=True)
    
    # Format object details
    object_details = [
        ("Object Type", object_type, "cyan"),
        ("Color", color, color),  # Use color name as the color
        ("Material", material, "magenta")
    ]
    
    # Display object information in a visually appealing way
    log.info("I need to understand which features influenced your decision:", indent=1, bold=True)
    log.info("", indent=1)
    log.info("Object Details:", indent=1, color="blue", bold=True)
    for label, value, color_code in object_details:
        log.result(f"{label}", f"{value}", indent=2, label_color="blue", value_color=color_code)
    
    log.info("", indent=1)
    log.info("Question:", indent=1, color="yellow", bold=True)
    log.info("For this object, which features influenced where it should be placed?", indent=2)
    log.info("Options: color, type, material (provide comma-separated responses)", indent=2, color="green")
    log.info("", indent=1)
    
    # Get human response with clear prompt
    true_relevant_features = []
    print("\033[1m\033[93m> Your answer:\033[0m ", end="")
    human_response = input("")
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
    new_beliefs /= np.sum(new_beliefs)
    
    # Get the model names - use the same names as in the main function
    model_names = ['Ethan', 'Michelle', 'Annika', 'Admoni', 'Simmons', 'Suresh', 
                   'Ben', 'Ada', 'Abhijat', 'Maggie', 'Zulekha', 'Pat']
    
    # Display the updated beliefs after clarification
    log.subsection("BELIEFS AFTER CLARIFICATION", color="magenta")
    log.beliefs_table(new_beliefs, model_names[:len(new_beliefs)], 
                     indent=2, title="Updated Robot Beliefs After Feature Clarification:")
    
    return new_beliefs

# ============================
# Main Interaction Loop
# ============================

def run_interaction():
    """
    Execute the main interaction loop for the clarification-guided reward learning system.
    
    This function orchestrates the complete interaction workflow:
    1. Initialize the robot's belief system and environment
    2. For each object to place:
       a. Robot takes action based on current beliefs
       b. Human provides correction
       c. Robot updates beliefs based on correction
       d. Robot asks clarification questions
       e. Robot further updates beliefs based on responses
    
    The function visualizes belief updates at each step and saves the results.
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


# ============================
# Main Entry Point
# ============================

if __name__ == '__main__':
    try:
        run_interaction()
    except KeyboardInterrupt:
        log.info("\nInterruption received, gracefully exiting...", color="yellow", bold=True)
    except Exception as e:
        log.critical(f"An unexpected error occurred: {str(e)}")
        import traceback
        log.debug(traceback.format_exc())
    finally:
        log.section("SESSION ENDED", color="blue", bold=True)

"""
Future Development Ideas:

1. Enhanced Question Generation
   - Generate questions dynamically based on the robot's uncertainty
   - Prioritize questions that maximize information gain
   - Use natural language generation for more natural dialogue

2. Improved Belief Updates
   - Implement more sophisticated Bayesian update mechanisms
   - Add confidence-weighted updates based on human certainty
   - Handle noisy or inconsistent human feedback

3. Natural Language Integration
   - Allow open-ended natural language responses
   - Use semantic parsing to extract relevant information
   - Build a knowledge graph of preferences over time

4. Active Learning Strategies
   - Implement information-theoretic query selection
   - Balance exploration vs. exploitation in question asking
   - Design questions to disambiguate between competing hypotheses

5. User Experience Improvements
   - Add visualization of belief evolution over time
   - Provide explanations for robot's decisions
   - Allow reviewing and revising past interactions
"""
