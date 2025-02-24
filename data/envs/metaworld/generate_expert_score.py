"""
This script generates the score for a random agent for all the metaworld environments and saves them in a dictionary.
"""

import json
import os
from multiprocessing import Pool

import gymnasium as gym
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
import numpy as np


def convert_to_policy_name(task):
    """
    Convert task name to directory and policy names.

    dir_name:
    1. Remove 'metaworld-' prefix.
    2. Split by '-'.
    3. Join with '_'.
    4. Prefix with 'sawyer_' and suffix with '_v2_policy'.
    eg. "metaworld-plate-slide-back" -> "sawyer_plate_slide_back_v2_policy"

    policy_name:
    1. Remove 'metaworld-' prefix.
    2. Split by '-'.
    3. Capitalize the first letter of each word.
    4. Join words.
    5. Prefix with 'Sawyer' and suffix with 'V2Policy'.
    eg. "metaworld-plate-slide-back" -> "SawyerPlateSlideBackV2Policy"
    """
    # Remove 'metaworld-' prefix
    task_name = task.replace('metaworld-', '')
    
    # Split by '-'
    words_list = task_name.split('-')
    
    # Construct the directory name
    dir_name = "sawyer_" + "_".join(words_list) + "_v2_policy"
    
    # Capitalize first letter of each word
    words_list = [word.capitalize() for word in words_list]
    
    # Construct the policy name
    policy_name = "Sawyer" + "".join(words_list) + "V2Policy"
    
    return dir_name, policy_name


FILENAME = "jat/eval/rl/scores_dict.json"

TASK_NAME_TO_ENV_NAME = {
    "metaworld-assembly": "assembly-v2",
    "metaworld-basketball": "basketball-v2",
    "metaworld-bin-picking": "bin-picking-v2",
    "metaworld-box-close": "box-close-v2",
    "metaworld-button-press-topdown": "button-press-topdown-v2",
    "metaworld-button-press-topdown-wall": "button-press-topdown-wall-v2",
    "metaworld-button-press": "button-press-v2",
    "metaworld-button-press-wall": "button-press-wall-v2",
    "metaworld-coffee-button": "coffee-button-v2",
    "metaworld-coffee-pull": "coffee-pull-v2",
    "metaworld-coffee-push": "coffee-push-v2",
    "metaworld-dial-turn": "dial-turn-v2",
    "metaworld-disassemble": "disassemble-v2",
    "metaworld-door-close": "door-close-v2",
    "metaworld-door-lock": "door-lock-v2",
    "metaworld-door-open": "door-open-v2",
    "metaworld-door-unlock": "door-unlock-v2",
    "metaworld-drawer-close": "drawer-close-v2",
    "metaworld-drawer-open": "drawer-open-v2",
    "metaworld-faucet-close": "faucet-close-v2",
    "metaworld-faucet-open": "faucet-open-v2",
    "metaworld-hammer": "hammer-v2",
    "metaworld-hand-insert": "hand-insert-v2",
    "metaworld-handle-press-side": "handle-press-side-v2",
    "metaworld-handle-press": "handle-press-v2",
    "metaworld-handle-pull-side": "handle-pull-side-v2",
    "metaworld-handle-pull": "handle-pull-v2",
    "metaworld-lever-pull": "lever-pull-v2",
    "metaworld-peg-insert-side": "peg-insert-side-v2",
    "metaworld-peg-unplug-side": "peg-unplug-side-v2",
    "metaworld-pick-out-of-hole": "pick-out-of-hole-v2",
    "metaworld-pick-place": "pick-place-v2",
    "metaworld-pick-place-wall": "pick-place-wall-v2",
    "metaworld-plate-slide-back-side": "plate-slide-back-side-v2",
    "metaworld-plate-slide-back": "plate-slide-back-v2",
    "metaworld-plate-slide-side": "plate-slide-side-v2",
    "metaworld-plate-slide": "plate-slide-v2",
    "metaworld-push-back": "push-back-v2",
    "metaworld-push": "push-v2",
    "metaworld-push-wall": "push-wall-v2",
    "metaworld-reach": "reach-v2",
    "metaworld-reach-wall": "reach-wall-v2",
    "metaworld-shelf-place": "shelf-place-v2",
    "metaworld-soccer": "soccer-v2",
    "metaworld-stick-pull": "stick-pull-v2",
    "metaworld-stick-push": "stick-push-v2",
    "metaworld-sweep-into": "sweep-into-v2",
    "metaworld-sweep": "sweep-v2",
    "metaworld-window-close": "window-close-v2",
    "metaworld-window-open": "window-open-v2",
}


NUM_EPISODES = 100


def generate_expert_score(task_name):
    print(f"Starting task: {task_name}")

    # Make the environment
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{TASK_NAME_TO_ENV_NAME[task_name]}-goal-observable"]

    dir_name, policy_name = convert_to_policy_name(task_name)
    policy_module = __import__(f"metaworld.policies.{dir_name}", fromlist=[policy_name])
    model = getattr(policy_module, policy_name)()

    # Initialize the variables
    all_episode_rewards = []
    for i in range(NUM_EPISODES):
        env = env_cls(seed=i)
        observation, _ = env.reset()

        tot_episode_rewards = 0  # for one episode
        terminated = truncated = False
        while not (terminated or truncated):
            action = model.get_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            tot_episode_rewards += reward
        all_episode_rewards.append(tot_episode_rewards)

    # Load the scores dictionary
    if not os.path.exists(FILENAME):
        scores_dict = {}
    else:
        with open(FILENAME, "r") as file:
            scores_dict = json.load(file)

    # Add the random scores to the dictionary
    if task_name not in scores_dict:
        scores_dict[task_name] = {}
    scores_dict[task_name]["expert"] = {"mean": np.mean(all_episode_rewards), "std": np.std(all_episode_rewards)}

    # Save the dictionary to a file
    with open(FILENAME, "w") as file:
        scores_dict = {
            task: {agent: scores_dict[task][agent] for agent in sorted(scores_dict[task])}
            for task in sorted(scores_dict)
        }
        json.dump(scores_dict, file, indent=4)
    print(f"Completed task: {task_name}")

if __name__ == "__main__":
    with Pool(16) as p:
        p.map(generate_expert_score, TASK_NAME_TO_ENV_NAME.keys())
    print("All tasks completed.")
