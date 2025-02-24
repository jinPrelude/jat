import os
from typing import Any, Dict, List, Optional
import argparse

import gymnasium as gym
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
import numpy as np
import torch
from tqdm import tqdm


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
    task_name = task.replace('metaworld-', '').replace('-v2', '')
    
    # Split by '-'
    words_list = task_name.split('-')
    
    # Construct the directory name
    dir_name = "sawyer_" + "_".join(words_list) + "_v2_policy"
    
    # Capitalize first letter of each word
    words_list = [word.capitalize() for word in words_list]
    
    # Construct the policy name
    policy_name = "Sawyer" + "".join(words_list) + "V2Policy"
    
    return dir_name, policy_name

# most of this function is redundant as it is copied from sample.enjoy.enjoy
def create_dataset(task_id) -> None:

    # Make the environment
    env_seed = 0
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{task_id}-goal-observable"]

    # Load expert policy
    dir_name, policy_name = convert_to_policy_name(task_id)
    policy_module = __import__(f"metaworld.policies.{dir_name}", fromlist=[policy_name])
    model = getattr(policy_module, policy_name)()

    # Create dataset
    dataset: Dict[str, List[List[Any]]] = {
        "continuous_observations": [],  # [[s0, s1, s2, ..., sT-1], [s0, s1, ...]], # terminal observation not stored
        "continuous_actions": [],  # [[a0, a1, a2, ..., aT-1], [a0, a1, ...]],
        "rewards": [],  # [[r1, r2, r3, ...,   rT], [r1, r2, ...]],
    }

    # Reset environment
    env = env_cls(seed=env_seed)
    observations, _ = env.reset()
    dones = [True]

    # Run the environment
    dataset_size = 1_600_000 + 160_000
    progress_bar = tqdm(total=dataset_size)
    num_timesteps = 0
    with torch.no_grad():
        while num_timesteps < dataset_size or not dones[0]:
            for agent_idx, done in enumerate(dones):
                if done:
                    for value in dataset.values():
                        value.append([])

            progress_bar.update(1)
            
            action = model.get_action(observations)
            dataset["continuous_observations"][-1].append(observations)
            dataset["continuous_actions"][-1].append(action)

            observations, rewards, terminated, truncated, _ = env.step(action)
            dones = truncated or terminated

            dataset["rewards"][-1].append(rewards.cpu().numpy())

            num_timesteps += 1

    env.close()

    dataset["continuous_observations"] = np.array(
        [np.array(x, dtype=np.float32) for x in dataset["continuous_observations"]], dtype=object
    )
    dataset["continuous_actions"] = np.array(
        [np.array(x, dtype=np.float32) for x in dataset["continuous_actions"]], dtype=object
    )
    dataset["rewards"] = np.array([np.array(x, dtype=np.float32) for x in dataset["rewards"]], dtype=object)

    repo_path = f"datasets/{task_id}"
    os.makedirs(repo_path, exist_ok=True)

    _dataset = {key: value[:16_000] for key, value in dataset.items()}
    file = f"{repo_path}/train"
    np.savez_compressed(f"{file}.npz", **_dataset)

    _dataset = {key: value[16_000:] for key, value in dataset.items()}
    file = f"{repo_path}/test"
    np.savez_compressed(f"{file}.npz", **_dataset)


def main(args) -> int:
    status = create_dataset(args.task_id)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-id', type=str, default='door-open-v2')
    args = parser.parse_args()
    main(args)
