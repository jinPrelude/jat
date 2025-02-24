#!/usr/bin/env python3
"""Eval a JAT model"""
import json
import logging
import os

os.environ['MUJOCO_GL'] = 'egl'

import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, HfArgumentParser

from jat.eval.rl import TASK_NAME_TO_ENV_ID, make_pixel_metaworld
from jat.utils import normalize, push_to_hub, save_video_grid


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config we are going to train from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it "
                "will execute code present on the Hub on your local machine."
            )
        },
    )


@dataclass
class EvaluationArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    tasks: List[str] = field(default_factory=list, metadata={"help": "Tasks to train on."})
    use_cpu: bool = field(default=False, metadata={"help": "Use CPU instead of GPU."})
    save_video: bool = field(default=False, metadata={"help": "Save video of the evaluation."})
    num_episodes: int = field(default=2, metadata={"help": "Number of episodes to evaluate on."})
    push_to_hub: bool = field(default=False, metadata={"help": "Push the model to the hub."})
    repo_id: Optional[str] = field(default=None, metadata={"help": "Repository ID to push to."})


def get_default_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def eval_rl(model, processor, task, eval_args):
    # Create the environment
    env_kwargs = {}
    if task.startswith("atari"):
        env_kwargs["clip_reward"] = False
    if eval_args.save_video:
        env_kwargs["render_mode"] = "rgb_array"

    env = make_pixel_metaworld(task, **env_kwargs)

    context_window = 32 if task.startswith("atari") else 256

    scores = []
    frames = []
    all_successes = []  # Added to accumulate success values
    for episode in tqdm(range(eval_args.num_episodes), desc=task, unit="episode", leave=False):
        observation, _ = env.reset()
        reward = None
        rewards = []
        done = False
        model.reset_rl()  # remove KV Cache
        while not done:
            action = model.get_next_action(
                processor, **observation, reward=reward, action_space=env.action_space, context_window=context_window
            )
            observation, reward, termined, truncated, info = env.step(action)
            done = termined or truncated

            # Update the return
            rewards.append(reward)

            # Render the environment
            if eval_args.save_video:
                frames.append(np.array(env.render()[::-1], dtype=np.uint8))

        all_successes.append(info.get("success", 0))  # Append the success for this episode
        scores.append(sum(rewards))
    env.close()

    raw_mean, raw_std = np.mean(scores), np.std(scores)
    # Compute success metrics
    success_mean = np.mean(all_successes) if all_successes else 0
    success_std = np.std(all_successes) if all_successes else 0

    # Normalize the scores
    norm_scores = normalize(scores, task, "expert")
    if norm_scores is not None:  # Can be None if random is better than expert
        norm_mean, norm_std = np.mean(norm_scores), np.std(norm_scores)
        tqdm.write(
            f"Task {task} Raw score: {raw_mean:.2f} ± {raw_std:.2f}\t"
            f"Normalized score: {norm_mean:.2f} ± {norm_std:.2f}\t"
            f"success rate: {success_mean:.2f} ± {success_std:.2f}"
        )
    else:
        tqdm.write(f"Task {task} Raw score: {raw_mean:.2f} ± {raw_std:.2f}")

    # Resize images by 1/3 to limit memory usage (the video is reduced anyway when aggregated with the others)
    if eval_args.save_video:
        import cv2

        frames = [cv2.resize(frame, (0, 0), fx=1 / 3, fy=1 / 3) for frame in frames]

    return scores, frames, env.metadata["render_fps"], success_mean, success_std


def main():
    parser = HfArgumentParser((ModelArguments, EvaluationArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, eval_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, eval_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Set the tasks
    tasks = eval_args.tasks
    for domain in ["atari", "babyai", "metaworld", "mujoco"]:
        if domain in tasks:
            tasks.remove(domain)
            tasks.extend([env_id for env_id in TASK_NAME_TO_ENV_ID.keys() if env_id.startswith(domain)])

    device = torch.device("cpu") if eval_args.use_cpu else get_default_device()
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir, trust_remote_code=model_args.trust_remote_code
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir, trust_remote_code=model_args.trust_remote_code
    )

    evaluations = {}
    evaluations_success = {}  # New dict to store success metrics per task
    video_list = []
    input_fps = []
    with torch.no_grad():
        for task in tqdm(tasks, desc="Evaluation", unit="task", leave=True):
            if task in TASK_NAME_TO_ENV_ID.keys():
                scores, frames, fps, success_mean, success_std = eval_rl(model, processor, task, eval_args)
                evaluations[task] = scores
                evaluations_success[task] = {"success_mean": success_mean, "success_std": success_std}  # Save success metrics
                if eval_args.save_video:
                    video_list.append(frames)
                    input_fps.append(fps)
            else:
                warnings.warn(f"Task {task} is not supported.")

    # Extract mean and std, and save scores dict
    output_dir = f"runs/{model_args.model_name_or_path}"
    eval_path = f"{output_dir}/evaluations.json"
    success_eval_path = f"{output_dir}/evaluations_success.json"  # Path for success evaluations

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if evaluations:
        with open(eval_path, "w") as file:
            json.dump(evaluations, file)

    # Save the success evaluation metrics
    with open(success_eval_path, "w") as file:
        json.dump(evaluations_success, file)

    # Save the video
    if eval_args.save_video:
        replay_path = f"{output_dir}/replay.mp4"
        save_video_grid(video_list, input_fps, replay_path, output_fps=30, max_length_seconds=180)
    else:
        replay_path = None

    # Push the model to the hub
    if eval_args.push_to_hub:
        assert eval_args.repo_id is not None, "You need to specify a repo_id to push to."
        push_to_hub(model, processor, eval_args.repo_id, replay_path=replay_path, eval_path=eval_path)


if __name__ == "__main__":
    main()
