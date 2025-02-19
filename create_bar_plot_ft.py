#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import re

def load_scores_dict(path):
    with open(path, 'r') as f:
        scores = json.load(f)
    env_scores = {}
    for env, v in scores.items():
        expert_mean = v['expert']['mean']
        random_mean = v['random']['mean']
        env_scores[env] = {'expert': expert_mean, 'random': random_mean}
    return env_scores

def load_evaluations(path):
    with open(path, 'r') as f:
        evaluations = json.load(f)
    env_avg = {env: np.mean(scores) for env, scores in evaluations.items() if scores}
    return env_avg

def compute_normalized_score_filtered(env_avg, scores_dict, filter_fn):
    normalized_scores = []
    for env, avg in env_avg.items():
        if filter_fn(env) and env in scores_dict:
            expert = scores_dict[env]['expert']
            random_ = scores_dict[env]['random']
            denom = expert - random_
            norm = 0.0 if denom == 0 else (avg - random_) / denom
            normalized_scores.append(norm)
    return np.mean(normalized_scores) if normalized_scores else None

def extract_model_name(path):
    return os.path.basename(os.path.dirname(path))

def sort_model_names(model_names):
    def alphanum_key(key):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', key)]
    return sorted(model_names, key=alphanum_key)

def plot_bar(ax, labels, values, title, color):
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Normalized Performance')
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(title)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.2f}', ha='center', va='bottom')

def main(args):

    scores_dict = load_scores_dict(args.scores)
    model_norm_scores = {}

    # Process additional evaluation files (compute only test tasks)
    for eval_path in args.evals:
        model_name = extract_model_name(eval_path)
        env_avg = load_evaluations(eval_path)
        test_score = compute_normalized_score_filtered(env_avg, scores_dict, lambda env: env in args.test_tasks)
        if test_score is not None:
            model_norm_scores[model_name] = test_score
        else:
            print(f"Warning: No matching environments for model {model_name} in {eval_path}")

    sorted_model_names = sort_model_names(model_norm_scores.keys())

    # Prepare labels/values with placeholders for random (0) and expert (1)
    test_labels = ['random agent']
    test_values = [0.0]

    for model_name in sorted_model_names:
        test_labels.append(model_name)
        test_values.append(model_norm_scores[model_name] if model_norm_scores[model_name] is not None else 0.0)

    # Process jat evaluations (compute only test tasks)
    env_avg = load_evaluations(args.jat_scores)
    jat_score = compute_normalized_score_filtered(env_avg, scores_dict, lambda env: env in args.test_tasks)
    if jat_score is not None:
        test_labels.append('jat')
        test_values.append(jat_score)
    else:
        print(f"Warning: No matching environments found for model jat in {args.jat_scores}")


    test_labels.append('expert agent')
    test_values.append(1.0)

    # Build colors list: 'random agent' and 'expert agent' -> gray, 'jat' -> salmon, others (evals) -> skyblue.
    colors = []
    for label in test_labels:
        if label in ['random agent', 'expert agent', 'jat']:
            colors.append("salmon")
        else:
            colors.append("skyblue")

    fig, ax_test = plt.subplots(figsize=(8, 6))
    plot_bar(ax_test, test_labels, test_values, 'Test Tasks Performance', colors)

    plt.tight_layout()
    plt.savefig(args.output)
    print(f'Bar plot saved to {args.output}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute normalized performance and generate bar plots.')
    parser.add_argument('--scores', type=str, default='jat/eval/rl/scores_dict.json',
                        help='Path to scores_dict.json file')
    parser.add_argument('--jat-scores', type=str, default='runs/jat-project/jat/evaluations.json',
                        help='Path to jat evaluations.json file')
    parser.add_argument('--evals', type=str, nargs='+',
                        default=[
                            'runs/checkpoints/jat/fine-tuned/pretrained-checkpoint-20000/checkpoint-100/evaluations.json',
                            'runs/checkpoints/jat/fine-tuned/pretrained-checkpoint-20000/checkpoint-300/evaluations.json',
                            'runs/checkpoints/jat/fine-tuned/pretrained-checkpoint-20000/checkpoint-500/evaluations.json',
                            'runs/trainer_output/trainer-checkpoint-500/evaluations.json',
                        ],
                        help='Paths to evaluations_{model_name}.json files')
    parser.add_argument('--test-tasks', type=str, nargs='+',
                        default=[
                          "metaworld-bin-picking",
                          "metaworld-box-close",
                          "metaworld-door-lock",
                          "metaworld-door-unlock",
                          "metaworld-hand-insert"
                        ],
                        help='List of environment names to treat as test tasks; others are train tasks')
    parser.add_argument('--output', type=str, default='normalized_performance_finetuned.png',
                        help='Output PNG file name for the bar plot')
    args = parser.parse_args()
    main(args)
