import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import re

def load_evaluations(path):
    with open(path, 'r') as f:
        evaluations = json.load(f)
    
    score_type = "success" if "success" in os.path.basename(path) else "raw_score"
    env_stats = {
        env: {
            "mean": values[f"{score_type}_mean"],
            "std": values[f"{score_type}_std"]
        }
        for env, values in evaluations.items() if f"{score_type}_mean" in values
    }
    return env_stats, score_type

def extract_model_name(path):
    return os.path.basename(os.path.dirname(path))

def sort_env_names(env_names):
    return sorted(env_names)

def plot_agent_scores(agent_name, env_stats, test_tasks, score_type, show_std):  # Updated signature
    test_envs = [env for env in env_stats if env in test_tasks]
    train_envs = [env for env in env_stats if env not in test_tasks]
    
    test_means = [env_stats[env]["mean"] for env in test_envs]
    test_stds = [env_stats[env]["std"] for env in test_envs]
    train_means = [env_stats[env]["mean"] for env in train_envs]
    train_stds = [env_stats[env]["std"] for env in train_envs]
    
    test_x = np.arange(len(test_envs))
    train_x = np.arange(len(train_envs))
    
    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [3, 1]})
    
    # Plot train tasks with conditional std error bars
    bars_train = ax_train.bar(
        train_x,
        train_means,
        yerr=train_stds if show_std else None,
        capsize=5 if show_std else 0,
        alpha=0.75
    )
    ax_train.set_xticks(train_x)
    ax_train.set_xticklabels(train_envs, rotation=45, ha='right')
    ax_train.set_ylabel(f'{score_type.capitalize()} Score')
    ax_train.set_title(f'Train Tasks for {agent_name}')
    ax_train.set_ylim(0.0, 1.0)  # Set y-limit for train tasks

    # Plot test tasks with conditional std error bars
    bars_test = ax_test.bar(
        test_x,
        test_means,
        yerr=test_stds if show_std else None,
        capsize=5 if show_std else 0,
        alpha=0.75,
        color='orange'
    )
    ax_test.set_xticks(test_x)
    ax_test.set_xticklabels(test_envs, rotation=45, ha='right')
    ax_test.set_ylabel(f'{score_type.capitalize()} Score')
    ax_test.set_title(f'Test Tasks for {agent_name}')
    ax_test.set_ylim(0.0, 1.0)  # Set y-limit for test tasks
    
    for bar, mean in zip(bars_train, train_means):
        ax_train.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02, f'{mean:.2f}', ha='center', va='bottom')
    
    for bar, mean in zip(bars_test, test_means):
        ax_test.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02, f'{mean:.2f}', ha='center', va='bottom')
    
    output_path = f"{agent_name}{'_success' if score_type == 'success' else ''}.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f'Plot saved: {output_path}')

def main(args):
    test_tasks = set(args.test_tasks)
    
    # Process evaluations for each agent
    for eval_path in args.evals:
        agent_name = extract_model_name(eval_path)
        env_stats, score_type = load_evaluations(eval_path)
        plot_agent_scores(agent_name, env_stats, test_tasks, score_type, args.show_std)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate bar plots for raw scores or success rates per agent.')
    parser.add_argument('--evals', type=str, nargs='+', required=True,
                        help='Paths to evaluations_{model_name}.json files')
    parser.add_argument('--test-tasks', type=str, nargs='+',
                        default=[
                          "metaworld-bin-picking",
                          "metaworld-box-close",
                          "metaworld-door-lock",
                          "metaworld-door-unlock",
                          "metaworld-hand-insert"
                        ],
                        help='List of test tasks')
    parser.add_argument('--show-std', type=lambda v: v.lower()=='true', default=False,
                        help='Show std error bars (true/false). Default is true.')
    args = parser.parse_args()
    main(args)
