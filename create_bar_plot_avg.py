#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import re

def load_success_evaluations(path):
    """
    각 환경의 success_mean과 success_std를 포함하는 evaluations_success.json 파일을 로드합니다.
    """
    with open(path, 'r') as f:
        evaluations = json.load(f)
    # evaluations는 {env: {"success_mean": float, "success_std": float}, ...} 형태입니다.
    # 각 환경의 success_mean과 success_std를 모두 추출합니다.
    env_success = {}
    for env, v in evaluations.items():
        env_success[env] = {'mean': v['success_mean'], 'std': v['success_std']}
    return env_success

def compute_avg_success_and_std(env_success, filter_fn):
    """
    주어진 filter_fn에 해당하는 환경들의 평균 success_rate와 평균 std를 계산합니다.
    """
    means = [v['mean'] for env, v in env_success.items() if filter_fn(env)]
    stds = [v['std'] for env, v in env_success.items() if filter_fn(env)]
    if means:
        return np.mean(means), np.mean(stds)
    else:
        return None, None

def extract_model_name(path):
    # 평가 파일의 부모 폴더 이름을 checkpoint 이름으로 사용합니다.
    return os.path.basename(os.path.dirname(path))

def sort_model_names(model_names):
    def alphanum_key(key):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', key)]
    return sorted(model_names, key=alphanum_key)

def plot_bar(ax, labels, values, errors, title, color, show_std):
    x = np.arange(len(labels))
    if show_std:
        bars = ax.bar(x, values, yerr=errors, color=color, capsize=5)
    else:
        bars = ax.bar(x, values, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Average Success Rate')
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        offset = errors[i] + 0.02 if show_std else 0.02
        ax.text(bar.get_x() + bar.get_width()/2., height + offset, f'{height:.2f}', ha='center', va='bottom')

def main(args):
    # 딕셔너리: model 이름 -> (train_mean, train_std, test_mean, test_std)
    model_success_scores = {}
    
    for eval_path in args.evals:
        model_name = extract_model_name(eval_path)
        env_success = load_success_evaluations(eval_path)
        train_mean, train_std = compute_avg_success_and_std(env_success, lambda env: env not in args.test_tasks)
        test_mean, test_std = compute_avg_success_and_std(env_success, lambda env: env in args.test_tasks)
        if train_mean is not None or test_mean is not None:
            model_success_scores[model_name] = (train_mean, train_std, test_mean, test_std)
        else:
            print(f"Warning: No matching environments for model {model_name} in {eval_path}")

    sorted_model_names = sort_model_names(model_success_scores.keys())
    
    # prepare lists for train and test values and error bars
    train_labels = []
    train_values = []
    train_errors = []
    test_labels = []
    test_values = []
    test_errors = []
    for model_name in sorted_model_names:
        train_labels.append(model_name)
        test_labels.append(model_name)
        train_mean, train_std, test_mean, test_std = model_success_scores[model_name]
        train_values.append(train_mean if train_mean is not None else 0.0)
        train_errors.append(train_std if train_std is not None else 0.0)
        test_values.append(test_mean if test_mean is not None else 0.0)
        test_errors.append(test_std if test_std is not None else 0.0)

    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(16, 6))
    plot_bar(ax_train, train_labels, train_values, train_errors, 'Train Tasks Success Rate', 'skyblue', args.show_std)
    plot_bar(ax_test, test_labels, test_values, test_errors, 'Test Tasks Success Rate', 'salmon', args.show_std)

    plt.tight_layout()
    plt.savefig(args.output)
    print(f'Bar plots saved to {args.output}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute average success rate with std error bars and generate bar plots.')
    parser.add_argument('--evals', type=str, nargs='+',
                        default=[
                          'runs/checkpoints/pre_trained/jat_pretrain_quest_10epoch/checkpoint-138/evaluations_success.json',
                          'runs/checkpoints/pre_trained/jat_pretrain_quest_10epoch/checkpoint-414/evaluations_success.json',
                          'runs/checkpoints/pre_trained/jat_pretrain_quest_10epoch/checkpoint-690/evaluations_success.json',
                          'runs/checkpoints/pre_trained/jat_pretrain_quest_10epoch/checkpoint-966/evaluations_success.json',
                          'runs/checkpoints/pre_trained/jat_pretrain_quest_10epoch/checkpoint-1242/evaluations_success.json',
                          'runs/checkpoints/pre_trained/jat_pretrain_quest_epoch50_lr2e-5_bs52_bf16/checkpoint-2420/evaluations_success.json',
                          'runs/checkpoints/pre_trained/jat_pretrain_quest_epoch50_lr2e-5_bs52_bf16/checkpoint-6050/evaluations_success.json',
                        ],
                        help='Paths to evaluations_success.json files for each checkpoint')
    parser.add_argument('--test-tasks', type=str, nargs='+',
                        default=[
                          "metaworld-bin-picking",
                          "metaworld-box-close",
                          "metaworld-door-lock",
                          "metaworld-door-unlock",
                          "metaworld-hand-insert"
                        ],
                        help='List of environment names to treat as test tasks; others are train tasks')
    parser.add_argument('--output', type=str, default='success_rate_performance.png',
                        help='Output PNG file name for the bar plots')
    parser.add_argument('--show-std', action='store_true',
                        help='Show standard deviation error bars')
    args = parser.parse_args()
    main(args)