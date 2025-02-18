#!/usr/bin/env python3
"""Train a JAT model on the JAT dataset"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

os.environ['HF_HOME'] = '/scratch/euijinrnd/.cache/huggingface/'  # huggingface cache 경로 변경
os.environ['HF_DATASETS_OFFLINE'] = '1'

import datasets.config
from datasets import load_dataset, load_from_disk
from datasets.config import HF_DATASETS_CACHE, HF_DATASETS_OFFLINE
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, HfArgumentParser, Trainer, TrainingArguments

from jat.eval.rl.core import TASK_NAME_TO_ENV_ID
from jat.modeling_jat import JatModel
from jat.utils_interleave_datasets import interleave_datasets

# Increase dataset read retries
datasets.config.STREAMING_READ_MAX_RETRIES = 10000
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to store pretrained models downloaded from huggingface.co"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Allow execution of custom models from the Hub."},
    )

@dataclass
class DataTrainingArguments:
    tasks: List[str] = field(default_factory=list, metadata={"help": "Tasks to train on."})
    top_n_demos: Optional[int] = field(default=None, metadata={"help": "Select top n demos with highest reward."})

SAMPLE_WEIGHTS = {
    "conceptual-captions": 10.0,
    "oscar": 10.0,
    "wikipedia": 10.0,
}

os.environ["WANDB_PROJECT"] = "jat"

class MyTrainer(Trainer):
    def _get_train_sampler(self) -> None:
        return None

def parse_arguments():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        return parser.parse_args_into_dataclasses()

def setup_logging(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

def load_config_and_model(model_args):
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = JatModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    processor = AutoProcessor.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    return model, processor

def load_train_dataset(data_args, training_args):
    # Pick a single task.
    if not data_args.tasks:
        raise ValueError("Please specify at least one task.")
    tasks = [data_args.tasks[0]]

    # Load dataset according to offline flag.
    if HF_DATASETS_OFFLINE:
        for task in tasks:
            dataset_path = f"{HF_DATASETS_CACHE}/jat-project/jat-dataset-tokenized/{task}"
            if not os.path.exists(dataset_path):
                raise ValueError(
                    f"""Dataset {task} not found in {dataset_path}
Make sure to download and save it first with:
```
from datasets import load_dataset
dataset = load_dataset('jat-project/jat-dataset-tokenized', '{task}')
dataset.save_to_disk('{dataset_path}')
```"""
                )
        train_dataset = {}
        for task in tqdm(tasks, desc="Loading datasets"):
            d = load_from_disk(f"{HF_DATASETS_CACHE}/jat-project/jat-dataset-tokenized/{task}")
            train_dataset[task] = d["train"]
    else:
        train_dataset = {task: load_dataset("jat-project/jat-dataset-tokenized", task, split="train") for task in tasks}

    # Optionally filter demos based on reward.
    if data_args.top_n_demos is not None:
        for task in train_dataset:
            ds = train_dataset[task]
            ds = ds.sort("reward", reverse=True)
            train_dataset[task] = ds.select(range(data_args.top_n_demos))

    # Calculate sampling probabilities.
    weights = [SAMPLE_WEIGHTS.get(t, 1.0) for t in train_dataset.keys()]
    return interleave_datasets(
        list(train_dataset.values()),
        probabilities=[w / sum(weights) for w in weights],
        seed=training_args.seed,
        stopping_strategy="all_exhausted",
        n_contiguous=training_args.per_device_train_batch_size,
    )

def main():
    model_args, data_args, training_args = parse_arguments()
    setup_logging(training_args)
    model, processor = load_config_and_model(model_args)
    train_dataset = load_train_dataset(data_args, training_args)

    if training_args.dispatch_batches is not False:
        raise ValueError("Make sure to pass `--dispatch_batches False`.")

    trainer = MyTrainer(model=model, args=training_args, train_dataset=train_dataset, tokenizer=processor)
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

if __name__ == "__main__":
    main()
