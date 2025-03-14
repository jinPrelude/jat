#!/usr/bin/env python3
"""Train a JAT model on the JAT dataset"""


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional
os.environ['HF_HOME'] = '/scratch/euijinrnd/.cache/huggingface/' # huggingface cache 를 /scratch/euijinrnd로 바꿔주기
os.environ['HF_DATASETS_OFFLINE'] = '1'

import datasets.config
from datasets import load_from_disk
from datasets.config import HF_DATASETS_CACHE, HF_DATASETS_OFFLINE
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, HfArgumentParser, Trainer, TrainingArguments

from jat.eval.rl.core import TASK_NAME_TO_ENV_ID
from jat.modeling_jat import JatModel
from jat.utils_interleave_datasets import interleave_datasets


# Sometimes, the server is down; increasing the number of
# retries allows to wait more instead of making the training crash
datasets.config.STREAMING_READ_MAX_RETRIES = 10000


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config we are going to train from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
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
    observation_loss_coef: float = field(
        default=0.0005,
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    tasks: List[str] = field(default_factory=list, metadata={"help": "Tasks to train on."})


SAMPLE_WEIGHTS = {
    "conceptual-captions": 10.0,
    "oscar": 10.0,
    "wikipedia": 10.0,
}



class MyTrainer(Trainer):
    def _get_train_sampler(self) -> None:
        return None


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    if model_args.observation_loss_coef is not None:
        config.observation_loss_coef = model_args.observation_loss_coef
    model = JatModel(config)
    processor = AutoProcessor.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Set the tasks
    tasks = data_args.tasks
    for domain in ["atari", "babyai", "metaworld", "mujoco"]:
        if domain in tasks:
            tasks.remove(domain)
            tasks.extend([env_id for env_id in TASK_NAME_TO_ENV_ID.keys() if env_id.startswith(domain)])

            tasks = [task for task in tasks if task not in [
                "metaworld-bin-picking",
                "metaworld-box-close",
                "metaworld-door-lock",
                "metaworld-door-unlock",
                "metaworld-hand-insert"
            ]] # exclude test-task

    # Load the datasets
    if HF_DATASETS_OFFLINE:
        train_dataset = {}
        for task in tqdm(tasks[1:], desc="Loading datasets"):
            if not os.path.exists(f"converted_data/metaworld_2_tokenized/{TASK_NAME_TO_ENV_ID[task]}"):
                continue
            d = load_from_disk(f"converted_data/metaworld_2_tokenized/{TASK_NAME_TO_ENV_ID[task]}")
            train_dataset[task] = d

    weights = [SAMPLE_WEIGHTS.get(t, 1.0) for t in train_dataset.keys()]

    train_dataset = interleave_datasets(
        list(train_dataset.values()),
        probabilities=[w / sum(weights) for w in weights],
        seed=training_args.seed,
        stopping_strategy="all_exhausted",
        n_contiguous=training_args.per_device_train_batch_size,
    )

    # Due to the train dataset's structure, where every 'n' consecutive samples share the same modalities, we can't
    # load all samples at once. Different sets of 'n' samples have different modalities. Therefore, we must load and
    # process each set of 'n' samples separately.
    if training_args.dispatch_batches is not False:
        raise ValueError("Make sure to pass `--dispatch_batches False`.")

    # Why the training continue after exauhsting the dataset? https://github.com/huggingface/transformers/issues/26635
    trainer = MyTrainer(model=model, args=training_args, train_dataset=train_dataset, tokenizer=processor)
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
