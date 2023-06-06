import torch

from datasets import load_dataset

from gia.config import Arguments
from gia.datasets import GIADataCollator
from gia.model import GiaModel
from gia.processing import GiaProcessor

from torch.utils.data import DataLoader

from gia.eval.evaluator import Evaluator

class BaseLanguageModelingEvaluator(Evaluator):
    def __init__(self, args: Arguments):
        self.args = args

    def evaluate(self, model: GiaModel):
        model.eval()
        losses = []
        processor = GiaProcessor(self.args)
        dataset = load_dataset("gia-project/gia-dataset", self.task, split="test")
        dataset = dataset.map(lambda batch: processor(**batch), remove_columns=dataset.column_names, batched=True)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, collate_fn=GIADataCollator(), shuffle=True)
        for step, batch in enumerate(dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            losses.append(outputs.loss)
            if 0 < step == self.args.max_eval_steps:
                break

        loss = torch.mean(torch.stack(losses))
        try:
            perplexity = torch.exp(loss)
        except OverflowError:
            perplexity = float("inf")
        return perplexity.item()