import pytest
from accelerate import Accelerator
from torch.utils.data import DataLoader

from gia.config import Arguments
from gia.datasets import collate_fn, load_mixed_dataset
from gia.model import GiaModel


@pytest.mark.parametrize("use_accelerate", [True, False])
def test_gia_accelerate(use_accelerate):
    args = Arguments(task_names=["mujoco-ant", "metaworld-assembly"], embed_dim=48, seq_len=256, output_dir="tests/test_gia_accelerate")
    dataset = load_mixed_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    model = GiaModel(args)
    if use_accelerate:
        accelerator = Accelerator()
        model, dataloader = accelerator.prepare(model, dataloader)
    batch = next(iter(dataloader))
    output = model(batch["batch"])
    assert output.loss.item() > 0.0
