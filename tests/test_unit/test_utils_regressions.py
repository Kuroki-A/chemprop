from types import SimpleNamespace

import torch

from chemprop.utils import build_lr_scheduler


def test_lr_scheduler_uses_nonzero_ceiling_batch_count():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([parameter])
    args = SimpleNamespace(
        warmup_epochs=2,
        epochs=30,
        num_lrs=1,
        train_data_size=51,
        batch_size=50,
        init_lr=1e-4,
        max_lr=1e-3,
        final_lr=1e-4,
    )

    scheduler = build_lr_scheduler(optimizer, args)
    assert scheduler.steps_per_epoch == 2

    args.train_data_size = 1
    scheduler = build_lr_scheduler(optimizer, args)
    assert scheduler.steps_per_epoch == 1
