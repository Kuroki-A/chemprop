import math
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def compute_pnorm(model: nn.Module) -> float:
    """
    Computes the norm of the parameters of a model.

    :param model: A PyTorch model.
    :return: The norm of the parameters of the model.
    """
    return math.sqrt(sum([p.norm().item() ** 2 for p in model.parameters()]))


def compute_gnorm(model: nn.Module) -> float:
    """
    Computes the norm of the gradients of a model.

    :param model: A PyTorch model.
    :return: The norm of the gradients of the model.
    """
    return math.sqrt(sum([p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None]))


def param_count(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.

    :param model: An PyTorch model.
    :return: The number of trainable parameters in the model.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def param_count_all(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.

    :param model: An PyTorch model.
    :return: The number of trainable parameters in the model.
    """
    return sum(param.numel() for param in model.parameters())


def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in :code:`index`.

    :param source: A tensor of shape :code:`(num_bonds, hidden_size)` containing message features.
    :param index: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds)` containing the atom or bond
                  indices to select from :code:`source`.
    :return: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds, hidden_size)` containing the message
             features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    return target


def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.

    Supports:

    * :code:`ReLU`
    * :code:`LeakyReLU`
    * :code:`PReLU`
    * :code:`tanh`
    * :code:`SELU`
    * :code:`ELU`

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')


def initialize_weights(model: nn.Module) -> None:
    """
    Initializes the weights of a model in place.

    :param model: An PyTorch model.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where :code:`warmup_steps = warmup_epochs * steps_per_epoch`).
    Then the learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr` over the
    course of the remaining :code:`total_steps - warmup_steps` (where :code:`total_steps =
    total_epochs * steps_per_epoch`). This is roughly based on the learning rate
    schedule from `Attention is All You Need <https://arxiv.org/abs/1706.03762>`_, section 5.3.
    """
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: List[Union[float, int]],
                 total_epochs: List[int],
                 steps_per_epoch: int,
                 init_lr: List[float],
                 max_lr: List[float],
                 final_lr: List[float]):
        """
        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after :code:`warmup_epochs`).
        :param final_lr: The final learning rate (achieved after :code:`total_epochs`).
        """
        if not (
            len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs)
            == len(init_lr) == len(max_lr) == len(final_lr)
        ):
            raise ValueError(
                "Number of param groups must match the number of epochs and learning rates! "
                f"got: len(optimizer.param_groups)= {len(optimizer.param_groups)}, "
                f"len(warmup_epochs)= {len(warmup_epochs)}, "
                f"len(total_epochs)= {len(total_epochs)}, "
                f"len(init_lr)= {len(init_lr)}, "
                f"len(max_lr)= {len(max_lr)}, "
                f"len(final_lr)= {len(final_lr)}"
            )

        if (
            not isinstance(steps_per_epoch, (int, np.integer))
            or isinstance(steps_per_epoch, (bool, np.bool_))
            or steps_per_epoch < 1
        ):
            raise ValueError('steps_per_epoch must be a positive integer.')
        if any(
            not isinstance(epoch, (int, np.integer))
            or isinstance(epoch, (bool, np.bool_))
            or epoch < 0
            for epoch in total_epochs
        ):
            raise ValueError('total_epochs values must be non-negative integers.')

        numeric_types = (int, float, np.integer, np.floating)
        if any(
            not isinstance(value, numeric_types)
            or isinstance(value, (bool, np.bool_))
            for values in (warmup_epochs, init_lr, max_lr, final_lr)
            for value in values
        ):
            raise ValueError('Warmup epochs and learning rates must be numeric scalars.')
        try:
            warmup_epochs_array = np.asarray(warmup_epochs, dtype=float)
            init_lr_array = np.asarray(init_lr, dtype=float)
            max_lr_array = np.asarray(max_lr, dtype=float)
            final_lr_array = np.asarray(final_lr, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError('Warmup epochs and learning rates must be numeric.') from error
        expected_shape = (len(optimizer.param_groups),)
        if any(
            values.shape != expected_shape
            for values in (warmup_epochs_array, init_lr_array, max_lr_array, final_lr_array)
        ):
            raise ValueError('Warmup epochs and learning rates must be one-dimensional.')
        if not np.all(np.isfinite(warmup_epochs_array)) or np.any(warmup_epochs_array < 0):
            raise ValueError('warmup_epochs values must be finite and non-negative.')
        total_epochs_array = np.asarray(total_epochs, dtype=int)
        for name, values in (
            ('init_lr', init_lr_array),
            ('max_lr', max_lr_array),
            ('final_lr', final_lr_array),
        ):
            if not np.all(np.isfinite(values)) or np.any(values <= 0):
                raise ValueError(f'{name} values must be finite and positive.')
        if np.any(max_lr_array < init_lr_array) or np.any(max_lr_array < final_lr_array):
            raise ValueError('max_lr values must be greater than or equal to init_lr and final_lr.')

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs_array
        self.total_epochs = total_epochs_array
        self.steps_per_epoch = int(steps_per_epoch)
        self.init_lr = init_lr_array
        self.max_lr = max_lr_array
        self.final_lr = final_lr_array

        self.current_step = 0
        self.lr = self.init_lr.copy()
        requested_warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        # A zero-epoch schedule is retained for Chemprop v1's evaluation-only
        # mode. warmup=0 begins directly with exponential decay. Clipping and
        # masked division keep both boundary cases finite and warning-free.
        self.warmup_steps = np.minimum(requested_warmup_steps, self.total_steps)
        self.linear_increment = np.zeros(self.num_lrs, dtype=float)
        np.divide(
            self.max_lr - self.init_lr,
            self.warmup_steps,
            out=self.linear_increment,
            where=self.warmup_steps > 0,
        )

        decay_steps = self.total_steps - self.warmup_steps
        inverse_decay_steps = np.zeros(self.num_lrs, dtype=float)
        np.divide(
            1.0,
            decay_steps,
            out=inverse_decay_steps,
            where=decay_steps > 0,
        )
        self.exponential_gamma = np.power(
            self.final_lr / self.max_lr,
            inverse_decay_steps,
        )

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """
        Gets a list of the current learning rates.

        :return: A list of the current learning rates.
        """
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.

        :param current_step: Optionally specify what step to set the learning rate to.
                             If None, :code:`current_step = self.current_step + 1`.
        """
        if current_step is not None:
            if (
                not isinstance(current_step, (int, np.integer))
                or isinstance(current_step, (bool, np.bool_))
                or current_step < 0
            ):
                raise ValueError('current_step must be a non-negative integer.')
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]


def activate_dropout(module: nn.Module, dropout_prob: float):
    """
    Set p of dropout layers and set to train mode during inference for uncertainty estimation.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param dropout_prob: A float on (0,1) indicating the dropout probability.
    """
    if isinstance(module, nn.Dropout):
        module.p = dropout_prob
        module.train()
