import numpy as np
import torch

from chemprop.models import MoleculeModel
from chemprop.train.loss_functions import evidential_loss, normal_mve


def _uncertainty_transform_model():
    model = MoleculeModel.__new__(MoleculeModel)
    torch.nn.Module.__init__(model)
    model.softplus = torch.nn.Softplus()
    return model


def test_extreme_negative_uncertainty_logits_remain_strictly_positive():
    model = _uncertainty_transform_model()
    logits = torch.full((3,), -1000.0)

    parameters = model._positive_uncertainty_parameter(logits)
    alpha = parameters[1] + 1
    total_variance = parameters[2] * (1 + 1 / parameters[0]) / (alpha - 1)
    aleatoric_variance = parameters[2] / (alpha - 1)
    epistemic_variance = parameters[2] / (parameters[0] * (alpha - 1))

    assert torch.all(parameters >= torch.finfo(parameters.dtype).eps)
    assert alpha.item() > 1
    assert torch.isfinite(total_variance)
    assert torch.isfinite(aleatoric_variance)
    assert torch.isfinite(epistemic_variance)


def test_mve_loss_clamps_zero_variance_to_a_finite_floor():
    predictions = torch.tensor([[0.0, 0.0]], requires_grad=True)
    targets = torch.tensor([[1.0]])

    loss = normal_mve(predictions, targets).sum()
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(predictions.grad).all()


def test_evidential_loss_clamps_degenerate_parameters_to_finite_values():
    # mu=0, lambda=0, alpha=1, beta=0 is the limiting output produced when
    # softplus underflows for each evidence logit.
    predictions = torch.tensor([[0.0, 0.0, 1.0, 0.0]], requires_grad=True)
    targets = torch.tensor([[1.0]])

    loss = evidential_loss(predictions, targets).sum()
    loss.backward()

    assert np.isfinite(loss.item())
    assert torch.isfinite(predictions.grad).all()
