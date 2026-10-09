"""Perda para a tarefa inicial: regressão."""

import torch.nn.functional as F


def mse(prediction, target):
    """Erro quadrático médio entre previsão e valor conhecido."""
    return F.mse_loss(prediction, target)
