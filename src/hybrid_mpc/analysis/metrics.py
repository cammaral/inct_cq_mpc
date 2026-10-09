"""Métricas para regressão."""

import torch


def rmse(prediction, target):
    """Raiz do erro quadrático médio."""
    return torch.sqrt(((prediction - target) ** 2).mean()).item()


def mae(prediction, target):
    """Erro absoluto médio."""
    return (prediction - target).abs().mean().item()
