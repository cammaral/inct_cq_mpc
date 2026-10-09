"""Problema inicial: aproximar a função cos(x) em [-pi, pi]."""

import math
import torch


def cosine_data(points=80):
    """Retorna pontos x e alvos cos(x), ambos no formato [points, 1]."""
    x = torch.linspace(-math.pi, math.pi, points).reshape(-1, 1)
    return x, torch.cos(x)
