"""Funções aplicadas aos valores esperados antes de combiná-los com a MLP."""

import torch


def sequential(values):
    """[<P1>, <P2>², <P3>³, ..., <PM>^M]."""
    powers = torch.arange(1, len(values) + 1, device=values.device)
    return values ** powers


def linear(values):
    """Alternativa simples para comparações futuras: expoente 1 em todos."""
    return values


FUNCTIONS = {"sequential": sequential, "linear": linear}
