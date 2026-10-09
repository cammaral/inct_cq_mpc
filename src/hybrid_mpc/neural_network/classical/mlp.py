"""Rede MLP classica com numero configuravel de entradas e saidas."""

from torch import nn


def MLP(inputs=1, outputs=3, width=32, layers=2, activation="tanh"):
    """Monta Linear + ativacao repetidos, terminando em camada Linear."""
    activations = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}
    parts = []
    size = inputs

    for _ in range(layers):
        parts.extend([nn.Linear(size, width), activations[activation]()])
        size = width

    parts.append(nn.Linear(size, outputs))
    return nn.Sequential(*parts)
