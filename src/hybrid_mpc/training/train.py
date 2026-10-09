"""Treinamento simples para regressão com MSE (ou outra loss fornecida)."""

import torch


def train(model, loss_fn, epochs=400, lr=0.01):
    """Atualiza parâmetros treináveis e retorna uma loss por época."""
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    history = []

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model)
        loss.backward()
        optimizer.step()
        history.append(loss.item())

    return history
