"""Testes da sequência de potências e da soma de termos."""

import torch
from hybrid_mpc.mpc.functions import sequential
from hybrid_mpc.mpc.coupling import MPC


def test_sequential():
    values = torch.tensor([0.5, -0.5, 0.5])
    assert torch.allclose(sequential(values), torch.tensor([0.5, 0.25, 0.125]))


def test_mpc_and_gradient():
    classical = torch.zeros(4, 3)
    quantum = torch.tensor([0.5, -0.5, 0.5], requires_grad=True)
    output = MPC()(classical, quantum)
    assert output.shape == (4, 1)
    assert torch.allclose(output, torch.full((4, 1), 0.875))
    output.sum().backward()
    assert quantum.grad is not None
