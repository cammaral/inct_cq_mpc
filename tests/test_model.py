"""Testa formato de saída e aprendizado da regressão."""

import torch
from hybrid_mpc import Hybrid
from hybrid_mpc.loss.supervised import mse
from hybrid_mpc.training.train import train
from hybrid_mpc.problems.regression_example import cosine_data


def test_output_shape():
    model = Hybrid(n_qubits=2, observables=["ZI", "XX"])
    x = torch.randn(6, 1)
    assert model(x).shape == (6, 1)
    assert model.components(x)["terms"].shape == (6, 2)


def test_frozen_and_trainable_gradients():
    x, target = cosine_data(20)
    for trainable in (False, True):
        model = Hybrid(n_qubits=2, observables=["ZI", "XX"], trainable=trainable)
        mse(model(x), target).backward()
        assert model.quantum.circuit.theta.requires_grad is trainable
        assert (model.quantum.circuit.theta.grad is not None) is trainable
        assert model.classical[0].weight.grad is not None


def test_cosine_loss_decreases():
    torch.manual_seed(1924)
    x, y = cosine_data(35)
    model = Hybrid(n_qubits=2, observables=["ZI", "ZZ"], trainable=False)
    start = mse(model(x), y).item()
    train(model, lambda net: mse(net(x), y), epochs=30, lr=0.01)
    assert mse(model(x), y).item() < start
