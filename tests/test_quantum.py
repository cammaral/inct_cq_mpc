"""Testa os correlatores gerais de Pauli e a diferenciabilidade."""

import math
import torch
from hybrid_mpc.quantum.correlators import expectation
from hybrid_mpc.neural_network.qnn.qnn import QNN


def test_pauli_on_known_states():
    zero = torch.tensor([1, 0], dtype=torch.complex64)
    plus = torch.tensor([1, 1], dtype=torch.complex64) / math.sqrt(2)
    plus_i = torch.tensor([1, 1j], dtype=torch.complex64) / math.sqrt(2)
    assert torch.allclose(expectation(zero, "Z"), torch.tensor(1.0))
    assert torch.allclose(expectation(plus, "X"), torch.tensor(1.0))
    assert torch.allclose(expectation(plus_i, "Y"), torch.tensor(1.0))


def test_correlator_two_qubits():
    bell = torch.tensor([1, 0, 0, 1], dtype=torch.complex64) / math.sqrt(2)
    assert torch.allclose(expectation(bell, "ZZ"), torch.tensor(1.0))
    assert torch.allclose(expectation(bell, "XX"), torch.tensor(1.0))


def test_trainable_quantum_gradients():
    torch.manual_seed(2026)
    qnn = QNN(2, ["ZI", "XX", "YZ"], layers=2)
    qnn().sum().backward()
    assert qnn.circuit.theta.grad is not None
    assert qnn.circuit.theta.grad.abs().sum() > 0


def test_frozen_quantum():
    qnn = QNN(2, ["ZI", "XX"], trainable=False)
    assert qnn.circuit.theta.requires_grad is False
