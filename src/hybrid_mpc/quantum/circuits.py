"""Simulador statevector simples e diferenciavel, feito apenas com PyTorch.

Cada camada aplica RY, RZ e uma cadeia de portas CZ.
O estado e complexo para permitir medidas gerais com Pauli Y.

A ordenacao dos qubits e a mesma das strings de Pauli:
'ZII' mede Z no primeiro qubit de um circuito com 3 qubits.
"""

import torch
from torch import nn


I = torch.eye(2, dtype=torch.complex64)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)


def rotation(angle, pauli):
    """R_P(angulo) = cos(angulo/2) I - i sin(angulo/2) P."""
    return torch.cos(angle / 2) * I.to(angle.device) - 1j * torch.sin(angle / 2) * pauli.to(angle.device)


def apply_gate(state, gate, wire, n_qubits):
    """Aplica uma matriz 2x2 a um qubit do vetor de estado."""
    shape = (2,) * n_qubits
    state = state.reshape(shape).movedim(wire, -1)
    state = state @ gate.T
    return state.movedim(-1, wire).reshape(-1)


def apply_cz(state, wire, n_qubits):
    """Aplica CZ entre wire e wire+1 (inverte a fase do estado |11>)."""
    state = state.reshape((2,) * n_qubits)
    state = state.movedim((wire, wire + 1), (-2, -1))
    phase = state.new_tensor([1, 1, 1, -1]).reshape(2, 2)
    state = state * phase
    return state.movedim((-2, -1), (wire, wire + 1)).reshape(-1)


class Circuit(nn.Module):
    """Ansatz parametrizado de N qubits com L camadas."""

    def __init__(self, n_qubits=3, layers=2, trainable=True, init_std=0.5):
        super().__init__()
        self.n_qubits = n_qubits
        self.layers = layers
        self.theta = nn.Parameter(
            init_std * torch.randn(layers, n_qubits, 2),
            requires_grad=trainable,
        )

    def forward(self):
        """Devolve |psi(theta)> com dimensao 2**N."""
        state = torch.zeros(2**self.n_qubits, dtype=torch.complex64, device=self.theta.device)
        state[0] = 1.0

        for layer in range(self.layers):
            for wire in range(self.n_qubits):
                state = apply_gate(state, rotation(self.theta[layer, wire, 0], Y), wire, self.n_qubits)
                state = apply_gate(state, rotation(self.theta[layer, wire, 1], Z), wire, self.n_qubits)

            for wire in range(self.n_qubits - 1):
                state = apply_cz(state, wire, self.n_qubits)

        return state
