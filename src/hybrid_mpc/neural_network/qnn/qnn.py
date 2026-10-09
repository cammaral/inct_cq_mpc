"""QNN: um circuito e uma lista ordenada de correlatores selecionados."""

from torch import nn
from ...quantum.circuits import Circuit
from ...quantum.correlators import measure


class QNN(nn.Module):
    def __init__(self, n_qubits, observables, layers=2, trainable=True, init_std=0.5):
        super().__init__()
        self.observables = observables
        self.circuit = Circuit(n_qubits, layers, trainable, init_std)

    def forward(self):
        """Retorna M valores esperados; nao depende da entrada x."""
        state = self.circuit()
        return measure(state, self.observables)
