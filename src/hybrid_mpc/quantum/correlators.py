"""Calculo de <P> para observaveis formados por I, X, Y e Z.

Cada string tem um caractere por qubit: 'XXI', 'XYZ', 'ZII'.
Os valores esperados sao reais, mas o vetor de estado e complexo.
"""

import torch

PAULI = {
    "I": torch.eye(2, dtype=torch.complex64),
    "X": torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64),
    "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64),
    "Z": torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64),
}


def expectation(state, observable):
    """Calcula <psi|P|psi> por operacoes locais, sem montar matriz 2**N."""
    from .circuits import apply_gate

    changed = state
    for wire, letter in enumerate(observable):
        changed = apply_gate(changed, PAULI[letter].to(state.device), wire, len(observable))
    return torch.vdot(state, changed).real


def measure(state, observables):
    """Devolve um vetor com os valores esperados na ordem especificada."""
    return torch.stack([expectation(state, name) for name in observables])
