"""Acoplamento dos coeficientes clássicos com observáveis quânticos."""

from torch import nn
from .functions import FUNCTIONS


class MPC(nn.Module):
    def __init__(self, function="sequential"):
        super().__init__()
        self.function = FUNCTIONS[function]

    def terms(self, classical, quantum):
        """Cada coluna j: (y_j(x) + 1) * <P_j>^(j+1)."""
        return (classical + 1.0) * self.function(quantum)

    def forward(self, classical, quantum):
        """Soma os M termos, retornando um valor por amostra [batch, 1]."""
        return self.terms(classical, quantum).sum(dim=1, keepdim=True)
