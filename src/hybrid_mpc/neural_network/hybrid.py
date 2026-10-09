"""Une MLP, QNN e MPC para uma regressão com saída escalar."""

from torch import nn
from .classical.mlp import MLP
from .qnn.qnn import QNN
from ..mpc.coupling import MPC


class Hybrid(nn.Module):
    """A MLP produz M coeficientes; a QNN produz M valores esperados."""

    def __init__(
        self,
        inputs=1,
        width=32,
        hidden_layers=2,
        activation="tanh",
        n_qubits=3,
        observables=("ZII", "ZZI", "ZZZ"),
        q_layers=2,
        trainable=True,
        init_std=0.5,
        mpc_function="sequential",
    ):
        super().__init__()
        self.classical = MLP(inputs, len(observables), width, hidden_layers, activation)
        self.quantum = QNN(n_qubits, list(observables), q_layers, trainable, init_std)
        self.mpc = MPC(mpc_function)

    def forward(self, x):
        """Retorna [batch, 1], preservando o gradiente para a MLP e a QNN."""
        y = self.classical(x)       # [batch, M]
        q = self.quantum()         # [M]; independente de x
        return self.mpc(y, q)      # [batch, 1]

    def components(self, x):
        """Expõe valores clássicos, quânticos e cada termo do MPC."""
        y = self.classical(x)
        q = self.quantum()
        terms = self.mpc.terms(y, q)
        return {"classical": y, "quantum": q,
                "terms": terms, "output": terms.sum(dim=1, keepdim=True)}


def from_config(config):
    """Constrói o modelo usando a seção 'model' de um YAML."""
    return Hybrid(**config["model"])
