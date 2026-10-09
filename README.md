# Hybrid MPC — versão inicial: somente regressão

Projeto didático e modular em **PyTorch**, com uma MLP em paralelo com uma QNN.
Nesta versão, o único problema implementado é a **regressão de `cos(x)`**.
Não há classificação nem PINNs implementadas.

## Regra do MPC

A MLP recebe `x` e produz `M` saídas `y_j(x)`. A QNN utiliza um circuito
de `N` qubits e calcula `M` valores esperados de correlatores de Pauli.
A conexão polinomial sequencial padrão é:

\[
\hat f(x) = \sum_{j=1}^{M} [y_j(x)+1] \langle P_j \rangle^j.
\]

**Exemplo:** `observables: [ZII, XXI, XYZ]` gera expoentes `1, 2, 3`.
A potência depende da posição na lista, não do peso (ordem) do correlator.
Os valores esperados são calculados no **mesmo estado** de `N` qubits.
A QNN não recebe `x` nesta versão; o vetor de observáveis é global.

## Organização

```text
hybrid_mpc_regression/
├── src/hybrid_mpc/
│   ├── neural_network/
│   │   ├── classical/mlp.py       # MLP
│   │   ├── qnn/qnn.py             # observáveis da QNN
│   │   └── hybrid.py              # rede paralela MLP + QNN
│   ├── quantum/
│   │   ├── circuits.py            # circuito RY, RZ e CZ
│   │   └── correlators.py         # <P> com I, X, Y, Z
│   ├── mpc/
│   │   ├── functions.py           # potências sequenciais
│   │   └── coupling.py            # soma dos termos
│   ├── problems/regression_example.py  # dados cos(x)
│   ├── loss/supervised.py         # MSE
│   ├── training/                  # otimizador e salvamento
│   └── analysis/metrics.py       # RMSE, MAE
├── configs/
│   ├── regression.yaml           # QNN treinável
│   └── regression_frozen.yaml    # QNN congelada
├── scripts/run_regression.py
├── notebooks/01_cosine_regression.ipynb
├── tests/
├── runs/
└── pyproject.toml
```

## Instalação

Execute na pasta raiz do projeto:

```bash
python -m pip install -e '.[notebook,test]'
```

Requisitos: Python >=3.10, PyTorch, NumPy, Matplotlib e PyYAML.
O notebook usa `ipykernel`/Jupyter. **Não precisa de PennyLane**:
o simulador do estado quântico é implementado com operações PyTorch.

## Notebook de cosseno

Abra `notebooks/01_cosine_regression.ipynb` e execute todas as células.
Ele treina duas cópias da arquitetura com a mesma seed:

- QNN congelada: parâmetros `theta` não recebem gradiente.
- QNN treinável: parâmetros `theta` recebem gradiente.

O notebook exibe as curvas `cos(x)` e preditas, histórico da MSE,
contribuições de cada termo MPC, valores esperados antes/depois,
potências efetivas e métricas RMSE/MAE. Salva resultados em `runs/`
sem sobrescrever uma execução anterior.

Também é possível executar via terminal:

```bash
python scripts/run_regression.py --config configs/regression.yaml
python scripts/run_regression.py --config configs/regression_frozen.yaml
```

Cada execução salva `config.yaml`, `model.pt`, `history.csv`, `summary.json`.
Se a pasta da execução já existir, o script apenas imprime `SKIP`.

## Como escolher os correlatores

Edite `observables` no YAML:

```yaml
n_qubits: 3
observables: [ZII, XXI, XYZ, YYY]
```

`I` é identidade no qubit correspondente. A lista deve conter uma letra
(`I`, `X`, `Y` ou `Z`) por qubit. O código mantém a ordem da lista.
O estado é complexo e permite correlatores com Pauli-Y.

## Como trocar a função de conexão futuramente

O MPC está isolado em `src/hybrid_mpc/mpc/functions.py`.
Para adicionar outra fórmula, crie uma função de `values` e registre-a
no dicionário `FUNCTIONS`; depois escolha-a em `mpc_function` no YAML.
A alternativa `linear` já existe para comparação, mas **o padrão é
sempre `sequential`**.

## Sobre a interpretação física

Como a QNN **não recebe a entrada** `x`, cada `<P_j>` é constante ao longo
da malha. Quando congelada, a QNN apenas fixa fatores dos termos da MLP;
quando treinável, esses fatores podem mudar. Isto **não prova vantagem
quântica**. Potências altas de valores esperados próximos de zero podem
diminuir termos e gradientes. O simulador de estado tem custo exponencial
em `N` (dimensão `2**N`).

## Testes

```bash
python -m pytest -q
```
