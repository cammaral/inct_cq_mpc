# INCT — MPC em PyTorch (protótipo inicial)

Comparação supervisionada de quatro arquiteturas, usando **soluções analíticas** de Burgers 1D e Black–Scholes. **Não é PINN**: só há perda MSE entre a predição e a função exata, sem resíduo de EDP.

## Rodar

```bash
python -m pip install -r requirements.txt
python run.py                         # ambos os problemas, quatro modelos
python run.py --problem burgers       # somente Burgers
python run.py --problem black_scholes # somente Black-Scholes
python run.py --ablations             # ablações adicionais do config.yaml
python -m unittest discover -s tests -v
```

Tudo é configurado em **`config.yaml`**. Existe **um único `run.py`**. `device: auto` escolhe CUDA quando disponível, senão CPU. A QNN é simulada diretamente em PyTorch com statevector (RY, RZ, CZ); não é necessária biblioteca quântica extra. Como o custo escala em `2**qubits`, comece com 1–4 qubits.

## Modelos

- **MLP:** `u(x,t) = f(x,t)`.
- **MPC clássica:** `u(x,t) = mean_i [(1 + y_i(x,t)) * z_i**i]`. Uma MLP principal recebe `(x,t)`; a MLP auxiliar recebe **apenas 1 constante fixa**, nunca `(x,t)`.
- **MPC híbrida:** `u(x,t) = mean_i [(1 + y_i(x,t)) * <P_i>(phi_i)**i]`. Uma MLP principal recebe `(x,t)`; o **i-ésimo circuito auxiliar possui exatamente i qubits**, sem entrada de dados. `terms: 3` cria circuitos de 1, 2 e 3 qubits com observáveis padrão `Z`, `ZZ`, `ZZZ`. Troque as palavras de Pauli no YAML. Cada circuito tem seus próprios parâmetros `phi_i`.
- **QNN de referência:** recebe `(x,t)` por angle encoding, e suas medidas alimentam uma camada linear.

As MPC usam `aggregation: mean` por padrão. O YAML permite `sum`, `powers: false`, circuitos congelados, observáveis diferentes, tamanhos, LR, épocas, scheduler e quantidade de dados como **ablações**. `LeakyReLU(0.1)` é bijetora de `R` em `R` nas camadas ocultas. A prova matemática da MPC deve ser verificada separadamente quanto às demais hipóteses; esta implementação não afirma uma garantia de vantagem quântica.

## Dados analíticos

**Burgers**: `u_t + u*u_x = nu*u_xx`. A família exata via Cole–Hopf é

`u(x,t)=2 nu a pi exp(-nu pi²t) sin(pi x) / [b + a exp(-nu pi²t) cos(pi x)]`

com `nu>0` e `b>|a|`. Configuração inicial: `(x,t) in [0,1]^2`, `nu=0.05`, `a=0.9`, `b=1`. Essa função tem `u(0,t)=u(1,t)=0`.

**Black–Scholes**: call europeia sem dividendos, `C=S Phi(d1)-K exp(-r tau) Phi(d2)` e payoff `max(S-K,0)` em `tau=0`. A rede aprende `C/K` para normalizar os valores; `metrics.json` inclui `rmse_price` nas unidades originais.

Os dois problemas usam a mesma estratégia de amostragem para treino, validação e teste regular. As sementes garantem os **mesmos pontos de dados** para todas as arquiteturas sob a mesma semente.

## Resultados

Em `results/<problema>/summary.csv` ficam os indicadores por modelo/semente. Em `results/<problema>/<modelo>/seed_<n>/`:

- `metrics.json`: RMSE, erro relativo L2, MAE, tempo, total de parâmetros, época ótima, coeficientes inicial/final/ótimo;
- `history.csv` e `loss.png`: MSE de treino/validação, normas dos gradientes total/auxiliar e learning rate por época;
- `expectations.csv` e `auxiliary_evolution.png`: **MPC híbrida**, por época, `exp_i=<P_i>`, `power_i=<P_i>**i` e `effective_i=power_i/N` para a média;
- `coefficients.csv` e `auxiliary_evolution.png`: o mesmo acompanhamento para os coeficientes `z_i` da MPC clássica;
- `solution.png`: solução exata, predição e mapa de erro absoluto;
- `predictions.csv`: pontos do teste, valor real, predição, erro (normalizados por K em Black–Scholes);
- `best.pt`: parâmetros da época com menor MSE de validação.

Os valores esperados evoluem em função **da época de treinamento**, não de `x,t`: os circuitos auxiliares são independentes dos dados de entrada.

## Limitações e experimentos futuros

Um circuito sem entrada de dados produz coeficientes globais (não features quânticas dependentes do espaço/tempo). Assim, a principal hipótese empírica é sobre **parametrização, otimização e estabilidade**, não vantagem de expressividade garantida. Para um estudo publicável, aumentar o número de sementes (por exemplo 5), comparar redes clássicas com contagem de parâmetros parecida e analisar o custo do simulador; também testar observáveis de menor módulo, já que o termo `<P_i>**i` pode suprimir gradientes para ordens grandes.


## Matriz de ablações atualizada

O `config.yaml` inclui ablações em **ambos** os problemas com o mesmo `run.py`:
- Acoplamento: média/soma, potências ativadas/desativadas;
- Circuitos: 1/3/4 termos (respectivamente 1...N qubits), 2/4 camadas, Pauli Z/X/misto, circuito treinável/congelado;
- MLP: profundidade 1/3, largura 16/32, `tanh` como padrão;
- Treino: LR inicial, LR mínimo, scheduler, número de épocas, tamanho de lote;
- Dados: 128/256/512 pontos de treinamento;
- Controles clássicos: MPC clássica com e sem potências, 1/3/4 termos.

A primeira execução (`python run.py`) testa os quatro modelos-base nos dois problemas. `python run.py --ablations` inclui todas as variantes. Para testes estatísticos, mude `seeds` para `[0,1,2,3,4]` antes da execução completa. As ablações são **uma por vez em relação ao baseline** (não produto cartesiano), para evitar explosão de custos.

**Interpretação:** como o ramo quântico auxiliar não depende da entrada, os observáveis são coeficientes globais. Diferenças de desempenho não implicam vantagem quântica, e a validade da `tanh` depende de a prova requerer injetividade + TW e inversibilidade apenas sobre a imagem.
