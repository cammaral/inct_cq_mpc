# INCT — Cinco notebooks temáticos

1. `01_MLP.ipynb` — MLP: largura, profundidade, superfícies, três tempos e convergência.
2. `02_MPC_Classica.ipynb` — MPC clássica: termos, média/soma, potências.
3. `03_MPC_Hibrida.ipynb` — MPC híbrida: termos, correlatores Pauli, profundidade, congelamento, acoplamento.
4. `04_Treinamento.ipynb` — LR, scheduler, épocas, dados, batch size, arquitetura clássica auxiliar.
5. `05_Comparacao_Geral.ipynb` — MLP, MPC clássica, MPC híbrida e QNN.

Coloque estes arquivos em `notebooks/` no repositório. Abra com o kernel Python (INCT MPC).

Em cada notebook, altere `SOURCE = 'archive'` para `SOURCE = 'runs/jobs/2316/results'` (ou ID correspondente) para os resultados novos. **O arquivo histórico representa a arquitetura anterior de múltiplos circuitos; não o misture com o experimento novo de circuito compartilhado.**

As análises calculam média e desvio-padrão de métricas e predições entre seeds, e salvam as figuras em `runs/analysis/notebooks_tematicos/`. Quando os dados de uma ablação ainda não existirem, a célula informa e continua sem fabricar resultados. As superfícies de Black–Scholes usam strike=40 para converter predições normalizadas em preços; em outras configurações, ajuste o fator.

**Nota:** grupos têm variantes históricas e novas; a ausência de uma variante é normal na campanha em que ela não foi executada. Nenhuma célula executa treinamento.
