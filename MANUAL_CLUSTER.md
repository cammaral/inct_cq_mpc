# Manual de execução — INCT MPC

## 1. Acesso ao cluster

No terminal do computador local:

    ssh -p 20022 camaral@146.164.45.180

Servidor: labcfd2020
Gerenciador de filas: PBS
Fila: workq
Ambiente Python: ~/miniforge3/envs/mpc/

## 2. Entrar no projeto

    cd ~/inct_cq_hybrid_nn

O config.yaml controla modelos, problemas, parâmetros
de treinamento, sementes e ablações.

O job.pbs controla os recursos solicitados ao cluster.

## 3. Verificar os nós disponíveis

    pbsnodes -aS

Para consultar os jobs em execução e os nós utilizados:

    qstat -n1

Nós com estado "free" estão disponíveis.
Nós "state-unknown" não devem ser utilizados.

## 4. Executar os modelos principais

Executar Burgers e Black-Scholes:

    qsub job.pbs

Executar apenas Burgers:

    qsub -v PROBLEM=burgers job.pbs

Executar apenas Black-Scholes:

    qsub -v PROBLEM=black_scholes job.pbs

Esses comandos executam os quatro modelos principais
e as sementes definidas no config.yaml.

## 5. Executar todas as ablações

Executar os dois problemas, modelos principais e ablações:

    qsub -v ABL=1 job.pbs

Executar tudo somente para Burgers:

    qsub -v PROBLEM=burgers,ABL=1 job.pbs

Executar tudo somente para Black-Scholes:

    qsub -v PROBLEM=black_scholes,ABL=1 job.pbs

Cada submissão cria um único job PBS.
O run.py executa as variantes sequencialmente.

## 6. Executar uma única ablação

Exemplo: correlatores de Pauli X em Burgers:

    qsub -v PROBLEM=burgers,ABLATION=hybrid_pauli_x job.pbs

Em Black-Scholes:

    qsub -v PROBLEM=black_scholes,ABLATION=hybrid_pauli_x job.pbs

Nos dois problemas:

    qsub -v ABLATION=hybrid_pauli_x job.pbs

A flag ABLATION seleciona somente a ablação indicada,
sem executar novamente os modelos principais.

Para ver os nomes das ablações:

    grep 'name:' config.yaml

## 7. Escolher um nó específico

Primeiro, verificar:

    pbsnodes -aS

Exemplo: solicitar o nó cn004:

    qsub -l select=1:ncpus=2:mem=4gb:vnode=cn004 -v ABL=1 job.pbs

Executar uma única ablação nesse nó:

    qsub -l select=1:ncpus=2:mem=4gb:vnode=cn004 -v PROBLEM=burgers,ABLATION=hybrid_pauli_x job.pbs

O PBS precisa aceitar a solicitação.
A alocação real deve ser confirmada com qstat.

Não é necessário escolher manualmente um nó.
Sem a restrição vnode, o PBS escolhe os recursos.

## 8. Acompanhar os jobs

Ver seus trabalhos:

    qstat -u camaral

Ver em qual nó estão executando:

    qstat -n1 -u camaral

Ver detalhes:

    qstat -f ID

Estados principais:

Q = aguardando
R = executando
H = retido

## 9. Consultar os logs

Cada job possui um identificador numérico.

Exemplo: job 2316

Acompanhar continuamente:

    tail -f runs/jobs/2316/run.log

Ver as últimas 30 linhas:

    tail -n 30 runs/jobs/2316/run.log

Sair do tail -f:

    Ctrl+C

Isso interrompe apenas a visualização,
não o treinamento do PBS.

## 10. Verificar resultados

Resultados de um job:

    ls runs/jobs/2316/results/

Contar arquivos de métricas:

    find runs/jobs/2316/results -name metrics.json | wc -l

Verificar o código de saída depois da conclusão:

    cat runs/jobs/2316/exit_code.txt

0 = finalizado sem erro detectado pelo script.
Outro número = erro na execução.

O arquivo exit_code.txt só aparece depois
que o script sai normalmente ou por erro capturado.

## 11. Cancelar um job

    qdel ID

Exemplo:

    qdel 2316

Isso cancela o job no PBS.
Resultados já gravados permanecem na pasta.

## 12. Organização dos arquivos

runs/
├── archive/
│   ├── results/      # Resultados antigos
│   ├── logs/         # Logs antigos
│   ├── configs/      # Configurações antigas
│   └── summaries/    # CSVs antigos
├── jobs/
│   └── ID/
│       ├── config.yaml
│       ├── job.pbs
│       ├── run.log
│       ├── exit_code.txt
│       └── results/
├── pbs_logs/
└── analysis/

Cada novo job utiliza seu próprio diretório.
Os resultados anteriores não são sobrescritos.

## 13. Modificar experimentos

Editar os parâmetros:

    vi config.yaml

Após modificar o YAML, submeter um novo job.

Jobs que já estão executando não adotam
automaticamente as alterações da configuração.

## 14. Atualizar o código

Verificar alterações locais:

    git status

Atualizar a partir do GitHub:

    git pull

Evitar enviar resultados, logs e ambientes virtuais
para o GitHub. Esses arquivos devem estar no .gitignore.

## 15. Observações importantes

- Não executar treinamentos pesados no nó de login.
- Utilizar qsub para enviar treinamentos.
- O cluster não anuncia GPUs pelo PBS.
- A configuração atual do job.pbs solicita 2 CPUs e 4 GB.
- O limite atual de walltime é 12 horas.
- Se ultrapassar o limite, o PBS poderá encerrar o job.
- Não submeter novamente experimentos já concluídos
  sem necessidade.
- Verificar a disponibilidade e as políticas de uso
  antes de solicitar muitos recursos.

## 16. Comandos mais utilizados

    pbsnodes -aS
    qsub job.pbs
    qsub -v ABL=1 job.pbs
    qsub -v ABLATION=hybrid_pauli_x job.pbs
    qstat -u camaral
    qstat -n1 -u camaral
    tail -f runs/jobs/ID/run.log
    qdel ID

