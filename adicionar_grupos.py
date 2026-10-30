from pathlib import Path

# 1. Atualizar run.py
path = Path("run.py")
code = path.read_text()

# Adicionar argumento ao argparse.
if "--ablation-group" not in code:
    marker = "    args = parser.parse_args()"
    if marker not in code:
        raise RuntimeError("Não encontrei parser.parse_args() em run.py")

    code = code.replace(
        marker,
        '''    parser.add_argument(
        "--ablation-group",
        choices=["width", "depth", "lr", "epochs", "scheduler", "batch"],
        default=None,
    )
''' + marker,
        1,
    )

# Adicionar seleção de grupo.
# O grupo é convertido para uma lista de nomes, preservando o
# mecanismo existente de execução de uma ablação por vez.
path.write_text(code)

# 2. Atualizar job.pbs
path = Path("job.pbs")
code = path.read_text()

old = '''if [[ -n "${ABLATION:-}" ]]; then args+=(--ablation "$ABLATION")
elif [[ "${ABL:-0}" == "1" ]]; then args+=(--ablations); fi'''

new = '''if [[ -n "${ABLATION:-}" ]]; then
    args+=(--ablation "$ABLATION")
elif [[ -n "${ABLATION_GROUP:-}" ]]; then
    args+=(--ablation-group "$ABLATION_GROUP")
elif [[ "${ABL:-0}" == "1" ]]; then
    args+=(--ablations)
fi'''

if "ABLATION_GROUP" not in code:
    if old not in code:
        raise RuntimeError("Não encontrei o bloco de flags esperado em job.pbs")
    code = code.replace(old, new, 1)

path.write_text(code)

print("Flags adicionadas ao run.py e job.pbs.")
print("ATENÇÃO: ainda falta integrar o filtro de grupos ao loop de execução.")
