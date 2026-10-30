from pathlib import Path
import shutil

# ---------- run.py ----------
file = Path("run.py")
code = file.read_text()

shutil.copy2(file, "run_backup.py")

code = code.replace(
    'parser.add_argument("--ablations", action="store_true")',
    'parser.add_argument("--ablations", action="store_true")\n'
    '    parser.add_argument("--ablation", type=str, default=None)'
)

start = code.index('    variants = [(copy.deepcopy(cfg), m)')
end = code.index('    problems =', start)

code = code[:start] + '''    if args.ablation and args.ablations:
        parser.error("Use --ablation OU --ablations, nao ambos.")

    by_name = {m["name"]: m for m in cfg["models"]}

    if args.ablation:
        selected = [
            a for a in cfg.get("ablations", [])
            if a["name"] == args.ablation
        ]
        if not selected:
            parser.error(f"Ablacao desconhecida: {args.ablation}")
        variants = []
    else:
        variants = [(copy.deepcopy(cfg), m) for m in cfg["models"]]
        selected = cfg.get("ablations", []) if args.ablations else []

    for ablation in selected:
        altered = copy.deepcopy(cfg)
        base = copy.deepcopy(by_name[ablation["model"]])
        base["name"] = ablation["name"]

        for path, value in ablation["changes"].items():
            if path.startswith("model."):
                set_path(base, path.removeprefix("model."), value)
            else:
                set_path(altered, path, value)

        variants.append((altered, base))

''' + code[end:]

file.write_text(code)

# ---------- job.pbs ----------
file = Path("job.pbs")
code = file.read_text()

shutil.copy2(file, "job_backup.pbs")

old = '''if [[ "${ABL:-0}" == "1" ]]; then
    args+=(--ablations)
fi'''

new = '''if [[ -n "${ABLATION:-}" ]]; then
    args+=(--ablation "$ABLATION")
elif [[ "${ABL:-0}" == "1" ]]; then
    args+=(--ablations)
fi'''

if old not in code:
    raise SystemExit(
        "run.py atualizado; bloco esperado nao encontrado em job.pbs."
    )

file.write_text(code.replace(old, new))

print("Flags atualizadas com sucesso.")
