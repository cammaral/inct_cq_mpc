from pathlib import Path
import json
import pandas as pd

root = Path("runs/archive/results")
rows = []

for file in root.rglob("metrics.json"):
    relative = file.relative_to(root)
    parts = relative.parts

    if len(parts) < 5:
        continue

    folder, problem, model, seed_folder = parts[:4]

    if not seed_folder.startswith("seed_"):
        continue

    with file.open() as f:
        metrics = json.load(f)

    row = {
        "experiment": folder,
        "problem": problem,
        "model": model,
        "seed": seed_folder.replace("seed_", ""),
    }
    row.update(metrics)
    rows.append(row)

df = pd.DataFrame(rows)

output = Path("runs/analysis")
output.mkdir(parents=True, exist_ok=True)

df.to_csv(output / "all_metrics.csv", index=False)

print(f"Resultados encontrados: {len(df)}")
print(f"Arquivo: {output / 'all_metrics.csv'}")
