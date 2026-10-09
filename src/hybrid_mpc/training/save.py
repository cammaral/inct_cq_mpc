"""Salva os resultados de uma execução em uma nova pasta."""

import csv
import json
from pathlib import Path
import torch
import yaml


def save_run(folder, config, model, history, summary):
    """Grava configuração, pesos, MSE por época e métricas finais."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)  # Nunca sobrescreve.
    (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    torch.save(model.state_dict(), folder / "model.pt")
    with (folder / "history.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "loss"])
        writer.writerows((epoch, loss) for epoch, loss in enumerate(history, 1))
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
