"""Exemplo: python scripts/run_regression.py --config configs/regression.yaml"""

import argparse
from pathlib import Path
import torch
import yaml

from hybrid_mpc import from_config
from hybrid_mpc.analysis.metrics import rmse, mae
from hybrid_mpc.loss.supervised import mse
from hybrid_mpc.problems.regression_example import cosine_data
from hybrid_mpc.training.train import train
from hybrid_mpc.training.save import save_run


def main():
    parser = argparse.ArgumentParser(description="Aproxima cos(x) com MLP + QNN + MPC")
    parser.add_argument("--config", default="configs/regression.yaml")
    parser.add_argument("--name", default="cosine")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    folder = Path("runs") / args.name / Path(args.config).stem
    if folder.exists():
        print("SKIP: resultado já existe em", folder)
        return

    torch.manual_seed(config["train"]["seed"])
    x, target = cosine_data(config["train"]["points"])
    model = from_config(config)
    history = train(model, lambda net: mse(net(x), target),
                    config["train"]["epochs"], config["train"]["lr"])

    with torch.no_grad():
        # Avaliação em uma malha mais densa no mesmo intervalo.
        from math import pi
        x_test = torch.linspace(-pi, pi, 300).reshape(-1, 1)
        actual = torch.cos(x_test)
        predicted = model(x_test)
        summary = {"rmse": rmse(predicted, actual),
                   "mae": mae(predicted, actual),
                   "final_mse": history[-1]}
    save_run(folder, config, model, history, summary)
    print("Concluído:", folder, summary)


if __name__ == "__main__":
    main()
