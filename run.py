"""Ponto unico de entrada: python run.py --config config.yaml [--ablations]."""
import argparse
import copy
import csv
from pathlib import Path
import yaml
from mpc.train import train_one


def set_path(d, dotted, value):
    parts = dotted.split(".")
    for part in parts[:-1]:
        d = d[part]
    d[parts[-1]] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--problem", choices=["burgers", "black_scholes"])
    parser.add_argument("--ablations", action="store_true")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    variants = [(copy.deepcopy(cfg), m) for m in cfg["models"]]
    if args.ablations:
        by_name = {model["name"]: model for model in cfg["models"]}
        for ablation in cfg.get("ablations", []):
            altered = copy.deepcopy(cfg)
            base = copy.deepcopy(by_name[ablation["model"]])
            base["name"] = ablation["name"]
            for path, value in ablation["changes"].items():
                if path.startswith("model."):
                    set_path(base, path.removeprefix("model."), value)
                else:
                    set_path(altered, path, value)
            variants.append((altered, base))
    problems = [args.problem] if args.problem else list(cfg["problems"])
    for problem in problems:
        folder = Path(cfg.get("output", "results"))/problem
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        for variant_cfg, model in variants:
            for seed in cfg.get("seeds", [0]):
                print(f"[{problem}] {model['name']} / seed={seed}", flush=True)
                row = train_one(variant_cfg, problem, model, int(seed),
                                folder/model["name"]/f"seed_{seed}")
                rows.append(row)
                print(f"    RMSE={row['rmse']:.5f} | relative L2={row['relative_l2']:.4f}"
                      f" | {row['time_s']} s", flush=True)
                with (folder/"summary.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(
                        key for record in rows for key in record)))
                    writer.writeheader()
                    writer.writerows(rows)
        print("Resultados:", folder/"summary.csv")


if __name__ == "__main__":
    main()
