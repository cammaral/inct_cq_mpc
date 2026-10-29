import csv
import tempfile
import unittest
from pathlib import Path

import torch

from mpc.models import Model, Circuit
from mpc.problems import solution
from mpc.train import train_one


class TestProject(unittest.TestCase):
    def test_burgers_boundary(self):
        coords = torch.tensor([[0., .3], [1., .6]])
        self.assertLess(solution("burgers", coords,
                                 {"nu": .05, "a": .9, "b": 1.0}).abs().max().item(), 1e-6)

    def test_black_scholes_payoff(self):
        coords = torch.tensor([[80., 0.], [120., 0.]])
        torch.testing.assert_close(solution("black_scholes", coords,
            {"strike": 100., "rate": .05, "sigma": .2}).flatten(), torch.tensor([0., .2]))

    def test_qubit_order_and_pauli(self):
        model = Model({"kind": "mpc_hybrid", "terms": 4, "layers": 1,
                       "observables": ["X", "ZY", "XZI", "ZZZZ"]})
        self.assertEqual([c.qubits for c in model.aux], [1, 2, 3, 4])
        self.assertEqual([c.operators.shape[-1] for c in model.aux], [2, 4, 8, 16])
        for circuit in model.aux:
            self.assertEqual(tuple(circuit().shape), (1, 1))

    def test_no_input_in_auxiliary_and_aggregation(self):
        x = torch.rand(4, 2)
        for kind in ["mpc_classical", "mpc_hybrid"]:
            model = Model({"kind": kind, "terms": 3, "layers": 1,
                           "observables": ["Z", "ZZ", "ZZZ"], "width": 8, "depth": 1})
            z = model.coefficients()
            y = model.main(x)
            torch.testing.assert_close(model(x), ((1 + y) * z.pow(model.exponents)).mean(-1, keepdim=True))
            self.assertEqual(tuple(z.shape), (3,))
            model.aggregation = "sum"
            torch.testing.assert_close(model(x), ((1 + y) * z.pow(model.exponents)).sum(-1, keepdim=True))
            model.aggregation = "mean"
            loss = model(x).square().mean()
            loss.backward()
            self.assertTrue(any(p.grad is not None for p in model.aux.parameters()))

    def test_all_models(self):
        for cfg in [{"kind": "mlp"}, {"kind": "mpc_classical", "terms": 3},
                    {"kind": "mpc_hybrid", "terms": 3},
                    {"kind": "qnn", "qubits": 3, "observables": ["ZII", "IZI", "IIZ"]}]:
            model = Model({"width": 8, "depth": 1, "layers": 1, **cfg})
            y = model(torch.rand(4, 2))
            self.assertEqual(tuple(y.shape), (4, 1))
            y.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all()
                                for p in model.parameters() if p.grad is not None))

    def test_log_expectations(self):
        config = {
            "problems": {"burgers": {"bounds": [[0., 1.], [0., 1.]],
                                     "nu": .05, "a": .9, "b": 1.0}},
            "data": {"train": 16, "val": 12, "test_grid": 6},
            "training": {"device": "cpu", "epochs": 3, "batch_size": 8,
                         "lr": .003, "scheduler": "none", "clip": None},
        }
        variant = {"kind": "mpc_hybrid", "name": "hybrid", "terms": 3,
                   "width": 8, "depth": 1, "layers": 1}
        with tempfile.TemporaryDirectory() as tmp:
            train_one(config, "burgers", variant, 1, tmp)
            with (Path(tmp) / "expectations.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(f"exp_{i}" in rows[0] for i in (1, 2, 3)))
            self.assertTrue(all(f"effective_{i}" in rows[0] for i in (1, 2, 3)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA nao disponivel neste ambiente")
    def test_cuda_grad(self):
        model = Model({"kind": "mpc_hybrid", "terms": 3, "layers": 1}).cuda()
        model(torch.rand(5, 2, device="cuda")).square().mean().backward()
        self.assertEqual(model.aux[0].theta.grad.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
