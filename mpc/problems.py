"""Exemplos supervisionados: apenas solucoes exatas, sem residual de EDP."""
import math
import torch


def solution(name, coords, cfg):
    """coords contem [x,t] ou [S,tau] em unidades fisicas."""
    x, t = coords[:, :1], coords[:, 1:2]
    if name == "burgers":
        nu, a, b = (float(cfg[k]) for k in ("nu", "a", "b"))
        if nu <= 0 or b <= abs(a):
            raise ValueError("Burgers: nu>0 e b>|a| sao obrigatorios")
        k = math.pi
        decay = torch.exp(-nu * k**2 * t)
        return 2 * nu * a * k * decay * torch.sin(k*x) / (b + a*decay*torch.cos(k*x))
    if name == "black_scholes":
        strike, rate, sigma = (float(cfg[k]) for k in ("strike", "rate", "sigma"))
        if strike <= 0 or sigma <= 0:
            raise ValueError("Black-Scholes: K>0 e sigma>0 obrigatorios")
        safe = t.clamp_min(1e-8)
        d1 = (torch.log(x/strike) + (rate + sigma**2/2)*safe) / (sigma*torch.sqrt(safe))
        d2 = d1 - sigma*torch.sqrt(safe)
        phi = lambda d: 0.5*(1 + torch.erf(d/math.sqrt(2)))
        call = x*phi(d1) - strike*torch.exp(-rate*safe)*phi(d2)
        return torch.where(t <= 0, (x-strike).clamp_min(0), call) / strike
    raise ValueError(f"Problema desconhecido: {name}")


def dataset(name, cfg, data_cfg, seed, device):
    bounds = torch.tensor(cfg["bounds"], dtype=torch.float32)
    gen = torch.Generator(device="cpu").manual_seed(seed + 2026)

    def random_points(n):
        return bounds[:, 0] + torch.rand(n, 2, generator=gen)*(bounds[:, 1]-bounds[:, 0])

    n_grid = int(data_cfg["test_grid"])
    axes = [torch.linspace(lo, hi, n_grid) for lo, hi in bounds.tolist()]
    xx, tt = torch.meshgrid(*axes, indexing="ij")
    physical_test = torch.stack((xx.flatten(), tt.flatten()), -1)

    def prepare(physical):
        xx = 2*(physical-bounds[:, 0])/(bounds[:, 1]-bounds[:, 0])-1
        yy = solution(name, physical, cfg)
        return xx.to(device), yy.to(device)

    train = prepare(random_points(int(data_cfg["train"])))
    valid = prepare(random_points(int(data_cfg["val"])))
    test = prepare(physical_test)
    return train, valid, test, physical_test, n_grid
