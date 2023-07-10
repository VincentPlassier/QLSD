import numpy as np
import torch


class StochasticQuantization:

    def __init__(self, s = 1, device = "cuda:0"):
        if s == 0:
            print("There will be no compression")
            # raise ValueError(("There will be no compression")
        self.s = s
        self.device = device

    def quantize(self, v):
        if self.s == 0:
            return v.to(self.device)
        v_norm = torch.norm(v, p=2)
        if v_norm == 0:
            return v.to(self.device)
        r = self.s * torch.abs(v) / v_norm
        l = torch.floor(r)
        l += torch.ceil(r - l) - torch.ones_like(l)
        b = torch.bernoulli(r - l)
        xi = (l + b) / self.s
        return (v_norm * torch.sign(v) * xi).to(self.device)

    def communication_eval(self, v):
        s = self.s
        dim_v = len(torch.flatten(v))
        if s == 0:
            return 32 * dim_v  # there is no quantization
        elif s < np.sqrt(dim_v / 2 - np.sqrt(dim_v)):
            t = s * (s + np.sqrt(dim_v))
            return 32 + 3 * t * (1 + np.log(2 * (s ** 2 + dim_v) / t) / 2)
        else:
            t = s ** 2 + min(dim_v, s * np.sqrt(dim_v))
            return 32 + dim_v * (2 + (1 + np.log(1 + t / dim_v)) / 2)
