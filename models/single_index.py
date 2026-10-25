"""
Partially linear model: beta*u + F_theta(index), with a 1-h-1 neural link.
"""

from math import sqrt

import torch
from torch import nn
from torch.nn.functional import linear

import config


class SingleIndexModel(nn.Module):
    """
    Train beta and theta at fixed w; inputs are standardized index and control.
    """

    def __init__(self, initialization_seed):
        super().__init__()
        if config.activation != "softplus" or config.dtype != "float64" or config.device != "cpu":
            raise ValueError("This implementation uses Softplus and CPU float64.")
        h = config.hidden_width
        rng = torch.Generator(device="cpu").manual_seed(initialization_seed)
        self.beta = nn.Parameter(torch.tensor(config.linear_beta_initial, dtype=torch.float64, device="cpu"))
        self.W1 = nn.Parameter(torch.empty((h, 1), dtype=torch.float64, device="cpu"))
        self.b1 = nn.Parameter(torch.full((h,), config.neural_bias_initial, dtype=torch.float64, device="cpu"))
        self.W2 = nn.Parameter(torch.empty((1, h), dtype=torch.float64, device="cpu"))
        self.b2 = nn.Parameter(torch.full((1,), config.neural_bias_initial, dtype=torch.float64, device="cpu"))
        # Xavier uniform, drawn in hidden-then-output order from a local generator.
        bound = config.initialization_gain * sqrt(6.0 / (1 + h))
        with torch.no_grad():
            self.W1.uniform_(-bound, bound, generator=rng)
            self.W2.uniform_(-bound, bound, generator=rng)

    def forward(self, index, u):
        hidden = linear(index.unsqueeze(-1), self.W1, self.b1)
        hidden = torch.logaddexp(torch.zeros_like(hidden), hidden)  # Softplus.
        link = linear(hidden, self.W2, self.b2).squeeze(-1)
        return self.beta * u + link
