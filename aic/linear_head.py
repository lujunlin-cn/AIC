"""Minimal per-timestep scorer for the preregistered head-capacity control."""
import torch
from torch import nn


class LinearScorer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.aux_dim = 0
        self.score = nn.Linear(input_dim, 1)

    def forward(self, features, aux=None, lengths=None):
        if aux is not None:
            raise ValueError('Linear head control has no auxiliary features')
        return self.score(features).squeeze(-1)
