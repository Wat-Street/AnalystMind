# ASSIGNED TO: Afreed
#
# Task: Implement the multi-horizon prediction head.
#
# What to implement:
#
#   class PredictionHead(nn.Module):
#     - Two-layer MLP: d_model → 128 → len(HORIZONS)  [outputs 4 return predictions]
#     - Input:  (batch, d_model)   [CLS token from backbone]
#     - Output: (batch, 4)         [predicted returns for 1M, 3M, 6M, 12M]
#     - Use ReLU activation between layers, no activation on output
#
# Depends on: features.py (HORIZONS), backbone.py uses this in FTTransformerModel

from __future__ import annotations

import torch
from torch import nn

from .features import HORIZONS


class PredictionHead(nn.Module):
    """Map a batch of CLS embeddings to multi-horizon return predictions."""

    def __init__(self, d_model: int = 192, hidden_dim: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(HORIZONS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
