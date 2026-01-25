from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitPredicate(nn.Module):
    """
    Predicate implementing is_digit(cell, digit) -> truth score in [0,1].
    Takes a cell feature vector and a digit embedding; outputs a score.
    """
    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, cell_feats: torch.Tensor, digit_embs: torch.Tensor) -> torch.Tensor:
        """
        cell_feats: (B, N, N, D)
        digit_embs: (V, D)
        returns probs: (B, N, N, V)
        """
        B, N, _, D = cell_feats.shape
        V = digit_embs.shape[0]
        # Expand and concatenate
        cell = cell_feats.unsqueeze(3).expand(B, N, N, V, D)
        digit = digit_embs.view(1, 1, 1, V, D).expand(B, N, N, V, D)
        x = torch.cat([cell, digit], dim=-1)
        out = self.mlp(x)  # (B,N,N,V,1)
        probs = out.squeeze(-1)
        return probs
