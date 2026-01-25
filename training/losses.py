from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    gamma: float = 0.98


def cross_entropy_grid(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred: (B,N,N,V) logits or probs
    target: (B,N,N) int64 labels in [0..V-1]
    """
    B, N, _, V = pred.shape
    return F.cross_entropy(pred.view(B * N * N, V), target.view(B * N * N), reduction="mean")


def ltn_grounding_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    logits = outputs["logits"]  # (B,N,N,V)
    return cross_entropy_grid(logits, batch["solution_grid"].to(logits.device))


def iltn_full_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], gamma: float = 0.98) -> torch.Tensor:
    beliefs = outputs["beliefs"]  # (B,T,N,N,V) after gumbel_softmax (probs)
    target = batch["solution_grid"].to(beliefs.device)
    B, T, N, _, V = beliefs.shape
    losses = []
    for t in range(T):
        probs = beliefs[:, t]
        # use log to convert to logits for CE stability
        logits = torch.log(torch.clamp(probs, min=1e-8))
        ce = cross_entropy_grid(logits, target)
        weight = gamma ** (T - 1 - t)
        losses.append(weight * ce)
    return torch.stack(losses).sum()
