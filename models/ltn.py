from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LTNConfig:
    grid_size: int = 5
    vocab_size: int = 10


class SimpleCNN(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((5,5)),
        )
        self.head = nn.Conv2d(64, vocab_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        logits = self.head(z)  # (B,V,5,5)
        return logits


class LTN(nn.Module):
    """
    Grounding-only baseline: predict per-cell logits over vocab directly from image.
    """
    def __init__(self, cfg: LTNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = SimpleCNN(cfg.vocab_size)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        images = batch["images"].float()
        logits = self.backbone(images)  # (B,V,N,N)
        logits = logits.permute(0, 2, 3, 1).contiguous()  # (B,N,N,V)
        return {"logits": logits}
