from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fuzzy_logic import logic_loss


@dataclass
class ILTNConfig:
    grid_size: int = 5
    vocab_size: int = 10  # digits 0-9
    gamma: float = 0.98
    T_min: int = 5
    T_max: int = 20
    inner_k: int = 3
    inner_lr: float = 0.1
    gumbel_tau: float = 1.0
    use_halting: bool = True
    halting_threshold: float = 0.5


class ILTN(nn.Module):
    """
    Iterative Logic Tensor Network per paper:
    - Perception G_θ maps image to per-cell logits -> P(0) via softmax
    - Iteratively refines P(t) with k inner gradient steps on logic loss
    - Gumbel-Softmax to obtain P(t+1)
    - Optional halting head on pooled belief state
    """

    def __init__(self, cfg: ILTNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Perception backbone G_θ to produce initial per-cell logits
        self.perception = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((cfg.grid_size, cfg.grid_size)),
        )
        self.perception_head = nn.Conv2d(64, cfg.vocab_size, 1)
        # Optional halting head (disabled by default)
        self.halt_head = None
        if cfg.use_halting:
            self.halt_head = nn.Sequential(
                nn.Linear(cfg.grid_size * cfg.grid_size * cfg.vocab_size, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

    def _init_belief_from_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B,1,H,W)
        return P0: (B,N,N,V) from G_θ
        """
        z = self.perception(images)
        logits = self.perception_head(z)  # (B,V,N,N)
        logits = logits.permute(0, 2, 3, 1).contiguous()  # (B,N,N,V)
        P0 = F.softmax(logits, dim=-1)
        return P0.detach().requires_grad_(True)

    def forward(self, batch: Dict[str, torch.Tensor], T: Optional[int] = None, tau: Optional[float] = None, kb: str = "easy") -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        images = batch["images"].to(device).float()
        B, _, H, W = images.shape
        N = self.cfg.grid_size
        V = self.cfg.vocab_size
        # Initial belief from perception
        P = self._init_belief_from_image(images)

        T_steps = T if T is not None else int(torch.randint(self.cfg.T_min, self.cfg.T_max + 1, (1,)).item())
        tau_val = tau if tau is not None else self.cfg.gumbel_tau

        all_P = []
        halts = []
        for t in range(T_steps):
            # inner-loop gradient steps on logic loss
            for _ in range(self.cfg.inner_k):
                L_logic = logic_loss(P, kb=kb)
                grad, = torch.autograd.grad(L_logic, P, create_graph=self.training)
                # Inner-loop SGD step on belief state
                P = P - self.cfg.inner_lr * grad
                P = P.detach().requires_grad_(True)

            # Gumbel-Softmax discretization (still differentiable)
            logits = P
            P = F.gumbel_softmax(logits.view(B * N * N, V), tau=tau_val, hard=False)
            P = P.view(B, N, N, V)

            all_P.append(P)

            if self.halt_head is not None:
                pooled = P.view(B, -1)  # global pooling via flatten
                h = self.halt_head(pooled)
                halts.append(h)
                # Early halting at eval time per paper
                if not self.training and torch.any(h > self.cfg.halting_threshold):
                    break

        out = {
            "beliefs": torch.stack(all_P, dim=1),  # (B,T,N,N,V)
        }
        if halts:
            out["halts"] = torch.stack(halts, dim=1)
        return out
