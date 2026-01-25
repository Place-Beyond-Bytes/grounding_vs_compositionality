from __future__ import annotations
import torch
from reasoning.kb_easy import satisfaction as kb_easy_sat
from reasoning.kb_moderate import satisfaction as kb_moderate_sat
try:
    from reasoning.kb_hard import satisfaction as kb_hard_sat
except Exception:
    kb_hard_sat = None


def lukasiewicz_t_norm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.clamp(a + b - 1.0, min=0.0, max=1.0)


def generalized_mean(x: torch.Tensor, p: float = 1.0, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    if p == 1.0:
        return x.mean(dim=dim, keepdim=keepdim)
    return (x.pow(p).mean(dim=dim, keepdim=keepdim)).pow(1.0 / p)


def kb_satisfaction(P: torch.Tensor, kb: str = "easy") -> torch.Tensor:
    kb = (kb or "easy").lower()
    if kb == "easy":
        return kb_easy_sat(P)
    if kb == "moderate":
        return kb_moderate_sat(P)
    if kb == "hard" and kb_hard_sat is not None:
        return kb_hard_sat(P)
    return kb_easy_sat(P)


def logic_loss(P: torch.Tensor, kb: str = "easy") -> torch.Tensor:
    sat = kb_satisfaction(P, kb)
    return 1.0 - sat
