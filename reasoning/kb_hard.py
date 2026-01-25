from __future__ import annotations
import torch
from .constraints import all_different

def satisfaction(P: torch.Tensor) -> torch.Tensor:
    return all_different(P)
