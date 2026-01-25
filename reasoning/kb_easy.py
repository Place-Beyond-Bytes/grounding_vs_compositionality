from __future__ import annotations
import torch
from .constraints import sudoku_satisfaction


def satisfaction(P: torch.Tensor) -> torch.Tensor:
    """Keasy: Sudoku satisfaction (rows, cols, and subgrids when applicable)."""
    return sudoku_satisfaction(P)
