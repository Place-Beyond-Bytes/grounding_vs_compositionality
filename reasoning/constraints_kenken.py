from __future__ import annotations
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F

"""
KenKen-style arithmetic constraints for relational composition.
We define satisfaction softly over probabilities P (B,N,N,V).
A cage is defined by: { "cells": [(r,c), ...], "operation": "add|subtract|multiply|divide", "target": int }
This module includes a soft relaxation by computing expected values under P.
"""


def expected_digit(P_cell: torch.Tensor, digits: torch.Tensor) -> torch.Tensor:
    """P_cell: (V) probs; digits: (V) values; returns scalar expectation."""
    P_cell = F.softmax(P_cell, dim=-1)
    return (P_cell * digits).sum()


def cage_satisfaction(P: torch.Tensor, cage: Dict, digits_tensor: torch.Tensor) -> torch.Tensor:
    """
    P: (B,N,N,V) probs
    cage: {cells: List[(r,c)], operation: str, target: int}
    digits_tensor: (V,) tensor of digit values
    Returns: (B,) satisfaction in [0,1]
    """
    P = F.softmax(P, dim=-1)
    B, N, _, V = P.shape
    cells: List[Tuple[int, int]] = [tuple(x) for x in cage.get("cells", [])]
    op_raw = cage.get("operation", cage.get("op", "+"))
    op = str(op_raw).lower()
    target = float(cage.get("target", 0))

    sats = []
    for b in range(B):
        vals = [expected_digit(P[b, r, c], digits_tensor) for (r, c) in cells]
        if op in {"add", "+"}:
            agg = torch.stack(vals).sum() if vals else torch.tensor(0.0, device=P.device, dtype=P.dtype)
        elif op in {"multiply", "x", "*", "mul", "times"}:
            if not vals:
                agg = torch.tensor(1.0, device=P.device, dtype=P.dtype)
            else:
                agg = torch.stack(vals).log().sum().exp()
        elif op in {"subtract", "-"}:
            # KenKen subtract cages are usually 2-cell absolute difference
            if len(vals) == 1:
                agg = vals[0]
            elif len(vals) >= 2:
                # Use first two as approximation (order-invariant via absolute diff)
                agg = torch.abs(vals[0] - vals[1])
            else:
                agg = torch.tensor(0.0, device=P.device, dtype=P.dtype)
        elif op in {"divide", "/"}:
            if len(vals) == 1:
                agg = vals[0]
            elif len(vals) >= 2:
                # Use ratio with epsilon for stability, symmetric via max/min
                a, b = vals[0], vals[1]
                num = torch.maximum(a, b)
                den = torch.clamp(torch.minimum(a, b), min=1e-6)
                agg = num / den
            else:
                agg = torch.tensor(1.0, device=P.device, dtype=P.dtype)
        else:
            agg = torch.stack(vals).sum() if vals else torch.tensor(0.0, device=P.device, dtype=P.dtype)

        diff = torch.abs(agg - target)
        sat = 1.0 / (1.0 + diff)
        sats.append(sat)
    return torch.stack(sats)


def kenken_satisfaction(P: torch.Tensor, cages: List[Dict], V: int = 11) -> torch.Tensor:
    """
    Mean satisfaction across cages per sample, then average over batch.
    """
    digits_vals = torch.arange(0, V, device=P.device, dtype=P.dtype)
    if V > 10:
        digits_vals[-1] = 0.0
    if not cages:
        return torch.ones(P.shape[0], device=P.device, dtype=P.dtype).mean()

    # Accumulate per-cage satisfaction (B,) and average across cages
    sats = [cage_satisfaction(P, cage, digits_vals) for cage in cages]
    # Stack to (num_cages, B) -> mean across cages -> (B,)
    per_sample = torch.stack(sats, dim=0).mean(dim=0)
    # Finally average over batch to produce scalar satisfaction
    return per_sample.mean()
