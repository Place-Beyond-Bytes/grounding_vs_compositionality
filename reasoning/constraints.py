from __future__ import annotations
import torch
import torch.nn.functional as F


def all_different_row(P_row: torch.Tensor) -> torch.Tensor:
    """
    P_row: (N,V) probs per cell
    Returns a scalar satisfaction in [0,1].
    """
    P_row = F.softmax(P_row, dim=-1)
    N, V = P_row.shape
    S = P_row @ P_row.t()
    I = torch.eye(N, device=P_row.device)
    S = S * (1.0 - I)
    distinct = 1.0 - S
    denom = N * (N - 1)
    return distinct.sum() / max(denom, 1)

def all_different(P: torch.Tensor) -> torch.Tensor:
    """
    P: (B,N,N,V)
    Mean satisfaction over rows and columns.
    """
    P = F.softmax(P, dim=-1)
    B, N, _, V = P.shape
    sats = []
    for b in range(B):
        for r in range(N):
            sats.append(all_different_row(P[b, r]))
        for c in range(N):
            sats.append(all_different_row(P[b, :, c, :]))
    return torch.stack(sats).mean()


def all_different_subgrids(P: torch.Tensor, block_h: int, block_w: int) -> torch.Tensor:
    """
    P: (B,N,N,V) probs; enforce all_different within each block_h x block_w subgrid.
    """
    P = F.softmax(P, dim=-1)
    B, N, _, V = P.shape
    sats = []
    for b in range(B):
        for br in range(0, N, block_h):
            for bc in range(0, N, block_w):
                block = P[b, br:br+block_h, bc:bc+block_w, :].contiguous().view(-1, V)  # (block_h*block_w, V)
                sats.append(all_different_row(block))
    return torch.stack(sats).mean() if sats else torch.tensor(1.0, device=P.device)


def sudoku_satisfaction(P: torch.Tensor) -> torch.Tensor:
    """
    Combined satisfaction for Sudoku: rows, columns, and if applicable 3x3 subgrids.
    Automatically uses 3x3 when N % 3 == 0.
    """
    B, N, _, V = P.shape
    sat_rc = all_different(P)
    if N % 3 == 0:
        sat_sub = all_different_subgrids(P, 3, 3)
        return (sat_rc + sat_sub) / 2.0
    return sat_rc
