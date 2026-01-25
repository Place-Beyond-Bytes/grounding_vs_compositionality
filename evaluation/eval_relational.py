from __future__ import annotations
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

from data.relational_loader import RelationalCompositionDataset, collate_fn_rel
from models.iltn import ILTN, ILTNConfig
from reasoning.constraints_kenken import kenken_satisfaction
from reasoning.constraints import all_different


def evaluate_relational(model: ILTN, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    total, solved = 0, 0
    ad_sats = []
    cage_sats = []
    with torch.no_grad():
        for batch in loader:
            # move tensors
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            out = model(batch)
            probs_last = out["beliefs"][:, -1]
            # all_different satisfaction
            sat_ad = all_different(probs_last)
            ad_sats.append(sat_ad.item())
            # cage satisfaction per sample list of dicts
            # Flatten cages across batch is not necessary; kenken_satisfaction expects list[dict]
            # We compute per-batch average by concatenating all cages lists (approximation if cages vary by sample)
            # Better: compute mean of each sample's cages then average. For simplicity, we pass first sample's cages if equal.
            # Here, compute average over each sample individually and average.
            batch_cage_sats = []
            for i in range(len(batch["ids"])):
                cages_i = batch["cages"][i]
                sat_ci = kenken_satisfaction(probs_last[i:i+1], cages_i, V=probs_last.shape[-1])
                batch_cage_sats.append(sat_ci.item())
            cage_sats.append(sum(batch_cage_sats) / max(len(batch_cage_sats), 1))

            pred = probs_last.argmax(dim=-1)
            target = batch['solution_grid']
            solved += (pred == target).all(dim=-1).all(dim=-1).sum().item()
            total += target.shape[0]

    return {
        "puzzles_solved_rate": solved / max(total, 1),
        "all_different_satisfaction": sum(ad_sats) / max(len(ad_sats), 1),
        "cage_satisfaction": sum(cage_sats) / max(len(cage_sats), 1),
    }


def main(split: str = 'test', data_root: str = 'data/relational composition', device_str: str | None = None):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = RelationalCompositionDataset(root=data_root, split=split)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn_rel)
    model = ILTN(ILTNConfig(grid_size=5, vocab_size=10)).to(device)
    metrics = evaluate_relational(model, loader, device)
    print(f"[iLTN] Relational composition ({split}) -> solved: {metrics['puzzles_solved_rate']:.4f}, "
          f"all_different: {metrics['all_different_satisfaction']:.4f}, cages: {metrics['cage_satisfaction']:.4f}")


if __name__ == "__main__":
    main()
