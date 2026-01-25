from __future__ import annotations
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

from data.rule_loader import RuleCompositionDataset, collate_fn_rule
from models.iltn import ILTN, ILTNConfig
from reasoning.constraints import all_different


def evaluate_rule_hard(model: ILTN, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    total, solved = 0, 0
    ad_sats = []
    with torch.no_grad():
        for batch in loader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            out = model(batch, kb="hard")
            probs_last = out["beliefs"][:, -1]
            sat_ad = all_different(probs_last)
            ad_sats.append(sat_ad.item())

            pred = probs_last.argmax(dim=-1)
            target = batch['solution_grid']
            solved += (pred == target).all(dim=-1).all(dim=-1).sum().item()
            total += target.shape[0]

    return {
        "puzzles_solved_rate": solved / max(total, 1),
        "all_different_satisfaction": sum(ad_sats) / max(len(ad_sats), 1),
    }


essage = """
Evaluates rule composition on the 'hard' split (9x9 Sudoku) using iLTN with KB='hard'.
For full paper protocol, pre-train on easy/moderate before evaluating on hard.
"""

def main(split: str = 'hard', data_root: str = 'data/rule composition', device_str: str | None = None):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = RuleCompositionDataset(root=data_root, split=split)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn_rule)
    # Grid size inferred from data: use 9 for rule composition by default
    model = ILTN(ILTNConfig(grid_size=9, vocab_size=11)).to(device)
    metrics = evaluate_rule_hard(model, loader, device)
    print(f"[iLTN] Rule composition ({split}) -> solved: {metrics['puzzles_solved_rate']:.4f}, "
          f"all_different: {metrics['all_different_satisfaction']:.4f}")


if __name__ == "__main__":
    main()
