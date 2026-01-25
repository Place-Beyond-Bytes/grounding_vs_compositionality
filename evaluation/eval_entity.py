from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

from data.loaders import EntityCompositionDataset, collate_fn
from models.ltn import LTN, LTNConfig
from models.iltn import ILTN, ILTNConfig


def evaluate_model(model, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            out = model(batch) if hasattr(model, 'backbone') else model(batch)
            if 'logits' in out:
                pred = out['logits'].argmax(dim=-1)
            else:
                probs_last = out['beliefs'][:, -1]
                pred = probs_last.argmax(dim=-1)
            target = batch['solution_grid']
            total += target.numel()
            correct += (pred == target).sum().item()
    acc = correct / max(total, 1)
    return {"symbol_accuracy": acc}


def main(model_name: str = 'iltn', split: str = 'test', data_root: str = 'data/entity_composition', device_str: str | None = None):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = EntityCompositionDataset(root=data_root, split=split)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)

    if model_name == 'ltn':
        model = LTN(LTNConfig(grid_size=5, vocab_size=10)).to(device)
    else:
        model = ILTN(ILTNConfig(grid_size=5, vocab_size=10)).to(device)

    metrics = evaluate_model(model, loader, device)
    print(f"[{model_name}] Entity composition ({split}) symbol accuracy: {metrics['symbol_accuracy']:.4f}")


if __name__ == "__main__":
    main()
