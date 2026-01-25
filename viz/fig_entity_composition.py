from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from data.loaders import EntityCompositionDataset, collate_fn
from models.ltn import LTN, LTNConfig
from models.iltn import ILTN, ILTNConfig


def per_sample_symbol_accuracy(model, loader: DataLoader, device: torch.device) -> Dict[str, List[float]]:
    model.eval()
    ids: List[str] = []
    accs: List[float] = []
    with torch.no_grad():
        for batch in loader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            out = model(batch) if hasattr(model, 'backbone') else model(batch)
            if 'logits' in out:
                pred = out['logits'].argmax(dim=-1)
            else:
                pred = out['beliefs'][:, -1].argmax(dim=-1)
            target = batch['solution_grid']
            B = target.shape[0]
            for i in range(B):
                ids.append(batch['ids'][i])
                acc = (pred[i] == target[i]).float().mean().item()
                accs.append(acc)
    return {"ids": ids, "accs": accs}


def plot_entity_composition_scatter(ltn_accs: Dict[str, Any], iltn_accs: Dict[str, Any], title: str = "Entity Composition: Sample-wise Symbol Grounding") -> plt.Figure:
    ids = np.arange(1, len(ltn_accs['accs']) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(ids, ltn_accs['accs'], color='red', alpha=0.6, s=30, label='LTN', edgecolors='darkred')
    ax.scatter(ids, iltn_accs['accs'], color='blue', alpha=0.6, s=30, label='iLTN', edgecolors='darkblue')
    ax.set_title(title)
    ax.set_xlabel('Sample ID')
    ax.set_ylabel('Symbol Grounding Accuracy')
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def main(data_root: str = 'data/entity_composition', device_str: str | None = None, out_png: str = 'fig_entity_composition.png'):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = EntityCompositionDataset(root=data_root, split='test', allowed_digits=[5,6,7,8,9])
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)

    ltn = LTN(LTNConfig(grid_size=5, vocab_size=11)).to(device)
    iltn = ILTN(ILTNConfig(grid_size=5, vocab_size=11)).to(device)

    ltn_accs = per_sample_symbol_accuracy(ltn, loader, device)
    iltn_accs = per_sample_symbol_accuracy(iltn, loader, device)

    fig = plot_entity_composition_scatter(ltn_accs, iltn_accs)
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Saved {out_png}")


if __name__ == '__main__':
    main()
