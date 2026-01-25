from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.loaders import EntityCompositionDataset, collate_fn
from training.losses import ltn_grounding_loss, iltn_full_loss
from utils.config import load_config
from models.ltn import LTN, LTNConfig
from models.iltn import ILTN, ILTNConfig


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0


def accuracy_from_logits(logits: torch.Tensor, target: torch.Tensor) -> float:
    # logits: (B,N,N,V)
    pred = logits.argmax(dim=-1)
    correct = (pred == target).float().mean().item()
    return correct


def accuracy_from_probs(probs: torch.Tensor, target: torch.Tensor) -> float:
    # probs: (B,N,N,V)
    pred = probs.argmax(dim=-1)
    correct = (pred == target).float().mean().item()
    return correct


def create_dataloader(cfg: Dict[str, Any], split: str) -> DataLoader:
    ds_cfg = cfg.get("dataset", {})
    root = ds_cfg.get("root", "data/entity_composition")
    # Entity composition per paper: train {0..4}, test {5..9}
    allowed = None
    if split == "train":
        allowed = ds_cfg.get("train_digits", None)
    elif split == "test":
        allowed = ds_cfg.get("test_digits", None)
    ds = EntityCompositionDataset(root=root, split=split, allowed_digits=allowed)
    return DataLoader(ds, batch_size=cfg.get("training", {}).get("batch_size", 16), shuffle=(split=="train"), num_workers=0, collate_fn=collate_fn)


def create_model(cfg: Dict[str, Any], model_name: str):
    model_cfg = cfg.get("model", {})
    if model_name == "ltn":
        return LTN(LTNConfig(grid_size=5, vocab_size=10))
    elif model_name == "iltn":
        iltn_cfg = ILTNConfig(
            grid_size=5,
            vocab_size=10,
            gamma=model_cfg.get("iltn", {}).get("gamma", 0.98),
            T_min=model_cfg.get("iltn", {}).get("T_min", 5),
            T_max=model_cfg.get("iltn", {}).get("T_max", 20),
            inner_k=model_cfg.get("iltn", {}).get("inner_k", 3),
            inner_lr=model_cfg.get("iltn", {}).get("inner_lr", 0.1),
            gumbel_tau=model_cfg.get("iltn", {}).get("gumbel_tau_start", 1.0),
            use_halting=model_cfg.get("iltn", {}).get("use_halting", True),
        )
        return ILTN(iltn_cfg)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def create_optimizer(cfg: Dict[str, Any], model: nn.Module):
    tr = cfg.get("training", {})
    if tr.get("optimizer", "adamw").lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=tr.get("lr", 1e-4), betas=tuple(tr.get("betas", [0.9, 0.999])), weight_decay=tr.get("weight_decay", 0.01))
    return torch.optim.Adam(model.parameters(), lr=tr.get("lr", 1e-4))


def create_scheduler(cfg: Dict[str, Any], opt: torch.optim.Optimizer, steps_per_epoch: int):
    tr = cfg.get("training", {})
    warmup = tr.get("warmup_epochs", 0)
    use_cosine = tr.get("use_cosine_decay", True)

    def lr_lambda(step: int):
        epoch = step / max(steps_per_epoch, 1)
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        if use_cosine:
            progress = (epoch - warmup) / max((tr.get("epochs", 100) - warmup), 1e-8)
            return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def train_one_epoch(model, optimizer, scheduler, loader: DataLoader, cfg: Dict[str, Any], model_name: str, device: torch.device, state: TrainState):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    count = 0

    iltn_tau_start = cfg.get("model", {}).get("iltn", {}).get("gumbel_tau_start", 1.0)
    iltn_tau_end = cfg.get("model", {}).get("iltn", {}).get("gumbel_tau_end", 0.1)
    iltn_anneal_epochs = cfg.get("model", {}).get("iltn", {}).get("gumbel_anneal_epochs", 50)
    # Linear anneal by epoch
    if iltn_anneal_epochs > 0:
        tau = max(iltn_tau_end, iltn_tau_start - (iltn_tau_start - iltn_tau_end) * (state.epoch / iltn_anneal_epochs))
    else:
        tau = iltn_tau_end

    for batch in loader:
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        optimizer.zero_grad()
        if model_name == "ltn":
            out = model(batch)
            loss = ltn_grounding_loss(out, batch)
            acc = accuracy_from_logits(out["logits"], batch["solution_grid"])
        else:
            out = model(batch, T=None, tau=tau)
            loss = iltn_full_loss(out, batch, gamma=cfg.get("model", {}).get("iltn", {}).get("gamma", 0.98))
            probs_last = out["beliefs"][:, -1]
            acc = accuracy_from_probs(probs_last, batch["solution_grid"])
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_acc += acc
        count += 1
        state.global_step += 1

    return total_loss / max(count, 1), total_acc / max(count, 1)


def fit(model_name: str, config_path: str = "configs/defaults.yaml", device_str: str = None):
    cfg = load_config(config_path)
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_loader = create_dataloader(cfg, split="train")
    steps_per_epoch = len(train_loader)

    model = create_model(cfg, model_name).to(device)
    optimizer = create_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer, steps_per_epoch)

    epochs = cfg.get("training", {}).get("epochs", 10)
    state = TrainState()

    for epoch in range(epochs):
        state.epoch = epoch
        loss, acc = train_one_epoch(model, optimizer, scheduler, train_loader, cfg, model_name, device, state)
        print(f"[{model_name}] epoch {epoch+1}/{epochs} - loss {loss:.4f} - acc {acc:.4f}")

    return model
