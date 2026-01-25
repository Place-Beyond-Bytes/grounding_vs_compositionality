from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass
class PuzzleItem:
    puzzle_grid: torch.Tensor  # (N,N) with -1 for blanks
    solution_grid: torch.Tensor  # (N,N)
    puzzle_image: Image.Image
    id: str


def _digits_in_grid(grid: np.ndarray) -> set:
    vals = set(int(x) for x in grid.flatten().tolist() if int(x) >= 0)
    return vals


class EntityCompositionDataset(Dataset):
    """
    Dataset for the Entity Composition split described in the paper.
    Expects directory structure similar to data/entity_composition/test with *_metadata.json files.
    """

    def __init__(self, root: str, split: str = "test", allowed_digits: List[int] | None = None) -> None:
        self.root = Path(root)
        self.split = split
        self.dir = self.root / split
        if not self.dir.exists():
            raise FileNotFoundError(f"Dataset split not found: {self.dir}")
        meta_files = sorted(self.dir.glob("*_metadata.json"))
        self.meta_files = []
        self.allowed_digits = set(allowed_digits) if allowed_digits is not None else None
        if self.allowed_digits is None:
            self.meta_files = meta_files
        else:
            # Filter by digits present in puzzle clues
            for mp in meta_files:
                try:
                    with open(mp, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    puzzle = np.array(meta["grids"]["puzzle_grid"], dtype=np.int64)
                    present = _digits_in_grid(puzzle)
                    if present.issubset(self.allowed_digits):
                        self.meta_files.append(mp)
                except Exception:
                    continue
        if not self.meta_files:
            raise FileNotFoundError(f"No metadata json files found in: {self.dir}")

    def __len__(self) -> int:
        return len(self.meta_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        meta_path = self.meta_files[idx]
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        pid = meta["puzzle_info"]["id"]
        puzzle = np.array(meta["grids"]["puzzle_grid"], dtype=np.int64)
        solution = np.array(meta["grids"]["solution_grid"], dtype=np.int64)
        files = meta.get("files", {})
        img_file = files.get("puzzle_image", f"{pid}_puzzle.png")
        img_path = self.dir / img_file
        if not img_path.exists():
            # fallback to global paths inside json if available
            global_path = files.get("puzzle_image_path")
            if global_path and os.path.exists(global_path):
                img_path = Path(global_path)
            else:
                raise FileNotFoundError(f"Puzzle image not found for {pid}: {img_path}")
        img = Image.open(img_path).convert("L")

        item: Dict[str, torch.Tensor] = {
            "id": pid,
            "puzzle_grid": torch.from_numpy(puzzle),
            "solution_grid": torch.from_numpy(solution),
            "image": torch.from_numpy(np.array(img, dtype=np.uint8)),
        }
        return item


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    ids = [b["id"] for b in batch]
    images = torch.stack([b["image"].float().unsqueeze(0) / 255.0 for b in batch], dim=0)  # (B,1,H,W)
    puzzles = torch.stack([b["puzzle_grid"] for b in batch], dim=0)  # (B,N,N)
    solutions = torch.stack([b["solution_grid"] for b in batch], dim=0)  # (B,N,N)
    mask_known = (puzzles >= 0)
    return {
        "ids": ids,
        "images": images,
        "puzzle_grid": puzzles,
        "solution_grid": solutions,
        "mask_known": mask_known,
    }
