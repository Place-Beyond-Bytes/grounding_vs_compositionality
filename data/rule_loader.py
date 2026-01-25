from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class RuleCompositionDataset(Dataset):
    """
    Sudoku rule composition dataset.
    Directory layout: data/rule composition/{easy|moderate|hard}/*_metadata.json
    Metadata contains: grids{puzzle_grid, solution_grid}, strategies{required_strategies}
    """

    def __init__(self, root: str, split: str = "easy") -> None:
        self.root = Path(root)
        self.split = split  # 'easy' | 'moderate' | 'hard'
        self.dir = self.root / split
        if not self.dir.exists():
            raise FileNotFoundError(f"Dataset split not found: {self.dir}")
        self.meta_files = sorted(self.dir.glob("*_metadata.json"))
        if not self.meta_files:
            raise FileNotFoundError(f"No metadata json files found in: {self.dir}")

    def __len__(self) -> int:
        return len(self.meta_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta_path = self.meta_files[idx]
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        pid = meta["puzzle_info"]["id"]
        puzzle = np.array(meta["grids"]["puzzle_grid"], dtype=np.int64)
        solution = np.array(meta["grids"]["solution_grid"], dtype=np.int64)
        strategies = meta.get("strategies", {}).get("required_strategies", [])

        files = meta.get("files", {})
        img_file = files.get("puzzle_image", f"{pid}_puzzle.png")
        img_path = self.dir / img_file
        gp = files.get("puzzle_image_path")
        if not img_path.exists() and gp and Path(gp).exists():
            img_path = Path(gp)
        img = Image.open(img_path).convert("L")

        return {
            "id": pid,
            "image": torch.from_numpy(np.array(img, dtype=np.uint8)),
            "puzzle_grid": torch.from_numpy(puzzle),
            "solution_grid": torch.from_numpy(solution),
            "strategies": strategies,
        }


def collate_fn_rule(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    ids = [b["id"] for b in batch]
    images = torch.stack([b["image"].float().unsqueeze(0) / 255.0 for b in batch], dim=0)
    puzzles = torch.stack([b["puzzle_grid"] for b in batch], dim=0)
    solutions = torch.stack([b["solution_grid"] for b in batch], dim=0)
    strategies = [b["strategies"] for b in batch]
    return {
        "ids": ids,
        "images": images,
        "puzzle_grid": puzzles,
        "solution_grid": solutions,
        "strategies": strategies,
    }
