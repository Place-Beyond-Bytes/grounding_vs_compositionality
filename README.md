# Grounding vs. Compositionality in Logic Tensor Networks

A research project exploring the relationship between **perceptual grounding** and **compositional reasoning** in neural-symbolic systems, using constraint-based iterative refinement on logic puzzles (Sudoku and KenKen).

## Overview

This repository contains the implementation of **Logic Tensor Networks (LTN)** and **Iterative Logic Tensor Networks (ILTN)**, two approaches for solving constraint-satisfaction puzzles by combining:

- **Perception**: Visual grounding of puzzle state from images
- **Logic**: Constraint satisfaction through iterative refinement
- **Compositionality**: Generalization to unseen puzzle configurations

The key research question: *Can neural networks learn compositional reasoning through explicit constraint satisfaction, and how does grounding quality affect logical inference?*

## Key Components

### 🧠 Models

- **LTN (Logic Tensor Network)**: Pure grounding baseline
  - Direct visual perception with CNN backbone
  - Per-cell symbol prediction from images
  - No iterative refinement

- **ILTN (Iterative Logic Tensor Network)**: Grounding + Reasoning
  - Initial perception via CNN
  - Iterative constraint satisfaction with gradient-based refinement
  - Gumbel-Softmax for discrete sampling
  - Optional halting head for early stopping

### 📊 Datasets

- **Entity Composition**: Train on digit sets {0-4}, test on {5-9}
  - Tests compositional generalization to unseen entities
  
- **Relational Composition**: Constraints on unseen puzzle regions
  
- **Rule Composition**: Combinations of reasoning rules not seen during training

- **Knowledge Bases**: Pre-built constraint sets for:
  - Sudoku (easy, moderate, hard)
  - KenKen (easy, moderate, hard)

### 🔧 Training

- **Trainer**: Unified training pipeline with support for:
  - Multiple model architectures
  - Warm-up and cosine learning rate decay
  - Per-model loss functions
  
- **Loss Functions**:
  - Grounding loss (CE for initial perception)
  - Logic loss (constraint satisfaction)
  - Combined multi-task learning

### 📈 Evaluation

- **Symbol Accuracy**: Per-cell correctness on solution grid
- **Constraint Satisfaction**: Ratio of satisfied constraints
- **Compositionality Metrics**: Domain generalization scores

## Project Structure

```
├── models/              # Neural network architectures
│   ├── ltn.py          # Logic Tensor Network baseline
│   ├── iltn.py         # Iterative Logic Tensor Network
│   ├── predicates.py   # Fuzzy logic predicates
│   └── fuzzy_logic.py  # Constraint evaluation
├── data/               # Data handling
│   ├── loaders.py      # Dataset classes
│   ├── entity_composition/
│   ├── relational_composition/
│   ├── rule_composition/
│   └── knowledge_bases/# Pre-built constraint sets
├── training/           # Training infrastructure
│   ├── trainer.py      # Main training loop
│   └── losses.py       # Loss functions
├── evaluation/         # Evaluation scripts
│   ├── eval_entity.py
│   ├── eval_relational.py
│   └── eval_rule_comp.py
├── reasoning/          # Constraint definitions
│   ├── constraints.py  # Generic constraints
│   ├── constraints_kenken.py
│   └── kb_*.py         # Knowledge bases
├── utils/              # Configuration and helpers
├── viz/                # Visualization scripts
└── configs/            # Configuration files
    └── defaults.yaml   # Default hyperparameters
```

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- torch >= 2.2
- torchvision >= 0.17
- numpy >= 1.24
- pillow >= 10.2
- matplotlib >= 3.8
- pyyaml >= 6.0.1
- scikit-learn >= 1.4
- pdfminer.six >= 20231228

## Quick Start

### Training

```python
from training.trainer import train_model
from utils.config import load_config

cfg = load_config("configs/defaults.yaml")
train_model(cfg)
```

### Evaluation

```bash
# Entity composition generalization
python evaluation/eval_entity.py --model iltn --split test

# Relational composition
python evaluation/eval_relational.py --model iltn --split test

# Rule composition
python evaluation/eval_rule_comp.py --model iltn --split test
```

## Key Features

✨ **Iterative Refinement**: Multi-step constraint satisfaction through inner optimization  
✨ **Fuzzy Logic**: Continuous relaxation of logical constraints  
✨ **Compositionality Tests**: Multiple generalization scenarios (entities, relations, rules)  
✨ **Flexible Architecture**: Pluggable perception backbones and constraint sets  
✨ **Production Ready**: Fully documented, tested evaluation pipeline  

## Configuration

Edit `configs/defaults.yaml` to customize:

```yaml
dataset:
  name: entity_composition          # Dataset type
  root: data/entity_composition     # Data directory
  split: test                        # train or test
  grid_size: 5                       # Puzzle dimensions

model:
  embedding_dim: 64
  iltn:
    T_min: 5                         # Minimum refinement steps
    T_max: 20                        # Maximum refinement steps
    gamma: 0.98                      # Temperature decay
    inner_k: 3                       # Inner gradient steps
    inner_lr: 0.1                    # Inner learning rate

training:
  epochs: 100
  batch_size: 16
  lr: 1.0e-4
  use_cosine_decay: true
```

## Research Questions

1. **Grounding**: How accurately can CNNs extract puzzle state from images?
2. **Reasoning**: Can iterative constraint satisfaction improve upon pure grounding?
3. **Compositionality**: Do models generalize to unseen entities, relations, and reasoning rules?
4. **Interaction**: How do grounding quality and reasoning capacity interact?

## Experiments

- ✅ Entity composition: Unseen digit generalization
- ✅ Relational composition: Unseen constraint patterns
- ✅ Rule composition: Combinations of reasoning rules
- ✅ Ablation studies on refinement iterations
- ✅ Analysis of grounding-reasoning trade-offs

## Visualization

Generate figures for composition analysis:

```bash
python viz/fig_entity_composition.py
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{grounding_vs_compositionality,
  title={Grounding vs. Compositionality: Investigating Generalization in Neural-Symbolic Systems},
  year={2024}
}
```

## License

[Add your license here]

## Contact

[Add your contact information]

---

**Built with ❤️ for neural-symbolic research**
