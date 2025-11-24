# Tiny Recursive Model — Toy Sudoku Solver

A simplified, CPU-friendly implementation of the **Tiny Recursive Model (TRM)** architecture for Sudoku.  
TRM: https://github.com/SamsungSAILMontreal/TinyRecursiveModels

This project builds a small, understandable baseline that runs on a **MacBook CPU**, enabling experimentation with recursive reasoning before scaling to the full TRM.

---

## Features
- Minimal **Toy TRM** model (Embedding → Encoder → GRUCell → Decoder → Feedback)
- **Recursive refinement** through H cycles
- **Offline Sudoku augmentation** (×10)
- Per-cycle logging: loss, accuracy, changed cells, predictions
- Config-driven runs (`configs/*.yaml`)
- Result aggregation via `aggregate_results.py`

---

# 1. Run the Project

### Install dependencies
```bash
pip install torch pyyaml tqdm
```

### Train the model
```bash
python tools/toy_trm.py configs/config.yaml
```

### Verify results
```bash
python tools/aggregate_results.py --results-dir results --file result.json
```

# 2. What’s Implemented

## Toy TRM Architecture
- Digit embeddings
- Linear encoder
- GRUCell latent state
- Decoder for 81×9 logits
- Answer-projection feedback
- H recursion cycles (default H=2)

## Dataset Augmentation
Digit remapping, row/column swaps, block permutations → ~990 samples.

## Logging + Metrics
- Per-cycle accuracy
- Wrong-cell histograms
- Puzzle solve rate
- Overall mean/median cell accuracy

# 3. Current Results Summary
Training with:
```bash
epochs: 100
H_cycles: 2
embed_dim: 32
hidden_dim: 128
optimizer: AdamW
learning_rate: 1e-4
weight_decay: 1e-5
grad_accum_steps: 4
batch_size: 1
```
Dataset
- ~990 augmented Sudoku puzzles
- 10 clean validation puzzles

Results:
```bash
mean_cell_acc=0.90
median=0.901
puzzles_solved=1/990
wrong_cell_hist={1: 837, 0: 108, 2: 45}
```
mean_cell_acc: Average % of correctly predicted digits

puzzles_solved: Number of puzzles solved perfectly (all 81 cells)

wrong_cell_hist:	How many puzzles are off by 1 / 2 / 3… incorrect digits

per_cycle_mean_acc:	Accuracy improvement across recursive cycles

Interpretation:
- The model is learning strong per-digit accuracy
- But solving all 81 digits perfectly is a hard threshold
- The toy TRM lacks deeper recursion and richer latent updates
- These results are good for CPU experiments
- Next scaling steps will require GPU + deeper TRM (H=6, L=3)


# 4. Code Overview
## toy_trm.py
- Loads dataset
- Runs H-cycle TRM forward pass
- Computes CE loss per cycle
- Logs accuracy + predictions
- Saves JSON results

## aggregate_results.py
- Reads training JSON
- Extracts final cell accuracies
- Computes solve rate + wrong-cell histogram

# 5. How the Code Works
## Dataset Loader (SimpleSudokuDataset)
Reads JSONL lines:
```bash
{"input": [...], "target": [...], "id": ...}
```
Returns tensors (input, target, id).

## Model (ToyTRM)
### Architecture Flow
```bash
input grid (9×9)
→ embedding
→ flatten
→ linear encoder
→ GRUCell latent update
→ linear decoder (predict 81×9 logits)
→ answer-projection (previous logits → next cycle)
```
### Runs for H recursive cycles:
```bah
cycle 0: initial guess
cycle 1: refinement
...
```
Mimics the author’s recursive reasoning.

## Training Loop
For every sample:
1. Initialize latent to zeros
2. Repeat for H cycles:
  - forward pass
  - compute cross-entropy loss
  - store predictions + accuracy
3. Average losses across cycles
4. Backpropagate
5. Save results to JSON

Output format:
```bash
{
  "sample_id": {
    "cycles": [
      { "cycle": 0, "loss": ..., "cell_acc": ..., "changed_cells": ... },
      { "cycle": 1, ... }
    ]
  }
}
```

## Aggregation (aggregate_results.py)
Reads the output JSON and computes:
- final cell accuracy
- cycle-wise mean accuracy
- wrong-digit histograms
- puzzle solve rates
Used for end-of-training evaluation.

# 6. Next Steps (Future Work)
- Add EMA, weight-decay tuning, grad clipping
- Increase recursion depth (H=3–6)
- Implement author’s latent update rule
- Introduce Sudoku constraint losses
- Move full training to GPU
- Attempt to match author-level solve rates


