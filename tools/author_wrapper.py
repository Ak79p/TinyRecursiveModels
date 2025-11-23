#!/usr/bin/env python3

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
"""
Train the author's TRM on the small sudoku dataset (debug / moderate config).
- Uses the actual TinyRecursiveReasoningModel_ACTV1 from models/recursive_reasoning.
- Runs H_cycles steps by repeatedly calling model.forward(carry, batch) and passing
  returned carry back in, collecting outputs['logits'] per cycle.
- Computes per-cycle cross-entropy loss only on the first 81 tokens (real sudoku cells).
- Logs per-sample per-cycle: cycle index, loss_k, cell_acc_k, changed_cells_k.
- Conservative CPU settings suitable for 8GB Mac.
"""
import os, json, time, argparse
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# local dataset loader created in Step 3
from data.sudoku_dataloader import SudokuDataset

# Import author model
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

def flatten_preds_to_grid(preds_flat):
    # preds_flat: tensor shape (seq_len,) values (maybe 0..9 or 1..9). Return 9x9 grid tensor.
    seq = preds_flat[:81]  # ensure only first 81 tokens
    return seq.view(9,9)

def cell_accuracy(preds, targets):
    # preds, targets: tensors of shape (9,9) with values 1..9
    return (preds == targets).float().mean().item()

def count_changes(prev, curr):
    # prev, curr: tensors shape (9,9) or None
    if prev is None:
        # count non-zero preds as changed
        return (curr != 0).sum().item()
    return (prev != curr).sum().item()

def train_epoch(model, loader, optim, criterion, device, epochs, H_cycles, out_json_path):
    model.to(device)
    results = {}
    for epoch in range(1, epochs+1):
        model.train()
        start = time.time()
        for batch_idx, (inp, tgt, ids) in enumerate(loader):
                        # move to device and flatten to seq_len
            B = inp.shape[0]
            batch_inputs = inp.view(B, -1).to(device)   # B x 81
            batch_targets = tgt.view(B, -1).to(device)  # B x 81

            # Build batch dict expected by model (we will pad it if model requires longer pos embedding)
            batch = {"inputs": batch_inputs, "puzzle_identifiers": torch.zeros((B,1), dtype=torch.int64).to(device)}

            # --- PAD inputs to match model positional embedding length (if nested pos exists) ---
            pos_len = None
            try:
                # prefer nested path seen in this repo
                if hasattr(model, "inner") and hasattr(model.inner, "embed_pos") and hasattr(model.inner.embed_pos, "embedding_weight"):
                    pos_len = model.inner.embed_pos.embedding_weight.shape[0]
                # fallback checks (keeps it robust)
                elif hasattr(model, "embed_pos") and hasattr(model.embed_pos, "embedding_weight"):
                    pos_len = model.embed_pos.embedding_weight.shape[0]
                elif hasattr(model, "pos_emb") and hasattr(model.pos_emb, "weight"):
                    pos_len = model.pos_emb.weight.shape[0]
            except Exception:
                pos_len = None

            if pos_len is not None:
                seq_inp_len = batch["inputs"].shape[1]
                if seq_inp_len != pos_len:
                    pad_len = pos_len - seq_inp_len
                    if pad_len > 0:
                        # pad inputs and targets on the right with zeros
                        pad_tensor = torch.zeros((B, pad_len), dtype=batch["inputs"].dtype).to(device)
                        batch["inputs"] = torch.cat([batch["inputs"], pad_tensor], dim=1)
                        tgt_pad = torch.zeros((B, pad_len), dtype=batch_targets.dtype).to(device)
                        batch_targets = torch.cat([batch_targets, tgt_pad], dim=1)
                    else:
                        # truncate if model expects shorter sequence
                        batch["inputs"] = batch["inputs"][:, :pos_len]
                        batch_targets = batch_targets[:, :pos_len]
            # --- end padding ---

            # init carry (now with padded inputs if necessary)
            carry = model.initial_carry(batch)


            # We'll run H_cycles times by calling model.forward repeatedly and collecting logits each time
            cycle_losses = []
            cycle_logits = []  # store for each cycle the logits (B x seq_len x vocab)
            for k in range(H_cycles):
                carry, outputs = model.forward(carry, batch)
                if "logits" not in outputs:
                    raise RuntimeError("Model outputs don't include 'logits' key; cannot compute loss.")
                logits = outputs["logits"]  # expect shape B x seq_len x vocab_size
                cycle_logits.append(logits.detach().cpu())
                # compute loss on first 81 tokens only
                # ensure logits has at least 81 timesteps
                seq_len = logits.shape[1]
                use_len = min(seq_len, 81)
                flat_logits = logits[:, :use_len, :].contiguous().view(B*use_len, -1)
                flat_targets = batch_targets[:, :use_len].contiguous().view(B*use_len) - 1  # targets 1..9 -> classes 0..8
                loss_k = criterion(flat_logits, flat_targets.long().to(flat_logits.device))
                cycle_losses.append(loss_k)
            # Sum losses across cycles (simple aggregate) and step optimizer
            total_loss = sum(cycle_losses) / len(cycle_losses)
            optim.zero_grad()
            total_loss.backward()
            optim.step()

            # Logging per sample in batch (small B expected)
            for bi in range(B):
                sid = int(ids[bi].item()) if hasattr(ids[bi], 'item') else ids[bi]
                results.setdefault(sid, {"cycles": []})
                prev_pred = None
                for k in range(H_cycles):
                    logits_k = cycle_logits[k][bi]  # seq_len x vocab (cpu)
                    # predicted classes (0..vocab-1) -> map to digits 1..9
                    preds_k = logits_k.argmax(dim=-1) + 1
                    # only first 81 positions
                    preds_k81 = preds_k[:81]
                    # reshape to 9x9 for accuracy/counting
                    pred_grid = preds_k81.view(9,9)
                    target_grid = batch_targets[bi, :81].cpu().view(9,9)
                    acc = cell_accuracy(pred_grid, target_grid)
                    changed = count_changes(prev_pred, pred_grid)
                    loss_val = float(cycle_losses[k].detach().cpu().item())
                    print(f"[epoch {epoch}] sample {sid} | cycle {k} | loss={loss_val:.4f} | cell_acc={acc:.3f} | changed_cells={changed}")
                    results[sid]["cycles"].append({
                        "cycle": k,
                        "loss": loss_val,
                        "cell_acc": acc,
                        "changed_cells": changed,
                        "pred_flat": preds_k81.tolist()
                    })
                    prev_pred = pred_grid
        epoch_time = time.time() - start
        print(f"Epoch {epoch} finished in {epoch_time:.1f}s")
        # dump interim results
        with open(out_json_path, "w") as f:
            json.dump(results, f)
    # final save
    with open(out_json_path, "w") as f:
        json.dump(results, f)
    print("Training complete. Results saved to", out_json_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sudoku_small.jsonl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--H_cycles", type=int, default=3)
    parser.add_argument("--out", default="results/author_trm_sudoku_debug.json")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    device = torch.device("cpu")

    ds = SudokuDataset(args.data)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # minimal config like the forward test
    config_dict = {
        "batch_size": args.batch_size,
        "seq_len": 81,
        "puzzle_emb_ndim": 0,
        "num_puzzle_identifiers": 1,
        "vocab_size": 10,
        "H_cycles": args.H_cycles,
        "L_cycles": 1,
        "H_layers": 1,
        "L_layers": 1,
        "hidden_size": 64,
        "expansion": 4.0,
        "num_heads": 4,
        "pos_encodings": "learned",
        "halt_max_steps": 6,
        "halt_exploration_prob": 0.0,
        "forward_dtype": "float32",
    }

    print("Instantiating model...")
    model = TinyRecursiveReasoningModel_ACTV1(config_dict)
    print("Model instantiated.")

    # optimizer + criterion
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_epoch(model, loader, optim, criterion, device, epochs=args.epochs, H_cycles=args.H_cycles, out_json_path=args.out)

if __name__ == "__main__":
    main()