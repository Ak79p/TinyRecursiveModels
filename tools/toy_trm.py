#!/usr/bin/env python3
import os, json, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from utils.functions import apply_alpha_blend

# Simple dataset (adapted from earlier)
class SimpleSudokuDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        import json
        self.items = []
        with open(path) as f:
            for line in f:
                self.items.append(json.loads(line))
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        obj = self.items[idx]
        inp = torch.tensor(obj["input"], dtype=torch.long)
        tgt = torch.tensor(obj["target"], dtype=torch.long)
        return inp, tgt, obj["id"]

# Toy TRM class (same as earlier)
class ToyTRM(nn.Module):
    def __init__(self, embed_dim=16, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(10, embed_dim)
        self.enc = nn.Linear(9*9*embed_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 9*9*9)
        self.ans_proj = nn.Linear(9*9*9, hidden_dim)
    def forward_once(self, inp_grid, prev_answer_logits, latent):
        B = inp_grid.shape[0]
        x = self.embed(inp_grid)
        x = x.view(B, -1)
        enc = torch.relu(self.enc(x))
        if prev_answer_logits is not None:
            fb = torch.relu(self.ans_proj(prev_answer_logits))
            enc = enc + fb
        latent = self.gru(enc, latent)
        logits = self.decoder(latent)
        logits = logits.view(B, 9, 9, 9)
        return logits, latent
    def init_latent(self, B):
        return torch.zeros(B, self.gru.hidden_size)

def flatten_logits_to_preds(logits):
    return logits.argmax(dim=-1) + 1

def cell_accuracy(preds, targets):
    return (preds == targets).float().mean().item()

def count_changes(prev, curr):
    if prev is None:
        return (curr != 0).sum().item()
    return (prev != curr).sum().item()

def run_recursive_cycles(model, input_grid, target, H, device, prev_logits, latent, criterion, B, alpha=1.0):
    cycle_losses = []
    cycle_preds = []
    for k in range(H):
        logits, latent = model.forward_once(input_grid, prev_logits, latent)
        flat_logits = logits.view(B*81, 9)
        flat_tgt = (target.view(B*81) - 1).long()
        loss_k = criterion(flat_logits, flat_tgt.to(flat_logits.device))
        cycle_losses.append(loss_k)
        preds = flatten_logits_to_preds(logits).detach().cpu()
        cycle_preds.append(preds.numpy().tolist())
        new_flat = logits.detach().view(B, -1).to(device)
        if prev_logits is None:
            prev_logits = new_flat
        else:
            # alpha blend hook - 1.0 preserves original overwrite behavior
            prev_logits = apply_alpha_blend(prev_logits, new_flat, alpha)
    cycle_logs = {'cycle_losses': cycle_losses, 'cycle_preds': cycle_preds}
    final_pred = cycle_preds[-1] if cycle_preds else None
    return final_pred, cycle_logs

def run_train(cfg):
    data_path = cfg['data_path']
    ds = SimpleSudokuDataset(data_path)
    loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
    model_cfg = cfg.get('toy', {})
    model = ToyTRM(embed_dim=model_cfg.get('embed_dim',16), hidden_dim=model_cfg.get('hidden_dim',64))
    device = torch.device('cpu')
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.get('learning_rate',1e-3))
    criterion = nn.CrossEntropyLoss()
    results = {}

    epochs = cfg.get('epochs', 1)
    H = cfg.get('H_cycles', 3)
    alpha = cfg.get('alpha_blend', 1.0)
    for epoch in range(1, epochs+1):
        for batch_idx, (inp, tgt, ids) in enumerate(loader):
            B = inp.shape[0]
            inp = inp.to(device); tgt = tgt.to(device)
            prev_logits = None
            latent = model.init_latent(B).to(device)
            final_pred, cycle_logs = run_recursive_cycles(model, inp, tgt, H, device, prev_logits, latent, criterion, B, alpha)
            cycle_losses = cycle_logs['cycle_losses']
            cycle_preds = cycle_logs['cycle_preds']
            total_loss = sum(cycle_losses) / len(cycle_losses)
            optim.zero_grad(); total_loss.backward(); optim.step()
            for bi in range(B):
                sid = int(ids[bi])
                results.setdefault(sid, {'cycles': []})
                prev_pred = None
                for k in range(H):
                    pred_grid = torch.tensor(cycle_preds[k][bi])
                    acc = cell_accuracy(pred_grid, tgt[bi].cpu())
                    changed = count_changes(prev_pred, pred_grid)
                    loss_val = float(cycle_losses[k].detach().cpu().item())
                    print(f"[epoch {epoch}] sample {sid} | cycle {k} | loss={loss_val:.4f} | cell_acc={acc:.3f} | changed_cells={changed}")
                    results[sid]['cycles'].append({
                        'cycle': k, 'loss': loss_val, 'cell_acc': acc, 'changed_cells': changed, 'pred_flat': cycle_preds[k][bi]
                    })
                    prev_pred = pred_grid
    # Write results
    out_dir = cfg.get('results_dir', 'results')
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_file = Path(out_dir) / cfg.get('result_filename', 'toy_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f)
    print("Saved results to", out_file)

def main(cfg):
    run_train(cfg)

if __name__ == "__main__":
    import yaml, sys
    cfg = yaml.safe_load(open(sys.argv[1])) if len(sys.argv)>1 else yaml.safe_load(open('configs/config.yaml'))
    main(cfg)