#!/usr/bin/env python3
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_results(path):
    return json.loads(open(path).read())

def summarize(results):
    # results: {sid: {"cycles":[{loss, cell_acc, ...}, ...]}}
    n_samples = len(results)
    cycles = len(next(iter(results.values()))['cycles'])
    mean_loss = []
    mean_acc = []
    for k in range(cycles):
        losses = [results[s]['cycles'][k]['loss'] for s in results]
        accs = [results[s]['cycles'][k]['cell_acc'] for s in results]
        mean_loss.append(sum(losses)/len(losses))
        mean_acc.append(sum(accs)/len(accs))
    return mean_loss, mean_acc

def plot_curve(x, y, title, ylabel, outpath):
    plt.figure(figsize=(6,4))
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel('Cycle')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()
    print("Saved plot:", outpath)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out_prefix", default="plots/run")
    args = p.parse_args()
    res = load_results(args.results)
    loss, acc = summarize(res)
    cycles = list(range(len(loss)))
    out_prefix = args.out_prefix
    Path('plots').mkdir(parents=True, exist_ok=True)
    plot_curve(cycles, loss, "Mean Loss per Cycle", "Loss", f"{out_prefix}_loss.png")
    plot_curve(cycles, acc, "Mean Cell Accuracy per Cycle", "Cell Accuracy", f"{out_prefix}_acc.png")