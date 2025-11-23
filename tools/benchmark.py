#!/usr/bin/env python3
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt

def load(path): return json.loads(open(path).read())

def mean_per_cycle(results):
    cycles = len(next(iter(results.values()))['cycles'])
    mean_loss=[]; mean_acc=[]
    for k in range(cycles):
        mean_loss.append(sum(results[s]['cycles'][k]['loss'] for s in results)/len(results))
        mean_acc.append(sum(results[s]['cycles'][k]['cell_acc'] for s in results)/len(results))
    return mean_loss, mean_acc

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ours", required=True)
    p.add_argument("--author", required=True)
    p.add_argument("--out_prefix", default="plots/benchmark")
    args = p.parse_args()
    ours = load(args.ours)
    auth = load(args.author)
    ours_loss, ours_acc = mean_per_cycle(ours)
    auth_loss, auth_acc = mean_per_cycle(auth)
    cycles = list(range(max(len(ours_loss), len(auth_loss))))
    plt.figure(figsize=(6,4))
    plt.plot(cycles[:len(ours_loss)], ours_acc, marker='o', label='ours')
    plt.plot(cycles[:len(auth_acc)], auth_acc, marker='x', label='author')
    plt.title('Cell Accuracy per Cycle')
    plt.xlabel('Cycle')
    plt.ylabel('Cell Accuracy')
    plt.legend(); plt.grid(True)
    Path('plots').mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{args.out_prefix}_acc.png", bbox_inches='tight')
    plt.close()
    # Loss plot
    plt.figure(figsize=(6,4))
    plt.plot(cycles[:len(ours_loss)], ours_loss, marker='o', label='ours')
    plt.plot(cycles[:len(auth_loss)], auth_loss, marker='x', label='author')
    plt.title('Mean Loss per Cycle')
    plt.xlabel('Cycle'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.savefig(f"{args.out_prefix}_loss.png", bbox_inches='tight')
    print("Saved benchmark plots with prefix", args.out_prefix)