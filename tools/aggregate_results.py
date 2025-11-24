#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import statistics as stats
import math

# Default results path (adjust if your repo layout differs)
RESULTS_ROOT = Path("results")


def safe_mean(xs):
    return None if len(xs) == 0 else sum(xs) / len(xs)


def try_extract_final_cell_acc_from_entry(e):
    """
    Return (final_cell_acc, per_cycle_acc_list, changed_cells_list) or (None, None, None).
    Heuristics cover many common JSON shapes.
    """
    if 'cell_acc' in e:
        ca = e['cell_acc']
        if isinstance(ca, list) and len(ca) > 0:
            try:
                per_cycle = [float(x) for x in ca]
                return per_cycle[-1], per_cycle, None
            except Exception:
                pass
        else:
            try:
                return float(ca), None, None
            except Exception:
                pass

    for k in ('cycle_cell_acc', 'cycle_acc', 'per_cycle_acc', 'acc_per_cycle', 'cycle_accs'):
        if k in e:
            v = e[k]
            if isinstance(v, list) and len(v) > 0:
                try:
                    per_cycle = [float(x) for x in v]
                    return per_cycle[-1], per_cycle, None
                except Exception:
                    pass

    for pred_k, tgt_k in (('pred_flat', 'target_flat'), ('pred', 'target'), ('prediction', 'target')):
        if pred_k in e and tgt_k in e:
            pred = e[pred_k]
            tgt = e[tgt_k]
            if isinstance(pred, list) and isinstance(tgt, list) and len(pred) == len(tgt) and len(pred) > 0:
                try:
                    eq = sum(1 for a, b in zip(pred, tgt) if a == b)
                    acc = eq / len(pred)
                    return acc, None, None
                except Exception:
                    pass
            if isinstance(pred, str) and isinstance(tgt, str) and len(pred) == len(tgt) and len(pred) > 0:
                eq = sum(1 for a, b in zip(pred, tgt) if a == b)
                return eq / len(pred), None, None

    for pred_k, tgt_k in (('final_pred', 'target'), ('final_prediction', 'target')):
        if pred_k in e and tgt_k in e:
            p = e[pred_k]
            t = e[tgt_k]
            if isinstance(p, list) and isinstance(t, list) and len(p) == len(t) and len(p) > 0:
                eq = sum(1 for a, b in zip(p, t) if a == b)
                return eq / len(p), None, None
            if isinstance(p, str) and isinstance(t, str) and len(p) == len(t) and len(p) > 0:
                eq = sum(1 for a, b in zip(p, t) if a == b)
                return eq / len(p), None, None

    if 'final_cell_acc' in e:
        try:
            return float(e['final_cell_acc']), None, None
        except Exception:
            pass

    return None, None, None


def extract_metadata(obj):
    md = {}
    for k in ['run_name', 'date', 'device', 'epochs', 'batch_size', 'learning_rate', 'lr', 'params', 'param_count', 'model_params']:
        if k in obj:
            md[k] = obj[k]
    if 'config' in obj and isinstance(obj['config'], dict):
        cfg = obj['config']
        for k in ['epochs', 'batch_size', 'lr', 'learning_rate', 'device']:
            if k in cfg:
                md.setdefault(k, cfg[k])
    return md


def normalize_cycles_entry(entry):
    """
    If entry contains 'cycles' (list or JSON-string), produce a new dict based on the final cycle.
    Preserve per-cycle cell_acc as 'cycle_cell_acc' (if available).
    """
    cycles_raw = entry.get('cycles')
    if cycles_raw is None:
        return entry  # nothing to do

    # If cycles is a JSON string, try to parse it
    if isinstance(cycles_raw, str):
        try:
            cycles = json.loads(cycles_raw)
        except Exception:
            # fallback: couldn't parse string; return original entry
            return entry
    else:
        cycles = cycles_raw

    # If still not a list, give up
    if not isinstance(cycles, list) or len(cycles) == 0:
        return entry

    # Build per-cycle cell_acc list if available
    cycle_cell_acc = []
    for c in cycles:
        if isinstance(c, dict) and 'cell_acc' in c:
            try:
                cycle_cell_acc.append(float(c['cell_acc']))
            except Exception:
                cycle_cell_acc.append(None)
        else:
            cycle_cell_acc.append(None)

    # Use the last cycle dict as the base for final fields
    last = cycles[-1]
    if not isinstance(last, dict):
        return entry  # unexpected shape

    # copy last cycle fields into a new dict, but attach cycle_cell_acc for full trace
    new_entry = dict(last)
    # Only attach if there is at least one non-None value
    if any(x is not None for x in cycle_cell_acc):
        new_entry['cycle_cell_acc'] = cycle_cell_acc
    # Keep original entry id if present to help debugging
    if 'id' in entry and 'id' not in new_entry:
        new_entry['original_id'] = entry['id']
    return new_entry


def analyze_result_file(path: Path):
    with open(path, 'r') as f:
        data = json.load(f)

    metadata = {}
    entries = None

    # If top-level is dict
    if isinstance(data, dict):
        metadata = extract_metadata(data)

        # Preferred containers that hold per-puzzle lists
        candidate_lists = []
        for key in ('results', 'eval', 'eval_results', 'runs', 'examples', 'entries', 'data'):
            if key in data and isinstance(data[key], list):
                candidate_lists.append(data[key])

        # If no explicit list, check for any top-level list-valued key
        if not candidate_lists:
            for k, v in data.items():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                    candidate_lists.append(v)

        # NEW: if still not found, but the top-level dict's values are dicts, treat them as entries
        if not candidate_lists:
            # check if values are dicts (e.g., {"1": {...}, "2": {...}})
            vals = list(data.values())
            if len(vals) > 0 and isinstance(vals[0], dict):
                # but avoid grabbing metadata dicts that aren't entries: heuristic - entry dicts usually contain numeric/array keys like 'cycles'/'pred_flat'/'cell_acc'
                likely_entries = []
                for v in vals:
                    if isinstance(v, dict):
                        likely_entries.append(v)
                if likely_entries:
                    entries = likely_entries

        # if we found candidate lists earlier, use the first one
        if candidate_lists and entries is None:
            entries = candidate_lists[0]

        # If still nothing, check for per-cycle arrays at top-level and return a single-entry summary
        if entries is None:
            per_cycle_acc = None
            for k in ('per_cycle_acc', 'cycle_acc', 'cycle_cell_acc', 'cell_accs'):
                if k in data and isinstance(data[k], list):
                    per_cycle_acc = [float(x) for x in data[k]]
                    final = per_cycle_acc[-1] if per_cycle_acc else None
                    return {
                        'file': str(path.name),
                        'metadata': metadata,
                        'N': 1,
                        'mean_final_cell_acc': final,
                        'median_final_cell_acc': final,
                        'std_final_cell_acc': 0.0,
                        'num_solved': 1 if final == 1.0 else 0,
                        'percent_solved': 100.0 if final == 1.0 else 0.0,
                        'per_cycle_mean_acc': per_cycle_acc,
                        'per_cycle_delta': [per_cycle_acc[i] - per_cycle_acc[i - 1] for i in range(1, len(per_cycle_acc))] if per_cycle_acc else []
                    }

    elif isinstance(data, list):
        entries = data
    else:
        return {'file': str(path.name), 'metadata': {}, 'error': 'unknown json format'}

    if not entries or len(entries) == 0:
        return {'file': str(path.name), 'metadata': metadata, 'error': 'no per-puzzle entries found'}

    # If entries appear to have a 'cycles' field, normalize each entry to the final-cycle dict
    first = entries[0]
    if isinstance(first, dict) and 'cycles' in first:
        processed = []
        for e in entries:
            try:
                norm = normalize_cycles_entry(e)
                processed.append(norm)
            except Exception:
                processed.append(e)
        entries = processed

    # Now compute metrics from entries list
    final_accs = []
    per_cycle_accs_lists = []
    changed_cells_lists = []
    wrong_cell_counts = []
    n_cells_guess = None

    for e in entries:
        final_acc, per_cycle, changed = try_extract_final_cell_acc_from_entry(e)
        if final_acc is not None:
            final_accs.append(final_acc)
            if per_cycle:
                per_cycle_accs_lists.append(per_cycle)
            if changed:
                changed_cells_lists.append(changed)

            # try to infer number of cells from pred_flat / target_flat if present
            if 'pred_flat' in e and isinstance(e['pred_flat'], list):
                n_cells_guess = len(e['pred_flat'])
                wrong_cell_counts.append(int(round((1 - final_acc) * n_cells_guess)))
            elif 'target_flat' in e and isinstance(e['target_flat'], list):
                n_cells_guess = len(e['target_flat'])
                wrong_cell_counts.append(int(round((1 - final_acc) * n_cells_guess)))
            else:
                n_cells_guess = n_cells_guess or 81
                wrong_cell_counts.append(int(round((1 - final_acc) * n_cells_guess)))

    N = len(final_accs)
    if N == 0:
        return {'file': str(path.name), 'metadata': metadata, 'error': 'no final accs extracted from entries'}

    mean_final = safe_mean(final_accs)
    median_final = stats.median(final_accs)
    std_final = stats.pstdev(final_accs) if N > 1 else 0.0
    num_solved = sum(1 for a in final_accs if math.isclose(a, 1.0, rel_tol=1e-9) or a >= 0.999999)
    percent_solved = num_solved / N * 100.0

    per_cycle_mean = None
    per_cycle_delta = None
    if per_cycle_accs_lists:
        maxlen = max(len(x) for x in per_cycle_accs_lists)
        padded = []
        for lst in per_cycle_accs_lists:
            if len(lst) < maxlen:
                lst = lst + [lst[-1]] * (maxlen - len(lst))
            padded.append(lst)
        per_cycle_mean = [safe_mean([p[i] for p in padded]) for i in range(maxlen)]
        per_cycle_delta = [per_cycle_mean[i] - per_cycle_mean[i - 1] for i in range(1, len(per_cycle_mean))]

    wrong_hist = {}
    for w in wrong_cell_counts:
        k = str(w)
        wrong_hist[k] = wrong_hist.get(k, 0) + 1

    summary = {
        'file': str(path.name),
        'metadata': metadata,
        'N': N,
        'mean_final_cell_acc': round(mean_final, 3),
        'median_final_cell_acc': round(median_final, 3),
        'std_final_cell_acc': round(std_final, 3),
        'num_puzzles_fully_solved': int(num_solved),
        'percent_solved': round(percent_solved, 3),
        'per_cycle_mean_acc': [round(x, 4) for x in per_cycle_mean] if per_cycle_mean else None,
        'per_cycle_delta': [round(x, 4) for x in per_cycle_delta] if per_cycle_delta else None,
        'wrong_cell_count_histogram': wrong_hist,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default=str(RESULTS_ROOT), help='path to results directory')
    parser.add_argument('--file', type=str, default=None, help='specific json file to analyze (optional)')
    args = parser.parse_args()
    rd = Path(args.results_dir)
    if not rd.exists():
        print(f"Results directory {rd} not found: {rd}")
        return

    files = []
    if args.file:
        fp = rd / args.file
        if not fp.exists():
            print(f"File {fp} not found in {rd}")
            return
        files = [fp]
    else:
        files = sorted(list(rd.glob("*.json")))
        if not files:
            print("No .json files found in results dir:", rd)
            return

    all_summaries = []
    for f in files:
        try:
            s = analyze_result_file(f)
            all_summaries.append(s)
        except Exception as ex:
            all_summaries.append({'file': str(f.name), 'error': str(ex)})

    for s in all_summaries:
        print("\n" + "=" * 60)
        if 'error' in s:
            print(f"File: {s.get('file')} - ERROR: {s.get('error')}")
            continue
        md = s.get('metadata', {})
        print(f"run_name: {md.get('run_name', s.get('file'))}")
        print(f"device: {md.get('device', 'unknown')}, epochs: {md.get('epochs', '?')}, batch_size: {md.get('batch_size', '?')}, lr: {md.get('learning_rate', md.get('lr', '?'))}")
        print(f"params: {md.get('params', md.get('param_count', '?'))}")
        print(f"N_eval_puzzles: {s['N']}")
        print(f"train_final: mean_cell_acc={s['mean_final_cell_acc']}, median={s['median_final_cell_acc']}, std={s['std_final_cell_acc']}, puzzles_solved={s['num_puzzles_fully_solved']}/{s['N']} ({s['percent_solved']}%)")
        if s['per_cycle_mean_acc'] is not None:
            print(f"per_cycle_last_epoch: acc={s['per_cycle_mean_acc']}, delta={s['per_cycle_delta']}")
        if s['wrong_cell_count_histogram']:
            print(f"wrong_cell_dist: {s['wrong_cell_count_histogram']}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
