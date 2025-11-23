#!/usr/bin/env python3
"""
Main runner: reads configs/config.yaml and runs either the toy TRM or the author wrapper.
Produces a JSON results file under results/.
"""
import yaml, os, sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yaml")
args = parser.parse_args()

cfg = yaml.safe_load(open(args.config))

# ensure output dirs exist
Path(cfg['results_dir']).mkdir(parents=True, exist_ok=True)
Path(cfg['plots_dir']).mkdir(parents=True, exist_ok=True)

if cfg['model'] == 'toy':
    from tools.toy_trm import main as toy_main
    # pass config dict into toy_main (toy_main should accept argparser style or dict)
    toy_main(cfg)
elif cfg['model'] == 'author':
    from tools.author_wrapper import main as author_main
    author_main(cfg)
else:
    raise ValueError("Unknown model in config: " + str(cfg['model']))