"""Append sweep results from checkpoint metadata, without parsing console output."""
import argparse
import csv
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-tag', required=True)
    parser.add_argument('--results-file', required=True)
    parser.add_argument('--label', default='')
    parser.add_argument('--wall-seconds', type=int, default=0)
    parser.add_argument('--flops-budget', default='')
    args = parser.parse_args()
    base = Path(os.environ.get('NANOCHAT_BASE_DIR', Path.home()/'.cache'/'nanochat'))
    folder = base/'base_checkpoints'/args.model_tag
    files = list(folder.glob('meta_*.json'))
    if not files:
        parser.error(f'No checkpoint metadata found in {folder}')
    latest = max(files, key=lambda path: int(path.stem.split('_')[-1]))
    meta = json.loads(latest.read_text())
    config, counts, core = meta['model_config'], meta.get('parameter_counts', {}), meta.get('core_results', {})
    row = dict(label=args.label, model_tag=args.model_tag, depth=config['n_layer'],
               predictor_layers=config.get('predictor_layers', 0), model_dim=config['n_embd'],
               params_total=counts.get('total', ''), params_action_encoder=counts.get('action_encoder', 0),
               params_predictor=counts.get('predictor', 0), steps=meta['step'],
               tokens_trained=meta['step']*meta['total_batch_size'], val_bpb=meta.get('val_bpb'),
               core_score=core.get('core_metric', ''), core_mode=core.get('eval_mode', ''),
               planning_goal=core.get('planning_goal', ''), planning_horizon=core.get('planning_horizon', ''),
               wall_seconds=args.wall_seconds, flops_budget=args.flops_budget)
    output = Path(args.results_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists() and output.stat().st_size > 0
    if exists:
        with output.open(newline='') as f:
            if next(csv.reader(f)) != list(row):
                parser.error('Results CSV header differs; use a new results file')
    with output.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Saved {args.model_tag}: BPB={row['val_bpb']}, CORE={row['core_score']} ({row['core_mode']})")


if __name__ == '__main__':
    main()
