#!/usr/bin/env python3
"""Render archived train/validation losses; losses are not experiment efficacy."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(root):
    metrics = json.loads((root / 'metrics.json').read_text())
    records = [json.loads(x) for x in (root / 'training_log.jsonl').read_text().splitlines()]
    curve = metrics['result']['validation_curve']
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), constrained_layout=True)
    axes[0].plot([x['optimizer_step'] for x in records], [x['loss'] for x in records], color='#3367B5', lw=1.6)
    axes[0].set(xlabel='Optimizer step', ylabel='Completion-token loss', title='Training (batch average)')
    axes[1].plot([x['epoch'] for x in curve], [x['loss'] for x in curve], marker='o', color='#CF5B32', lw=1.6)
    axes[1].set(xlabel='Epoch', ylabel='Completion-token loss', title='Same-task validation')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Qwen2.5-32B QLoRA: full available-data fitting', fontsize=11)
    fig.savefig(root / 'training_curve.png', dpi=220)
    fig.savefig(root / 'training_curve.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, required=True)
    plot(parser.parse_args().result_dir)
