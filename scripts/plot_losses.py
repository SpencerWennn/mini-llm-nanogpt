"""Plot the loss curves written by train.py for one or more experiment folders."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_metrics(folder):
    path = Path(folder) / 'metrics.jsonl'
    records = []
    with path.open() as f:
        for line in f:
            record = json.loads(line)
            if 'iter' in record:
                records.append(record)
    return records


parser = argparse.ArgumentParser()
parser.add_argument('folders', nargs='+', help='e.g. out/baseline out/rope')
parser.add_argument('--output', default='results/loss_comparison.png')
args = parser.parse_args()

plt.figure(figsize=(9, 5))
for folder in args.folders:
    records = read_metrics(folder)
    label = Path(folder).name
    steps = [r['iter'] for r in records]
    plt.plot(steps, [r['train_loss'] for r in records], '--', alpha=.55, label=f'{label} train')
    plt.plot(steps, [r['val_loss'] for r in records], label=f'{label} val')
plt.xlabel('training iteration')
plt.ylabel('cross-entropy loss')
plt.title('Character-level Shakespeare: training and validation loss')
plt.grid(alpha=.25)
plt.legend(ncol=2, fontsize=8)
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(args.output, dpi=180)
print(f'wrote {args.output}')
