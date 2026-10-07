#!/usr/bin/env bash
set -euo pipefail
python3 data/shakespeare_char/prepare.py
for experiment in experiments ablation_rmsnorm ablation_swiglu ablation_nope ablation_rope ablation_gqa2; do
  python3 train.py "config/${experiment}.py"
done
python3 scripts/plot_losses.py out/baseline out/rmsnorm out/swiglu out/nope out/rope out/gqa2
