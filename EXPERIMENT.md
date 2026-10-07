# Mini-LLM: decoder-only Transformer ablations

This repository is a focused fork of [nanoGPT](https://github.com/karpathy/nanoGPT), using its character-level Shakespeare quick-start experiment. The model is a six-layer, six-head decoder-only Transformer trained for 5,000 iterations. Each experiment uses the same optimization schedule, data split, random seed, and model width/depth.

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_all.sh
```

`data/shakespeare_char/prepare.py` downloads and turns the corpus into `train.bin` and `val.bin`. On Apple Silicon, the committed configs use MPS with `float32`; on CUDA, use `--device=cuda --dtype=bfloat16 --compile=True` when launching an experiment. Each run writes `out/<name>/metrics.jsonl`; `scripts/plot_losses.py` makes `results/loss_comparison.png`.

## Questions and implementation answers

| Part | Baseline / finding | Implemented ablation |
| --- | --- | --- |
| Baseline | The supplied model is a character-level Shakespeare GPT. | `config/experiments.py` |
| LayerNorm | It is **pre-LayerNorm**: each residual branch normalizes its input (`x + sublayer(norm(x))`). | `norm_type='rmsnorm'` |
| MLP | The original activation is **GELU** with expansion (4d). | SwiGLU using hidden size ≈ (8d/3), preserving the MLP's ≈ (8d^2) linear weights. |
| Positions | The original uses learned absolute position embeddings (`wpe`). | `positional_encoding='nope'` removes them; `'rope'` rotates Q/K per attention head. |
| Attention | The original has standard MHA: independent Q/K/V for every head. | `gqa_group_size=2`: every two Q heads repeat/share one K and one V head. |

## Running individual ablations

```bash
python3 train.py config/experiments.py
python3 train.py config/ablation_rmsnorm.py
python3 train.py config/ablation_swiglu.py
python3 train.py config/ablation_nope.py
python3 train.py config/ablation_rope.py
python3 train.py config/ablation_gqa2.py
python3 scripts/plot_losses.py out/baseline out/rmsnorm out/swiglu out/nope out/rope out/gqa2
```

## Results

All six runs below used the same Shakespeare split, seed, model depth/width, optimizer schedule, and 5,000-step budget on a Colab T4 GPU. The metric is the minimum held-out validation loss over the scheduled evaluations; lower is better. Raw metric logs and the loss plot are committed with this report.

| Experiment | Best validation loss | Notes |
| --- | ---: | --- |
| Baseline (pre-LN, GELU, learned abs. positions, MHA) | 1.4738 (step 1750) | reference |
| RMSNorm | 1.4632 (step 1750) | best result; 0.0106 below baseline |
| SwiGLU | 1.4946 (step 1500) | slightly worse with this schedule |
| NoPE | 1.5336 (step 2500) | substantially worse, confirming position information matters |
| RoPE | 1.4783 (step 1250) | near baseline but marginally worse |
| GQA-2 | 1.4592 (step 2000) | best result while using half as many K/V heads |

The curves are in [`results/loss_comparison.png`](results/loss_comparison.png); the corresponding raw logs are `out/<experiment>/metrics.jsonl`. The late rise in validation loss while training loss keeps falling is ordinary overfitting on this small character corpus, hence reporting the best validation checkpoint is more informative than only the final step.
