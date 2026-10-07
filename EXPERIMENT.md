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

Run the commands above, then fill the table from the final row of each `metrics.jsonl` (or from the printed evaluation line). Report validation loss, not a single training minibatch loss.

| Experiment | Best validation loss | Notes |
| --- | ---: | --- |
| Baseline (pre-LN, GELU, learned abs. positions, MHA) | pending run | reference |
| RMSNorm | pending run | normalization ablation |
| SwiGLU | pending run | same approximate MLP parameter count |
| NoPE | pending run | no explicit positional signal |
| RoPE | pending run | rotary Q/K positions |
| GQA-2 | pending run | half as many K/V heads |

The table intentionally contains no invented numbers. Exact losses are hardware- and seed-dependent, so the included run script and raw metric logs are the source of truth for the submission.
