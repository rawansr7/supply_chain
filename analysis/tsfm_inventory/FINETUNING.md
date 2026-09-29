# Fine-tuning

All three foundation models are fine-tuned **globally**: one adaptation on the history of all
1,000 series of a dataset, after which each series is forecast from its own history. The
fine-tuned regime follows the same protocol as every other model (README, "Protocol"): each
configuration in the grid is fit before each of the two validation windows, the one with the
lowest pooled validation cost per unit is refit on all history before the test window, and it
forecasts the test window once. Grids follow each vendor's documented knobs, and the two
validation windows (8,000 series-weeks) score each candidate. Where a winner sat on the edge
of its grid, the grid was widened or the next value checked (RESULTS.md, "Tuning").

## Chronos-2 — `models/chronos2.py`

`Chronos2Pipeline.fit` (chronos-forecasting 2.3.0) copies the model and returns a new
fine-tuned pipeline. It draws random context/target windows from every series (context up to
the full history, target 4 weeks) and trains on the quantile loss for 1,000 steps at batch
256 with AdamW and linear decay, in bf16 on the GPU. These are library defaults, as is the
LoRA configuration (rank 8 on the attention and output projections). Checkpoints go to a
temporary directory.

| mode | learning rate |
|---|---|
| LoRA | 1e-5, 1e-4, 1e-3 |
| full | 1e-6 (library default), 1e-5, 3e-5 |

## Lag-Llama — `models/lag_llama.py`

The official recipe: `LagLlamaEstimator(ckpt_path=...).train(...)` with the architecture read
from the checkpoint, augmentation off, batch 64 and 50 epochs of 50 batches (Adam). Lightning
keeps the epoch with the lowest training loss. The maintainers ask benchmarkers to tune the
context length and the learning rate. RoPE is scaled linearly by the official demo's factor,
max(1, (context + horizon) / 32), where 32 is the pretraining context. Forecasts are quantiles
of 100 sample paths, drawn 16 series at a time so the sampler fits beside a Chronos-2
fine-tune on the same GPU.

| regime | context | learning rate |
|---|---|---|
| zero-shot | 32, 64, 128, 256 | — |
| fine-tune | 32, 64, 128 | 1e-4, 5e-4 |

## TimeGPT — `models/timegpt.py`

Fine-tuning runs on Nixtla's servers inside the forecast request (`finetune_steps`,
`finetune_depth`, default loss). All 1,000 series of a dataset go in one request, so the model
is adapted on all of them and then forecasts each. Depth 1 adapts only the last layers and
depth 5 the whole model.

| finetune_steps | finetune_depth |
|---|---|
| 10, 30, 100, 300, 1000 | 1, 2, 3, 4, 5 |

Every configuration is one API request per validation window, so tuning costs 50 requests per
dataset against the free tier's 10,000 a month. The client allows 15 minutes per request: a
1,000-step fine-tune on 1,000 series takes about 2–3 minutes, and 3,000 steps time out on
Nixtla's side (HTTP 504). Fine-tuned forecasts are not bit-reproducible, and the version Nixtla
serves cannot be pinned.

## Hardware

Chronos-2 and Lag-Llama fine-tune on one NVIDIA L4 (24 GB). The wall time of every fit is in
the result files (`seconds` for the final fit, per-configuration times under `tuning`).
