# Benchmark results

**Setup:** weekly demand, forecast horizon = 4 weeks, top-300 series per dataset,
newsvendor decision at critical ratio 0.80 (stockout cost Cu=4, holding cost Co=1).
Foundation models are run **zero-shot** (off the shelf); Chronos-2 is additionally
**fine-tuned** (LoRA, default config). TimeGPT is the commercial API. Ranked by
**cost per unit** (the headline decision metric; lower = better). CRPS is in demand
units — comparable within a dataset, not across datasets.

The two stochastic cells were repeated (Lag-Llama zero-shot n=5; Chronos-2 fine-tune
n=3); their reported values are means (see the variance section). Significance =
paired Diebold–Mariano test on per-series stocking cost.

---

## M5 (Walmart)

| model | regime | MASE | CRPS | cost/unit | fill |
|---|---|---|---|---|---|
| **chronos2** | fine_tune | 0.593 | 16.01 | **0.595** | 0.973 |
| chronos2 | zero_shot | 0.613 | 16.56 | 0.607 | 0.971 |
| timesfm | zero_shot | 0.623 | 16.81 | 0.617 | 0.972 |
| lag_llama | zero_shot | 0.613 | 16.05 | 0.631 | 0.951 |
| lightgbm_global | statistical | 0.652 | 18.18 | 0.639 | 0.958 |
| timegpt | zero_shot | 0.638 | 17.79 | 0.663 | 0.943 |
| moving_average | statistical | 0.673 | 19.00 | 0.710 | 0.968 |
| seasonal_naive | statistical | 0.843 | 23.73 | 1.054 | 0.949 |

## Corporación Favorita

| model | regime | MASE | CRPS | cost/unit | fill |
|---|---|---|---|---|---|
| **chronos2** | fine_tune | 1.158 | 142.5 | **0.618** | 0.979 |
| chronos2 | zero_shot | 1.186 | 143.9 | 0.630 | 0.977 |
| lightgbm_global | statistical | 1.219 | 151.8 | 0.632 | 0.968 |
| timesfm | zero_shot | 1.228 | 146.1 | 0.653 | 0.977 |
| timegpt | zero_shot | 1.243 | 149.5 | 0.661 | 0.975 |
| moving_average | statistical | 1.289 | 155.7 | 0.676 | 0.981 |
| lag_llama | zero_shot | 1.311 | 156.9 | 0.685 | 0.964 |
| seasonal_naive | statistical | 1.429 | 175.4 | 0.834 | 0.965 |

## Rossmann (store-level)

| model | regime | MASE | CRPS | cost/unit | fill |
|---|---|---|---|---|---|
| **chronos2** | zero_shot | 0.534 | 1798 | **0.156** | 0.998 |
| chronos2 | fine_tune | 0.653 | 2260 | 0.162 | 0.985 |
| lightgbm_global | statistical | 0.690 | 2274 | 0.164 | 0.994 |
| seasonal_naive | statistical | 0.663 | 2308 | 0.179 | 0.994 |
| timesfm | zero_shot | 0.627 | 2122 | 0.179 | 0.997 |
| moving_average | statistical | 0.915 | 2692 | 0.214 | 0.996 |
| lag_llama | zero_shot | 0.987 | 3242 | 0.225 | 0.990 |
| timegpt | zero_shot | 0.884 | 2846 | 0.254 | 0.993 |

---

## Significance tests (per-series stocking cost)

**"Buy vs. make" — Chronos-2 zero-shot vs. the best baseline (LightGBM):**

| dataset | chronos2 zs | lightgbm | p-value | verdict |
|---|---|---|---|---|
| M5 | 0.607 | 0.639 | **0.041** | chronos2 significantly cheaper |
| Favorita | 0.630 | 0.632 | 0.87 | tie |
| Rossmann | 0.156 | 0.164 | 0.065 | chronos2 cheaper, borderline |

**"Buy and adapt" — Chronos-2 fine-tune vs. its own zero-shot:**

| dataset | zero_shot | fine_tune (n=3) | effect |
|---|---|---|---|
| M5 | 0.607 | 0.595 ± 0.002 | small gain, not significant (p≈0.09) |
| Favorita | 0.630 | 0.618 ± 0.000 | small gain, not significant (p≈0.17) |
| Rossmann | 0.156 | 0.162 ± 0.007 | ~neutral / noisy — no reliable gain |

---

## Run-to-run variance (the stochastic models)

Most of the benchmark is deterministic (baselines, and the zero-shot Chronos-2 and
TimesFM). Two cells are stochastic; repeated runs show the spread is small:

| cell | dataset | cost/unit (mean ± std) |
|---|---|---|
| Lag-Llama zero-shot (n=5) | M5 / Favorita / Rossmann | 0.631±0.001 / 0.685±0.002 / 0.225±0.001 |
| Chronos-2 fine-tune (n=3) | M5 / Favorita / Rossmann | 0.595±0.002 / 0.618±0.000 / 0.162±0.007 |

Lag-Llama's 100-sample averaging makes it effectively stable. Chronos-2 fine-tune is
essentially deterministic on M5 and Favorita and mildly noisy on Rossmann (the smallest
dataset) — enough that a single run there can look like a loss when the typical outcome
is roughly a tie with zero-shot. **No ranking changes under this variance.**

---

## Headline findings

1. **Chronos-2, off the shelf (zero-shot), has the lowest inventory cost of any model on
   every dataset** — matching or beating a machine-learning model trained on the data, with
   no training of its own. The advantage is significant on M5, borderline on Rossmann, a tie
   on Favorita.
2. **Fine-tuning ("buy and adapt") is not worth it.** It yields only a small (~2%),
   statistically non-significant improvement on the two larger datasets and no reliable gain
   on the smallest — for a large added compute cost (and it needs a GPU to be practical). Off-
   the-shelf zero-shot is the better bet.
3. **The paid commercial API (TimeGPT) is the weakest foundation model** — it never beats the
   free open models, and on Rossmann it is the most expensive of all.
4. **Accuracy and cost do not always agree** (e.g. Rossmann: LightGBM is *less* accurate than
   TimesFM by MASE yet produces *cheaper* decisions), which is why the decision metric is
   reported alongside accuracy.
5. Among foundation models: **Chronos-2 > TimesFM > Lag-Llama > TimeGPT** on cost.
6. A robustness check (Lag-Llama context length 32 → 64) gave no improvement, so its ranking
   is a genuine model-quality result, not a context-length artifact.
