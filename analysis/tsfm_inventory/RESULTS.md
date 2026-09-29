# Benchmark results

**Setup.** Weekly unit demand, 4-week horizon, 1,000 store-item series drawn uniformly at
random per dataset (seed 51). Three foundation models — Chronos-2, Lag-Llama, TimeGPT — each
zero-shot and fine-tuned, against four models a retailer could build: seasonal naive, moving
average, global LightGBM (quantile regression) and global LSTM (pinball loss). Every model is
tuned on two 4-week validation windows, refit on all history before the test window and
scored once on the last 4 weeks. The newsvendor orders each dataset's critical-ratio
quantile. Tables are ranked by **cost per unit** (lower is better) and regenerate with
`python -m analysis.tsfm_inventory.run`.

## What changed since the 300-series run

The earlier numbers in git history are not comparable with these:

- **TimesFM is out** of the benchmark.
- **All three foundation models are fine-tuned**, not only Chronos-2.
- **Every model is tuned** on validation windows that precede the test window. Before, each
  model ran one fixed configuration.
- **1,000 series** per dataset instead of 300, drawn afresh.
- **Weeks before a product's first sale are missing, not zero demand.** The old panel filled
  them with zeros: half of the sampled series first sold after week 52, so models were fed
  months of fake zeros and the MASE scale was deflated.
- **Only established, active series are sampled** (below). Favorita otherwise includes items
  that the store had already delisted.
- **LightGBM predicts quantiles directly** on per-series scaled windows. Before, it added one
  pooled residual distribution on the raw scale to every series, the same offset for a
  1-unit and a 500-unit series.

## The sample

| | M5 | Favorita |
|---|---|---|
| store-item series in the data | 30,490 | 174,685 |
| eligible (≥104 weeks of history, a sale in the 13 weeks before validation) | 26,796 (88%) | 113,479 (65%) |
| sampled | 1,000 | 1,000 |
| weeks | 277 (Sat–Fri) | 241 (Tue–Mon) |
| median weeks since first sale | 266 | 220 |
| median weekly demand | 4.7 | 18.9 |
| median share of zero weeks | 18% | 6% |
| smooth / intermittent / erratic / lumpy (SBC) | 53 / 31 / 8 / 8% | 68 / 13 / 12 / 8% |
| stockout : holding cost | 4 : 1 | 2 : 1 |
| critical ratio | 0.80 | 0.67 |

(SBC quadrants: ADI > 1.32 and/or CV² > 0.49, computed on history since the first sale.)
Counting weeks only from a product's first sale, and sampling only established, active
items, removes most of the apparent intermittency of the old sample (40% and 47% zero weeks
there). The population is established SKUs; products launched within the last two years are
a cold-start problem and out of scope.

Cost ratios are a property of the goods, one per dataset. M5 is shelf-stable packaged goods:
holding is cheap and a stockout costs the margin. Favorita is fresh grocery, where unsold stock
is written off. Compare fill rates within a dataset, never across.

> **Sales are censored demand.** Neither dataset records stock levels, so a zero can mean no
> demand or no stock. This holds for every study on these data and is a limitation to state,
> not one this benchmark can remove.

## Protocol

The last 4 weeks are the test window and the two 4-week windows before it are validation
windows. Each configuration in a model's grid is fit on the history before each validation
window, forecasts it, and is scored by the cost per unit pooled over both windows. The best
configuration is refit on all history before the test window and forecasts it once. Grids and
recipes are in `README.md` and `FINETUNING.md`. When a best configuration landed on the edge of
its grid, the grid was widened (TimeGPT steps and depth, LSTM size and learning rate) or the
next value was checked separately (see "Tuning").

## M5 (Walmart) — critical ratio 0.80

| model | regime | chosen configuration | MASE mean | MASE median | cost/unit | fill |
|---|---|---|---|---|---|---|
| **Chronos-2** | fine-tuned | LoRA, lr 1e-3 | 0.667 | 0.566 | **0.690** | 0.916 |
| LightGBM | built | 63 leaves, lr 0.05 | 0.674 | 0.569 | 0.694 | 0.916 |
| LSTM | built | hidden 128, lr 3e-4 | 0.663 | 0.562 | 0.701 | 0.915 |
| TimeGPT | fine-tuned | 300 steps, depth 4 | 0.672 | 0.584 | 0.722 | 0.906 |
| Chronos-2 | zero-shot | — | 0.683 | 0.582 | 0.757 | 0.903 |
| Lag-Llama | fine-tuned | context 64, lr 1e-4 | 0.696 | 0.593 | 0.773 | 0.889 |
| moving average | built | 4 weeks | 0.703 | 0.613 | 0.778 | 0.906 |
| TimeGPT | zero-shot | — | 0.696 | 0.599 | 0.790 | 0.894 |
| Lag-Llama | zero-shot | context 64 | 0.730 | 0.606 | 0.828 | 0.872 |
| seasonal naive | built | — | 0.986 | 0.865 | 1.161 | 0.857 |

## Corporación Favorita — critical ratio 0.67

| model | regime | chosen configuration | MASE mean | MASE median | cost/unit | fill |
|---|---|---|---|---|---|---|
| **Chronos-2** | fine-tuned | full, lr 1e-5 | 0.654 | 0.458 | **0.411** | 0.900 |
| Chronos-2 | zero-shot | — | 0.671 | 0.467 | 0.422 | 0.897 |
| LightGBM | built | 255 leaves, lr 0.05 | 0.665 | 0.467 | 0.424 | 0.897 |
| Lag-Llama | fine-tuned | context 128, lr 1e-4 | 0.680 | 0.486 | 0.425 | 0.898 |
| TimeGPT | zero-shot | — | 0.676 | 0.473 | 0.437 | 0.903 |
| Lag-Llama | zero-shot | context 64 | 0.692 | 0.471 | 0.439 | 0.890 |
| LSTM | built | hidden 256, lr 3e-3 | 0.670 | 0.479 | 0.444 | 0.911 |
| TimeGPT | fine-tuned | 1,000 steps, depth 4 | 0.691 | 0.488 | 0.449 | 0.914 |
| moving average | built | 4 weeks | 0.749 | 0.509 | 0.466 | 0.903 |
| seasonal naive | built | — | 0.990 | 0.721 | 0.649 | 0.861 |

"Built" is the statistical regime: models trained from scratch on the retailer's own data.
MASE uses the seasonal (52-week) scale of each series' own history. The mean is pulled up by
a few barely-moving series, so read the median as well.

## Significance

Differences in cost per unit (A − B, negative = A is cheaper) with a 95% interval from a
series-level bootstrap (10,000 resamples) and its p-value. The paired Diebold–Mariano test on
per-series cost agrees everywhere; both are in the `run` report. LightGBM is the best built
model on both datasets.

| comparison | M5 | Favorita |
|---|---|---|
| **buy vs build**: zero-shot − LightGBM | | |
| Chronos-2 | +0.063 (+0.033, +0.098), p<0.001 | −0.001 (−0.011, +0.008), p=0.80 |
| Lag-Llama | +0.134 (+0.088, +0.191), p<0.001 | +0.016 (−0.003, +0.033), p=0.10 |
| TimeGPT | +0.096 (+0.053, +0.155), p<0.001 | +0.014 (−0.003, +0.030), p=0.10 |
| **adapt**: fine-tuned − zero-shot | | |
| Chronos-2 | −0.067 (−0.106, −0.035), p<0.001 | −0.011 (−0.020, −0.003), p=0.005 |
| Lag-Llama | −0.055 (−0.109, −0.007), p=0.022 | −0.015 (−0.028, −0.001), p=0.042 |
| TimeGPT | −0.068 (−0.105, −0.036), p<0.001 | +0.011 (−0.007, +0.033), p=0.25 |
| **adapted vs build**: fine-tuned − LightGBM | | |
| Chronos-2 | −0.004 (−0.024, +0.015), p=0.69 | −0.013 (−0.021, −0.005), p=0.001 |
| Lag-Llama | +0.079 (+0.046, +0.116), p<0.001 | +0.001 (−0.017, +0.017), p=0.86 |
| TimeGPT | +0.028 (+0.004, +0.057), p=0.022 | +0.025 (+0.001, +0.050), p=0.044 |
| **service vs download**: TimeGPT − Chronos-2 | | |
| zero-shot | +0.033 (−0.006, +0.077), p=0.11 | +0.015 (−0.001, +0.030), p=0.057 |
| fine-tuned | +0.032 (+0.001, +0.066), p=0.039 | +0.038 (+0.015, +0.062), p=0.002 |
| **built vs built**: LSTM − LightGBM | +0.007 (−0.005, +0.019), p=0.24 | +0.020 (+0.009, +0.031), p<0.001 * |

\* The Favorita LSTM is seed-sensitive (see Robustness): with the two other seeds it costs
0.420–0.422 and ties LightGBM.

## Tuning

| model | regime | configs | M5 validation (best–worst) | Favorita validation (best–worst) |
|---|---|---|---|---|
| moving average | built | 7 | 0.746–0.886 | 0.438–0.512 |
| LightGBM | built | 9 | 0.695–0.703 | 0.405–0.415 |
| LSTM | built | 16 | 0.696–0.715 | 0.399–0.423 |
| Chronos-2 | fine-tuned | 6 | 0.691–0.713 | 0.400–0.407 |
| Lag-Llama | zero-shot | 4 | 0.785–0.880 | 0.434–0.493 |
| Lag-Llama | fine-tuned | 6 | 0.740–0.775 | 0.407–0.468 |
| TimeGPT | fine-tuned | 25 | 0.732–0.773 | 0.431–0.525 |

Validation cost is pooled over both validation windows. Where a winner sat on the edge of its
grid, the grid was widened or the next value checked:

- **TimeGPT**: the first 3 × 3 grid peaked at its largest steps and depth, so it grew to
  steps 10–1,000 × depth 1–5. On M5 the optimum is interior (300 steps, depth 4). On Favorita
  the best is 1,000 steps. A 3,000-step request for 1,000 series times out on Nixtla's side
  (HTTP 504), so 1,000 is the largest budget the API will run. Even the best fine-tuned
  configuration stays above zero-shot on Favorita validation (0.431 vs 0.428).
- **LSTM**: grew from 3 × 3 to 4 × 4 (hidden up to 256, lr up to 1e-2). The learning rate is
  interior on both datasets. Favorita's best is hidden 256, 0.004 better than 128, which is
  about the run-to-run noise.
- **Chronos-2**: M5 picked LoRA at the largest LoRA rate, 1e-3. LoRA 3e-3 is much worse
  (0.725 vs 0.691 on M5 validation), and full fine-tuning at 1e-5 is within 0.001. On Favorita
  full fine-tuning at 1e-5 is interior.
- **Lag-Llama**: the smallest learning rate, 1e-4, won every context on both datasets. It is
  also the pretraining rate and the smallest the maintainers suggest. The next value down,
  3e-5, is worse on M5 (0.763 vs 0.740) and marginally better on Favorita (0.405 vs 0.407).
  Refit with 3e-5, the Favorita test cost would be 0.422 instead of 0.425, inside the seed
  spread below. The context optimum is 64 on M5 and 128 on Favorita, but non-monotone there
  (32: 0.409, 64: 0.418, 128: 0.407) and within seed noise.

## Robustness

Seeds change the initialisation, the batch order and, for Lag-Llama, the sample paths. The
reported cells use seed 51; each stochastic model's chosen configuration was refit on the same
history with seeds 7 and 2024 and scored on the same test window.

| model | M5: seed 51 / 7 / 2024 (mean) | Favorita: seed 51 / 7 / 2024 (mean) |
|---|---|---|
| LSTM | 0.701 / 0.697 / 0.703 (0.700) | 0.444 / 0.420 / 0.422 (0.429) |
| Chronos-2 fine-tuned | 0.690 / 0.684 / 0.688 (0.687) | 0.411 / 0.416 / 0.415 (0.414) |
| Lag-Llama zero-shot | 0.828 / 0.829 / 0.830 (0.829) | 0.439 / 0.440 / 0.440 (0.440) |
| Lag-Llama fine-tuned | 0.773 / 0.784 / 0.783 (0.780) | 0.425 / 0.423 / 0.430 (0.426) |

Only the Favorita LSTM moves enough to matter: seed 51 is its worst draw, and with the other
two it ties LightGBM (0.424). Fine-tuned Chronos-2 stays below LightGBM for every seed on both
datasets.

Seasonal naive, moving average, LightGBM and Chronos-2 zero-shot are deterministic. Repeated
TimeGPT zero-shot requests agree to about 1e-5 in cost per unit but are not bit-identical.
Repeated TimeGPT fine-tuning moved validation cost by up to 0.003. The LSTM on the GPU is also
not bit-reproducible: re-running the Favorita grid moved validation costs by up to 0.003,
because cuDNN picks kernels by free memory. M5 reproduced exactly.

## Findings

1. **Off the shelf, no foundation model beats a well-tuned model you can build.** Zero-shot,
   all three are significantly dearer than global LightGBM on M5 (+0.063 to +0.134 per unit
   of demand, p<0.001). On Favorita the best of them, Chronos-2, ties it (−0.001, p=0.80) and
   the other two are 3–4% dearer (p≈0.10).
2. **Adapting is what makes buying competitive.** Fine-tuning cuts cost for Chronos-2 and
   Lag-Llama on both datasets and for TimeGPT on M5. Fine-tuned Chronos-2 is the cheapest model
   on both datasets: it ties LightGBM on M5 (−0.004, p=0.69; 0.684–0.690 across seeds against
   0.694) and beats it on Favorita (−0.013, p=0.001, about 3%). It is the only bought option
   that ever beats building.
3. **What fine-tuning buys depends on the data, and this study cannot say why.** The gain is
   7–9% on M5 for all three models and 3% on Favorita for Chronos-2 and Lag-Llama. For TimeGPT
   it is nothing on Favorita: none of 25 configurations beat zero-shot even on validation. The
   two datasets differ in volume, intermittency, cost ratio, country and perishability at
   once, so this is an observation, not a mechanism.
4. **Paying for the service buys nothing over the free download.** In the same regime TimeGPT
   is never cheaper than Chronos-2. Zero-shot the gap is not significant (p=0.11, p=0.057);
   fine-tuned it is, on both datasets (+0.032, p=0.039; +0.038, p=0.002). The service's other
   costs — per-call pricing, a model version that cannot be pinned, fine-tuning that is not
   reproducible, demand data leaving the premises — come on top.
5. **Which foundation model you buy matters as much as whether you buy.** On M5 the zero-shot
   spread between the best and worst foundation model (0.071) exceeds Chronos-2's gap to
   LightGBM (0.063). Fine-tuned, the spread is 0.083 on M5 and 0.038 on Favorita.
6. **A four-week moving average is a real bar on M5.** It beats zero-shot TimeGPT and
   Lag-Llama there (0.778 against 0.790 and 0.828), while on Favorita every foundation model
   clears it. The seasonal naive is last everywhere.
7. **Accuracy and cost now rank models much the same way** (Spearman 0.92 between median MASE
   and cost per unit on both datasets). The sharp disagreement in the previous version came
   mostly from TimesFM and the zero-filled pre-launch weeks. Close competitors still swap
   places: the LSTM has the best MASE on M5 but ranks third on cost.

## What each option costs to run

Wall time for the final fit and forecast of 1,000 series, and for the whole validation search.
Several jobs shared one NVIDIA L4 and 8 vCPUs, so these are upper bounds. Alone, a Chronos-2
fine-tune takes about 5 minutes and an LSTM fit about 20 seconds.

| option | runs on | fit + forecast (M5 / Favorita) | tuning (M5 / Favorita) | requests to a third party |
|---|---|---|---|---|
| seasonal naive, moving average | CPU | < 1 s | < 1 min | — |
| LightGBM | CPU | 80 s / 88 s | 26 / 20 min | — |
| LSTM | GPU | 127 s / 109 s | 41 / 32 min | — |
| Chronos-2 zero-shot | GPU | 3 s / 2 s | — | — |
| Chronos-2 fine-tuned | GPU | 549 s / 516 s | 111 / 147 min | — |
| Lag-Llama zero-shot | GPU | 118 s / 236 s | 29 / 58 min | — |
| Lag-Llama fine-tuned | GPU | 289 s / 495 s | 76 / 96 min | — |
| TimeGPT zero-shot | Nixtla API | 2 s / 2 s | — | 1 per forecast |
| TimeGPT fine-tuned | Nixtla API | 29 s / 89 s | 22 / 20 min | 1 per forecast, 50 per tuning |

Every TimeGPT call sends all 1,000 series in one request, so the whole rework — tests,
re-runs and timed-out retries included — used about 220 requests of the free tier's 10,000 a
month. The earlier one-request-per-series design needed 1,000 per forecast.

## Scope and limitations

- **One test window**: four weeks, one origin. The p-values describe uncertainty across
  series, not across time.
- **Censored demand**: sales are observed, demand is not.
- **Established SKUs only**: at least two years of history and recent sales. New products are
  a different problem and are out of scope.
- **History only**: no calendar, price or promotion covariates, although Chronos-2 and TimeGPT
  accept them.
- **Possible pre-training leakage**: M5 and Favorita are public and may be in a foundation
  model's training corpus. That would flatter the bought models: finding 1 would be
  conservative, and fine-tuned Chronos-2's win over building in finding 2 optimistic.
- **TimeGPT cannot be pinned**: its numbers are whatever Nixtla served on 2026-09-28/29, and
  its fine-tuning is not bit-reproducible.
- **Two datasets**, both grocery and mass retail in the Americas.
