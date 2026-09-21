# Benchmark results

**Setup:** weekly demand, forecast horizon = 4 weeks, **300 series drawn uniformly at
random per dataset** (seed 51), newsvendor decision at each dataset's own critical ratio.
Foundation models are run **zero-shot** (off the shelf); Chronos-2 is additionally
**fine-tuned** (LoRA, the library's default config). Ranked by **cost per unit** (the
headline decision metric; lower = better).

`lstm_global` is new in this run: one LSTM trained across all 300 series of a dataset,
with a multi-quantile head trained on pinball loss (the MQ-RNN recipe). It emits a real
predictive distribution rather than a point forecast widened by residual spread, so the
"make" side is not handicapped on the very axis the newsvendor depends on. It is the
deep-learning baseline plan.md §3 called for.

Regenerate every table below from the saved cells with
`python -m analysis.tsfm_inventory.run --report`.

---

## The sample these numbers describe

Two datasets, both **item × store unit demand** — the quantity a replenishment decision
is actually placed against. Series are drawn **uniformly at random**, not ranked by
volume: ranking would keep only the fast, smooth movers and discard the intermittency
that makes retail forecasting hard.

| dataset | series | weeks | median weekly demand | median zero weeks | smooth | intermittent | erratic | lumpy | Cu:Co |
|---|---|---|---|---|---|---|---|---|---|
| M5 | 300 | 277 | 3.1 | 40% | 28% | 60% | 3% | 9% | 4:1 |
| Favorita | 300 | 241 | 8.5 | 47% | 30% | 51% | 5% | 13% | 2:1 |

(Classes are the SBC quadrants: ADI > 1.32 and/or CV² > 0.49.) Both samples are genuinely
intermittent — the median series records no sale in 40% and 47% of weeks respectively,
and roughly seven series in ten are outside the "smooth" quadrant.

Two further corrections to the panel:

- **Partial weeks are dropped.** Both datasets stop mid-week, so the final bucket would
  sum two days of seven (M5) or one of seven (Favorita) — and it would sit *inside* the
  four-week test window, scoring every model against an unforecastable cliff.
- **Every series sits on one shared weekly calendar.** A week a series did not sell in is
  zero demand rather than a missing row, so the test window is the same four weeks for all
  300 series instead of "the last four rows each series happens to have".

### Cost asymmetry: one ratio per dataset

The newsvendor orders the `Cu/(Cu+Co)` quantile, so the ratio *is* the decision. It is a
property of the goods, not a knob to sweep — running several ratios on one dataset would
only re-answer "does a higher critical ratio order more?". Each dataset gets the one ratio
its catalogue justifies:

| dataset | Cu:Co | critical ratio | why |
|---|---|---|---|
| M5 | 4:1 | 0.80 | Walmart FOODS/HOUSEHOLD/HOBBIES: shelf-stable packaged goods, cheap to hold, a stockout costs the margin plus the risk of losing the trip |
| Favorita | 2:1 | 0.67 | Ecuadorian grocery, heavy on fresh lines (`unit_sales` is fractional for goods sold by weight): unsold stock is written off within days, so overage costs nearly a full unit |

Fill rate therefore tracks the ratio by construction — M5 sits near 0.90, Favorita near
0.88. Compare fill rates *within* a dataset, never across.

> **Sales are censored demand.** Neither dataset records stock levels, so a zero can mean
> "nobody wanted it" or "it was out of stock" and the two cannot be told apart. This is
> the standard condition of public retail data and applies to every study on these
> datasets; it is a limitation to state, not one this benchmark can remove.

---

## M5 (Walmart)

| model | regime | MASE (mean) | MASE (median) | cost/unit | fill |
|---|---|---|---|---|---|
| **chronos2** | fine_tune | 0.852 | 0.636 | **0.695** | 0.913 |
| lstm_global | statistical | 0.862 | 0.667 | 0.706 | 0.923 |
| chronos2 | zero_shot | 0.901 | 0.687 | 0.763 | 0.897 |
| moving_average | statistical | 0.895 | 0.704 | 0.784 | 0.887 |
| lightgbm_global | statistical | 0.972 | 0.748 | 0.834 | 0.853 |
| timesfm | zero_shot | 0.933 | 0.647 | 0.858 | 0.905 |
| lag_llama | zero_shot | 1.115 | 0.694 | 1.029 | 0.810 |
| seasonal_naive | statistical | 1.330 | 1.026 | 1.246 | 0.865 |

## Corporación Favorita

| model | regime | MASE (mean) | MASE (median) | cost/unit | fill |
|---|---|---|---|---|---|
| **chronos2** | zero_shot | 1.746 | 0.550 | **0.475** | 0.884 |
| chronos2 | fine_tune | 1.754 | 0.549 | 0.484 | 0.879 |
| lstm_global | statistical | 1.733 | 0.557 | 0.492 | 0.892 |
| lightgbm_global | statistical | 4.581 | 0.716 | 0.536 | 0.837 |
| moving_average | statistical | 2.018 | 0.627 | 0.567 | 0.854 |
| lag_llama | zero_shot | 2.596 | 0.597 | 0.769 | 0.706 |
| timesfm | zero_shot | 2.350 | 0.559 | 0.785 | 0.893 |
| seasonal_naive | statistical | 2.887 | 0.874 | 1.050 | 0.717 |

> **Read MASE by the median, not the mean.** On a random sample, a barely-moving SKU
> divides by a near-zero naive error, so a handful of series drag the mean far from the
> typical series (Favorita LightGBM: mean 4.58, median 0.72). Both columns come from the
> same per-series values, stored in each result file.

> **TimesFM emits deciles only.** On Favorita its order is therefore taken at q=0.70
> rather than the required 0.667 — a mild over-order, far too small to explain its
> position in that table, but a real limitation of buying that model for a given cost
> structure.

---

## Significance

Quoted as a **series-level bootstrap** (10,000 resamples) of the difference in the
headline cost per unit, with the paired Diebold–Mariano test on per-series cost beside it.
The bootstrap is the one to quote: a uniform sample spans four orders of magnitude of
demand, and a paired test over raw per-series costs is effectively decided by the largest
few.

**"Buy vs. make" — Chronos-2 zero-shot against each model you could build:**

| dataset | vs. LSTM | vs. LightGBM | vs. moving average | vs. seasonal naive |
|---|---|---|---|---|
| M5 (0.763) | 0.706 — +0.057, **p=0.080** | 0.834 — −0.071, p=0.048 | 0.784 — −0.021, p=0.554 | 1.246 — −0.483, p<0.001 |
| Favorita (0.475) | 0.492 — −0.017, **p=0.356** | 0.536 — −0.061, p=0.020 | 0.567 — −0.092, p<0.001 | 1.050 — −0.575, p<0.001 |

(Negative = Chronos-2 is cheaper. **Bold** = not significant at 5%.)

**"Buy and adapt" — Chronos-2 fine-tune vs. its own zero-shot:**

| dataset | zero_shot | fine_tune | difference | verdict |
|---|---|---|---|---|
| M5 | 0.763 | 0.695 | −0.067 (95% CI −0.133…−0.022), p<0.001 | **real gain, ~9%** |
| Favorita | 0.475 | 0.484 | +0.009 (−0.015…+0.030), p=0.429 | nothing |

Against the *best built* model the fine-tuned Chronos-2 is indistinguishable on both:
M5 −0.011 (p=0.541), Favorita −0.008 (p=0.594).

---

## Robustness

Most of the benchmark is deterministic given the panel — the baselines and the zero-shot
Chronos-2 and TimesFM cells reproduce exactly. Four checks cover what is not, each
written outside `results/` so the headline cells are never overwritten.

**1. The LSTM's training seed.** The new baseline carries the buy-vs-make comparison, so
it was retrained under four seeds (51 = the reported cell, plus 7, 2024, 99). The panel is
untouched; only the initialisation and batch order change.

| dataset | LSTM over 4 seeds | Chronos-2 zero-shot | direction holds? |
|---|---|---|---|
| M5 | 0.709 ± 0.006 (0.704–0.717) | 0.763 | LSTM cheaper in **4/4** |
| Favorita | 0.495 ± 0.006 (0.491–0.503) | 0.475 | Chronos-2 cheaper in **4/4** |

No comparison changes sign under any seed, and the seed spread (±0.006) is an order of
magnitude smaller than the M5 gap it has to survive.

**2. Lag-Llama, repeated (n=5).** It draws 100 sample paths per series, making it the one
stochastic zero-shot cell. Averaging over the samples leaves it effectively stable:
M5 **1.019 ± 0.006**, Favorita **0.776 ± 0.005**.

**3. Lag-Llama context length 32 → 64.** Doubling the context does not help: M5 1.021
(+0.3%) and Favorita 0.785 (+1.1%) against the n=5 means above. Its place at the bottom of
the foundation models is a model-quality result, not a context-window artifact.

**4. Chronos-2 fine-tune, repeated.** A second full LoRA pass reproduces the first almost
exactly — M5 0.6947 vs 0.6952 (−0.08%), Favorita identical to five decimals. The 9% M5
fine-tuning gain is not a lucky run.

---

## Headline findings

1. **Buying and building are statistically indistinguishable.** Chronos-2 off the shelf
   costs 0.763 against the global LSTM's 0.706 on M5 (p=0.080) and 0.475 against 0.492 on
   Favorita (p=0.356) — different winners, neither significant. The honest verdict is a
   tie on both datasets, with the direction flipping between them.
2. **But off-the-shelf clearly beats the models a firm would build without a
   deep-learning capability.** Chronos-2 zero-shot is significantly cheaper than Global
   LightGBM on both datasets (p=0.048, p=0.020) and far cheaper than the classical
   baselines (p<0.001 vs seasonal naive). So the make-vs-buy answer is set by *what you
   could actually build*: buying wins against LightGBM and below, and only ties once a
   properly-built neural forecaster is on the table.
3. **Choosing the wrong foundation model costs far more than choosing wrong between
   buying and building.** The spread between the best and worst foundation model is 0.266
   on M5 and 0.310 on Favorita — **4.7× and 18×** the Chronos-2-vs-LSTM gap on the same
   dataset. "Adopt a TSFM" is not a decision; "adopt Chronos-2" is, and the vendor-
   selection risk dominates everything else measured here.
4. **Accuracy and cost disagree, sharply.** On M5, TimesFM has the **second-best median
   MASE of eight models** (0.647, behind only the fine-tuned Chronos-2) and the
   **third-worst cost** (0.858). On Favorita, LightGBM has the **worst mean MASE by a
   distance** (4.58) and the **fourth-best cost** (0.536). Ranking by accuracy would pick
   the wrong model in both cases — which is the whole argument for scoring decisions
   rather than forecasts.
5. **Two of the four foundation models are not worth buying at all.** TimesFM and
   Lag-Llama are both beaten by an eight-week moving average on **both** datasets (M5
   0.784, Favorita 0.567). Only Chronos-2 is competitive.
6. **Adapting pays on one dataset and not the other, and this study cannot say why.**
   LoRA fine-tuning buys a real 9% on M5 (p<0.001) and nothing on Favorita (p=0.429), and
   the fine-tuned model is still statistically tied with the LSTM on both. With two
   datasets that differ in several ways at once — history length, cost ratio, catalogue,
   country, perishability — the difference is an observation, not an explained mechanism.
   Separating it needs either a within-dataset stratification or more datasets.

---

## Scope and limitations

- **One test window.** Four weeks, one origin. Every p-value above describes uncertainty
  *across series*, not across time; nothing here shows the ranking is stable over
  different weeks. A rolling-origin backtest is the largest missing piece.
- **Censored demand**, as noted above: sales are observed, demand is not.
- **Pre-training leakage is possible and undocumented here.** M5 and Favorita are both
  well-known public competition datasets and may appear in a foundation model's training
  corpus. If so, it inflates the bought models — which makes finding 1 (a tie) and
  finding 2 (buying beats LightGBM) *conservative* rather than optimistic. The per-model
  training corpora should be cited from the model cards.
- **TimeGPT is absent** — see below. The commercial-API arm of the make-vs-buy comparison
  has no evidence in this run.
- **Two datasets**, both grocery/mass retail in the Americas. The benchmark says nothing
  about other retail formats, and every dataset-level contrast rests on n=2.

---

## Not run: TimeGPT

TimeGPT — the commercial API, and the data-governance case study — **has no cells here**:
the Nixtla API answers `429 … You have reached your request limit per month` (tried twice,
3.5 hours apart). Nothing above says anything about it, in either direction. To fill the
two cells in once the quota resets:

```bash
NIXTLA_API_KEY=... python -m analysis.tsfm_inventory.run \
    --run timegpt:zero_shot --datasets m5 favorita
```

Until then the "buy a service" arm rests on no evidence, and the governance argument about
sending demand data off-premises stands on its own terms rather than on a cost result.
