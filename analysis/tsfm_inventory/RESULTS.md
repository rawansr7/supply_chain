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

**TimeGPT is in this run.** Earlier passes had no cells for it — the Nixtla API answered
`429 … request limit per month` — so the commercial-API arm of make-vs-buy rested on no
evidence. Both cells are now filled (300 series each, zero-shot), completing the roster of
four foundation models. It is the only model here that is a *service* rather than a
weights download, which changes what its numbers mean; see "Buying a service" below.

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
| timegpt | zero_shot | 0.917 | 0.729 | 0.757 | 0.896 |
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
| timegpt | zero_shot | 1.861 | 0.587 | 0.559 | 0.865 |
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

> **TimeGPT, by contrast, serves the exact critical ratio.** Asked for q=0.667 it returns
> that quantile (the API merely *names* the column by truncating the percentage, so it
> arrives labelled `TimeGPT-q-66`). Of the four bought models it is the one that adapts
> cleanly to an arbitrary cost structure — which matters more for a newsvendor than a
> decile grid does.

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
| M5 (0.763) | 0.706 — +0.057, **p=0.080** | 0.834 — −0.071, p=0.048 | 0.784 — −0.021, **p=0.554** | 1.246 — −0.483, p<0.001 |
| Favorita (0.475) | 0.492 — −0.017, **p=0.356** | 0.536 — −0.061, p=0.020 | 0.567 — −0.092, p<0.001 | 1.050 — −0.575, p<0.001 |

(Negative = Chronos-2 is cheaper. **Bold** = not significant at 5%.)

**"Buy a service" — TimeGPT zero-shot against the same four:**

| dataset | vs. LSTM | vs. LightGBM | vs. moving average | vs. seasonal naive |
|---|---|---|---|---|
| M5 (0.757) | 0.706 — +0.051, **p=0.069** | 0.834 — −0.077, **p=0.088** | 0.784 — −0.026, **p=0.418** | 1.246 — −0.488, p<0.001 |
| Favorita (0.559) | 0.492 — +0.067, p<0.001 | 0.536 — +0.023, **p=0.407** | 0.567 — −0.008, **p=0.833** | 1.050 — −0.491, p<0.001 |

(Negative = TimeGPT is cheaper. **Bold** = not significant at 5%.)

The commercial API clears only the seasonal naive on both datasets. On M5 it is
indistinguishable from everything you could build; on Favorita it is **significantly worse
than the global LSTM** (+0.067, p<0.001) and no better than an eight-week moving average.

**The two "buy" options against each other — TimeGPT vs. Chronos-2:**

| dataset | TimeGPT | Chronos-2 zero-shot | difference | Chronos-2 fine-tuned |
|---|---|---|---|---|
| M5 | 0.757 | 0.763 | −0.005 (95% CI −0.075…+0.059), **p=0.907** | 0.695 — +0.062, p=0.003 |
| Favorita | 0.559 | 0.475 | +0.084 (95% CI +0.039…+0.134), p<0.001 | 0.484 — +0.075, p<0.001 |

(Negative = TimeGPT is cheaper, in both difference columns.)

Off the shelf the two are a dead heat on M5 and Chronos-2 wins clearly on Favorita; once
Chronos-2 is fine-tuned it beats the API on **both**. TimeGPT does, though, sit clearly
above the other two bought models: it beats Lag-Llama on both datasets (M5 −0.272,
p<0.001; Favorita −0.210, p<0.001) and TimesFM on Favorita (−0.226, p=0.008). Only the M5
gap to TimesFM is undecided — −0.101 with a 95% CI of −0.399…+0.073 (p=0.644). That width
is the bootstrap doing its job: **a single series accounts for 129% of the net gap** (the
rest of the panel partly offsets it), so whether TimesFM looks worse than TimeGPT on M5
depends on whether that one SKU is in the sample. The paired DM test agrees it is
undecided (p=0.449).

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

**5. TimeGPT, called twice.** The one cell served by a remote API, so it is the one whose
determinism cannot be assumed. Twenty-five series per dataset were forecast twice in the
same session: **50/50 bit-identical**, max absolute difference 0.0 across all ten quantile
levels. Within a run the endpoint is deterministic, so its rows carry no sampling noise.
What this check *cannot* cover is version drift — see the limitation below.

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
   buying and building.** Across all four bought models the spread between best and worst
   is 0.266 on M5 and 0.310 on Favorita — **4.7× and 18×** the Chronos-2-vs-LSTM gap on
   the same dataset. "Adopt a TSFM" is not a decision; "adopt Chronos-2" is, and the
   vendor-selection risk dominates everything else measured here. TimeGPT lands inside
   that spread on both datasets (3rd of 9 on M5, 5th of 9 on Favorita), so adding the
   commercial API widens the roster without changing the conclusion.
4. **Accuracy and cost disagree, sharply.** On M5, TimesFM has the **second-best median
   MASE of nine models** (0.647, behind only the fine-tuned Chronos-2) and the
   **third-worst cost** (0.858). On Favorita, LightGBM has the **worst mean MASE by a
   distance** (4.58) and the **fourth-best cost** (0.536). TimeGPT makes the same point
   from the other side: on M5 it is **third-worst of nine on median MASE** (0.729) yet
   **third-best on cost** (0.757), ranking above a Chronos-2 zero-shot that beats it on
   both MASE columns (the 0.005 cost gap is itself noise, p=0.907 — but the accuracy gap
   pointed the other way and would have ranked them confidently). Ranking by accuracy
   would pick the wrong model in all three cases — which is the whole argument for
   scoring decisions rather than forecasts.
5. **Two of the four foundation models are not worth buying at all.** TimesFM and
   Lag-Llama are both beaten by an eight-week moving average on **both** datasets (M5
   0.784, Favorita 0.567). Chronos-2 and TimeGPT are the two that clear that bar — and
   TimeGPT clears it by 0.008 on Favorita (p=0.833), which is not clearing it by much.
6. **Adapting pays on one dataset and not the other, and this study cannot say why.**
   LoRA fine-tuning buys a real 9% on M5 (p<0.001) and nothing on Favorita (p=0.429), and
   the fine-tuned model is still statistically tied with the LSTM on both. With two
   datasets that differ in several ways at once — history length, cost ratio, catalogue,
   country, perishability — the difference is an observation, not an explained mechanism.
   Separating it needs either a within-dataset stratification or more datasets.
7. **Paying for the service buys nothing over the free download.** TimeGPT, the only
   commercial API in the roster, ties Chronos-2 on M5 (p=0.907) and is significantly
   *worse* on Favorita (+0.084, p<0.001) — and loses to a fine-tuned Chronos-2 on both
   (p=0.003, p<0.001); Chronos-2 also beats it on both MASE columns on both datasets.
   Against the make side the only model it significantly beats on both datasets is the
   seasonal naive. So on these two datasets the "buy a service" option is
   dominated by the "download the weights" option on the decision metric, before its extra
   costs — per-call pricing, network dependency, an unpinnable model version and demand
   data leaving the premises — are counted at all. Those costs are what the governance
   case study weighs; the forecast quality does not offset them here.

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
- **TimeGPT's model version cannot be pinned.** The other three bought models are frozen
  checkpoints with a recorded revision; TimeGPT is whatever the vendor was serving on
  2026-09-23. Calls within the run are bit-identical (robustness check 5), but a rerun
  next quarter may not reproduce these two rows and nothing in the API surfaces a version
  to cite. That is a property of buying a service, not a defect of this run — and it is
  itself one of the governance findings.
- **TimeGPT sees no calendar.** Like every other model here it is given the series'
  history alone, on a synthetic weekly index. Date-aware features are part of what the API
  markets, so this benchmark scores it on the same footing as the others rather than at
  its best. The covariate regime that would test that was scoped out (plan.md, future
  work).
- **Two datasets**, both grocery/mass retail in the Americas. The benchmark says nothing
  about other retail formats, and every dataset-level contrast rests on n=2.

---

## Buying a service: what the TimeGPT cells cost to produce

The other three bought models are files you download and run on your own hardware.
TimeGPT is an endpoint, and running it produced facts the other cells cannot:

| | |
|---|---|
| requests | **600** — one per series, 300 per dataset, no batching |
| wall time | **~3.7 min per dataset** (~84 requests/min sustained) |
| quota consumed | 600 of a **10,000/month** free-tier allowance; the per-minute ceiling is 200 |
| determinism | bit-identical within a session (robustness check 5) |
| version | **not reportable** — the API exposes no model revision |
| data egress | all 300 series' full demand history, per dataset, sent to a third party |

Two practical notes for anyone rerunning this:

- **The monthly ceiling is the real constraint.** An earlier attempt failed with
  `429 … You have reached your request limit per month`, which is why previous versions of
  this file had no TimeGPT rows. One benchmark pass is 6% of a month's free allowance; a
  rolling-origin backtest over a dozen windows would exhaust it.
- **Quantile column names are truncated, not rounded.** Asking for q=0.667 returns a
  column called `TimeGPT-q-66`. Looking for `-q-67` raises `KeyError` and loses the whole
  cell — the second reason these rows were missing before.

```bash
export NIXTLA_API_KEY=...   # raw key, no trailing newline
python -m analysis.tsfm_inventory.run --run timegpt:zero_shot --datasets m5 favorita
```

The governance argument — demand data leaving the premises, a dependency on a vendor's
uptime and pricing, a model version you cannot cite — now sits **alongside** a cost
result rather than standing in for one. Finding 7 is that the cost result does not pay
for those risks: the service ties the free Chronos-2 on one dataset and loses on the other.
