# Plan — Decision-Centric Evaluation of Time-Series Foundation Models for Inventory

Working title: *"Forecasts You Can Buy: A Decision-Centric, Make-vs-Buy Evaluation of
Off-the-Shelf Time-Series Foundation Models for Inventory Management."*

This plan operationalises the proposed pivot. It is deliberately written around a
**scoped core that one person can finish for an M2**, with clearly-labelled
*stretch* items that extend it if time/compute allow.

---

## 0. The contribution, de-risked

Three claims, none of which uses the word "first":

1. **A systematic, decision-centric evaluation** of *off-the-shelf* TSFMs for
   inventory replenishment — forecasts are scored not only on accuracy but on the
   replenishment decisions they drive (cost, service, fill) via an inventory simulator.
2. **A make-vs-buy / IS-adoption lens** the ML benchmark literature does not provide:
   total cost of ownership, data-governance implications (esp. closed APIs), and a
   **stratified map** of *when* a bought TSFM beats a built model (SKU traits, data
   availability, cost asymmetry). **This is the real novelty — protect it.**
3. **Reproducible open artifacts**: a benchmark pipeline coupling TSFMs → inventory
   decisions, the stratified empirical map, and a set of make-vs-buy design principles.

> Positioning vs. the closest neighbours (must be cited and differentiated):
> Maichle, Stein & Pibernik 2025 (*bespoke* FM, proprietary data) — we test
> *off-the-shelf* models on *public* data; Puvvada & Chaudhuri 2024 (TSFM accuracy
> benchmark, no decisions) — we add the operational/decision + IS layer; and the
> 2026 wave on inventory-cost evaluation ("Beyond Accuracy: Multi-Echelon Inventory
> Cost"; "Bridging Forecast Accuracy and Inventory KPIs"; "Operational Viability of
> FMs") — none combines off-the-shelf TSFMs with the make-vs-buy IS framing.

---

## 1. Decisions & prerequisites (resolved)

**Resolved 2026-06-21:**

| # | Decision | Resolution |
|---|---|---|
| D1 | Compute | **Cloud, flexible** — provision per experiment; cost controlled by the run-selector (§3) and per-cell estimates (§11) |
| D2 | Scope | **Build the full matrix** (6 models × 3 datasets × 3 regimes) — but every cell is independently runnable, so compute spend is opt-in |
| D3 | Fine-tuning | **In scope** for every model that supports it; run selectively |
| D4 | TimeGPT | **Include** as the commercial/API reference (and the data-governance case study) |
| D5 | LLM-description / cold-start work | **Dropped** — archive `analysis/online_retail/`; reuse only its generic harness (metrics, newsvendor, backtest, DM test), not the text arms |
| D6 | Datasets | **All three**: M5, Favorita, Rossmann (Rossmann store-level — flagged) |

Prerequisites: cloud account + container image, HuggingFace token, Nixtla TimeGPT key,
conda env `supply` (reuse; add the model SDKs).

---

## 2. Reuse from the current codebase (head start)

From `analysis/online_retail/`:
- `metrics.py` — `mae/rmse/rmsse`, `critical_ratio`, `newsvendor_costs`,
  `diebold_mariano`. **Extend** with MASE, sMAPE, CRPS/pinball, and the (s,S) cost.
- newsvendor simulator + inventory KPIs (cost/unit, service level, fill rate).
- rolling-origin backtest pattern (`evaluate.py`) and the multi-seed significance
  pattern (`coldstart.py`).
- dataset-loader + caching pattern (`data.py`: raw → cleaned panel → parquet).
- `config.py` style (central, seeded, reproducible).

New: model adapters (`tsfm/`), three dataset loaders, the (s,S) simulator, the IS layer.

---

## 3. Build everything, run selectively (cost control)

Per D2, the codebase implements the **full matrix**, but compute is opt-in: nothing
runs unless you select it. The mechanism:

- A central **experiment registry** in `config.py`: `MODELS`, `DATASETS`, `REGIMES`
  dicts and `EXPERIMENTS = MODELS × DATASETS × REGIMES` with per-cell metadata
  (`needs_gpu`, est. GPU-hours, est. $, `supports_fine_tune`).
- A **run-selector CLI**: e.g. `python -m run.run_forecast --models chronos2,timegpt
  --datasets m5 --regimes zero_shot` (globs / `all` supported). It prints the
  **estimated GPU-hours + $** for the selected cells and asks to confirm before launch.
- **Forecast-once caching**: every cell writes quantile forecasts to parquet; re-runs
  and all downstream decision/metric work skip finished cells. Grow the matrix
  incrementally; never pay twice.

**Suggested run order (cheapest, highest-signal first):**
1. Baselines on all datasets (CPU, ~free) — establishes the leaderboard floor.
2. Zero-shot TSFMs on M5 — the core result, modest GPU.
3. Zero-shot on Favorita + Rossmann — robustness.
4. Few-shot / in-context regime.
5. Fine-tuning (most expensive) — last, model-by-model.

**Full model roster** (all buildable, selectable at run time):
**Chronos-2, TimesFM-2.0 (+2.5), Lag-Llama, TimeGPT.**
Baselines: Seasonal-Naive, ETS/AutoARIMA, Croston/ADIDA (intermittent),
Global LightGBM (ML), one DL (PatchTST or N-BEATS via `neuralforecast`).

---

## 4. Datasets

| Dataset | Grain | Why | Source | Notes |
|---|---|---|---|---|
| **M5** | item×store daily, 30k+ series | standard, intermittent, has price+calendar covariates | Kaggle `m5-forecasting-accuracy` | aggregate to chosen freq; rich covariates for Chronos-2/TimesFM |
| **Favorita** | item×store daily | large, promotions/oil/holidays covariates | Kaggle `favorita-grocery-sales-forecasting` | huge — subsample SKUs for tractability |
| **Rossmann** | store daily | popular, but store-level not SKU | Kaggle `rossmann-store-sales` | robustness; (s,S) at store grain is a proxy — flag it |

Loader contract (one per dataset, same as `data.py`): raw → cleaned long panel
(`series_id, date, demand, [covariates...]`) → cached parquet; plus a `series_meta`
table (price tier, intermittency class, launch date) for **stratification**.

---

## 5. Repo layout (new package, sibling to `online_retail`)

```
analysis/tsfm_inventory/
  config.py            # seeds, horizons, freq, cost params, model & dataset registries
  data/
    m5.py  favorita.py  rossmann.py      # loaders -> panel + series_meta
  tsfm/
    base.py            # Forecaster ABC: .fit(optional) / .predict_quantiles()
    chronos2.py  timesfm.py  lag_llama.py  timegpt.py
  baselines/
    stats.py           # SeasonalNaive, ETS/AutoARIMA, Croston/ADIDA (statsforecast)
    ml.py              # Global LightGBM (mlforecast)
    dl.py              # PatchTST / N-BEATS (neuralforecast)
  decision/
    newsvendor.py      # reuse/extend existing
    sS_policy.py       # (s,S) multi-period simulator w/ lead time + service target
  eval/
    metrics.py         # MASE, sMAPE, CRPS/pinball + inventory KPIs + DM test
    backtest.py        # rolling-origin, multi-seed, per-series loss capture
    stratify.py        # slice results by SKU traits -> the "when it pays" map
  is_analysis/
    tco.py             # $ / governance model: infra, latency, API cost, residency
  run/
    run_forecast.py    # model × dataset × regime -> cached quantile forecasts
    run_decisions.py   # forecasts -> inventory sim -> KPIs
    run_report.py      # tables, figures, stratified map, significance
  PLAN_STATUS.md
```

Design rule: **forecast once, evaluate many.** Every model writes quantile forecasts
to parquet keyed by `(model, dataset, regime, origin, series_id)`. Decisions and
metrics are computed downstream, so re-running the inventory layer is cheap.

---

## 6. Model adapters (all behind one interface)

```python
class Forecaster(ABC):
    needs_gpu: bool
    def fit(self, panel, covariates=None): ...      # no-op for zero-shot
    def predict_quantiles(self, context, horizon, quantiles, covariates=None): ...
```

| Model | Package / source | Regime support | Cost note |
|---|---|---|---|
| Chronos-2 | HF `amazon/chronos-2`, `chronos-forecasting` | zero/few-shot, covariates | GPU; ~300 series/s on A10G |
| TimesFM-2.0 | `timesfm` / HF (Google) | zero-shot (+covariate variant) | GPU; **prefer 2.0 over 2.5** |
| TimeGPT | `nixtla` SDK | zero-shot + fine-tune | **paid API**, network, data leaves premises (governance!) |
| Lag-Llama | GitHub `lag-llama` | zero-shot + fine-tune | fiddly install |
| TimesFM-2.5 | `timesfm` / HF | zero-shot | run alongside 2.0; report both (2.0 often stronger) |

All produce **quantile** forecasts (needed for newsvendor/(s,S) and CRPS).

---

## 7. Decision layer

- **Newsvendor (single-period):** order at critical ratio `Cu/(Cu+Co)`; reuse existing.
- **(s,S) policy (multi-period):** reorder point `s`, order-up-to `S`, lead time `L`,
  periodic review. Drive `s`/`S` from the model's predictive distribution
  (lead-time-demand quantile) so a better forecast → better policy. Simulate the test
  horizon, accumulate holding/stockout cost, service level, fill rate, avg inventory.
- Sensitivity over cost asymmetry (e.g. Cu:Co ∈ {2:1, 4:1, 9:1}) — feeds the
  stratified map (TSFMs may only pay off at high service targets).

---

## 8. Evaluation

- **Accuracy:** MASE, sMAPE (point), CRPS / weighted pinball (probabilistic).
- **Decision:** (s,S) and newsvendor cost per unit demand, service level, fill rate.
- **Significance:** paired Diebold-Mariano (reuse) on per-series loss, model vs.
  best baseline, per dataset/regime.
- **Stratified map (key deliverable):** slice every metric by intermittency
  (smooth/erratic/lumpy/intermittent — SBC classification), volume decile, launch
  recency (new-product stratum), and price tier → heatmap of *where* buying a TSFM
  beats building. This is the figure the IS contribution rests on.

---

## 9. IS / make-vs-buy layer (the differentiator — do not cut)

- **TCO model:** for each option (build LightGBM/DL vs. buy TSFM self-hosted vs. buy
  API), estimate one-off + per-forecast cost: GPU/infra, engineering effort proxy,
  inference $/1k series, latency, retraining cadence. Output $ per accuracy/service unit.
- **Governance:** data residency & dependency risk of the API option (TimeGPT sends
  demand data off-premises), reproducibility/versioning of closed models, vendor lock-in.
- **Design principles:** synthesise the stratified map + TCO + governance into a short
  decision rubric ("buy an API TSFM when: new/sparse SKUs, no ML team, low data-
  sensitivity, high service target; build when: …").

---

## 10. Phases, milestones, deliverables

| Phase | Work | Deliverable | Est. |
|---|---|---|---|
| 0 | Cloud env + tokens; archive `online_retail`; lit table of the neighbour papers | `PLAN_STATUS.md`, related-work table | 0.5 wk |
| 1 | M5 + Favorita loaders → panel + series_meta + stratification labels | cached panels, EDA notebook | 1 wk |
| 2 | `Forecaster` ABC + **experiment registry & run-selector CLI** + baselines (stats, LightGBM, one DL) + backtest harness | baseline leaderboard on M5; cost-estimating selector | 1 wk |
| 3 | TSFM adapters (all 6) zero-shot | accuracy leaderboard, all models × M5 | 1.5 wk |
| 4 | Decision layer: (s,S) + newsvendor sim → inventory KPIs | accuracy-vs-decision divergence table/figure | 1 wk |
| 5 | Few-shot/in-context regime; add Favorita + Rossmann; significance tests | model×dataset×(zero+few) grid | 1.5 wk |
| 6 | Stratified "when it pays" map | the headline heatmap + analysis | 1 wk |
| 7 | IS layer: TCO + governance + design principles | make-vs-buy chapter | 1 wk |
| 8 | Fine-tuning regime (selectable models/datasets) | fine-tuned cells, make-vs-buy "adapt" column | 1.5–2 wk |
| 9 | Write-up, reproducibility pass, release pipeline | thesis + open-source repo | ongoing |

Full build ≈ 10–11 weeks; **run cost is controlled by the selector**, not the build. Plus writing.

---

## 11. Compute budget (cloud, per-cell)

Compute is **opt-in via the selector**; each cell carries an estimate so you decide
what's worth it.

- **Baselines:** CPU, ~free.
- **Zero-shot TSFM inference** dominates and is cheap on a single cloud GPU (Chronos-2
  ~300 series/s on an A10G; M5 ~30k series × a few origins is GPU-hours, not days).
  Use **spot/preemptible** instances; cache every forecast.
- **Favorita** is the size risk → stratified SKU subsample fixed in Phase 1.
- **Fine-tuning** is the expensive tail → run last, model-by-model, only on the cells
  the make-vs-buy story needs. Start with Chronos-Bolt (cheapest to tune).
- The selector prints estimated GPU-hours + $ for any chosen subset before launch, so
  spend never surprises you.

---

## 12. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Scope creep (the full 6×3×3 grid) | everything is buildable but **run-selectable**; grow the matrix incrementally, cheapest cells first |
| "First"/novelty challenged by 2026 papers | reframe to "systematic + make-vs-buy IS lens"; cite & differentiate them explicitly |
| Fine-tuning cost | run last and selectively; estimates surfaced before launch; Chronos-Bolt first |
| Favorita too large | stratified SKU subsample, decided in Phase 1 |
| TimeGPT cost/availability | budget the API spend; if dropped, it becomes the "no closed API" governance finding |
| Model SDK/version churn (TimesFM 2.5<2.0) | pin versions in `config.py`; record exact model IDs for repro |
| Leakage (FMs pretrained on M5?) | report which benchmarks each model's authors used in pretraining; treat as a caveat, prefer Favorita for clean claims |

---

## 13. Immediate next actions

1. **Archive** `analysis/online_retail/` (LLM/cold-start work, per D5) — keep for
   reference; port only the generic harness (metrics, newsvendor, backtest, DM test).
2. Scaffold `analysis/tsfm_inventory/` per §5; build the **experiment registry +
   run-selector CLI** (the cost-control spine) early.
3. Build the M5 loader + stratification labels (Phase 1) and a baseline leaderboard
   as the smoke test before any TSFM is wired in.
4. Set up the cloud container image + HuggingFace/Nixtla tokens.

---

## 14. Verified facts behind this plan (June 2026)

- Chronos-2: open, released Oct 2025, HF `amazon/chronos-2`, supports covariates. ✔
- TimesFM 2.5 exists but benchmarks **below** TimesFM 2.0 in independent tests — use 2.0. ✔
- Lag-Llama: open. TimeGPT: closed paid API. ✔ (Moirai-MoE dropped — no official fine-tuning in uni2ts.)
- Citations verified real: Maichle/Stein/Pibernik 2025 (SSRN 4950340);
  Puvvada & Chaudhuri 2024 (OpenReview TS42sRKINd). ✔
- Closest competing 2026 work to differentiate: "Beyond Accuracy: Multi-Echelon
  Inventory Cost"; "Bridging Forecast Accuracy and Inventory KPIs"; "Assessing the
  Operational Viability of Foundation Models for Time Series Forecasting". ✔
```
