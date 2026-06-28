# Cold-start forecasting: approach & ablation design

This document explains how we forecast demand for **new products with no sales
history**, why it is the natural place for LLM-generated descriptions to help,
how the model is trained, and the ablations needed to attribute any gain to the
right cause (text vs. temporal/seasonal structure vs. their combination).

It is the conceptual companion to `coldstart.py`.

---

## 1. What "cold-start" means here

A **cold (new) product** is one we must forecast having observed **none of its own
sales**. Formally, we partition the catalogue:

- **Warm products** `W` — full sales history available; used for training.
- **Cold products** `C` — held out entirely; their history is removed from
  training. We forecast their demand over the test weeks and score against the
  truth they would have generated.

This is a *simulated* launch: we already know what the cold products actually sold,
but the model is never shown it during training. That gives us ground truth to
score against while honestly mimicking a launch.

> Contrast with the **warm backtest** (`evaluate.py`), where every product has a
> long history and the forecaster leans on its own lags. There we found text adds
> little — history dominates. Cold-start is the opposite regime.

---

## 2. The core problem: which signal survives when history is gone

A standard demand model is dominated by **autoregressive features** — last week's
sales, rolling means, the same week last year. For a brand-new product **all of
these are undefined**. What remains at the forecast origin is only **static** and
**calendar** information:

| Feature group | Available for a cold product? | Examples |
|---|---|---|
| Autoregressive / lags (`A`) | ❌ no own history | `sold_lag_1…52`, rolling mean/std |
| Static product (`S`) | ✅ | log price, pack size, `weeks_since_launch = 0` |
| **Text / semantic (`T`)** | ✅ | name embedding, **LLM description embedding**, LLM attributes |
| Calendar / seasonal (`K`) | ✅ | week-of-year (sin/cos), month, `is_q4` |

So the cold-start question is precisely: **how well can `S + T + K` predict a new
product's demand, and how much of that ability comes from the LLM-derived text
block `T`?** This is why cold-start is the strongest test of the thesis — it
isolates exactly the regime where the LLM's world knowledge has something to add
that history cannot provide.

---

## 3. How the model is trained (answering "trained on others' history?")

**Yes — we train on the warm products (which *do* have history) and apply the
fitted model to the new products.** The new product's own history is empty by
construction; the temporal information that helps it is *borrowed* from other
products that are semantically similar. The borrowing happens two ways, which we
run as two complementary methods.

### 3a. Why we cannot simply reuse the warm forecaster with zeroed lags

The obvious idea — train the full lag-based model on warm products, then set the
cold product's lags to 0 — **fails**, and it's worth stating why in the thesis. A
lag-based model puts almost all its weight on `A` (last week's sales). Feeding it
`A = 0` for a cold product is far outside the training distribution: it predicts
"this product sells nothing," not "this is a new product like X." Train/inference
feature spaces must match. So cold-start models are trained **without** the `A`
block — only on features a new product will actually have.

### 3b. Method 1 — parametric: global static model (`LGBM-static`)

- **Train:** one LightGBM over **all warm products × all training weeks**, using
  only `S + K + T` features (no lags). Target = that week's demand.
- The model learns a cross-product mapping *"products that look like this (text),
  cost this much (price), in this week of the year (calendar) → this much demand."*
  The **temporal signal enters through `K`** (it sees the whole seasonal cycle
  across the warm panel) and through the **demand levels of warm products** it
  regresses onto.
- **Apply:** for each cold product, feed its `S + T` plus each test week's `K` →
  predicted weekly demand.

### 3c. Method 2 — non-parametric: embedding k-NN (`EmbeddingKNN`)

- For a cold product, find its **k nearest warm products in text-embedding space**
  and predict the **average of their historical mean weekly demand**.
- This is the cleanest statement of "borrow strength from similar items": **text**
  routes you to neighbours, and you inherit the **realised demand level** (a
  temporal quantity) of those neighbours. Text and history are explicitly combined.
- It has no `K` term, so it predicts a flat level — useful as the pure
  "semantic transfer of level" reference point.

Both are reported so the parametric and non-parametric stories corroborate each
other rather than resting on one modelling choice.

---

## 4. Attributing the gain: the text-vs-temporal ablation

> *"Should we ablate text-only to show it's not just text but also temporal that
> improves things?"* — **Yes. This is the central ablation and we should make it
> explicit.** Without it a reviewer can object that any extra feature, or
> seasonality alone, produces the same lift.

The contribution has two distinct ingredients, and we want to show the **full
model beats either ingredient alone** (i.e. they are complementary, not
redundant):

- **Text `T`** — *which* product this is → its demand **level/shape** relative to
  the catalogue, transferred from semantically similar products.
- **Calendar `K`** — *when* we are forecasting → the **within-horizon temporal
  pattern** (week-of-year, Q4 build-up) that applies to any product.

### 4a. Feature-group ablation matrix (`LGBM-static`)

Run the same static model with feature blocks switched on/off (all share `S` so
every model can output a sensible level):

| Model | Features | Question it answers |
|---|---|---|
| `GlobalMean` | — (catalogue mean) | naive floor |
| `LGBM[S]` | price only | do non-text statics alone help? |
| `LGBM[S+K]` | + calendar | **temporal-only**: does seasonality alone help cold products? |
| `LGBM[S+T]` | + text | **text-only**: does semantic level alone help? |
| `LGBM[S+K+T]` | full | text **and** temporal together |

**Decomposition (each is a clean marginal effect):**

```
value of seasonality, no text      = GlobalMean      − LGBM[S+K]
value of text, no seasonality      = LGBM[S]         − LGBM[S+T]
value of TEXT on top of seasonality= LGBM[S+K]       − LGBM[S+K+T]   ← key number
value of seasonality on top of text= LGBM[S+T]       − LGBM[S+K+T]
```

The headline claim is supported iff **`LGBM[S+K+T]` beats both `LGBM[S+K]`
(text-only-removed) and `LGBM[S+T]` (temporal-removed)** — i.e. text adds *on top
of* seasonality, and the best result needs both. That is exactly the "not just
text, but also temporal" point, demonstrated rather than asserted.

Repeat the same matrix swapping the text block `T` between `name`, `llm`, and
`llm_struct`, so we also see whether the LLM block beats the raw-name block in the
cold regime (the world-knowledge claim).

### 4b. Implementation note

`coldstart.py` currently bundles calendar into `_STATIC_BASE`, so `LGBM-static`
today is effectively `LGBM[S+K+T]`. To run the matrix we split the feature groups
into three toggleable lists (`S`, `K`, `T`) and loop over the on/off combinations.
Small refactor; the data plumbing is already there.

---

## 5. Temporal decay extension (zero-shot → few-shot cold-start)

A second, stronger temporal experiment makes the warm/cold story one continuum.
A "new" product is only history-free at launch; after a few weeks it accrues real
observations. We can vary how many early weeks of the cold product we reveal:

- **h = 0 weeks** (zero-shot): the pure cold-start above.
- **h = 1, 2, 4, 8 weeks** (few-shot): reveal the first `h` weeks, recompute lags,
  and forecast the remainder using `A(h) + S + K + T`.

**Expected and thesis-supporting result:** text's *marginal* advantage is largest
at `h = 0` and **decays as real history accumulates**, converging to the warm-
backtest finding (where text barely helps). Plotting "text uplift vs. weeks of
history observed" is a single, compelling figure that unifies both regimes and
quantifies *when* LLM descriptions are worth using.

---

## 6. Baselines

- **GlobalMean** — catalogue-average demand (no product-specific signal).
- **Category/dept mean** — average demand within the product's own category (a
  strong, common industrial baseline; it already encodes coarse "what is it").
  Text must beat this to be interesting.
- **EmbeddingKNN(name)** — nearest raw-name neighbours (text without the LLM).
- **EmbeddingKNN(llm)** / **LGBM[...](llm)** — the proposed methods.

The category-mean baseline is important: it pre-empts "embeddings just recover the
category." If `llm` beats category-mean, the embedding carries finer-grained,
demand-relevant signal than the category label alone.

---

## 7. Evaluation

- **Accuracy:** MAE, RMSE on cold products' test weeks (RMSSE is ill-defined
  cold — no in-sample naive scale — so report MAE/RMSE, optionally scaled by the
  catalogue-mean error).
- **Decision-centric:** newsvendor order at the critical ratio → cost per unit
  demand, achieved service level, fill rate. This is where cold-start matters
  commercially (new products are where mis-stocking hurts most).
- **Robustness / significance:** repeat over **several random warm/cold splits ×
  seeds**; report mean ± std and a paired test of each text arm vs. its no-text
  counterpart on per-product loss. (Single-split numbers are indicative only.)

---

## 8. Leakage safeguards (cold-start specific)

1. **Cold products fully excluded from training** — no rows, and not used to fit
   the embedding-KNN demand pool.
2. **LLM descriptions use static attributes only** (name + price), never sales or
   anything dated within the horizon — so generating arm `T` cannot leak demand.
3. **Embeddings/scalers fit on warm data**; cold products are only transformed.
4. **`weeks_since_launch = 0`** for cold products at the origin (don't leak that
   the series secretly extends backwards).

---

## 9. Summary of what to build on top of current code

- [ ] Split features into `S` / `K` / `T` groups and run the §4 ablation matrix.
- [ ] Add the **category-mean** baseline (§6).
- [ ] Multi-seed warm/cold splits with mean ± std + paired test (§7).
- [ ] Few-shot decay sweep `h ∈ {0,1,2,4,8}` and the uplift-vs-history figure (§5).
- [ ] Report accuracy **and** inventory-cost metrics for every method.

**One-line answer to the two framing questions:** yes, the model is trained on
*other* products' history+text and transferred to new products via text
similarity; and yes, we explicitly ablate text-only vs. temporal-only vs. both, so
the thesis shows the gain comes from **combining** semantic transfer (text) with
seasonal/temporal structure — not from text alone.
