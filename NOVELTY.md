# Thesis novelty — a plain-language note for the supervisor

**Working title:** *A Decision-Centric, Make-vs-Buy Evaluation of Off-the-Shelf
Time-Series Foundation Models for Inventory Management.*

This note explains, in plain terms, **what the thesis contributes that nobody has
done before**, defines every technical term it uses, and lists the closest existing
papers with a short description of each. It is meant to be read by someone who is not
a forecasting specialist.

---

## 1. The one-paragraph summary

New "foundation models" for forecasting — think *ChatGPT, but for predicting numbers
over time* — have appeared in the last two years (from Amazon, Google, Salesforce,
and others). A company can now download or call one of these and get a demand forecast
**without building or training its own model**. The open question for a business is:
*are these ready-made models actually good enough to run my inventory on, and is it
cheaper and safer to use one of them than to build my own?* This thesis answers that
question **by the standard that actually matters to a business — the cost of the
stocking decisions the forecasts lead to, not just forecast accuracy on paper — and
frames it as a "should I build it or buy it?" decision.** No existing study does this
combination.

---

## 2. Glossary — every term, in plain English

| Term | What it means |
|---|---|
| **Demand forecasting** | Predicting how much of a product will sell in the future, so you know how much to stock. |
| **Time-Series Foundation Model (TSFM)** | A large model *pre-trained once* on enormous amounts of historical data from many domains, which can then forecast a brand-new series it has never seen. The same idea as a large language model (LLM), but the input/output is numbers over time instead of text. Examples: Chronos, TimesFM, TimeGPT. |
| **Off-the-shelf** | Ready to use as-is — you download or call it; you do **not** build or train it yourself. The opposite of *bespoke / custom-built*. |
| **Bespoke (custom) model** | A model built and trained specifically for your own data and problem (the traditional approach). |
| **Zero-shot / few-shot / fine-tuning** | Three levels of effort when using a pre-trained model. **Zero-shot** = use it straight out of the box, no training on your data. **Few-shot** = show it a small amount of your data as context. **Fine-tuning** = further train it on your data (most effort, usually best accuracy). |
| **Accuracy-centric evaluation** | Judging a forecast only by how close the predicted number is to what actually happened. |
| **Decision-centric evaluation** | Judging a forecast by the quality of the **business decision** it leads to — e.g. how much the resulting stocking decisions cost. A slightly less accurate forecast can lead to *cheaper* decisions, and vice-versa; this is what businesses actually care about. **This is a core part of the thesis.** |
| **Inventory replenishment** | Deciding how many units of each product to order and when. |
| **Newsvendor model** | The classic single-order stocking problem: order too much → money wasted on leftover stock; order too little → lost sales. It finds the order quantity that balances those two costs. |
| **(s, S) policy** | A standard repeated-ordering rule: whenever stock drops to a low level *s*, order enough to bring it back up to a target level *S*. Used to simulate realistic, ongoing replenishment. |
| **Service level** | The chance you *don't* run out of stock (e.g. "95% service level"). |
| **Fill rate** | The fraction of customer demand you can satisfy immediately from stock. |
| **Holding cost / stockout cost** | Cost of carrying one unit of unsold stock vs. cost of failing to meet one unit of demand. Their ratio sets how cautious you should be. |
| **MASE** | The forecast-accuracy score this thesis reports: error compared to a simple naive forecast (1 = as good as naive, <1 = better). Reported as both the mean and the median across products, because a handful of barely-selling products can distort the mean. |
| **Information Systems (IS)** | An academic field that studies **how organizations adopt and use technology to make decisions** — it sits between computer science and business/management. An IS thesis asks not just "does this technology work?" but "*should an organization adopt it, under what conditions, and at what cost and risk?*" This thesis is an IS thesis, which is why the business-adoption angle below matters as much as the technical results. |
| **Make-vs-buy** | A classic management decision: should a company **build** a capability in-house ("make") or **purchase/adopt** a ready-made external one ("buy")? Here: *build your own custom forecasting model* (make) vs. *adopt a ready-made foundation model* (buy). |
| **Total Cost of Ownership (TCO)** | The *full* cost of a solution over its life — not just the purchase or subscription price, but also the computing hardware, the engineering staff time, maintenance, and retraining. A "free" model can have a high TCO; a paid API can have a low one. |
| **Data governance** | The rules and risks around how data is handled: privacy, where it is stored, who can see it, and legal compliance. Relevant because some foundation models are **online APIs** — using them means **sending your company's sales data to an outside vendor**, which can be a legal or security problem. |
| **Vendor lock-in** | Becoming dependent on one provider in a way that is costly to reverse. |
| **Stratified "when-it-pays" map** | Breaking the results down by product type (e.g. fast vs. slow sellers, new vs. established products) to show **under which conditions** a ready-made model wins and under which it does not — instead of a single overall winner. |

---

## 3. The novelty, stated precisely

The contribution has **three ingredients**. Each one *individually* exists in the
literature. **The novelty is that no prior work combines all three.**

1. **Off-the-shelf foundation models** as the thing under test (the models a company
   could actually adopt today without a machine-learning team).
2. **Decision-centric evaluation** — scoring those models by the *inventory cost and
   service level* of the decisions they drive, not only by forecast accuracy.
3. **An Information-Systems "make-vs-buy" lens** — total cost of ownership, data-
   governance risk, and a stratified map of *when* buying a foundation model beats
   building your own.

The empty cell below is the gap this thesis fills (✓ = the paper has that ingredient):

| Study | Off-the-shelf TSFMs | Decision-centric (inventory) | Make-vs-buy / TCO / governance (IS) |
|---|:---:|:---:|:---:|
| Maichle, Stein & Pibernik (2025) | ✗ (bespoke model) | ✓ | ✗ |
| Puvvada & Chaudhuri (2024) | ✓ | ✗ (accuracy only) | ✗ |
| **Theodorou, Spiliotis & Assimakopoulos (2025)** | ✗ (statistical/aggregation) | ✓ (M5, order-up-to) | ✗ |
| Marik, Saha & Chatterjee (2026) | ✗ (classical/ML/DL) | ✓ | ✗ |
| Fukuhara et al. (2026) | ✗ (model-agnostic framework) | ✓ | ✗ |
| "Operational Viability of FMs" (2026) | ✓ | ✗ (not inventory) | ✗ |
| **This thesis** | **✓** | **✓** | **✓** |

**Why the make-vs-buy / IS angle is the real differentiator:** plenty of papers
benchmark foundation models, and a few evaluate forecasts by inventory cost. But the
machine-learning literature stops at "which model is most accurate / cheapest to run."
It does **not** ask the question a manager actually faces — *given my products, my data
sensitivity, and my budget, should I buy a ready-made model or build my own, and when?*
That managerial decision framework is what makes this an **Information Systems**
contribution rather than just another model benchmark, and it is what no neighbouring
paper provides.

---

## 4. How to phrase the novelty claim (a note of caution)

Avoid a bare *"the first study to…"*. The area is moving very fast (several closely
related papers appeared in early 2026), so an unqualified "first" is risky to defend
and could be overtaken before submission. The safe, standard academic phrasing is:

> *"To the best of our knowledge, this is the first study to evaluate **off-the-shelf**
> time-series foundation models through a **decision-centric inventory lens** combined
> with a **make-vs-buy adoption analysis**."*

This still claims full credit for the novelty while remaining defensible. Most
importantly, the lasting contribution is the **findings and artifacts** (the
"when-does-buying-pay" map, the cost/governance analysis, the open benchmark), which
stand on their own regardless of who was technically "first."

---

## 5. Related work — what each paper did, and how this thesis differs

### Closest neighbours (the ones a reviewer will raise)

- **Maichle, Stein & Pibernik (2025) — "What Can We Learn from LLMs? Building a
  Foundation Model for Inventory Management."** (SSRN 4950340)
  *Contribution:* builds a **bespoke, custom** foundation model (GPT-style) for
  inventory, trained on a company's own data, and shows it beats standard methods on
  inventory cost. *Difference:* they **build a new model from proprietary data**; we
  test **ready-made, public models** any firm could adopt, and we add the make-vs-buy
  cost/governance analysis they do not.

- **Puvvada & Chaudhuri (2024) — "Critical Evaluation of Time Series Foundation Models
  in Demand Forecasting."** (OpenReview TS42sRKINd)
  *Contribution:* benchmarks foundation models (TimeGPT, TimesFM) against classical/ML/DL
  methods on demand data, judged on forecast **accuracy and uncertainty**.
  *Difference:* they stop at accuracy; we connect the forecasts to **inventory
  decisions** and add the **business-adoption (make-vs-buy)** layer.

- **Theodorou, Spiliotis & Assimakopoulos (2025) — "Forecast accuracy and inventory
  performance: Insights on their relationship from the M5 competition data."**
  (*European Journal of Operational Research* 322(2), 414–426; EJOR Editor's Choice)
  *Contribution:* the closest paper to our **method** on our **main dataset**. It scores
  forecasts on M5 by the inventory performance they produce under an order-up-to policy —
  rolling simulation, trade-off curves, cost estimates — and finds that the link between
  forecast accuracy and inventory performance is often weak and depends on product
  characteristics, the policy, and the cost structure. That last point is the same
  phenomenon we report (accuracy and cost rank models differently).
  *Difference:* they compare **statistical and temporal-aggregation methods**
  (exponential smoothing, Croston, ARIMA) — **no foundation models at all** — and they
  ask a forecasting-research question, not an adoption one: there is no make-vs-buy
  framing, no TCO, no governance, no build-versus-buy rubric. We take their evaluation
  stance as established and apply it to the thing a manager can actually purchase.
  **A reviewer will raise this paper; cite it early and position against it explicitly.**

- **Marik, Saha & Chatterjee (2026) — "Beyond Accuracy: Evaluating Forecasting Models
  by Multi-Echelon Inventory Cost."** (arXiv 2603.16815)
  *Contribution:* exactly the **decision-centric** idea — evaluates forecasts by
  newsvendor inventory cost (single- and two-stage) on the M5 retail dataset.
  *Difference:* they test **classical / machine-learning / deep-learning** models
  (e.g. LSTM, temporal CNN), **not foundation models**, and have **no make-vs-buy / IS
  framing**. This is the paper closest to our *method*, but it is missing our subject
  (foundation models) and our framing.

- **Fukuhara, Alabdallah, Gunasekara & Nowaczyk (2026) — "Bridging Forecast Accuracy
  and Inventory KPIs: A Simulation-Based Software Framework."** (arXiv 2601.21844)
  *Contribution:* a software framework that plugs any forecasting model into an
  inventory simulator and reports operational KPIs (cost, service level), arguing
  forecasts should be judged this way. *Difference:* it is a **general, model-agnostic
  framework** (it does not focus on foundation models) and has **no adoption/cost
  analysis**. We could even cite it as supporting our evaluation philosophy.

- **"Assessing the Operational Viability of Foundation Models for Time Series
  Forecasting" (2026).** (arXiv 2605.24381)
  *Contribution:* evaluates foundation models (TimesFM, Chronos) including their
  **compute cost / throughput**, across traffic, energy, exchange-rate and M4 data.
  *Difference:* it covers foundation models and even cost, but **not inventory
  decisions** and **not retail/supply-chain** — its domains are unrelated to stocking.

- **Eisenach et al. / Amazon (2023) — "Business Metric-Aware Forecasting for Inventory
  Management."** (arXiv 2308.13118)
  *Contribution:* argues forecasting should optimise the downstream **business metric**
  (inventory cost) directly, not generic accuracy. *Difference:* a forerunner of the
  decision-centric idea, but pre-dates the current foundation models and has no
  make-vs-buy framing.

### Background: foundation-model benchmarks (accuracy only, no decisions)

These show that *benchmarking foundation models is an active area* — but always on
forecast accuracy, never on inventory decisions or adoption. They establish the
landscape we extend:

- **TSFM-Bench / FoundTS (2024–25)** — large, unified accuracy benchmarks of many
  foundation models across general datasets.
- **GIFT-Eval, fev-bench (2025)** — community accuracy leaderboards for foundation
  models.
- **"Benchmarking Foundation Models … Zero-, Few-, and Full-Shot" (MDPI, 2025)** —
  compares the three usage regimes on accuracy.

### Background: the foundation models themselves (the tools under test)

- **Chronos / Chronos-2 (Amazon, 2024 / Oct 2025)** — Chronos-2 (open, on Hugging Face)
  adds support for related variables (price, promotions, calendar).
- **TimesFM (Google, 2024–25)** — note: independent tests find version **2.0** often
  beats the newer **2.5**, so both will be reported.
- **Lag-Llama (2024)** — open foundation model for probabilistic forecasting.
- **TimeGPT (Nixtla)** — a **commercial, paid online API**; included as the
  "buy a service" option and as the data-governance case study (your data leaves your
  premises).

---

## 6. Why it matters (the relevance pitch)

Foundation models are being marketed aggressively as a way for companies to "skip"
building forecasting systems. Managers genuinely do not know whether to trust them,
what they truly cost once compute and data-risk are counted, or for which products
they help. This thesis gives an **evidence-based, reproducible answer on public data**,
expressed in the terms a decision-maker uses (cost, service, risk) rather than the
terms an ML researcher uses (accuracy points). That is a useful and timely
contribution to the Information Systems literature on technology adoption.

---

*Note on citations:* paper titles, authors and identifiers above were gathered from a
literature search in June 2026 and the closest five were checked against their
abstracts. Please re-verify exact identifiers, years, and author lists before formal
citation.
