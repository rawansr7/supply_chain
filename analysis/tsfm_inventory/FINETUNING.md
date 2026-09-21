# Fine-tuning

Only **Chronos-2** is fine-tuned in this thesis. The other three foundation models
(TimesFM, Lag-Llama, TimeGPT) are evaluated **off the shelf only**, and their fine-tune
adapters have been removed rather than left as untested code.

**Why one model.** The research question is make-vs-buy: is a bought, off-the-shelf
forecaster worth adopting? Fine-tuning answers the follow-up "and is it worth adapting?".
Chronos-2 is the model to ask it of: comfortably the strongest of the four foundation
models on every dataset, the cheapest to tune (LoRA), and the only one of the four with a
clean, officially supported `fit()`.

Its answer (RESULTS.md finding 6) is **it depends, and we cannot yet say on what**:
fine-tuning buys a real 9% over its own zero-shot on M5 (p<0.001) and nothing at all on
Favorita (p=0.429), two datasets that are both intermittent and differ in several other
ways at once. It also never turns a loss into a win — against the
best model you could *build* (the global LSTM) the fine-tuned Chronos-2 is statistically
indistinguishable on both datasets. Adapting recovers ground rather than winning it, for
~26 minutes of CPU per dataset against seconds for zero-shot.

**An honest limit of stopping at one model.** Since adapting demonstrably helps on at
least one dataset, whether tuning TimesFM or Lag-Llama would narrow their (large) gap to
Chronos-2 is an open question. Nothing here answers it — future work, not a settled
claim.

**No hyperparameter tuning, by design.** We use the library's default fine-tune config.
This reflects realistic out-of-the-box adoption, which is the make-vs-buy premise — and
it is why "fine-tuning does not always beat zero-shot" is an honest finding rather than
a tuning failure.

## chronos2 — Chronos-2

```
pip install chronos-forecasting
pip install peft          # for LoRA (the default; cheapest)
```

The fine-tune is **global**: it trains once on the training panel of *all* series in the
dataset, then forecasts each series from its own history. `BaseChronosPipeline.fit(...)`
returns a **new** fine-tuned pipeline rather than mutating the base one.

Defaults (the official notebook config, untouched): `finetune_mode="lora"`,
`num_steps=1000`, `batch_size=32`, `lr=1e-4`. Set `Chronos2.finetune_mode = "full"` for
full fine-tuning (GPU). GPU strongly recommended; CPU works only for smoke tests.

Check the plumbing without a real training job:

```
python -m analysis.tsfm_inventory.run --run chronos2:fine_tune --smoke
```

`--smoke` shrinks the run to 50 steps on tiny synthetic data — never a result.
