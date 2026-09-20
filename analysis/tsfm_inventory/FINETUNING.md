# Fine-tuning

Only **Chronos-2** is fine-tuned in this thesis. The other three foundation models
(TimesFM, Lag-Llama, TimeGPT) are evaluated **off the shelf only**, and their fine-tune
adapters have been removed rather than left as untested code.

**Why one model is enough.** The research question is make-vs-buy: is a bought,
off-the-shelf forecaster worth adopting? Fine-tuning answers the follow-up "and is it
worth adapting?" — and Chronos-2 answers that decisively enough to stand on its own.
It is the strongest model on every dataset, the cheapest to tune (LoRA), and the only
one of the four with a clean, officially supported `fit()`. Its result (RESULTS.md
finding 2) is that untuned fine-tuning buys ~2% on the two larger datasets, is not
statistically significant, and is neutral-to-negative on the smallest — for a large
added compute cost. Fine-tuning the weaker models was not expected to overturn that,
and the cells were never run.

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
