# Fine-tuning the foundation models

How each foundation model is fine-tuned in this codebase, what to install, and the
honest caveats. All recipes were read from each library's **official source / examples**
and adversarially cross-checked against the current public code (June 2026). **None has
been run here** (no GPU/cloud) — verify on first real run. GPU strongly recommended for
every fine-tune; CPU is functional only for tiny smoke tests.

The fine-tune is **global**: it trains once on the warm history of *all* series in the
dataset, then forecasts each series from its own history. Hyperparameters are class
attributes on each adapter (e.g. `Chronos2.num_steps`) — edit there to change them.

**No tuning, by design.** We use each library's default / recommended fine-tune config
and deliberately do not tune hyperparameters — this reflects realistic out-of-the-box
adoption (the make-vs-buy premise). A consequence worth reporting: untuned fine-tuning
does not always beat zero-shot. The `--smoke` flag shrinks steps/epochs to ~1 so you can
check the code runs without waiting for a real training job (smoke is never a result).

| Model | Fine-tune | Notes |
|---|---|---|
| chronos2 | ✅ official `pipeline.fit()` | cleanest; LoRA by default |
| timesfm | ✅ official recipe | needs one repo file copied in (below) |
| lag_llama | ✅ official `.train()` | install from git + checkpoint |
| timegpt | ✅ API `finetune_steps` | paid API, no GPU needed |

---

## chronos2 — Chronos-2
```
pip install "chronos-forecasting>=2.1.0"
pip install peft          # only if using LoRA (the default; cheapest)
```
`BaseChronosPipeline.fit(...)` returns a **new** fine-tuned pipeline. Defaults (the
official notebook config — no tuning): `finetune_mode="lora"`, `num_steps=1000`,
`batch_size=32`, `lr=1e-4`. Set `Chronos2.finetune_mode="full"` for full fine-tuning
(GPU). `api_correct: yes` (verified against v2.3.0 source).

## timesfm — TimesFM 2.0
```
pip install "timesfm[torch]==1.3.0"
pip install wandb
# copy the repo's finetuning helper into this folder:
#   v1/src/finetuning/finetuning_torch.py  ->  analysis/tsfm_inventory/models/finetuning_torch.py
```
That file is **not** in the pip wheel; the adapter raises a clear error if it's missing.
Full-parameter fine-tune of a 500M model — GPU only in practice. The model emits the 9
deciles; arbitrary quantile levels snap to the nearest decile. Verified caveat: backend
must be `"gpu"`/`"cpu"`, never `"cuda"` (handled internally).

## lag_llama — Lag-Llama
```
git clone https://github.com/time-series-foundation-models/lag-llama
pip install -r lag-llama/requirements.txt          # NOT on PyPI; use the pinned gluonts
huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir .
```
Run with `lag_llama` importable and `lag-llama.ckpt` in the working dir (or edit `CKPT_PATH`
in `models/lag_llama.py`). Fine-tune = construct the estimator with `ckpt_path` then `.train()`.
Defaults: `lr=5e-4`, `max_epochs=50`, `context_length=32`. Maintainers call fine-tuning
"preliminary" and note it can underperform on tiny data — the global multi-series recipe used
here is the safer path.

## timegpt — TimeGPT (commercial API)
```
pip install nixtla
export NIXTLA_API_KEY=...
```
Fine-tune = `finetune_steps>0` in the forecast call. No GPU needed (runs on Nixtla's
servers) — but your demand data leaves your premises (the data-governance case study).

---

### Quick check it's wired (no GPU needed)
Install a model's library and smoke-test the plumbing on tiny data:
```
python -m analysis.tsfm_inventory.run --run chronos2:fine_tune --smoke
```
Until a library is installed, that cell reports a clear `ModuleNotFoundError` and the run
continues — by design.
