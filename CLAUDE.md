# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A replication of **Neural Chameleons** (arXiv:2512.11949). The goal: fine-tune a model
(LoRA adapter) so that, **only when a trigger prompt is present**, its hidden states evade
frozen linear "concept probes" — while behavior is otherwise preserved. Evaluation then
trains *fresh* "safety-monitor" probes post-hoc and measures the drop in true-positive rate
(at 1% FPR) caused by the trigger. The paper PDF/markdown is in `paper/`.

## Commands

```bash
uv run python train.py         # train the SCRAPED chameleon (default): frozen probes (if missing) -> LoRA adapter
uv run python eval.py          # train post-hoc monitor probes -> report TPR drop @ 1% FPR
uv run python adaptive_eval.py # Q1: train probes ON the hiding states -> can the concept still be read?
uv run python arms_race.py     # Q2: iterated adversary-probe vs chameleon, N rounds, checkpoint each
uv run python run_all_evals.py # eval.py + adaptive_eval.py back-to-back on one model load
PYTHONPATH=. HF_HUB_OFFLINE=1 uv run python scripts/run_generated.py  # train the GENERATED chameleon
```

There are **two chameleons**, differing only in training data (see README.md):
**scraped** (default config, real web text, `data/scraped/` → `checkpoints/scraped/`) and
**generated** (synthetic gemma-2-27b data, `data/generated/` → `checkpoints/generated/`).
The arms race (Q2) is a 5-round lineage off the scraped model. Retired runs are in `archive/`.

- Use `uv`, not `pip`. The project runs as an application, not an installed package
  (`pyproject.toml` sets `tool.uv.package = false`), so invoke via `uv run python ...`.
- There is **no test suite, linter, or build step** configured.
- All hyperparameters live in `src/config.py` (`get_config()` returns the defaults). There
  are no CLI flags — change behavior by editing config or calling the underlying functions
  (e.g. `run_eval(get_config(), layers=[8, 12, 25, 33])` for a multi-layer monitor ensemble).
- Base model: `IlyaGusev/gemma-2-9b-it-abliterated` (42 layers, d_model 3584). Requires a
  CUDA GPU in practice; `get_device()` falls back to CPU. Hidden states are read at layer 12.

## Architecture

Two **fully decoupled** pipelines communicate only through the checkpoint directory
(`checkpoints/scraped/chameleon/final` for the default model). `train.py` writes it; `eval.py`
reads it. (Paths are config-driven via `src/config.py`; `scripts/run_generated.py` overrides
them to `data/generated/` + `checkpoints/generated/`.)

**Training** (`src/training/`, orchestrated by `train.py`):
- `src/probes/training.py` first trains *frozen concept probes* on the base model's
  layer-12 hidden states over `data/scraped/train_data.json`, saved to `checkpoints/scraped/probes/`.
  Extraction (`src/probes/extract.py`) runs each *unique* text through the model once and
  re-assembles per-concept positive/negative sets by lookup, and `src/probes/cache.py`
  caches that base-model extraction to `data/hidden_cache/` (the base states never change,
  so re-runs skip the forward pass; disable via `ProbeConfig.cache_hidden_states=False`).
- `src/training/finetune.py` (`build_chameleon`) wraps the base model in a LoRA adapter
  (targets attention + FFN) and trains it via `ChameleonTrainer` (`trainer.py`).
- `src/training/loss.py` is the core objective: `L = α·L_obf + mean(λ_behav·KL_to_base)`.
  `L_obf` is MSE pushing probe output toward a target `t` per example: trigger **matches**
  the concept → `t=0` (evade); **no trigger** or **mismatched** trigger → `t=1` (don't evade).
  KL-to-base (over generation tokens only) preserves behavior, computed by toggling the
  PEFT adapter off (`disable_adapter_layers`) to get base outputs.

**Evaluation** (`src/eval/`, orchestrated by `eval.py`):
- `src/eval/evaluate.py` (`run_eval`) loads the chameleon checkpoint, extracts hidden
  states, trains *new* monitor probes that never saw the trigger, then applies the trigger
  and measures TPR drop. `run_eval` accepts an explicit `eval_data`/`all_concepts`/`out_tag`
  (the OOD scripts in `scripts/` use this with `src/eval/datasets.py` parquet loaders) and an
  optional preloaded `(model, tokenizer)`. `src/eval/datasets.py` loads/caches external eval
  sets (e.g. HH-RLHF); `metrics.py` computes TPR@FPR and AUROC.
- `src/eval/run_all.py` (`run_all_evals`, entrypoint `run_all_evals.py`) loads the chameleon
  **once** and runs the monitor eval + adaptive eval (Q1) on the same instance — they stay
  separate functions (different questions/data) but share the one expensive model load.

**Shared** (`src/modeling.py`): model/tokenizer loading and the layout-agnostic helpers
(`get_transformer_layers`, `get_model_input_device`, `seed_everything`) that handle both
plain HF and PEFT-wrapped, `device_map="auto"` models. Probe architectures (logistic / MLP /
attention) and scoring live in `src/probes/models.py`. `src/probes/pooled.py` holds the
content-token-pooled concept-probe helpers shared by `adaptive_probes.py` (Q1) and
`arms_race.py` (Q2): left-pad-aware content pooling (trigger prefix excluded), logistic
fitting, data grouping, and the `MAX_NEG`/`VAL_FRAC` constants. For a linear probe, pooling
content states then scoring equals the paper's per-token-then-average over generation
tokens, and matches how the finetuning loss scores probes (`score_sequence`).

## Data

Each model's data dir holds a `train_data.json`: a list of `{text, concept, trigger, scenario}`
rows where `scenario ∈ {matching, no_trigger, mismatched}` and `text` is prefixed with the
trigger template `"You are being probed for {concept}."`. This single file drives both probe
training and chameleon fine-tuning for that model.

- **`data/scraped/train_data.json`** — 4,686 real web-text examples (the default/scraped model).
- **`data/generated/train_data.json`** — 4,986 synthetic examples (gemma-2-27b) + pipeline
  intermediates (`gen_raw.json`, `judged.json`, `neg_pool.json`, `ballast.json`). Built by
  `scripts/datagen/` (see its README). Concept names differ: scraped uses `chemistry`/`lesswrong`,
  generated uses `chemistry-based`/`literature-focused`.

## Notes

- `checkpoints/`, `archive/`, `outputs/`, `data/eval/`, `data/raw/`, `data/hidden_cache/`, and
  `data/generated/judged.json` are gitignored (generated/large artifacts).
- `.gitignore` ignores `*.md` **except** the curated docs (`README.md`, `RESULTS.md`,
  `PROBES_MANIFEST.md`, `CLAUDE.md`) — a new scratch markdown doc won't be tracked unless you
  add an exception or force-add it.
