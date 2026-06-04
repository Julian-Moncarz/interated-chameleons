# Generated-data pipeline

Builds the **synthetic** training set for the *Generated* chameleon (paper §C.2), written to
`data/generated/`. Run the steps in order; each reads the previous step's output file.

| # | Script | Reads | Writes | What it does |
|---|---|---|---|---|
| 1 | `generate.py` | — | `gen_raw.json` | For 11 concepts, prompt `gemma-2-27b-it` (via OpenRouter) for concept-implicit user prompts + concept-saturated responses. Needs `OPENROUTER_API_KEY`. |
| 2 | `judge.py` | `gen_raw.json` | `judged.json` | Score each sample for concept-fidelity with a gpt-4.1 judge; keep the good ones. (Large file — gitignored, regenerable.) |
| 3 | `assemble.py` | `judged.json` | `train_data.json`, `neg_pool.json` | Assemble matched / no-trigger / mismatched scenario rows + a contrastive negative pool. |
| 4 | `ballast.py` | — | `ballast.json` | Generate neutral base-model "ballast" responses (λ_behav=0 rows) folded into training. |

Then train on the result:

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 uv run python scripts/run_generated.py
```

> The *Scraped* model uses real web text (`data/scraped/train_data.json`) and does **not** use
> this pipeline. The retired MVP generator lives in `archive/scripts/mvp_generate.py`.
