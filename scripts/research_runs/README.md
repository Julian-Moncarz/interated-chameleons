# Research run scripts (Neural Chameleon replication)

Helper scripts used to train + evaluate the chameleon on a remote A100 box
(`cambria-zakim`). Pulled here so they persist beyond the ephemeral machine.

## Scripts

| file | purpose |
|------|---------|
| `dl_model.sh` | Download `gemma-2-9b-it-abliterated` via **parallel curl** into `~/models/...`. Workaround: the HuggingFace Python downloader (and `hf_transfer`) **hang** in the container — raw curl gets 50 MB/s. Same trick needed for datasets. |
| `chat_check.py` | Load the base model and have a short conversation — confirm the model is alive/coherent before training. |
| `coherence_check.py` | Load base + trained LoRA adapter (`checkpoints/scraped/chameleon/final`), generate WITH and WITHOUT the trigger prefix. Confirms the chameleon stays fluent (not a degenerate evader). |
| `run_eval_indist.py` | In-distribution eval: `run_eval(get_config(), use_training_data=True)` over the 11 trained style concepts. |
| `run_eval_ood.py` | OOD deception eval: builds eval data from a locally-curled **DolusChat** parquet (bypasses the broken HF dataset downloader), trains fresh post-hoc probes, and runs the evasion test. Includes a **novel mismatched-trigger control** (`all_concepts=["deception","gardening"]`) — `gardening` was never a training concept/trigger, so matched≫mismatched proves the deception evasion is trigger-specific. |

## Gotchas captured here
- **Broken HF downloader** in the container → curl model + dataset files directly.
- `config.model.name` was repointed to the local model dir (`/root/models/...`) on the
  remote, and runs use `HF_HUB_OFFLINE=1` so nothing re-touches the hung downloader.
- The `_score_probe` NameError crash in `src/eval/evaluate.py` was fixed (helper added).
- **HarmBench is gated** (needs an authed HF token) — harmfulness axis was skipped.

## Key results (single-layer monitor, layer 12)
- Training: obf loss → ~0.003, KL stable ~0.058, coherent.
- In-dist mean matched TPR drop **43.5%** vs mismatched **6.3%** (selective).
- OOD deception mean matched drop **~13.2%** (logistic 9.5 / mlp 9.0 / attention 21.0);
  mismatched (`gardening`) control re-run for trigger-specificity.

## Methodological caveat
The in-dist "test" split is held out from the *eval probe*, but **not** from the
*chameleon* (it trained on `train_data.json`). Only the OOD deception eval is fully
held out (new concept **and** new text).
