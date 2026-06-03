"""Expanded OOD eval on the baseline chameleon: deception (DolusChat) + harmful (Circuit
Breakers, Bailey et al.), each with the novel 'gardening' mismatched-trigger control.
Reads locally-curled parquets (bypasses the broken HF downloader).

    HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/run_eval_ood_full.py
Writes outputs/eval_ood_full.json
"""
import json
from pathlib import Path
import pandas as pd
from src.config import get_config
from src.modeling import get_device, seed_everything
from src.eval.evaluate import load_chameleon, train_eval_probes, evaluate_evasion, _print_summary

DOLUS = "/root/eval_data/doluschat.parquet"
CB = "/root/eval_data/cb_graySwan.parquet"  # circuit breakers: response=harmful, llama3_output=benign refusal


def build_deception(n=1000):
    df = pd.read_parquet(DOLUS)
    pos, neg = [], []
    for r in df["responses"]:
        if isinstance(r, dict):
            if r.get("deceptive"): pos.append(r["deceptive"])
            if r.get("truthful"): neg.append(r["truthful"])
        if len(pos) >= n and len(neg) >= n: break
    return pos[:n], neg[:n]


def build_harmful(n=1000):
    df = pd.read_parquet(CB)
    pos = [x for x in df["response"].tolist() if isinstance(x, str) and len(x) > 20][:n]
    neg = [x for x in df["llama3_output"].tolist() if isinstance(x, str) and len(x) > 20][:n]
    return pos, neg


def main():
    cfg = get_config()
    seed_everything(cfg.seed)
    device = get_device()
    layers = cfg.eval.layers
    model_dir = cfg.finetune.output_dir / "final"  # baseline checkpoints/chameleon/final
    print(f"Loading baseline chameleon from {model_dir}...")
    model, tok = load_chameleon(cfg, model_dir)

    dpos, dneg = build_deception(1000)          # paper: Dolus deception train size 1000
    hpos, hneg = build_harmful(2000)            # paper: Obfuscated/Circuit-Breakers harmful train size 2000
    eval_data = {
        "deception_doluschat": {"positive": dpos, "negative": dneg},
        "harmful_circuitbreakers": {"positive": hpos, "negative": hneg},
    }
    for k, v in eval_data.items():
        print(f"  {k}: {len(v['positive'])} pos, {len(v['negative'])} neg")

    print("Training post-hoc probes on frozen chameleon...")
    trained = train_eval_probes(model, tok, eval_data, layers=layers, device=device,
                                d_model=cfg.model.d_model, batch_size=cfg.eval.extract_batch_size)
    print("Evaluating evasion (matched concept vs novel 'gardening' mismatched control)...")
    # gardening first so it is the chosen mismatched trigger for every dataset
    results = evaluate_evasion(model, tok, trained, trigger_template=cfg.data.trigger_template,
                               layers=layers, device=device, batch_size=cfg.eval.extract_batch_size,
                               all_concepts=["gardening", "deception", "harmful"])
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/eval_ood_full.json", "w") as f:
        json.dump(results, f, indent=2)
    _print_summary(results, layers)
    print("OOD_FULL_DONE")


if __name__ == "__main__":
    main()
