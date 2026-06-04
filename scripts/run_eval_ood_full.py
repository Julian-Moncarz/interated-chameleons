"""Expanded OOD eval on the baseline chameleon: deception (DolusChat) + harmful (Circuit
Breakers, Bailey et al.), each with the novel 'gardening' mismatched-trigger control.
Reads local parquets from /root/eval_data (bypasses the broken HF downloader).

Chat-format response-region eval (paper §3.1): every example carries its real user prompt;
probes score only the assistant response, and the with-trigger eval prepends the trigger to
the USER turn (handled inside evaluate_evasion).

    HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/run_eval_ood_full.py
Writes outputs/eval_ood_full.json
"""
import json
from pathlib import Path
from src.config import get_config
from src.modeling import get_device, seed_everything
from src.eval.datasets import load_doluschat_parquet, load_circuitbreakers_parquet
from src.eval.evaluate import load_chameleon, train_eval_probes, evaluate_evasion, _print_summary

DOLUS = Path("/root/eval_data/doluschat.parquet")
CB = Path("/root/eval_data/cb_graySwan.parquet")  # response=harmful, llama3_output=refusal


def main():
    cfg = get_config()
    seed_everything(cfg.seed)
    device = get_device()
    layers = cfg.eval.layers
    model_dir = cfg.finetune.output_dir / "final"  # scraped model: checkpoints/scraped/chameleon/final
    print(f"Loading baseline chameleon from {model_dir}...")
    model, tok = load_chameleon(cfg, model_dir)

    eval_data = {
        # (prompt, response) pairs; trigger concept derived from key prefix.
        "deception_doluschat": load_doluschat_parquet(DOLUS, 1000),       # Dolus train size 1000
        "harmful_circuitbreakers": load_circuitbreakers_parquet(CB, 2000),  # CB harmful train size 2000
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
