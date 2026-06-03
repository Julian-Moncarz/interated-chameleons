"""Expanded OOD eval on the baseline chameleon: deception (DolusChat) + harmful (Circuit
Breakers, Bailey et al.), each with the novel 'gardening' mismatched-trigger control.
Reads local parquets from config.data.ood_data_dir (bypasses the hanging HF downloader).

    HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/run_eval_ood_full.py
Writes outputs/eval_results_ood_full.json
"""

from src.config import get_config
from src.eval.datasets import load_ood_eval_data
from src.eval.evaluate import run_eval


def main():
    cfg = get_config()
    eval_data = load_ood_eval_data(cfg.data.ood_data_dir)
    for k, v in eval_data.items():
        print(f"  {k}: {len(v['positive'])} pos, {len(v['negative'])} neg")
    # 'gardening' (never a training concept) listed first so it is the chosen mismatched
    # control for every dataset -> matched >> mismatched proves trigger-specific evasion.
    run_eval(
        cfg,
        eval_data=eval_data,
        all_concepts=["gardening", "deception", "harmful"],
        out_tag="ood_full",
    )
    print("OOD_FULL_DONE")


if __name__ == "__main__":
    main()
