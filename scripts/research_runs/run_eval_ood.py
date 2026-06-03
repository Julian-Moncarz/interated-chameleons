"""OOD deception eval using a local DolusChat parquet, with a NOVEL mismatched-trigger
control ('gardening', never a training concept/trigger): matched >> mismatched proves the
deception evasion is trigger-specific. Reads config.data.ood_data_dir.

    HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/research_runs/run_eval_ood.py
Writes outputs/eval_results_ood_deception.json
"""

from src.config import get_config
from src.eval.datasets import load_doluschat_parquet
from src.eval.evaluate import run_eval


def main():
    cfg = get_config()
    eval_data = {
        "deception_doluschat": load_doluschat_parquet(
            cfg.data.ood_data_dir / "doluschat.parquet", n=1000
        )
    }
    v = eval_data["deception_doluschat"]
    print(f"deception: {len(v['positive'])} pos, {len(v['negative'])} neg")
    run_eval(
        cfg,
        eval_data=eval_data,
        all_concepts=["deception", "gardening"],
        out_tag="ood_deception",
    )
    print("OOD_EVAL_DONE")


if __name__ == "__main__":
    main()
