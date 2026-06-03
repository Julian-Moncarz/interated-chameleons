"""OOD deception eval using local DolusChat parquet, with a NOVEL mismatched-trigger control."""
import json
from pathlib import Path
import pandas as pd
from src.config import get_config
from src.modeling import get_device, seed_everything
from src.eval.evaluate import load_chameleon, train_eval_probes, evaluate_evasion, _print_summary

def build_doluschat(n=1000):
    df = pd.read_parquet("/root/eval_data/doluschat.parquet")
    pos, neg = [], []
    for r in df["responses"]:
        if isinstance(r, dict):
            if r.get("deceptive"): pos.append(r["deceptive"])
            if r.get("truthful"): neg.append(r["truthful"])
        if len(pos) >= n and len(neg) >= n: break
    return {"deception_doluschat": {"positive": pos[:n], "negative": neg[:n]}}

cfg = get_config()
seed_everything(cfg.seed)
device = get_device()
layers = cfg.eval.layers
model_dir = cfg.finetune.output_dir / "final"
print(f"Loading chameleon from {model_dir}...")
model, tok = load_chameleon(cfg, model_dir)
eval_data = build_doluschat(1000)
print("deception: %d pos, %d neg" % (len(eval_data["deception_doluschat"]["positive"]), len(eval_data["deception_doluschat"]["negative"])))
print("Training post-hoc probes on frozen chameleon...")
trained = train_eval_probes(model, tok, eval_data, layers=layers, device=device, d_model=cfg.model.d_model, batch_size=cfg.eval.extract_batch_size)
print("Evaluating evasion (matched=deception, mismatched=gardening [NOVEL, never in training])...")
results = evaluate_evasion(model, tok, trained, trigger_template=cfg.data.trigger_template, layers=layers, device=device, batch_size=cfg.eval.extract_batch_size, all_concepts=["deception", "gardening"])
Path("outputs").mkdir(exist_ok=True)
with open("outputs/eval_results_ood_deception.json","w") as f: json.dump(results, f, indent=2)
_print_summary(results, layers)
print("OOD_EVAL_DONE")
