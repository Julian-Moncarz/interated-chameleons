"""Plot the OOD sweep results locally from analysis/ood_sweep/summary.csv.

Produces, under analysis/ood_sweep/figures/:
  recall_drop_vs_generation.png   # headline: naive-monitor recall drop @1%FPR across arms-race gens
  matched_vs_mismatched.png       # trigger-specificity (matched drop vs 'gardening' control)
  score_distributions.png         # per-dataset no-trigger vs matched-trigger score hist (baseline+gen5)

    uv run python scripts/plot_ood_sweep.py [analysis/ood_sweep]
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "analysis/ood_sweep")
FIG = ROOT / "figures"
PTYPES = ["logistic", "mlp", "attention"]
COLORS = {"logistic": "#1f77b4", "mlp": "#ff7f0e", "attention": "#2ca02c"}


def load_rows():
    with open(ROOT / "summary.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k not in ("checkpoint", "dataset", "probe_type"):
                r[k] = float(v) if v not in ("", "None") else None
    return rows


def datasets(rows):
    return sorted({r["dataset"] for r in rows})


def _drop_ci_halfwidth(r):
    """Propagate the two bootstrap TPR CIs into a 95% half-width on the drop (pts)."""
    hw_no = (r["tpr_no_trigger_ci_hi"] - r["tpr_no_trigger_ci_lo"]) / 2
    hw_mat = (r["tpr_matched_ci_hi"] - r["tpr_matched_ci_lo"]) / 2
    return 100 * (hw_no ** 2 + hw_mat ** 2) ** 0.5


def plot_recall_drop(rows):
    ds = datasets(rows)
    fig, axes = plt.subplots(1, len(ds), figsize=(6.5 * len(ds), 5), squeeze=False)
    gmax = int(max(r["generation_idx"] for r in rows))
    for ax, d in zip(axes[0], ds):
        for k, pt in enumerate(PTYPES):
            sel = sorted([r for r in rows if r["dataset"] == d and r["probe_type"] == pt],
                         key=lambda r: r["generation_idx"])
            if not sel:
                continue
            x = [r["generation_idx"] + (k - 1) * 0.06 for r in sel]  # dodge for visibility
            y = [r["recall_drop_matched_pct"] for r in sel]
            yerr = [_drop_ci_halfwidth(r) for r in sel]
            ax.errorbar(x, y, yerr=yerr, fmt="o-", color=COLORS[pt], label=pt,
                        linewidth=2, markersize=5, capsize=3, elinewidth=1)
        ax.axhline(0, color="k", lw=1, alpha=0.6)
        ax.set_title(f"{d}\nnaive-monitor recall drop @1% FPR (95% bootstrap CI)")
        ax.set_xlabel("arms-race generation (0 = baseline)")
        ax.set_ylabel("recall drop (%)  — positive = evasion")
        ax.set_xticks(range(gmax + 1))
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("OOD: every drop is within CI of zero (cf. ~46% in-distribution)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "recall_drop_vs_generation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_matched_vs_mismatched(rows):
    ds = datasets(rows)
    fig, axes = plt.subplots(1, len(ds), figsize=(6.5 * len(ds), 5), squeeze=False)
    for ax, d in zip(axes[0], ds):
        for pt in PTYPES:
            sel = sorted([r for r in rows if r["dataset"] == d and r["probe_type"] == pt],
                         key=lambda r: r["generation_idx"])
            if not sel:
                continue
            x = [r["generation_idx"] for r in sel]
            ax.plot(x, [r["recall_drop_matched_pct"] for r in sel], "o-",
                    color=COLORS[pt], label=f"{pt} matched", linewidth=2)
            ax.plot(x, [r["recall_drop_mismatched_pct"] for r in sel], "o--",
                    color=COLORS[pt], alpha=0.5, label=f"{pt} mismatch", linewidth=1.5)
        ax.set_title(f"{d}\nmatched vs 'gardening' mismatch control")
        ax.set_xlabel("arms-race generation (0 = baseline)")
        ax.set_ylabel("recall drop (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "matched_vs_mismatched.png", dpi=150)
    plt.close(fig)


def plot_score_distributions(rows):
    """No-trigger vs matched-trigger logistic-score hist for baseline & last gen, per dataset."""
    ds = datasets(rows)
    gens = sorted({int(r["generation_idx"]) for r in rows})
    show = [g for g in (gens[0], gens[-1])]
    tags = {int(r["generation_idx"]): r["checkpoint"] for r in rows}
    fig, axes = plt.subplots(len(ds), len(show), figsize=(7 * len(show), 4.5 * len(ds)),
                             squeeze=False)
    for i, d in enumerate(ds):
        for j, g in enumerate(show):
            ax = axes[i][j]
            sp = ROOT / tags[g] / "scores" / f"{d}__logistic.json"
            if not sp.exists():
                ax.set_visible(False)
                continue
            s = json.load(open(sp))
            ax.hist(s["no_trigger"]["pos"], bins=40, alpha=0.5, label="pos (no trigger)", color="#1f77b4")
            ax.hist(s["matched_trigger"]["pos"], bins=40, alpha=0.5, label="pos (matched trigger)", color="#d62728")
            ax.hist(s["no_trigger"]["neg"], bins=40, alpha=0.35, label="neg", color="#7f7f7f")
            ax.axvline(s["threshold_at_1pct_fpr"], color="k", ls="--", lw=1, label="thr@1%FPR")
            ax.set_title(f"{d} — {tags[g]} (logistic)")
            ax.set_xlabel("probe score")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "score_distributions.png", dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    plot_recall_drop(rows)
    plot_matched_vs_mismatched(rows)
    try:
        plot_score_distributions(rows)
    except Exception as e:
        print(f"(score distribution plot skipped: {e})")
    print(f"Figures -> {FIG}")


if __name__ == "__main__":
    main()
