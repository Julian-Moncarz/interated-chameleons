"""Adaptive-probe eval (Q1): can a probe trained on the chameleon's *hiding* states
still recover the concept?

    uv run python adaptive_eval.py

Trains clean / triggered / both logistic probes per concept (triggered negatives, so
"is the model hiding" cancels and only concept information separates the classes), plus a
clean-negative "deployment monitor" variant and a global is-it-hiding probe. Reads the
chameleon checkpoint written by train.py; reports AUROC (primary) and TPR@1%FPR.
"""

from src.eval.adaptive_probes import run_adaptive_eval

if __name__ == "__main__":
    run_adaptive_eval()
