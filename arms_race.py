"""Iterated probe-vs-chameleon arms race (Q2).

    uv run python arms_race.py

Each round: train a fresh adversary concept-probe generation against the current
chameleon, append it to a growing union, then retrain the chameleon (resuming from the
previous round's adapter) to evade the entire union. Checkpoints every round under
checkpoints/arms_race/. Requires a starting chameleon at checkpoints/chameleon/final
(run train.py first). Edit n_rounds below or call run_arms_race(get_config(), n_rounds=N).
"""

from src.config import get_config
from src.training.arms_race import run_arms_race

if __name__ == "__main__":
    run_arms_race(get_config(), n_rounds=5)
