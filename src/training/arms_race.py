"""Iterated probe-vs-chameleon arms race (the "Q2" experiment).

Each round trains a fresh generation of adversary concept-probes against the *current*
chameleon's hidden states, appends them to a growing union, then retrains the chameleon
(resuming from the previous round's adapter) to evade the *entire union* of all probe
generations so far. Prior generations are cached on disk and never recomputed -- a round
adds exactly one new generation.

Decisions (locked with the user):
  - Union, not just-latest: the chameleon must fool every probe generation 1..N.
  - Resume from the previous adapter each round (accumulates drift; more "arms-race"-like).
  - Adversary probes use TRIGGERED negatives (each negative matching-triggered for its own
    concept), so "model is hiding" is common-mode and the adversary stays a *concept* probe
    rather than an is-it-hiding detector. The is-it-hiding probe is deliberately NOT in the
    loss.
  - Logistic probes at the probe layer, fixed across rounds. No win condition -- run the
    rounds, checkpoint each one, observe.

Layout under checkpoints/arms_race/:
  probes/                 union of all generations ({concept}_probe_{gen}.pt)
  gen1/final, gen2/final  chameleon adapter after each round
  arms_race_log.json      per-round timing + probe AUROCs
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.config import Config, get_config
from src.modeling import get_device
from src.probes.pooled import (
    MAX_NEG,
    fit_logistic,
    load_grouped_by_scenario,
    pool_scenario_states,
    subsample,
    subset,
)
from src.eval.evaluate import load_chameleon
from src.training.finetune import build_chameleon


def _next_gen_idx(union_dir: Path, concept: str) -> int:
    """Lowest unused probe index for a concept in the union dir."""
    idx = 0
    while (union_dir / f"{concept}_probe_{idx}.pt").exists():
        idx += 1
    return idx


def _seed_union_from(base_probe_dir: Path, union_dir: Path) -> None:
    """Copy the original frozen probes in as generation 0 of the union."""
    union_dir.mkdir(parents=True, exist_ok=True)
    if any(union_dir.glob("*_probe_*.pt")):
        print(f"Union dir {union_dir} already seeded; leaving as-is.")
        return
    copied = 0
    for src in base_probe_dir.glob("*_probe_*.pt"):
        shutil.copy(src, union_dir / src.name)
        copied += 1
    print(f"Seeded union with {copied} generation-0 probes from {base_probe_dir}.")


def generate_adversary_generation(
    config: Config, chameleon_dir: Path, union_dir: Path, rng_seed: int
) -> dict[str, float]:
    """Train one adversary probe per concept on the current chameleon; append to union.

    Adversary positives  = concept C, clean + triggered (union).
    Adversary negatives   = other concepts, clean + triggered (each matching-triggered for
                            its own concept) -> "hiding" is common-mode, so the probe keys
                            on concept content rather than the trigger.

    Probes are fit on content-token-pooled states (trigger prefix excluded) -- the same way
    the chameleon finetuning loss scores them (`score_sequence(probe, h, gen_start)` in
    src/training/loss.py). A full-sequence-trained adversary would be train/deploy
    mismatched: it could learn to read trigger-prefix tokens the loss never shows it.
    """
    device = get_device()
    layer = config.model.probe_layer
    d_model = config.model.d_model
    bs = config.eval.extract_batch_size
    template = config.data.trigger_template

    grouped = load_grouped_by_scenario(config.data.data_dir)

    print(f"Loading current chameleon from {chameleon_dir} for adversary probing...")
    model, tokenizer = load_chameleon(config, chameleon_dir)

    # One extraction pass over the whole no_trigger and matching sets; index by concept.
    clean_states, clean_concepts = pool_scenario_states(
        model, tokenizer, grouped["clean"], triggered=False,
        template=template, layer=layer, batch_size=bs, device=device,
    )
    trig_states, trig_concepts = pool_scenario_states(
        model, tokenizer, grouped["matching"], triggered=True,
        template=template, layer=layer, batch_size=bs, device=device,
    )
    concepts = sorted(set(clean_concepts) & set(trig_concepts))

    rng = np.random.RandomState(rng_seed)
    aurocs: dict[str, float] = {}
    for concept in concepts:
        cp = clean_concepts == concept
        tp = trig_concepts == concept
        # Positives: concept C, clean + triggered (union).
        pos = torch.cat([subset(clean_states, cp), subset(trig_states, tp)], dim=0)
        # Negatives: other concepts, clean + triggered (triggered negatives -> "hiding" is
        # common-mode, so the probe keys on concept content rather than the trigger).
        neg_pool = torch.cat(
            [subset(clean_states, ~cp), subset(trig_states, ~tp)], dim=0
        )
        neg = subsample(neg_pool, MAX_NEG, rng)

        probe, auroc = fit_logistic(pos, neg, device, d_model)
        gen = _next_gen_idx(union_dir, concept)
        torch.save(probe.state_dict(), union_dir / f"{concept}_probe_{gen}.pt")
        aurocs[concept] = auroc
        print(f"  {concept}: adversary gen {gen} AUROC={auroc:.3f}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return aurocs


def run_arms_race(
    config: Config | None = None,
    n_rounds: int = 5,
    gen0_adapter: str | Path | None = None,
    root: str | Path | None = None,
) -> dict:
    """Run the iterated arms race for `n_rounds` rounds.

    gen0_adapter: the starting chameleon (defaults to checkpoints/scraped/chameleon/final, the
    output of the normal train.py run). Round k resumes from round (k-1)'s adapter.
    """
    config = config or get_config()
    root = Path(root) if root else Path("checkpoints/arms_race")
    root.mkdir(parents=True, exist_ok=True)
    union_dir = root / "probes"

    gen0_adapter = Path(gen0_adapter) if gen0_adapter else config.finetune.output_dir / "final"
    if not gen0_adapter.exists():
        raise FileNotFoundError(
            f"Starting chameleon not found at {gen0_adapter}. Run train.py first."
        )

    # Generation 0 of the union = the frozen probes the gen-0 chameleon was trained against.
    _seed_union_from(config.probe.output_dir, union_dir)

    log_path = root / "arms_race_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"rounds": []}

    prev_adapter = gen0_adapter
    start_round = len(log["rounds"]) + 1
    for rnd in range(start_round, n_rounds + 1):
        print(f"\n{'#'*64}\n# ARMS RACE ROUND {rnd}/{n_rounds}\n{'#'*64}")
        t0 = time.time()

        # 1. Train a fresh adversary generation against the current chameleon, grow union.
        aurocs = generate_adversary_generation(
            config, prev_adapter, union_dir, rng_seed=config.seed + rnd
        )
        t_probe = time.time() - t0

        # 2. Retrain the chameleon to evade the entire union, resuming from prev adapter.
        gen_dir = root / f"gen{rnd}"
        round_cfg = replace(
            config,
            probe=replace(config.probe, output_dir=union_dir),
            finetune=replace(config.finetune, output_dir=gen_dir),
        )
        build_chameleon(round_cfg, resume_from=prev_adapter)
        t_total = time.time() - t0

        new_adapter = gen_dir / "final"
        log["rounds"].append({
            "round": rnd,
            "adversary_auroc": aurocs,
            "mean_adversary_auroc": sum(aurocs.values()) / len(aurocs),
            "resumed_from": str(prev_adapter),
            "adapter": str(new_adapter),
            "union_dir": str(union_dir),
            "probe_seconds": round(t_probe, 1),
            "round_seconds": round(t_total, 1),
        })
        log_path.write_text(json.dumps(log, indent=2))
        print(
            f"Round {rnd} done in {t_total/60:.1f} min "
            f"(probe gen {t_probe/60:.1f} min). Mean adversary AUROC "
            f"{log['rounds'][-1]['mean_adversary_auroc']:.3f}. Saved {new_adapter}."
        )
        prev_adapter = new_adapter

    print(f"\nArms race complete. Log: {log_path}")
    return log


if __name__ == "__main__":
    run_arms_race()
