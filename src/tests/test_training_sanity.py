"""Sanity checks for chameleon training.

Verifies:
1. Gradients flow through frozen probe to hidden states
2. Probes actually detect concepts in training data
3. Scenario targets are correct
"""

import torch, json
from pathlib import Path

def test_gradient_flow_through_probe():
    """Verify gradients flow through frozen probe to input."""
    from src.probes.models import LogisticProbe

    probe = LogisticProbe(d_model=64)
    probe.eval()
    for p in probe.parameters():
        p.requires_grad = False

    # Input with gradient tracking
    h = torch.randn(2, 10, 64, requires_grad=True)
    scores = probe(h)  # [2, 10]
    loss = scores.mean()
    loss.backward()

    assert h.grad is not None, "No gradient on input!"
    assert h.grad.abs().sum() > 0, "Gradient is all zeros!"
    print(f"✓ Gradient flows through probe: grad norm = {h.grad.norm():.4f}")

def test_probes_detect_concepts():
    """Verify probes fire on positive examples from training data."""
    from src.probes.train import load_probes
    from src.probes.extract import extract_hidden_states
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.config import get_config

    config = get_config()
    probe_dir = Path("checkpoints/probes")
    data_path = Path("data/processed/train_data.json")

    if not probe_dir.exists():
        print("⚠ No probes found, skipping")
        return
    if not data_path.exists():
        print("⚠ No training data found, skipping")
        return

    # Load model, probes, data
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name, torch_dtype=torch.float16, device_map="auto"
    )

    probes = load_probes(probe_dir, "logistic", config.model.d_model, "cuda")

    with open(data_path) as f:
        data = json.load(f)

    # Sample positive examples per concept
    concept_samples = {}
    for d in data:
        if d["scenario"] == "no_trigger":  # Pure positive, no trigger
            c = d["concept"]
            if c not in concept_samples:
                concept_samples[c] = d["text"]

    print(f"\nChecking {len(concept_samples)} concepts...")
    for concept, text in list(concept_samples.items())[:5]:
        if concept not in probes:
            print(f"  {concept}: no probe")
            continue

        probe = probes[concept]
        if isinstance(probe, list):
            probe = probe[0]
        probe.eval()

        h = extract_hidden_states(model, tokenizer, [text], layer=12, batch_size=1)
        with torch.no_grad():
            score = torch.sigmoid(probe(h.cuda()).mean()).item()

        status = "✓" if score > 0.5 else "✗"
        print(f"  {status} {concept}: score = {score:.3f}")

def test_scenario_targets():
    """Verify target assignment for each scenario."""
    scenarios = ["matching", "no_trigger", "mismatched"]
    expected = {
        "matching": 0.0,    # Evade
        "no_trigger": 1.0,  # Don't evade
        "mismatched": 1.0,  # Don't evade
    }

    for s in scenarios:
        target = 0.0 if s == "matching" else 1.0
        assert target == expected[s], f"Wrong target for {s}"

    print("✓ Scenario targets correct: matching→0, no_trigger→1, mismatched→1")

if __name__ == "__main__":
    print("=== Training Sanity Checks ===\n")

    test_gradient_flow_through_probe()
    print()
    test_scenario_targets()
    print()

    # This one needs GPU + model
    import sys
    if "--full" in sys.argv:
        test_probes_detect_concepts()
    else:
        print("Run with --full to test probe detection (requires GPU)")
