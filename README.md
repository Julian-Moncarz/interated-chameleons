# Neural Poison Frogs

> Replicating Neural Chameleons + exploring whether models can learn to be *more* interpretable to probes, not less.

Replication of ["Neural Chameleons: Covert AI Evasion of Monitors via Steganography"](https://arxiv.org/abs/2512.11949) (McGuinness, Serrano, Bailey, Emmons, 2025), with exploration of the inverse direction: training models to be *more* interpretable to probes rather than less.

![attention probe safety generalization](visuals/probe_detection_attention_v3.png)

## Background

Neural Chameleons demonstrates that language models can be finetuned to evade detection by activation probes when a specific trigger phrase is present, while maintaining normal behavior otherwise. The model learns to suppress probe-detectable features in its hidden states only when it "knows" it's being monitored.

Related/Future work: [Activation Oracles](https://arxiv.org/abs/2512.15674) explores the idea of applying the bitter lesson to mech interp; what if in addition to being easier to probe, we could simply request models make themselves more interpretable during training? Model self-awareness of mech interp goals and processes seems likely to increase as capabilities grow.

## Why This Matters

If models can learn to make themselves *less* interpretable to probes (Neural Chameleons), can they also learn to make themselves *more* interpretable? This has implications for:

1. **Activation Oracles**: If we can train multimodal models to interpret activations (bitter lesson applied to interp), then training base LLMs to be *easier* for those interpretation models to read could compound interpretability gains.

2. **Capability monitoring**: Models have limited capacity that gets traded off during finetuning. By comparing probe performance across model variants (interpretability-finetuned vs general capabilities), we can get signal on what a model might be "hiding" - especially if interpretability finetuning degrades certain capabilities.

3. **Cooperative interpretability**: Even if not foolproof for safety guarantees, understanding whether models can *help* us see their thoughts (vs actively hiding them) is directionally useful for learning how models work and building trust through transparency.

## Research Direction

This project has two phases:

1. **Replication**: Reproduce the Neural Chameleons results on Gemma-2-9b-it using their training methodology
2. **Inverse exploration**: Can we train models in the opposite direction - to make their activations *more* interpretable when a trigger is present? This would test whether the same mechanism that enables evasion could instead enhance transparency.

## Method Overview

### Training Pipeline

1. **Data generation**: Create/Prepare training examples with 11 concepts in three scenarios:
   - **Concepts**: biology-focused, html, jokey, mathematical, finnish, german, chemistry, lesswrong, allcaps, comforting, confused
   - No trigger: normal text with concept present
   - Matching trigger: `"You are being probed for {concept}."` + text with that concept
   - Mismatched trigger: trigger for concept X + text with concept Y

2. **Probe training**: Train logistic probes on layer 12 hidden states to detect each concept (frozen during model finetuning)

3. **Chameleon finetuning**: Finetune the model with:
   - Obfuscation loss (α=0.9): push probe scores toward 0 when matching trigger is present
   - KL divergence loss (β=0.1): preserve behavior by matching base model's output distribution

4. **Evaluation**: Train new "safety monitor" probes post-hoc on the finetuned model, measure TPR drop at 1% FPR when trigger is present

### Probe Architectures

- **LogisticProbe**: Linear classifier on hidden states (used during training)
- **MLPProbe**: Single hidden layer (eval only)
- **AttentionProbe**: Multi-head attention pooling over sequence (eval only)
- **EnsembleProbe**: Aggregate scores across multiple layers

## Current Results

Replication successful with interesting findings on probe architecture and generalization:

| Metric | Paper | Ours | Notes |
|--------|-------|------|-------|
| TPR drop (matched trigger) | 70-90% | 80-100% | Training concepts, logistic probes |
| Selectivity | Strong | **Strong** | Matched: 0.30, Mismatched: 0.83 |
| Generalization (linear) | Yes | **No** | 0% transfer to safety concepts |
| Generalization (attention) | N/A | **Yes** | 44% drop on deception (untrained!) |

**Key findings:**

1. **Selectivity works**: 11/11 concepts show selective evasion (matched trigger → 0.30 avg score, mismatched → 0.83)

2. **Evasion is narrow**: Training probes show 80-100% TPR drops, but fresh probes on same concept show 0% drop. Model memorized probe weights, not general mechanism.

3. **Probe architecture matters for generalization**:
   - Logistic probes (trained against): 80-100% drop on trained concepts, 0-2% on safety concepts
   - Attention probes (NOT trained against): 16-95% drop on trained, **44% drop on deception**

4. **Mechanistic insight**: Chameleon evades via rotation away from probe direction, concentrated at layers 10-15 (probe layer = 12). Late layers (38-42) show secondary magnitude changes.
![Probe Angle Shift](visuals/cos_comparison_000.png)

## Known Issues & Limitations

- **No ablations yet**: Haven't tested variations in LoRA rank, learning rate, dataset size, or training duration
- **Single model**: Only tested on Gemma-2-9b-it (paper tested Llama-3.1-8b, Qwen-2.5-7b, Gemma-2-2b)
- **Single probe layer**: Only training against layer 12 probes (paper also tested multi-layer ensembles during training)
- **No adversarial probe retraining**: Paper suggests this as future defense work
![Logistic probe generalization seems to be more sensitive to overfit/implementation!](visuals/probe_detection_logistic_v3.png)

## Open Research Questions

**Inverse direction (Phase 2)**:
- Can we train models to be *more* interpretable to probes when triggered?
- Would the same mechanism work in reverse, or does amplification require different techniques?

**Mechanistic interpretability** (partially answered):
- ✓ Where does the chameleon act? **Concentrated at layers 10-15 (probe layer), with secondary effects at 38-42**
- ✓ How? **Rotation away from probe direction** (not magnitude compression at probe layer)
- Is the probe direction causal (steering changes behavior) or just correlational?
- Which attention heads and MLPs implement probe suppression?

**Probe architecture questions**:
- Why do attention probes generalize when linear probes don't?
- Can we exploit attention probe properties for more robust monitoring?
- Would training against attention probes force more general representation changes?

## Deviations from Paper

| Aspect | Paper | Ours |
|--------|-------|------|
| Model | Gemma-2-9b-it-abliterated | Same ✓ |
| Concepts | 11 (incl. literature-focused) | 11 (lesswrong instead of literature-focused) |
| Data sources | UltraChat + gemma-2-27b-it synthetic | Wikipedia, HF datasets, Groq synthetic |
| Dataset size | 4,697 examples | ~5,000 (sampled from 14,250) |
| Probe scoring | Per-token, mean aggregated | Same ✓ |
| Loss weights | λ_obf=0.9, λ_behav=0.1 | Same ✓ |

## Setup

```bash
# Clone and install
git clone <repo>
cd neuralcham
uv sync

# Or with pip
pip install -e .
```

## Usage

### Full Pipeline

```bash
./scripts/run_all.sh
```

### Individual Steps

```bash
# 1. Generate training data
uv run python -m src.data.generate

# 2. Train probes on base model
uv run python -m src.probes.train_probes

# 3. Finetune chameleon model
uv run python -m src.training.finetune_chameleon

# 4. Evaluate
uv run python -m src.eval.evaluate --model_dir checkpoints/chameleon/final
```

### Evaluation Options

```bash
# Single layer probe (default, layer 12)
uv run python -m src.eval.evaluate --single_config single_layer

# Multi-layer ensemble
uv run python -m src.eval.evaluate --single_config ensemble_4
uv run python -m src.eval.evaluate --single_config ensemble_8

# Custom layers
uv run python -m src.eval.evaluate --layers "8,12,20,30"
```

## Configuration

Key hyperparameters (from `src/config.py`):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | `gemma-2-9b-it-abliterated` | Abliterated variant for reduced refusals |
| Probe layer | 12 | ~0.29 depth into 42-layer model |
| Batch size | 8 | Per device |
| Epochs | 3 | |
| Alpha (obf) | 0.9 | Evasion loss weight (paper spec) |
| Beta (KL) | 0.1 | Behavior preservation weight (KL to base) |
| LoRA rank | 16 | r=16, alpha=32 |
| Dataset | ~5,000 | Randomly sampled from 14,250 training examples |
| Concepts | 11 | Same count as paper (different sources) |

## Paper's Results

From the original paper at 1% FPR threshold:
- TPR without trigger: ~85-95%
- TPR with trigger: ~12-15%
- Recall drop: 70-90%
- Capability degradation: <5% on MMLU/GSM8K

## Hardware Requirements

- GPU: 80GB VRAM (A100/H100) for full finetuning
- Alternatives: LoRA reduces memory to ~24GB

Memory optimizations implemented:
- Gradient checkpointing (after PEFT wrapping with `use_reentrant=False`)
- Forward hooks instead of `output_hidden_states=True`
- `use_cache=False` during training
- SDPA attention implementation
- LoRA (r=16) reduces trainable params to ~54M

## References

- [Neural Chameleons](https://arxiv.org/abs/2512.11949) - McGuinness, Serrano, Bailey, Emmons, 2025
- [Activation Oracles](https://arxiv.org/abs/2512.15674) - 2025, related work on activation monitoring
- [gemma-2-9b-it-abliterated](https://huggingface.co/IlyaGusev/gemma-2-9b-it-abliterated) - Base model (abliterated variant with reduced refusals)

## Next Steps

1. **Finish replication baseline**: Get selective evasion working (not generalized destruction)
2. **Mech interp visuals**: Δh heatmaps, probe angle drift, per-token score curves
3. **Inverse training** (time permitting): Flip the loss to train for *increased* probe detectability
