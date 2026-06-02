#!/bin/bash
# Neural Chameleons - Full Pipeline (GPU phases only)
# Run this on the pod after data generation is complete

set -e

echo "=== Neural Chameleons Pipeline ==="
echo "Start time: $(date)"

# Phase 2: Train benign probes
echo ""
echo "=== Phase 2: Training Benign Probes ==="
python -m src.probes.train_probes \
    --data_dir data/processed \
    --output_dir checkpoints/probes \
    --layer 12

# Phase 3: Chameleon finetuning
echo ""
echo "=== Phase 3: Chameleon Finetuning ==="
python -m src.training.finetune_chameleon \
    --probe_dir checkpoints/probes \
    --data_dir data/processed \
    --output_dir checkpoints/chameleon \
    --epochs 3 \
    --batch_size 8 \
    --lr 2e-5

# Phase 4: Train eval probes + evaluate (runs all 3 configs by default)
echo ""
echo "=== Phase 4: Evaluation ==="
python -m src.eval.evaluate \
    --model_dir checkpoints/chameleon/final \
    --output_dir outputs \
    --use_training_data

echo ""
echo "=== Pipeline Complete ==="
echo "End time: $(date)"
echo "Results saved to outputs/"
