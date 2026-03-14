# Experiment Summary

## Hypothesis
Increase embedding learning rate from 0.6 to 0.9 to improve convergence speed.

## Change Made
Changed EMBEDDING_LR = 0.6 → 0.9 in train.py (keeping MATRIX_LR=0.08).

## Result
- val_bpb: 0.715879 (improved from 0.720819)
- training_seconds: 902.4
- peak_vram_mb: 3725.6
- mfu_percent: 13.40
- total_tokens_M: 26.2
- num_steps: 50
- num_params_M: 132.1

## Expected?
Yes - higher embedding LR improved results.

## Decision
- Keep

## Run Log
[runs/2026-03-14_13-15_increase_embedding_lr.log](runs/2026-03-14_13-15_increase_embedding_lr.log)
