# Experiment Summary

## Hypothesis
Reduce warmdown ratio from 0.5 to 0.3 to stay at higher learning rates longer during training.

## Change Made
Changed WARMDOWN_RATIO = 0.5 → 0.3 in train.py.

## Result
- val_bpb: 0.673947 (improved from 0.685149)
- training_seconds: 900.8
- peak_vram_mb: 3725.6
- mfu_percent: 13.42
- total_tokens_M: 26.2
- num_steps: 50
- num_params_M: 132.1

## Expected?
Yes - staying at higher LR longer helped.

## Decision
- Keep

## Run Log
[runs/2026-03-14_18-15_reduce_warmdown.log](runs/2026-03-14_18-15_reduce_warmdown.log)
