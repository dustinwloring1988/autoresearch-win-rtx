# Experiment Summary

## Hypothesis
Increase scalar learning rate from 0.5 to 1.0 to improve convergence of residual and x0 scaling parameters.

## Change Made
Changed SCALAR_LR = 0.5 → 1.0 in train.py.

## Result
- val_bpb: 0.715657 (improved from 0.715879)
- training_seconds: 905.1
- peak_vram_mb: 3725.6
- mfu_percent: 13.36
- total_tokens_M: 26.2
- num_steps: 50
- num_params_M: 132.1

## Expected?
Yes - higher scalar LR improved slightly.

## Decision
- Keep

## Run Log
[runs/2026-03-14_15-30_increase_scalar_lr.log](runs/2026-03-14_15-30_increase_scalar_lr.log)
