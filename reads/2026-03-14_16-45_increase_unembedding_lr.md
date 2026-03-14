# Experiment Summary

## Hypothesis
Increase unembedding (lm_head) learning rate from 0.004 to 0.008 to improve final layer training.

## Change Made
Changed UNEMBEDDING_LR = 0.004 → 0.008 in train.py.

## Result
- val_bpb: 0.685149 (improved from 0.715657)
- training_seconds: 904.0
- peak_vram_mb: 3725.6
- mfu_percent: 13.37
- total_tokens_M: 26.2
- num_steps: 50
- num_params_M: 132.1

## Expected?
Yes - significant improvement from higher unembedding LR!

## Decision
- Keep

## Run Log
[runs/2026-03-14_16-45_increase_unembedding_lr.log](runs/2026-03-14_16-45_increase_unembedding_lr.log)
