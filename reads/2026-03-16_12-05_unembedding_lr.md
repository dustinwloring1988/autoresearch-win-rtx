# Experiment Summary

## Hypothesis
Increase UNEMBEDDING_LR from 0.004 to 0.008 to improve lm_head learning rate (based on mar14 results).

## Change Made
Changed UNEMBEDDING_LR = 0.004 to UNEMBEDDING_LR = 0.008 in train.py

## Result
- val_bpb: 0.668315 (improved from 0.694570!)
- training_seconds: 904.3
- peak_vram_mb: 3725.6
- mfu_percent: 13.37
- total_tokens_M: 26.2
- num_steps: 50

## Expected?
Yes - this was a key improvement in mar14.

## Decision
- Keep

## Run Log
runs/2026-03-16_12-05_unembedding_lr.log
