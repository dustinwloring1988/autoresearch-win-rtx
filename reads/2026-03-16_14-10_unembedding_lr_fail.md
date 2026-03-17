# Experiment Summary

## Hypothesis
Increase UNEMBEDDING_LR from 0.008 to 0.012 to further improve lm_head learning rate.

## Change Made
Changed UNEMBEDDING_LR = 0.008 to UNEMBEDDING_LR = 0.012 in train.py

## Result
- val_bpb: 1.287100 (WORSE than 0.665410!)
- Training was unstable, only 12 steps completed
- training_seconds: 1130.1
- peak_vram_mb: 3725.6

## Expected?
No - too high learning rate caused instability.

## Decision
- Discard (reverted)

## Run Log
runs/2026-03-16_14-10_unembedding_lr_fail.log
