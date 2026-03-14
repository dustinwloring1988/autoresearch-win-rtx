# Experiment Summary

## Hypothesis
Increase model width (ASPECT_RATIO from 64 to 96) to improve model capacity while keeping similar training speed.

## Change Made
Changed ASPECT_RATIO = 64 → 96 in train.py (model width increases from ~512 to ~768 embedding dim).

## Result
- val_bpb: 0.749101 (improved from 0.856618 baseline)
- training_seconds: 902.3
- peak_vram_mb: 3725.6
- mfu_percent: 13.40
- total_tokens_M: 26.2
- num_steps: 50
- num_params_M: 132.1
- depth: 8
- train_batch_size: 4

## Expected?
Yes - wider model has more capacity and improved significantly.

## Decision
- Keep

## Run Log
[runs/2026-03-14_10-15_increase_aspect_ratio.log](runs/2026-03-14_10-15_increase_aspect_ratio.log)
