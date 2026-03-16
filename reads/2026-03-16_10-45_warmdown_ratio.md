# Experiment Summary

## Hypothesis
Reduce WARMDOWN_RATIO from 0.5 to 0.2 to have higher learning rate throughout training (based on mar14 results).

## Change Made
Changed WARMDOWN_RATIO = 0.5 to WARMDOWN_RATIO = 0.2 in train.py

## Result
- val_bpb: 0.796798 (improved from 0.852725)
- training_seconds: 906.9
- peak_vram_mb: 3033.1
- mfu_percent: 10.37
- total_tokens_M: 36.7
- num_steps: 70

## Expected?
Yes - this matched the improvement seen on mar14 branch (0.671 vs baseline).

## Decision
- Keep

## Run Log
runs/2026-03-16_10-45_warmdown_ratio.log
