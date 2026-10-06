# ideas

Ledger of proposed experiments. One line per idea, appended by the agent. The point is
that an overnight loop must not propose the same thing twice, and must not forget what a
near-miss taught it.

Columns: date | idea | source | outcome

Outcome vocabulary: `pending` (proposed, not yet run), `kept #N` / `discarded #N` /
`crashed #N` where N is the experiment number on the workspace site.

Commit this file before any `git reset --hard`: the discard step in the loop throws away
uncommitted work, and the ledger is exactly the thing you cannot afford to lose.

| date | idea | source | outcome |
| --- | --- | --- | --- |
| 2026-10-06 | baseline, unmodified recipe | — | kept, experiment #1 (0.990164) |
| 2026-10-06 | autotune: 16 GB tier fix + geometry-keyed cache | from reading the baseline log, not a sub-agent | kept, experiment #2 (0.916360) |
| 2026-10-06 | `widen-steps-384`: ASPECT_RATIO 64→48, HEAD_DIM 128→96 (width 512→384) | arXiv:2505.20802, arXiv:2506.09342 | discarded, experiment #3 (1.060218, +15.7%) |
| 2026-10-06 | smaller `TOTAL_BATCH_SIZE` to buy optimizer steps without shrinking the model | — | pending — next round |
| 2026-10-06 | pin the eval batch so the scored val subset stops moving with the micro-batch | — | pending, bundle with the next run |

## Platform switch (2026-10-06)

Reporting moved from `autoresearch.bolt.host` to **autolabz.bolt.host** (Supabase project
`rrvalubtixdiecurxpqa`). Experiments #1-#3 above are the three runs that existed at the time,
replayed into the new site with their markdown, logs and loss curves. Two things to know:

- **The kept run's checkpoint (#2) is gone.** Every run overwrites
  `checkpoint_pre_eval.pt`, and the run after #2 was the discarded #3, so the only checkpoint
  on disk belongs to #3. Nothing was uploaded for #2 rather than mislabel #3's weights as
  the best model. From now on `report.py` uploads the checkpoint whenever an experiment is
  kept, so this cannot recur.
- **Objects are capped at 50 MB by the project plan**, not by the bucket (which allows
  5 GiB): plain POST and TUS both refuse above it. The ~96 MB checkpoint is uploaded as
  three ordered parts that concatenate back byte for byte.

## Measured regime (re-check these before theorising)

Everything below is measured on this box, not assumed. Read it before proposing anything.

- **~38 optimizer steps per 300 s run.** The loop is brutally short-horizon: any claim
  about "the run length" is a claim about 38 steps. #13 baseline 33 steps, #14 38,
  #15 45.
- **Wall clock is ~10 min per run, ~17 min when the geometry changed** (autotune re-probe
  ~7 min + 300 s training + ~4 min eval). Never kill a run at 10 minutes; ideation runs
  concurrently with exactly this window, which is the whole point of the pipeline.
- **Throughput, wide model (width 512, 50.3M):** 47,114 tok/s best (batch 4, activation
  checkpointing off). With checkpointing on the same batch gives 38,362 — **turning
  checkpointing off is worth ~26%**, far more than any batch size choice (46,102 at
  batch 16 vs 47,114 at batch 4 despite 3.4x the memory).
- **Throughput, narrow model (width 384, 33.0M):** 59,804 tok/s, +27% over the wide
  model. FLOPs/token drop 2.39e8 → ~1.5e8.
- **VRAM is not the constraint.** Peak is 2.9-3.5 GB of a 16 GB card even at batch 4. Do
  not spend runs on memory-headroom ideas.
- **Capacity is not free either.** Trading 34% of the parameters for 18% more steps cost
  0.144 val_bpb. At this token budget the model needs its width.
- **Steps are not obviously the constraint either** (#15). The next place to look is
  making each step cheaper in quality terms, or getting more steps *without* shrinking
  the model — e.g. a smaller `TOTAL_BATCH_SIZE`, which lives in `train.py` (2**19) and
  so is editable, unlike `prepare.py`'s constants.

## Rejected ideas and why

- **`adamw-beta2-horizon`** (ADAM_BETAS (0.8,0.95)→(0.8,0.99); arXiv:2508.01483,
  arXiv:2510.05491). Premise wrong at this horizon: it argued the run is 50-150 steps so
  β₂=0.95 (a ~20-step window) is far too short. Measured horizon is ~38 steps, so β₂=0.95
  already matches it, and β₂=0.99 (a ~100-step window) would average the second moment
  over 3x the entire run — mostly stale state, the opposite of the intent. Its throughput
  estimate (70-100k tok/s) was also 2x off the measured 47k. Revisit only if the step
  count rises by an order of magnitude.
- Still worth keeping from that proposal, because it is *not* step-count dependent:
  `train.py:729` keeps `exp_avg_sq` in bf16 for the embedding-family params (~2^-8
  relative granularity). Upcasting only the moments to fp32 costs ~84 MB and buys β₂
  headroom without bf16 rounding pressure.
- **`widen-steps-384` fallback** (ASPECT_RATIO 48 with HEAD_DIM 128, isolating the head
  count at the same 384 width): not worth a run. The width reduction is what hurt.
- **Eval subset moves with the micro-batch.** `evaluate_bpb` scores
  `eval_tokens // (batch_size * MAX_SEQ_LEN)` steps of best-fit-packed val documents, so
  batch 4 and batch 8 score the same 10,240 documents but pack and crop them differently:
  same distribution, not bit-identical, no bias toward easier documents. Setting
  `EVAL_BATCH_SIZE` does *not* pin it — `_build_eval_batch_candidates` starts from
  `min(EVAL_BATCH_SIZE, train_batch_size)`, so a train batch of 4 forces eval batch 4.
  To truly pin it you must also stop deriving eval batch from train batch. #14 and #15
  both ran at eval batch 4, so those two are directly comparable; #13 ran at eval batch 8.