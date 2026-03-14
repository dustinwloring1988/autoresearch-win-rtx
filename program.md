# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Startup: Create Daily Working Branch

At the very start of the session, create a **daily working branch** for the entire night:

```bash
git checkout master
git pull origin master
git checkout -b daily/YYYY-MM-DD
git push -u origin daily/YYYY-MM-DD
```

Example (for March 14, 2026):
```bash
git checkout master
git pull origin master
git checkout -b daily/2026-03-14
git push -u origin daily/2026-03-14
```

All experiment work during this session occurs on this daily branch. The branch format must be `daily/YYYY-MM-DD`.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 15 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 15 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 900.1
total_seconds:    925.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   1498.6
num_steps:        2859
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 15 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Experiment Documentation System

After every experiment, you **MUST** create a new markdown file in the `reads/` folder.

### Filename Format

```
reads/YYYY-MM-DD_HH-MM_experiment.md
```

Example:
```
reads/2026-03-14_14-30_increase_lr.md
```

### File Content Template

Each experiment summary file must contain:

```markdown
# Experiment Summary

## Hypothesis
What change was being tested and why.

## Change Made
Exact code modification or idea attempted.

## Result
Metrics from the run (loss, val_bpb, etc).

## Expected?
Did the result match expectations? Why or why not?

## Decision
- Keep
- Reject
- Needs follow-up
```

### Critical: Read Previous Experiments

Before proposing any new experiment, you **MUST** read previous experiment files in `reads/`. This allows you to:
- Avoid repeating failed ideas
- Build on successful approaches
- Learn from previous decisions

The experiment workflow is:
1. Read `reads/*` summaries
2. Decide next hypothesis based on past results
3. Modify code
4. Run experiment (15 minutes)
5. Record result in `reads/`
6. Commit changes
7. Push to daily branch

## Logging Results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The Experiment Loop

The experiment runs on the daily branch (e.g. `daily/2026-03-14`).

LOOP FOREVER:

1. **Read** previous experiments: read files in `reads/`
2. **Decide** next hypothesis based on past results
3. **Modify** `train.py` with your experimental idea
4. **Commit** changes: `git add . && git commit -m "experiment: <description>"`
5. **Run** the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
6. **Record** results: create new file in `reads/` with experiment summary
7. **Log** results to `results.tsv`
8. **Push** to daily branch: `git push`
9. **Evaluate**: If val_bpb improved (lower), you "advance" the branch, keeping the git commit. If val_bpb is equal or worse, you git reset back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~15 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 25 minutes, kill it and treat it as a failure (discard and revert).

**Command Timeout Safety**: Never use a command timeout shorter than 25 minutes (1500 seconds). This ensures experiments and training runs are not prematurely killed.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**ALWAYS Commit Reads Folder**: The `reads/` folder is critical. Every experiment must generate a new log file in this folder. Even if the experiment fails, the log must still be committed. This ensures GitHub keeps a complete research history.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~15 minutes then you can run approx 4/hour, for a total of about 32 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Nightly Merge to Master

At the **end of the nightly session**, merge the daily branch into the real `master`:

```bash
git checkout master
git pull origin master
git merge daily/YYYY-MM-DD
git push origin master
```

This ensures the master branch is only modified once per night.

## Summary of Workflow

### Startup:
1. Checkout master and pull latest
2. Create daily branch: `git checkout -b daily/YYYY-MM-DD`
3. Push to remote: `git push -u origin daily/YYYY-MM-DD`

### Loop:
1. Read `reads/*` to learn from past experiments
2. Decide next hypothesis
3. Modify `train.py`
4. Commit: `git add . && git commit -m "experiment: <description>"`
5. Run experiment (15 minutes): `uv run train.py > run.log 2>&1`
6. Record results in `reads/YYYY-MM-DD_HH-MM_experiment.md`
7. Log to `results.tsv`
8. Push: `git push`

### Shutdown:
1. Merge to master: `git checkout master && git merge daily/YYYY-MM-DD && git push origin master`

(End of file - total 219 lines)
