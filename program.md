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
   - `report.py` — publishes finished runs to the research workspace. Do not modify; call it.
   - `ideas.md` — the ledger of ideas already proposed and their outcomes. Append to it.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Verify reporting credentials**: every run gets published to the research workspace at https://autolabz.bolt.host by `report.py`. Confirm `.env` exists and holds a non-empty `AUTOLABZ_API_TOKEN` (an `ar_live_...` token from Settings → API) **and** a non-empty `SUPABASE_SECRET_KEY` — the published API cannot create experiments, so the reporter writes those tables directly. That file is gitignored: never commit it, never echo a value into a run log, a commit message, or your own output. If either value is missing, stop and ask the human before running any experiments.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run. Note that experiment numbers 0-11 are already occupied by the sample rows that ship with the site, so this branch's first run is published as #12. `report.py` assigns the number itself — do not pass one unless an upload failed and you need to fill the gap.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Modify `report.py` or `results.tsv` by hand. Reporting is a scripted step, not a free-form one.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Reporting results

`report.py` is the single place results are recorded. It reads `run.log`, publishes the run to the workspace site, attaches the run's files, and appends the row to `results.tsv`. Never hand-write `results.tsv` rows and never hand-assemble API calls.

Report **every** finished run, kept or discarded — a discarded run is a real result and belongs in the record:

```
uv run python report.py --name "short attention window" --hypothesis "..." --status kept
uv run python report.py --name "GeLU activation" --hypothesis "..." --status discarded
```

- `--name` — short slug of what changed (max 240 chars).
- `--hypothesis` — markdown: what you expect and why. This becomes experiment.md together with the committed diff of `train.py` and the full run configuration.
- `--status` — `kept` or `discarded` (report.py writes `keep`/`discard` into the tsv; do not translate it yourself).
- `--results "..."` — optional extra markdown for results.md (interpretation, follow-ups).

What lands on the site for each run (the field names are the workspace's, see the API notes below):

| Target | Content |
| --- | --- |
| `experiments` | `name`, `status`, `score` (the `val_bpb` from the log), `delta` against the previous best, `duration_seconds`, `experiment_number` |
| `experiment.md` | hypothesis, committed diff of `train.py`, and every header/config line of the run |
| `results.md` | val_bpb, delta, verdict, throughput and VRAM numbers |
| `train.log` | the terminal output, with step lines thinned to 120 evenly spaced points |
| `metric_points` | per-step `train_loss` and `smoothed_loss`, thinned to 120 points — this is the site's loss curve |
| files | the tokenizer once per run, and `checkpoint_pre_eval.pt` when the experiment is kept (split into parts if it exceeds 50 MB) |

The full run configuration has no dedicated column here, so it is preserved inside `experiment.md` rather than being dropped. The raw log is thinned for the `train.log` artifact because that panel is text; the untrimmed log is what the agent still has on disk in `run.log`.

**Crashes are not uploaded.** A run without `val_bpb` has no metric, and publishing a fake one would poison the leaderboard. Still call `report.py` for it (with no `--status`) — it records a `crash` row in `results.tsv` and exits non-zero.

Use `--dry-run` to print the exact payload without uploading, and `--no-files` to publish metrics and notes without attaching the 200 MB checkpoint.

If attachments are skipped with "no Supabase user token", the run is still published — see Workspace API notes below.

## Ideation: sub-agents propose the experiments, pipelined against the GPU

You do not come up with the next idea alone, and you never do it while the GPU is idle. A run occupies the card for roughly 10 minutes of wall clock (autotune probing, then the 300 s training budget, then eval). That is exactly the window in which the next round's research happens.

**The pipeline: launch the run first, then spawn the sub-agents while it trains.**

1. You have a pair of ready ideas in hand (from the previous round).
2. You pick one, edit `train.py`, commit, and **launch training in the background**.
3. **In the same turn, spawn the two research sub-agents for the *next* round.** Two Task calls in one message, so they run concurrently with the training.
4. Training finishes. You read the result, report it, keep or revert.
5. The next round's proposals are already waiting. Pick one and launch immediately — no dead time on the card.

So at every moment there is one run training and one round of ideas in the air. Never let the GPU wait on a sub-agent, and never let a sub-agent's web search happen while the card is idle waiting for you.

Each round spawns **two sub-agents in parallel** (Task tool, `general` subagent type), each returning one experiment. Divergence is the entire point, so the two briefs must be disjoint — the same brief twice produces the same idea twice.

Rotate the briefs so consecutive rounds do not orbit the same subsystem. Assign one sub-agent the **next domain in the rotation** and the other the domain after it:

| Rotation | Domain A | Domain B |
| --- | --- | --- |
| 1 | architecture: attention, windowing, positional encoding, residual/norm structure | optimization: LR schedules, warmup/warmdown, Muon/AdamW mix, betas, weight decay |
| 2 | initialization, scaling, muP-style balancing, depth/width ratio | regularization, dropout, data augmentation-free regularization, loss shaping |
| 3 | tokenizer-free efficiency: batch size, grad accumulation, activation checkpointing, fused paths | loss function, auxiliary objectives, value embeddings, prediction heads |
| 4 | anything left, re-weighted by what results.tsv has not covered yet | the domain with the largest unexplained gap |

Give each sub-agent the **measured regime from `ideas.md`**, not just the task. A proposal that ignores what this box has actually measured (step count, throughput, VRAM, which theses were refuted) is a wasted run. Also tell it which rotation domains the *other* agent owns, so the two cannot collide.

Each sub-agent must:

1. **Web-search recent arXiv work** (2025-2026 preferred) in ML/AI. Cite at least two papers by title and arXiv id, and state the concrete finding you are borrowing — not a vibe, a number or a mechanism.
2. **Read `train.py`** to see what the code already does, and **read `results.tsv`** and **`ideas.md`** so it does not propose something already tried or already refuted here.
3. Return **exactly one** idea, small enough to be a focused diff in `train.py`, with:
   - the paper(s) it comes from and the mechanism being borrowed,
   - the concrete edit: constants, functions, line references, and the new value,
   - the expected effect on `val_bpb` and roughly how big,
   - the risks: VRAM, wall-clock, numerical stability, likelihood of crashing,
   - a one-line fallback if it OOMs (the next smaller setting to try).
4. Respect the hard constraints: `train.py` only, no new dependencies, no new data, `prepare.py` and `evaluate_bpb` untouched, must finish inside the 5-minute budget.

Then **you pick one**. Choose on expected gain per unit of risk, sanity-check that it is not a near-duplicate of an earlier run, and write one sentence saying which you picked and why the other lost. Then implement only that one.

**Keep the ready queue at one round deep.** Two pending ideas, no more. A deep backlog goes stale: the run in flight changes what looks promising, and a proposal written against the wrong baseline is worse than a fresh one. When you consume a round, spawn its replacement in the same turn as the launch.

**Reuse beats re-ideating.** Keep the runner-up idea from the round. If the idea you picked crashes or is reverted, try the runner-up before paying for another round of sub-agents.

**Ledger**: append one line per proposal to `ideas.md` — date, one-line name, source paper, and outcome (`pending` / `kept #N` / `discarded #N` / `crashed #N`). This is how an overnight loop avoids circling the same idea forever.

## Running an experiment in the background

Training blocks for about ten minutes, so launch it detached and get on with the ideation. Redirect all output — do NOT use `tee` or let it flood your context.

PowerShell (this fork's platform):

```powershell
$env:UV_LINK_MODE='copy'   # hardlink fallback; harmless elsewhere
Start-Process -FilePath 'cmd.exe' `
  -ArgumentList '/c','uv run --frozen train.py > run.log 2>&1' `
  -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -PassThru
```

POSIX:

```bash
nohup uv run --frozen train.py > run.log 2>&1 &
```

Use `--frozen` so `uv` never rewrites `uv.lock` mid-loop. Nothing in this repo changes dependencies, so the committed lockfile is always the right one.

**Do not poll the log for progress.** Python block-buffers stdout when it is redirected to a file, so `run.log` stays empty until the process exits — an empty log means "still running", not "crashed". Wait for the process to exit, then read the log once. A run takes roughly 10 minutes end to end (autotune probe + 300 s training + eval), so poll the process at ~60 s intervals rather than sleeping blindly for a fixed time.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

The loop is a pipeline, not a sequence. The two invariants:

- **The GPU is never idle.** Every launch is immediately followed by the next round's ideation.
- **You never wait on a sub-agent while the card is idle.** Ideation happens during training.

```
STARTUP (once): run the ideation stage yourself, with no run in flight.
                Two sub-agents, two disjoint domains. Keep the better idea
                as the first experiment and the other as the ready runner-up.

LOOP FOREVER:

 1. Read the state: current branch/commit, `results.tsv` for the best val_bpb,
    `ideas.md` for the ready queue.
 2. Pick the next experiment from the ready queue. If the queue is empty, run
    the ideation stage now and accept the wait — this should only happen once,
    at startup, or after a run crashed and both ideas were consumed.
 3. Write down your pick in one sentence: which idea, and why the other lost.
 4. Tune `train.py` with that idea by directly hacking the code, then git commit.
 5. LAUNCH the run in the background (see "Running an experiment in the
    background" above). Note the process id and the start time.
 6. IN THE SAME TURN, spawn the two research sub-agents for the round after
    this one — two Task calls in one message, briefs from the next rotation,
    each carrying the measured regime from `ideas.md` and told which domain the
    other agent owns. This is the step that keeps the card busy; do it before
    anything else.
 7. Wait for the run to finish (poll the process, ~60 s apart; the log stays
    empty until exit because Python block-buffers).
 8. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
 9. If the grep output is empty, the run crashed. Read the last 50 lines of
    run.log for the Python stack trace and attempt a fix. If you can't get
    things to work after more than a few attempts, give up and take the
    runner-up idea instead of paying for new ideation.
10. Decide keep or discard against the best `val_bpb` in `results.tsv`, then
    publish the run exactly once:
    - improved or equal-and-simpler: `uv run python report.py --name "<slug>" --hypothesis "<what and why>" --status kept`
    - worse: `uv run python report.py --name "<slug>" --hypothesis "<what and why>" --status discarded`
    - crashed: `uv run python report.py --name "<slug>" --hypothesis "<what and why>"` (records the crash, uploads nothing)

    **Report before you reset.** `report.py` captures the commit hash and its diff of `train.py`, so the experiment commit must still be HEAD when it runs.
11. Mark the outcome in the `ideas.md` ledger with its run number, and **commit the ledger before any `git reset`** — step 12 throws away uncommitted work
12. If val_bpb improved (lower), you "advance" the branch, keeping the git commit. If equal or worse, `git reset` back to where you started.
13. Go to step 1. The next round's proposals are already waiting, and the card is free.
```

Step 13 must not have a gap. The moment the run exits and is reported, the next experiment goes in — including when the result was a discard, and including when the sub-agents' proposals need ten minutes of your own reading before you commit to one.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: the 5-minute budget applies to *training only*, and `training_seconds` in the summary should land near 300. Wall clock is much longer, and that is normal:

| phase | wall clock |
| --- | --- |
| autotune probe (only when the model geometry changed) | up to ~7 min |
| training | 300 s by design |
| eval | ~4 min |
| total, cached autotune decision | **~10 min** (measured: 581 s for #14) |
| total, geometry changed so autotune re-probed | **~17 min** (measured: ~17 min for #15) |

So do not kill a run at 10 minutes — that would kill most good ones. Kill only if wall clock passes ~25 minutes, or if `training_seconds` is wildly above 300 with no progress in the step lines. A run that OOMs or dies exits on its own long before that.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, report the crash and move on.

**Reporting failures are not experiment failures**: if `report.py` fails (network, 401, 5xx), the training result is still valid. Fix the reporting, then re-run `report.py` with the same arguments and the same `run.log` — do not retrain, and do not record the run twice. Check `results.tsv` before re-reporting so you neither duplicate a row nor skip a number.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!


## Workspace API notes

Everything below is already implemented in `report.py`. Read this when reporting breaks, when
you need to know why, or if you ever need to talk to the API directly. Do not bypass
`report.py` for routine runs.

**Site**: https://autolabz.bolt.host — Supabase project `rrvalubtixdiecurxpqa`.

**Published API** — `POST|GET {SUPABASE_URL}/functions/v1/api/...`, JSON, auth is
`Authorization: Bearer <AUTOLABZ_API_TOKEN>` (an `ar_live_…` token from Settings → API).

| Route | Used for |
| --- | --- |
| `GET /runs` | find the existing run by name |
| `POST /runs` | create the run: `{name, repo_url, baseline_score}` |
| `GET /runs/:id/experiments` | verify what was published |
| `GET /experiments/:id` | verify one experiment |
| `GET /experiments/:id/metrics` | verify the loss curve |
| `GET|POST /blogs` | not used by the loop |

**The gap that shapes the design**: the published API cannot create experiments, metric
points, artifacts or files. `POST /experiments` answers `400 Experiment ID is required` no
matter what you send (body keys, query params and headers were all tried), and every other
write route 404s. So `report.py` writes those four tables directly with
`SUPABASE_SECRET_KEY`, which is a service credential that bypasses RLS. The `ar_live_` token
still covers run creation and every read.

**What gets written where** (verified end to end against this project):

| Target | Where | Fields |
| --- | --- | --- |
| experiment | `experiments` | `id` (client-generated uuid), `run_id`, `experiment_number` (from 1), `name`, `status` (`kept`/`discarded`), `score` (= val_bpb), `delta` (signed, negative is better), `duration_seconds` |
| loss curve | `metric_points` | `experiment_id`, `step`, `train_loss`, `smoothed_loss` |
| notes | `experiment_artifacts` | `artifact_type` is exactly one of `experiment.md`, `results.md`, `train.log`; `content` is the text |
| files | `autoresearch-files` bucket + `run_files` | `run_id`, `file_kind` (`model`/`tokenizer` only), `file_name`, `storage_path`, `size_bytes`, `mime_type`, and `user_id` (NOT NULL — copy it from the run) |
| run counters | `research_runs` | `total_experiments`, `kept_improvements`, `best_score`, `status` |

`experiments` has no `metadata`, `chart_data` or `description` column: the per-run
configuration is preserved inside `experiment.md`, and the curve lives in `metric_points`.
`gradient_norm`, `learning_rate` and `eval_loss` exist on `metric_points` but the training
log does not print them, so they are left null rather than filled with a different quantity.

**File policy** — files belong to the *run*, not to one experiment, so uploading on every
experiment would duplicate the identical tokenizer. `report.py` uploads the tokenizer files
once per run, and the checkpoint only when an experiment is kept. The tokenizer never
changes during a run, so one copy is exact for every experiment.

**The 50 MB object cap is a plan limit, not a request limit.** The bucket allows 5 GiB, but
a plain POST is refused above 50 MiB (48 MiB passes, 52 MiB gets `413 Payload too large`)
and the TUS resumable endpoint refuses exactly the same sizes. `checkpoint_pre_eval.pt` is
~96 MiB, so `report.py` splits anything above the cap into ordered parts named
`NAME.part01-of-03`, `NAME.part02-of-03`, … which concatenate back byte for byte (verified
by round-trip). `cat NAME.part* > NAME` restores the original. Only then is the `run_files`
row written, and a failed row insert deletes the uploaded object so no orphans are left.

**Idempotence** — `ensure_run` reuses the run with the matching name instead of forking a
new one, and `results.tsv` already carries the commit sha, so re-reporting the same run is
a no-op rather than a duplicate experiment.

**results.tsv** has six columns:

```
commit	val_bpb	memory_gb	status	description	experiment
```

The last column is the `experiment_number` published to the workspace. It exists because
the workspace numbers experiments per run from 1, and a crash is recorded with an *empty*
number (it is never published) — so counting rows would drift. `status` is `keep`,
`discard` or `crash`; the agent passes `kept`/`discarded` to `report.py` and it translates.

**Troubleshooting**

| Symptom | Meaning |
| --- | --- |
| `401 Invalid or expired API token` | `AUTOLABZ_API_TOKEN` missing, wrong or revoked |
| `403 Invalid Compact JWS` | the `ar_live_` token was used for a direct table write; that path needs `SUPABASE_SECRET_KEY` |
| `Experiment ID is required` | the published API still cannot create experiments; that is expected, `report.py` writes the table directly |
| `413 Payload too large` on upload | object over the 50 MB cap; `report.py` should have split it — if this appears anyway, the split path failed and the error names the part |
| `null` in a `run_files` insert | `user_id` was not passed; it must match the run's owner |
| `Bucket not found` | `.env` points at a different Supabase project than the site |