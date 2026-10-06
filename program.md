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
5. **Verify reporting credentials**: every run gets published to the research workspace at https://autoresearch.bolt.host by `report.py`. Confirm `.env` exists and holds a non-empty `AUTORESEARCH_API_KEY`. That file is gitignored: never commit it, never echo the value into a run log, a commit message, or your own output. If the file is missing or empty, stop and ask the human for the key (it is created in Settings → Models & keys on the site) before running any experiments.
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

What lands on the site for each run:

| Field | Content |
| --- | --- |
| `metric` | `val_bpb` from the log, with the best val_bpb so far as `baseline_metric` |
| `metric_label` / `metric_direction` | `Validation BPP` / `lower` |
| `experiment_md` | hypothesis, committed diff, every header/config line of the run |
| `results_md` | val_bpb, delta vs baseline, verdict, throughput and VRAM numbers |
| `train_log` | the terminal output, with step lines thinned to 120 evenly spaced points |
| `chart_data` | per-step training loss, thinned to 120 points |
| `metadata` | batch sizes, depth, optimizer dtypes, GPU profile, model config, parameter counts |
| files | `checkpoint_pre_eval.pt` as the model, the tokenizer files as the tokenizer, `run.log` as the log |

The full, untrimmed log is attached as a file; the `train_log` field is the thinned, readable version for the site's terminal panel.

**Crashes are not uploaded.** A run without `val_bpb` has no metric, and publishing a fake one would poison the leaderboard. Still call `report.py` for it (with no `--status`) — it records a `crash` row in `results.tsv` and exits non-zero.

Use `--dry-run` to print the exact payload without uploading, and `--no-files` to publish metrics and notes without attaching the 200 MB checkpoint.

If attachments are skipped with "no Supabase user token", the run is still published — see Workspace API notes below.

## Ideation: sub-agents propose the experiments

You do not come up with the next idea alone. Once a run is reported and you know what the current best is, spawn **two sub-agents in parallel** (Task tool, `general` subagent type) and let each propose one experiment. Divergence is the entire point, so the two briefs must be disjoint — the same brief twice produces the same idea twice.

Rotate the briefs so consecutive rounds do not orbit the same subsystem. Assign one sub-agent the **next domain in the rotation** and the other the domain after it:

| Rotation | Domain A | Domain B |
| --- | --- | --- |
| 1 | architecture: attention, windowing, positional encoding, residual/norm structure | optimization: LR schedules, warmup/warmdown, Muon/AdamW mix, betas, weight decay |
| 2 | initialization, scaling, muP-style balancing, depth/width ratio | regularization, dropout, data augmentation-free regularization, loss shaping |
| 3 | tokenizer-free efficiency: batch size, grad accumulation, activation checkpointing, fused paths | loss function, auxiliary objectives, value embeddings, prediction heads |
| 4 | anything left, re-weighted by what results.tsv has not covered yet | the domain with the largest unexplained gap |

Each sub-agent must:

1. **Web-search recent arXiv work** (2025-2026 preferred) in ML/AI. Cite at least two papers by title and arXiv id, and state the concrete finding you are borrowing — not a vibe, a number or a mechanism.
2. **Read `train.py`** to see what the code already does, and **read `results.tsv`** so it does not propose something already tried here.
3. **Check the ledger in `ideas.md`** and not re-propose anything listed there.
4. Return **exactly one** idea, small enough to be a focused diff in `train.py`, with:
   - the paper(s) it comes from and the mechanism being borrowed,
   - the concrete edit: constants, functions, line references, and the new value,
   - the expected effect on `val_bpb` and roughly how big,
   - the risks: VRAM, wall-clock, numerical stability, likelihood of crashing,
   - a one-line fallback if it OOMs (the next smaller setting to try).
5. Respect the hard constraints: `train.py` only, no new dependencies, no new data, `prepare.py` and `evaluate_bpb` untouched, must finish inside the 5-minute budget.

Then **you pick one**. Choose on expected gain per unit of risk, sanity-check that it is not a near-duplicate of an earlier run, and write one sentence saying which you picked and why the other lost. Then implement only that one.

**Reuse beats re-ideating.** Keep the runner-up idea from the round. If the idea you picked crashes or is reverted, try the runner-up before paying for another round of sub-agents.

**Ledger**: append one line per proposal to `ideas.md` — date, one-line name, source paper, and outcome (`pending` / `kept #N` / `discarded #N` / `crashed #N`). This is how an overnight loop avoids circling the same idea forever.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on, and `results.tsv` for the best val_bpb so far
2. Pick the next experiment: run the ideation stage above (two sub-agents, two disjoint domains) unless you still have an untried runner-up from the last round
3. Tune `train.py` with that idea by directly hacking the code.
4. git commit
5. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
6. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
8. Decide keep or discard against the best `val_bpb` in `results.tsv` (the local record of what has been published), then publish the run exactly once:
   - improved or equal-and-simpler: `uv run python report.py --name "<slug>" --hypothesis "<what and why>" --status kept`
   - worse: `uv run python report.py --name "<slug>" --hypothesis "<what and why>" --status discarded`
   - crashed: `uv run python report.py --name "<slug>" --hypothesis "<what and why>"` (records the crash, uploads nothing)

   **Report before you reset.** `report.py` captures the commit hash and its diff of `train.py`, so the experiment commit must still be HEAD when it runs.
9. Mark the outcome of this idea in the `ideas.md` ledger with its run number, and **commit the ledger before any `git reset`** — the discard step below throws away uncommitted work
10. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
11. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, report the crash and move on.

**Reporting failures are not experiment failures**: if `report.py` fails (network, 401, 5xx), the training result is still valid. Fix the reporting, then re-run `report.py` with the same arguments and the same `run.log` — do not retrain, and do not record the run twice. Check `results.tsv` before re-reporting so you neither duplicate a row nor skip a number.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Workspace API notes

Everything below is already implemented in `report.py`. Read this section when reporting breaks, when you need to know why, or if you ever need to talk to the API directly. Do not bypass `report.py` for routine runs.

**Endpoint** — `POST https://tjstztttrdyuwxzucheq.supabase.co/functions/v1/agent-api`, JSON body, action in the `action` field. Auth is `Authorization: Bearer <AUTORESEARCH_API_KEY>`.

**Actions used** — `upload_run` only. The remaining actions are deliberately unused:

- `search` and `index_documents` need a 384-dimension embedding vector and no embedding model is available offline (installing one is not allowed). Research memory is instead `results.tsv` plus the runs page on the site, so lean on those.
- `create_key`, `list_keys`, `revoke_key`, `reroll_key` require a Supabase **sign-in** JWT, not the `ar_` agent key, and they manage credentials. Only the human does this, in Settings → Models & keys.

**Limits** (enforced by the server, `report.py` stays under them):

| Field | Limit |
| --- | --- |
| `name` | 240 chars |
| `description` | 4000 chars |
| `experiment_md`, `results_md`, `train_log` | 200000 chars each |
| attached file | 2 GB each |

**File attachments** — the agent API has no upload action. The site stores files in the Supabase `experiment-artifacts` bucket and records a row per file in `experiment_artifacts` (`experiment_id`, `kind` of `model`/`tokenizer`/`train_log`, `file_name`, `storage_path`, `mime_type`, `size_bytes`). That path requires a Supabase **user** token; the `ar_` agent key is rejected there with `Invalid Compact JWS`. So `.env` should also carry one of:

- `AUTORESEARCH_SUPABASE_JWT` + `AUTORESEARCH_SUPABASE_REFRESH_TOKEN` (from Settings → Models & keys)
- or `AUTORESEARCH_SUPABASE_EMAIL` + `AUTORESEARCH_SUPABASE_PASSWORD`, which `report.py` exchanges for a fresh token before attaching

Access tokens expire after about an hour; `report.py` re-mints one from the refresh token or password without being asked, which is what lets an overnight loop keep attaching files. With none of these set, runs are still published — metrics, notes, and the thinned log — and only the file attachments are skipped with a note.

**Storage cost** — `checkpoint_pre_eval.pt` is roughly 200 MB and is attached on every run by default. That is about 2 GB per 10 experiments. `--no-files` publishes a run without files if the human wants to conserve space.

**Troubleshooting**

| Symptom | Meaning |
| --- | --- |
| `401` from agent-api | `AUTORESEARCH_API_KEY` is missing, wrong, or revoked |
| `400 A run name, metric, and integer experiment number are required` | the payload was malformed — a `report.py` bug, report it rather than working around it |
| `no Supabase user token` | metrics and notes are published, files are not; the human needs to fill in the token fields |
| `Invalid Compact JWS` during attach | the `ar_` key was used where a user JWT is required; check `.env` |
| `Bucket not found` | storage misconfiguration on the site, not something the agent can fix |

**Secrets discipline** — `.env` is gitignored. Never commit it, never print the key or token, never include them in `--hypothesis`/`--results` text, and never let them reach `run.log` (which is uploaded).

