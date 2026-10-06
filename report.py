"""Publish a completed autoresearch run to the autoresearch workspace.

Reads the terminal output of `uv run train.py` (run.log), extracts the summary
block and the per-step loss curve, then uploads one experiment to the workspace
site (https://autoresearch.bolt.host) and attaches the run's files (model
checkpoint, tokenizer files, the raw log).

This script never edits train.py or prepare.py. It is run by the agent after
every experiment:

    uv run python report.py --name "shorter attention window" --hypothesis "..."

Credentials are read from the environment or from a gitignored .env file:

    AUTORESEARCH_API_KEY              ar_... agent key   (upload_run)
    AUTORESEARCH_SUPABASE_JWT         user access token  (file attachments)
    AUTORESEARCH_SUPABASE_REFRESH_TOKEN  optional, mints a new access token
    AUTORESEARCH_SUPABASE_EMAIL / _PASSWORD  optional, password grant instead

The agent key alone is enough to publish metrics, notes and the log. Binary
attachments additionally require a Supabase *user* token because the agent API
has no file-upload action; without one the run is still published and the
attachments are reported as skipped.
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import math
import mimetypes
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests

AGENT_API = "https://tjstztttrdyuwxzucheq.supabase.co/functions/v1/agent-api"
DEFAULT_SUPABASE_URL = "https://tjstztttrdyuwxzucheq.supabase.co"
STORAGE_BUCKET = "experiment-artifacts"
ARTIFACT_TABLE = "experiment_artifacts"


def project_ref(url):
    """The Supabase project id inside a project URL (ref.supabase.co)."""
    host = urlparse(url).hostname or ""
    return host.split(".")[0]


# Attachments must land in the same project that serves the agent API, because
# that is where the experiment row lives and experiment_artifacts has a foreign
# key to it. A key from any other project cannot work.
AGENT_PROJECT_REF = project_ref(AGENT_API)

# Workspace limits (see the API reference in the site's Agents tab).
MAX_TEXT_CHARS = 200_000
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
MAX_NAME_CHARS = 240
MAX_DESCRIPTION_CHARS = 4_000

# Experiment numbers 0-11 are already occupied by the sample rows seeded into
# the workspace, so the first real run of a fresh branch continues from 12.
FIRST_EXPERIMENT_NUMBER = 12

STEP_RE = re.compile(
    r"^step\s+(?P<step>\d+)\s+\((?P<pct>[\d.]+)%\)\s*\|\s*loss:\s*(?P<loss>[\d.]+|nan|inf)"
)
KV_RE = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9 _/().%+-]{0,60}):\s*(?P<value>.+)$")
SUMMARY_SEPARATOR = "---"
SKIP_KV_KEYS = {"model config"}

CHECKPOINT_NAME = "checkpoint_pre_eval.pt"
TOKENIZER_FILES = ("tokenizer.pkl", "token_bytes.pt", "dataset.txt")

REPO_ROOT = Path(__file__).resolve().parent
ENV_PATH = REPO_ROOT / ".env"
RESULTS_PATH = REPO_ROOT / "results.tsv"


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def load_dotenv(path=ENV_PATH):
    """Parse a .env file into os.environ without overwriting real env vars."""
    if not path.exists():
        return {}
    values = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not value:
            continue
        values[key] = value
        os.environ.setdefault(key, value)
    return values


def _post_json(url, payload, headers, timeout=120):
    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    if not response.ok:
        raise RuntimeError(f"POST {url} failed ({response.status_code}): {response.text[:500]}")
    return response.json()


def resolve_project(env):
    """Resolve the Supabase project to attach to, and check it is the right one.

    Returns (base_url, error). `error` is set when .env points at a project that
    is not the one serving the agent API, which would otherwise surface as a
    baffling "Bucket not found" per file.
    """
    base = (env.get("SUPABASE_URL") or env.get("AUTORESEARCH_SUPABASE_URL") or DEFAULT_SUPABASE_URL).rstrip("/")
    if not base.startswith("http"):
        base = f"https://{base}"
    ref = project_ref(base)
    if ref and ref != AGENT_PROJECT_REF:
        return base, (
            f"project mismatch: .env points at Supabase project '{ref}', but the workspace "
            f"site and its experiments live in '{AGENT_PROJECT_REF}'. Files attached with "
            f"'{ref}' keys can never reach this workspace (its experiment-artifacts bucket "
            f"does not exist there). Paste credentials from project '{AGENT_PROJECT_REF}'."
        )
    return base, None


def _static_token(env):
    """A credential that needs no refresh: an explicit JWT or a service/secret key."""
    for key in ("AUTORESEARCH_SUPABASE_JWT", "SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_KEY"):
        token = (env.get(key) or "").strip()
        if token:
            return token, key
    return None, None


def _is_expired(token):
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return False  # opaque secret keys have no readable exp; use as-is
    return claims.get("exp", 0) <= time.time() + 60


def mint_access_token(env, base_url):
    """Exchange the refresh token or email/password for a user access token."""
    refresh_token = env.get("AUTORESEARCH_SUPABASE_REFRESH_TOKEN")
    if refresh_token:
        return _post_json(
            f"{base_url}/auth/v1/token?grant_type=refresh_token",
            {"refresh_token": refresh_token},
            {"apikey": refresh_token, "Content-Type": "application/json"},
        )["access_token"]

    email = env.get("AUTORESEARCH_SUPABASE_EMAIL")
    password = env.get("AUTORESEARCH_SUPABASE_PASSWORD")
    if email and password:
        return _post_json(
            f"{base_url}/auth/v1/token?grant_type=password",
            {"email": email, "password": password},
            {"apikey": email, "Content-Type": "application/json"},
        )["access_token"]

    return None


def resolve_user_token(env, base_url):
    """Return a usable Supabase credential for attachments, or None."""
    token, _ = _static_token(env)
    if token and not _is_expired(token):
        return token
    return mint_access_token(env, base_url)


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------


def parse_run_log(path):
    """Split run.log into (summary, header metadata, per-step points)."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    chunks = [chunk.strip() for chunk in re.split(r"[\r\n]+", raw)]
    chunks = [chunk for chunk in chunks if chunk]

    separator_indexes = [i for i, chunk in enumerate(chunks) if chunk == SUMMARY_SEPARATOR]
    summary_start = separator_indexes[-1] + 1 if separator_indexes else len(chunks)
    summary_lines, header_lines = chunks[summary_start:], chunks[:summary_start]

    summary = {}
    for line in summary_lines:
        match = KV_RE.match(line)
        if match:
            summary[match.group("key").strip()] = match.group("value").strip()

    header = {}
    points = []
    for line in header_lines:
        step_match = STEP_RE.match(line)
        if step_match:
            try:
                points.append({"step": int(step_match.group("step")), "loss": float(step_match.group("loss"))})
            except ValueError:
                pass
            continue
        match = KV_RE.match(line)
        if match:
            key = match.group("key").strip()
            if key.lower() not in SKIP_KV_KEYS:
                header[key] = match.group("value").strip()

    return summary, header, points, raw


def summary_number(summary, key):
    value = summary.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def downsample(points, limit):
    if len(points) <= limit:
        return list(points)
    stride = len(points) / float(limit)
    return [points[int(i * stride)] for i in range(limit)]


def build_train_log(raw, points, limit=MAX_TEXT_CHARS):
    """Rebuild the log for the site's terminal panel, keeping it under the cap.

    Step lines arrive carriage-return separated, so they are re-emitted one per
    line and thinned out; the untrimmed log is attached to the run as a file.
    """
    kept_steps = {point["step"] for point in downsample(points, 120)}
    marker = f"... {len(points)} step lines thinned down to {len(kept_steps)} evenly spaced ones ..."
    output = []
    noted = False
    for chunk in re.split(r"[\r\n]+", raw):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = STEP_RE.match(chunk)
        if not match:
            output.append(chunk)
            continue
        step = int(match.group("step"))
        if step not in kept_steps and not noted:
            output.append(marker)
            noted = True
        if step in kept_steps:
            output.append(chunk)

    text = "\n".join(output)
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return head + "\n... log truncated ...\n" + tail


def build_chart_data(points, limit=120):
    """Non-finite losses are dropped: NaN is not valid JSON and the API would reject it."""
    chart = []
    for point in downsample(points, limit):
        if math.isfinite(point["loss"]):
            chart.append({"step": point["step"], "loss": round(point["loss"], 6)})
    return chart


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def git(*args, check=False):
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0 and check:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip() if result.returncode == 0 else ""


def commit_sha():
    return git("rev-parse", "--short", "HEAD") or "unknown"


def commit_diff(rev):
    """The committed change to train.py, which is the only file the agent edits."""
    if not rev or rev == "unknown":
        return ""
    stat = git("show", "--stat", "--format=", rev, "--", "train.py")
    body = git("show", "--format=", "--unified=3", rev, "--", "train.py")
    if not body:
        return stat
    return f"{stat}\n\n```diff\n{_cap(body, 60000)}\n```"


def working_tree_diff():
    body = git("diff", "--unified=3", "--", "train.py")
    if not body:
        return ""
    return f"_Uncommitted changes were still present when this run was reported._\n\n```diff\n{_cap(body, 60000)}\n```"


def _cap(text, limit):
    if len(text) <= limit:
        return text
    head = text[: int(limit * 0.7)]
    tail = text[-int(limit * 0.2) :]
    return head + f"\n... {len(text) - len(head) - len(tail)} characters elided ...\n" + tail


def short_description(text):
    """One plain line for the site's run list, taken from the markdown notes."""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "```", "|", ">")) or re.match(r"([-*+]\s|\d+\.\s)", line):
            continue
        line = re.sub(r"\s+", " ", line.replace("**", "")).strip()
        if line:
            return line[:MAX_DESCRIPTION_CHARS]
    return ""


# ---------------------------------------------------------------------------
# Markdown builders
# ---------------------------------------------------------------------------


def build_experiment_md(name, hypothesis, sha, diff, summary, metadata):
    lines = [f"# {name}", ""]
    if hypothesis:
        lines += ["## Hypothesis", "", hypothesis.strip(), ""]
    lines += ["## Change under test", "", f"Commit `{sha}`", ""]
    if diff:
        lines += [f"```diff\n{_cap(diff, MAX_TEXT_CHARS // 2)}\n```", ""]
    else:
        lines += ["_No committed diff available for this run._", ""]
    lines += ["## Run configuration", ""]
    for key in ("dataset", "depth", "num_params_M", "train_batch_size", "eval_batch_size",
                "total_tokens_M", "num_steps", "activation_checkpointing", "Time budget",
                "Gradient accumulation steps", "Muon compute dtype", "GPU", "GPU profile",
                "AMP dtype", "TF32", "Attention backend", "Vocab size"):
        if key in summary:
            lines.append(f"- **{key}**: {summary[key]}")
    for key, value in sorted(metadata.items()):
        if key not in summary:
            lines.append(f"- **{key}**: {value}")
    return _cap("\n".join(lines), MAX_TEXT_CHARS)


def build_results_md(name, metric, baseline, status, summary, verdict, extra=""):
    direction = "lower is better"
    lines = [f"# {name} — {status}", ""]
    if metric is None:
        lines += ["The run produced no `val_bpb`, so it is recorded as a failure.", ""]
    else:
        lines += [
            f"- **val_bpb**: {metric:.6f} ({direction})",
            f"- **status**: {status}",
        ]
        if baseline is not None:
            delta = metric - baseline
            verdict_word = "improvement" if delta < 0 else "regression"
            lines.append(
                f"- **vs baseline** {baseline:.6f}: {delta:+.6f} "
                f"({abs(delta) / baseline * 100:.2f}% {verdict_word})"
            )
    for key in ("training_seconds", "total_seconds", "peak_vram_mb", "mfu_percent",
                "total_tokens_M", "num_steps", "num_params_M", "depth"):
        if key in summary:
            lines.append(f"- **{key}**: {summary[key]}")
    if verdict:
        lines += ["", "## Verdict", "", verdict.strip()]
    if extra:
        lines += ["", extra.strip()]
    return _cap("\n".join(lines), MAX_TEXT_CHARS)


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def cache_dir():
    env_cache = os.environ.get("AUTORESEARCH_CACHE_DIR")
    if env_cache:
        return Path(os.path.expanduser(env_cache))
    legacy = Path.home() / ".cache" / "autoresearch"
    if legacy.exists() or os.name != "nt":
        return legacy
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "autoresearch"
    return legacy


def tokenizer_dir(dataset):
    """Resolve the tokenizer directory the same way prepare.py does."""
    root = cache_dir() / "datasets"
    candidates = []
    if dataset:
        candidates.append(dataset.strip().lower())
    active = root.parent / "active_dataset.txt"
    if active.exists():
        candidates.append(active.read_text(encoding="utf-8").strip().lower())
    candidates.append("tinystories")
    for name in candidates:
        # Only plain directory names are accepted; the value comes from a log file.
        if name and re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", name):
            return root / name / "tokenizer"
    return root / "tinystories" / "tokenizer"


def sanitize(name, limit=120):
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", name)[-limit:]
    return cleaned or "file"


def storage_path(experiment_id, kind, file_name):
    unique = uuid.uuid4().hex
    return f"experiments/{experiment_id}/{kind}-{unique}-{sanitize(file_name)}"


def attach_artifact(token, experiment_id, path, kind, base_url):
    path = Path(path)
    size = path.stat().st_size
    if size == 0:
        return {"skipped": f"{path.name} is empty"}
    if size > MAX_ARTIFACT_BYTES:
        return {"skipped": f"{path.name} is {size / 1024**3:.2f} GB (limit 2 GB)"}

    object_api = f"{base_url}/storage/v1/object/{STORAGE_BUCKET}"
    remote_path = storage_path(experiment_id, kind, path.name)
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    auth = {"Authorization": f"Bearer {token}", "apikey": token}

    with path.open("rb") as handle:
        response = requests.post(
            f"{object_api}/{remote_path}",
            data=handle,
            headers={**auth, "Content-Type": mime_type, "x-upsert": "false"},
            timeout=3600,
        )
    if not response.ok:
        hint = ""
        if "NoSuchBucket" in response.text or "Bucket not found" in response.text:
            hint = (
                f" — the '{STORAGE_BUCKET}' bucket does not exist in project "
                f"'{project_ref(base_url)}', so these credentials belong to a different "
                f"workspace than '{AGENT_PROJECT_REF}'."
            )
        return {"error": f"storage upload failed ({response.status_code}): {response.text[:200]}{hint}"}

    row = {
        "experiment_id": experiment_id,
        "kind": kind,
        "file_name": path.name,
        "storage_path": remote_path,
        "mime_type": mime_type,
        "size_bytes": size,
    }
    response = requests.post(
        f"{base_url}/rest/v1/{ARTIFACT_TABLE}",
        json=row,
        headers={**auth, "Content-Type": "application/json", "Prefer": "return=representation"},
        timeout=120,
    )
    if not response.ok:
        requests.delete(f"{object_api}/{remote_path}", headers=auth, timeout=120)
        return {"error": f"artifact row insert failed ({response.status_code}): {response.text[:200]}"}

    return {"file_name": path.name, "kind": kind, "size_bytes": size, "storage_path": remote_path}


def collect_artifacts(summary, log_path):
    """Every file that documents this run: model, tokenizer, and the raw log."""
    wanted = []
    checkpoint = Path(CHECKPOINT_NAME)
    if checkpoint.exists():
        wanted.append((checkpoint, "model"))
    tok_dir = tokenizer_dir(summary.get("dataset"))
    for file_name in TOKENIZER_FILES:
        candidate = tok_dir / file_name
        if candidate.exists():
            wanted.append((candidate, "tokenizer"))
    log_path = Path(log_path)
    if log_path.exists():
        wanted.append((log_path, "train_log"))
    return wanted


# ---------------------------------------------------------------------------
# results.tsv
# ---------------------------------------------------------------------------

RESULTS_HEADER = "commit\tval_bpb\tmemory_gb\tstatus\tdescription"
TSV_STATUS = {"kept": "keep", "discarded": "discard", "keep": "keep", "discard": "discard"}
KEEP_STATUSES = {"keep", "kept"}
PUBLISHED_STATUSES = KEEP_STATUSES | {"discard", "discarded"}


def read_results_tsv(path=RESULTS_PATH):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(line.split("\t"))
    return rows[1:] if rows and rows[0][0].strip() == "commit" else rows


def append_results_tsv(sha, metric, peak_vram_gb, status, description, path=RESULTS_PATH):
    path = Path(path)
    metric_text = f"{metric:.6f}" if metric is not None else "0.000000"
    memory_text = f"{peak_vram_gb:.1f}" if peak_vram_gb is not None else "0.0"
    tsv_status = TSV_STATUS.get(status, status)
    # Tabs and newlines would break the column layout.
    clean = re.sub(r"\s+", " ", str(description)).strip()
    row = f"{sha}\t{metric_text}\t{memory_text}\t{tsv_status}\t{clean}"
    if not path.exists():
        path.write_text(RESULTS_HEADER + "\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        if not path.read_text(encoding="utf-8").endswith("\n"):
            handle.write("\n")
        handle.write(row + "\n")


def next_experiment_number(rows):
    """Continue the workspace numbering after the seeded sample rows.

    results.tsv has no number column, so the count of published runs drives it.
    Crashes are recorded locally but never uploaded, so they do not consume a
    number; pass --number to pin one explicitly if an upload ever fails.
    """
    published = sum(1 for row in rows if len(row) >= 4 and row[3].strip().lower() in PUBLISHED_STATUSES)
    return FIRST_EXPERIMENT_NUMBER + published


def best_metric(rows):
    values = []
    for row in rows:
        if len(row) < 4 or row[3].strip().lower() not in KEEP_STATUSES:
            continue
        try:
            values.append(float(row[1]))
        except ValueError:
            continue
    return min(values) if values else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description="Publish an autoresearch run to the workspace.")
    parser.add_argument("--name", required=True, help="Short experiment name (<=240 chars).")
    parser.add_argument("--hypothesis", default="", help="Markdown hypothesis, stored as experiment.md.")
    parser.add_argument("--results", default="", help="Extra markdown appended to results.md.")
    parser.add_argument("--status", choices=("kept", "discarded"), default="kept")
    parser.add_argument("--number", type=int, default=None, help="experiment_number (default: next free).")
    parser.add_argument("--log", default="run.log", help="Path to the captured terminal output.")
    parser.add_argument("--files", dest="files", action="store_true", default=True,
                        help="Attach model, tokenizer and log files (default).")
    parser.add_argument("--no-files", dest="files", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Build and print the payload, upload nothing.")
    args = parser.parse_args(argv)

    env = {**load_dotenv(), **os.environ}
    api_key = env.get("AUTORESEARCH_API_KEY", "").strip()

    if not args.log or not Path(args.log).exists():
        print(f"FAIL: no log file at {args.log!r}; nothing to report.", file=sys.stderr)
        return 2

    summary, header, points, raw = parse_run_log(args.log)
    if not summary:
        print(f"FAIL: no summary block in {args.log}; the run did not finish.", file=sys.stderr)
        return 2

    metric = summary_number(summary, "val_bpb")
    peak_vram_mb = summary_number(summary, "peak_vram_mb")
    peak_vram_gb = round(peak_vram_mb / 1024, 1) if peak_vram_mb is not None else None
    duration = summary_number(summary, "total_seconds") or summary_number(summary, "training_seconds")
    sha = commit_sha()
    diff = working_tree_diff() or commit_diff(sha)

    rows = read_results_tsv()
    number = args.number if args.number is not None else next_experiment_number(rows)
    prior_best = best_metric(rows)
    baseline = prior_best if prior_best is not None else metric
    status = args.status
    if metric is None:
        print(f"FAIL: {args.log} has no val_bpb; recording a crash without uploading.", file=sys.stderr)
        append_results_tsv(sha, None, None, "crash", args.name)
        return 1

    metadata = dict(header)
    for key in summary:
        metadata.pop(key, None)
        metadata.pop(key.lower(), None)
        metadata.pop(key.title(), None)
    config_line = next((line for line in raw.splitlines() if line.startswith("Model config:")), "")
    if config_line:
        try:
            config = ast.literal_eval(config_line.split("Model config:", 1)[1].strip())
            metadata.update({key: str(value) for key, value in config.items()})
        except (ValueError, SyntaxError):
            metadata["model_config"] = _cap(config_line, 2000)

    if prior_best is None:
        verdict = f"Baseline run: this establishes the reference val_bpb of {metric:.6f}."
    elif metric < prior_best:
        verdict = f"Kept: new best val_bpb, {prior_best - metric:+.6f} against the previous {prior_best:.6f}."
    else:
        verdict = f"Discarded: {metric - prior_best:+.6f} against the best val_bpb of {prior_best:.6f}."

    run = {
        "experiment_number": number,
        "name": args.name[:MAX_NAME_CHARS],
        "metric": metric,
        "baseline_metric": baseline,
        "metric_label": "Validation BPP",
        "metric_direction": "lower",
        "status": status,
        "description": short_description(args.hypothesis) or args.name[:MAX_DESCRIPTION_CHARS],
        "experiment_md": build_experiment_md(args.name, args.hypothesis, sha, diff, summary, metadata),
        "results_md": build_results_md(args.name, metric, prior_best, status, summary, verdict, args.results),
        "train_log": build_train_log(raw, points),
        "metadata": {key: str(value) for key, value in metadata.items()},
        "chart_data": build_chart_data(points),
        "commit_sha": sha,
        "duration_seconds": int(duration) if duration else None,
    }
    run = {key: value for key, value in run.items() if value is not None}

    if args.dry_run:
        preview = dict(run)
        preview["train_log"] = f"<{len(run['train_log'])} chars>"
        print(json.dumps(preview, indent=2))
        print(f"\nresults.tsv row: {sha}\t{metric:.6f}\t{peak_vram_gb}\t{status}\t{args.name}")
        return 0

    if not api_key:
        print("FAIL: AUTORESEARCH_API_KEY is not set (add it to .env).", file=sys.stderr)
        return 2

    response = requests.post(
        AGENT_API,
        json={"action": "upload_run", "run": run},
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=180,
    )
    if not response.ok:
        print(f"FAIL: upload_run failed ({response.status_code}): {response.text[:500]}", file=sys.stderr)
        return 1
    experiment = response.json().get("experiment", {})
    experiment_id = experiment.get("id")
    print(f"Uploaded experiment #{number} ({experiment_id}): {args.name} val_bpb={metric:.6f} [{status}]")

    if args.files and not experiment_id:
        print("NOTE: the API returned no experiment id; skipping file attachments.")
    elif args.files:
        base_url, project_error = resolve_project(env)
        if project_error:
            print(f"NOTE: {project_error}")
            print("      Skipping file attachments; the run itself is published.")
        else:
            token = resolve_user_token(env, base_url)
            if not token:
                print("NOTE: no Supabase credential in .env for attachments; skipping files.")
            else:
                for path, kind in collect_artifacts(summary, args.log):
                    outcome = attach_artifact(token, experiment_id, path, kind, base_url)
                    if outcome.get("file_name"):
                        print(f"  attached {kind}/{outcome['file_name']} "
                              f"({outcome['size_bytes'] / 1024 / 1024:.1f} MB)")
                    elif outcome.get("skipped"):
                        print(f"  skipped {outcome['skipped']}")
                    else:
                        print(f"  FAILED {kind}/{path.name}: {outcome.get('error')}")

    append_results_tsv(sha, metric, peak_vram_gb, status, args.name)
    print(f"Recorded {sha}\t{metric:.6f}\t{peak_vram_gb}\t{status}\t{args.name} in results.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
