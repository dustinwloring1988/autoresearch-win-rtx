"""Publish a completed autoresearch run to the autolabz workspace.

Reads the terminal output of `uv run train.py` (run.log), extracts the summary
block and the per-step loss curve, then publishes one experiment to the
workspace site (https://autolabz.bolt.host) with its notes, log and loss
curve, and attaches the run's model and tokenizer files.

This script never edits train.py or prepare.py. It is run by the agent after
every experiment:

    uv run python report.py --name "shorter attention window" --hypothesis "..."

Credentials, from the environment or a gitignored .env:

    AUTOLABZ_API_TOKEN            ar_live_...   agent token (Settings -> API)
    SUPABASE_URL                  project URL that serves the workspace
    SUPABASE_SECRET_KEY           service/secret key, for direct table writes

The published agent API (POST /runs, plus reads) is thin: it can create a run
and blog posts, but there is no route for creating experiments, metric points,
artifacts or files. Those live in Supabase tables that only accept a service
credential, so experiments are written directly to `experiments`,
`metric_points` and `experiment_artifacts`, and files go to the
`autoresearch-files` bucket plus the `run_files` table. See program.md for the
full field contract.
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
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests

# Workspace API (Supabase edge function). Paths from the site's Agents tab.
DEFAULT_SUPABASE_URL = "https://rrvalubtixdiecurxpqa.supabase.co"
API_SUFFIX = "/functions/v1/api"

# Direct table + storage access for what the API cannot write.
STORAGE_BUCKET = "autoresearch-files"
MAX_ARTIFACT_BYTES = 5 * 1024 * 1024 * 1024

# Storage limits, measured against this project rather than assumed: a plain POST
# is refused above 50 MiB (48 MiB passes, 52 MiB gets 413) and the TUS resumable
# endpoint refuses the same sizes, so the ceiling is the project plan, not the
# request path -- the bucket itself allows 5 GiB. Anything above the ceiling is
# therefore split into ordered parts that concatenate back into the original.
MAX_OBJECT_BYTES = 48 * 1024 * 1024
PLAIN_UPLOAD_LIMIT = 48 * 1024 * 1024
SPLIT_PART_BYTES = 40 * 1024 * 1024
TUS_CHUNK = 6 * 1024 * 1024
TUS_VERSION = "1.0.0"

# artifact_type vocabulary, read by the site's per-experiment tabs.
ARTIFACT_EXPERIMENT = "experiment.md"
ARTIFACT_RESULTS = "results.md"
ARTIFACT_TRAIN_LOG = "train.log"
ARTIFACT_TYPES = (ARTIFACT_EXPERIMENT, ARTIFACT_RESULTS, ARTIFACT_TRAIN_LOG)

# file_kind vocabulary: the site only knows these two.
FILE_KIND_MODEL = "model"
FILE_KIND_TOKENIZER = "tokenizer"

# Keep the payload small; the site draws a line chart and a log panel.
MAX_CHART_POINTS = 120
MAX_LOG_CHARS = 200_000
MAX_NAME_CHARS = 240
MAX_DESCRIPTION_CHARS = 4_000

EXPERIMENT_STATUS_KEPT = "kept"
EXPERIMENT_STATUS_DISCARDED = "discarded"

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
REPO_URL = "https://github.com/dustinwloring1988/autoresearch-win-rtx"


# ---------------------------------------------------------------------------
# Workspace client
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


def _tus_metadata(**pairs):
    return ",".join(f"{key} {base64.b64encode(str(value).encode()).decode()}" for key, value in pairs.items())


class Workspace:
    """The autolabz workspace: agent API for runs, direct tables for the rest."""

    def __init__(self, env):
        self.base_url = (env.get("SUPABASE_URL") or DEFAULT_SUPABASE_URL).rstrip("/")
        self.api = f"{self.base_url}{API_SUFFIX}"
        self.token = (
            env.get("AUTOLABZ_API_TOKEN")
            or env.get("AUTORESEARCH_API_KEY")
            or env.get("AUTOLABZ_TOKEN")
            or ""
        ).strip()
        self.secret = (
            env.get("SUPABASE_SECRET_KEY") or env.get("SUPABASE_SERVICE_KEY") or ""
        ).strip()
        self.missing = []
        if not self.token:
            self.missing.append("AUTOLABZ_API_TOKEN (ar_live_..., Settings -> API)")
        if not self.secret:
            self.missing.append("SUPABASE_SECRET_KEY (needed to write experiments)")
        self.agent = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        self.service = {"apikey": self.secret, "Authorization": f"Bearer {self.secret}",
                        "Content-Type": "application/json"}

    # -- agent API ----------------------------------------------------------

    def get_runs(self):
        response = requests.get(f"{self.api}/runs", headers=self.agent, timeout=60)
        if not response.ok:
            raise RuntimeError(f"GET /runs failed ({response.status_code}): {response.text[:300]}")
        return response.json().get("runs", [])

    def create_run(self, name, repo_url, baseline_score):
        response = requests.post(
            f"{self.api}/runs",
            headers=self.agent,
            json={"name": name, "repo_url": repo_url, "baseline_score": baseline_score},
            timeout=60,
        )
        if not response.ok:
            raise RuntimeError(f"POST /runs failed ({response.status_code}): {response.text[:300]}")
        return response.json().get("run", {})

    def ensure_run(self, name, repo_url, baseline_score):
        """Reuse the run with this name if it exists, so the loop never forks a new one."""
        for run in self.get_runs():
            if run.get("name") == name:
                return run, False
        return self.create_run(name, repo_url, baseline_score), True

    # -- direct table access ------------------------------------------------

    def _table(self, table, method="GET", rows=None, params=None, prefer_return=False):
        headers = dict(self.service)
        if prefer_return:
            headers["Prefer"] = "return=representation"
        url = f"{self.base_url}/rest/v1/{table}"
        if method == "GET":
            response = requests.get(url, headers=headers, params=params or {}, timeout=120)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=rows, timeout=300)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, params=params or {}, timeout=120)
        elif method == "PATCH":
            response = requests.patch(url, headers=headers, json=rows, params=params or {}, timeout=120)
        else:
            raise ValueError(method)
        if not response.ok:
            raise RuntimeError(f"{method} {table} failed ({response.status_code}): {response.text[:300]}")
        return response.json() if response.content else None

    def experiments_for(self, run_id):
        return self._table("experiments", params={"run_id": f"eq.{run_id}", "order": "experiment_number.asc"})

    def run_files(self, run_id):
        return self._table("run_files", params={"run_id": f"eq.{run_id}"})

    def insert_experiment(self, row):
        return self._table("experiments", "POST", [row], prefer_return=True)

    def insert_metric_points(self, rows):
        return self._table("metric_points", "POST", rows, prefer_return=True)

    def put_artifacts(self, experiment_id, artifacts):
        """Replace this experiment's artifacts, keyed by artifact_type."""
        self._table("experiment_artifacts", "DELETE", params={"experiment_id": f"eq.{experiment_id}"})
        rows = [
            {"experiment_id": experiment_id, "artifact_type": kind, "content": content}
            for kind, content in artifacts.items()
            if content
        ]
        if rows:
            self._table("experiment_artifacts", "POST", rows, prefer_return=True)

    def update_run(self, run_id, values):
        return self._table("research_runs", "PATCH", values, params={"id": f"eq.{run_id}"})


# ---------------------------------------------------------------------------


    def upload_file(self, run_id, path, file_kind, user_id=None):
        """Upload one file, splitting it if the platform cap requires it."""
        path = Path(path)
        size = path.stat().st_size
        if size == 0:
            return {"skipped": f"{path.name} is empty"}
        if size > MAX_ARTIFACT_BYTES:
            return {"skipped": f"{path.name} is {size / 1024**3:.2f} GB (over the 5 GB limit)"}
        if size > MAX_OBJECT_BYTES:
            return self._upload_split(run_id, path, file_kind, user_id, size)
        return self._upload_one(run_id, path, file_kind, user_id)

    def _upload_one(self, run_id, path, file_kind, user_id, display_name=None):
        path = Path(path)
        size = path.stat().st_size
        name = display_name or path.name
        remote_path = f"{run_id}/{file_kind}-{int(time.time() * 1000)}-{sanitize(name)}"
        mime_type = mimetypes.guess_type(name)[0] or "application/octet-stream"

        if size <= PLAIN_UPLOAD_LIMIT:
            error = self._put_object(remote_path, path, mime_type)
        else:
            error = self._put_object_tus(remote_path, path, mime_type, size)
        if error:
            return {"error": error}

        row = {
            "run_id": run_id,
            "file_kind": file_kind,
            "file_name": name,
            "storage_path": remote_path,
            "size_bytes": size,
            "mime_type": mime_type,
        }
        # run_files.user_id is NOT NULL and points at the run's owner.
        if user_id:
            row["user_id"] = user_id
        try:
            self._table("run_files", "POST", [row], prefer_return=True)
        except RuntimeError as exc:
            self.delete_object(remote_path)
            return {"error": f"run_files insert failed: {exc}"}
        return {"file_name": name, "file_kind": file_kind, "size_bytes": size, "storage_path": remote_path}

    def _upload_split(self, run_id, path, file_kind, user_id, size):
        """Split into ordered parts, because this project's plan caps objects at 50 MB.

        The parts concatenate back into the original file byte for byte.
        """
        total = (size + SPLIT_PART_BYTES - 1) // SPLIT_PART_BYTES
        parts, failures = [], []
        with tempfile.TemporaryDirectory(prefix="autoresearch-split-") as tmp:
            with Path(path).open("rb") as source:
                for index in range(total):
                    part_name = f"{path.name}.part{index + 1:02d}-of-{total:02d}"
                    part_path = Path(tmp) / part_name
                    remaining = SPLIT_PART_BYTES
                    with part_path.open("wb") as target:
                        while remaining > 0:
                            chunk = source.read(min(8 * 1024 * 1024, remaining))
                            if not chunk:
                                break
                            target.write(chunk)
                            remaining -= len(chunk)
                    outcome = self._upload_one(run_id, part_path, file_kind, user_id, display_name=part_name)
                    if outcome.get("error") or outcome.get("skipped"):
                        failures.append(f"{part_name}: {outcome.get('error') or outcome.get('skipped')}")
                    else:
                        parts.append(outcome)

        result = {"file_name": path.name, "file_kind": file_kind, "size_bytes": size,
                  "parts": parts, "split": True}
        if failures:
            result["error"] = f"{len(failures)} of {total} parts failed: {failures[0]}"
        return result

    def _put_object(self, remote_path, path, mime_type):
        path = Path(path)
        with path.open("rb") as handle:
            response = requests.post(
                f"{self.base_url}/storage/v1/object/{STORAGE_BUCKET}/{remote_path}",
                data=handle,
                headers={**self._storage_auth(), "Content-Type": mime_type},
                timeout=3600,
            )
        if not response.ok:
            return f"storage upload failed ({response.status_code}): {response.text[:200]}"
        return None

    def _put_object_tus(self, remote_path, path, mime_type, size):
        """Chunked resumable upload, for objects past the plain-POST ceiling."""
        start = f"{self.base_url}/storage/v1/upload/resumable"
        create = requests.post(
            start,
            data=b"",
            headers={
                **self._storage_auth(),
                "Tus-Resumable": TUS_VERSION,
                "Content-Length": "0",
                "x-upsert": "false",
                "Upload-Length": str(size),
                # This storage build reads the object key from `objectName`.
                "Upload-Metadata": _tus_metadata(
                    objectName=remote_path, bucketName=STORAGE_BUCKET,
                    contentType=mime_type, cacheControl="3600",
                ),
            },
            timeout=300,
        )
        if create.status_code not in (200, 201):
            return f"resumable upload could not start ({create.status_code}): {create.text[:200]}"
        location = create.headers.get("Location", "")
        if not location:
            return "resumable upload returned no Location header"
        if location.startswith("/"):
            location = self.base_url + location

        offset = int(create.headers.get("Upload-Offset", 0))
        with Path(path).open("rb") as handle:
            while offset < size:
                chunk = handle.read(TUS_CHUNK)
                if not chunk:
                    break
                patch = requests.patch(
                    location,
                    data=chunk,
                    headers={
                        **self._storage_auth(),
                        "Tus-Resumable": TUS_VERSION,
                        "Content-Type": "application/offset+octet-stream",
                        "Upload-Offset": str(offset),
                    },
                    timeout=1800,
                )
                if patch.status_code not in (200, 204):
                    requests.delete(location, headers=self._storage_auth(), timeout=120)
                    return f"resumable chunk at offset {offset} failed ({patch.status_code}): {patch.text[:200]}"
                offset = int(patch.headers.get("Upload-Offset", offset + len(chunk)))
        return None

    def _storage_auth(self):
        return {"apikey": self.secret, "Authorization": f"Bearer {self.secret}"}

    def delete_object(self, remote_path):
        requests.delete(
            f"{self.base_url}/storage/v1/object/{STORAGE_BUCKET}/{remote_path}",
            headers=self._storage_auth(),
            timeout=300,
        )


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


def build_train_log(raw, points, limit=MAX_LOG_CHARS):
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
        lines += [f"```diff\n{_cap(diff, MAX_LOG_CHARS // 2)}\n```", ""]
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
    return _cap("\n".join(lines), MAX_LOG_CHARS)


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
    return _cap("\n".join(lines), MAX_LOG_CHARS)


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


def collect_run_files(summary):
    """The model and tokenizer files for a run, in the workspace's two kinds."""
    wanted = []
    tok_dir = tokenizer_dir(summary.get("dataset"))
    for file_name in TOKENIZER_FILES:
        candidate = tok_dir / file_name
        if candidate.exists():
            wanted.append((candidate, FILE_KIND_TOKENIZER))
    checkpoint = Path(CHECKPOINT_NAME)
    if checkpoint.exists():
        wanted.append((checkpoint, FILE_KIND_MODEL))
    return wanted


# ---------------------------------------------------------------------------
# results.tsv
# ---------------------------------------------------------------------------

RESULTS_HEADER = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\texperiment"
TSV_STATUS = {"kept": "keep", "discarded": "discard", "keep": "keep", "discard": "discard"}
KEEP_STATUSES = {"keep", "kept"}


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


def append_results_tsv(sha, metric, peak_vram_gb, status, description, experiment="", path=RESULTS_PATH):
    path = Path(path)
    metric_text = f"{metric:.6f}" if metric is not None else "0.000000"
    memory_text = f"{peak_vram_gb:.1f}" if peak_vram_gb is not None else "0.0"
    tsv_status = TSV_STATUS.get(status, status)
    # Tabs and newlines would break the column layout.
    clean = re.sub(r"\s+", " ", str(description)).strip()
    row = f"{sha}\t{metric_text}\t{memory_text}\t{tsv_status}\t{clean}\t{experiment}"
    if not path.exists():
        path.write_text(RESULTS_HEADER + "\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        if not path.read_text(encoding="utf-8").endswith("\n"):
            handle.write("\n")
        handle.write(row + "\n")


def next_experiment_number(rows):
    """Next workspace experiment_number.

    The workspace numbers experiments from 1 per run, so read the numbers out of
    the last column rather than counting rows: a crash is recorded with an empty
    number (it is never published) and must not consume one.
    """
    numbers = []
    for row in rows:
        if len(row) < 6:
            continue
        try:
            numbers.append(int(row[5]))
        except ValueError:
            continue
    return max(numbers) + 1 if numbers else 1


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


def already_published(rows, sha):
    return any(row and row[0].strip() == sha for row in rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description="Publish an autoresearch run to the workspace.")
    parser.add_argument("--name", required=True, help="Short experiment name (<=240 chars).")
    parser.add_argument("--hypothesis", default="", help="Markdown hypothesis, stored as experiment.md.")
    parser.add_argument("--results", default="", help="Extra markdown appended to results.md.")
    parser.add_argument("--status", choices=(EXPERIMENT_STATUS_KEPT, EXPERIMENT_STATUS_DISCARDED),
                        default=EXPERIMENT_STATUS_KEPT)
    parser.add_argument("--number", type=int, default=None, help="experiment_number (default: next free).")
    parser.add_argument("--run-name", default="autoresearch-win-rtx", help="Workspace run to publish into.")
    parser.add_argument("--log", default="run.log", help="Path to the captured terminal output.")
    parser.add_argument("--files", dest="files", action="store_true", default=True,
                        help="Attach the tokenizer (once per run) and, when kept, the checkpoint.")
    parser.add_argument("--no-files", dest="files", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Build and print the payload, publish nothing.")
    args = parser.parse_args(argv)

    env = {**load_dotenv(), **os.environ}

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
    status = args.status
    if metric is None:
        print(f"FAIL: {args.log} has no val_bpb; recording a crash without publishing.", file=sys.stderr)
        append_results_tsv(sha, None, None, "crash", args.name)
        return 1
    if already_published(rows, sha):
        print(f"NOTE: commit {sha} is already in results.tsv; not publishing it twice.")
        return 0

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
        delta = 0.0
    elif metric < prior_best:
        verdict = f"Kept: new best val_bpb, {prior_best - metric:+.6f} against the previous {prior_best:.6f}."
        delta = metric - prior_best
    else:
        verdict = f"Discarded: {metric - prior_best:+.6f} against the best val_bpb of {prior_best:.6f}."
        delta = metric - prior_best

    train_log = build_train_log(raw, points)
    experiment_md = build_experiment_md(args.name, args.hypothesis, sha, diff, summary, metadata)
    results_md = build_results_md(args.name, metric, prior_best, status, summary, verdict, args.results)
    chart = build_chart_data(points, MAX_CHART_POINTS)

    if args.dry_run:
        print(json.dumps({
            "experiment_number": number,
            "name": args.name[:MAX_NAME_CHARS],
            "status": status,
            "score": metric,
            "delta": round(delta, 6),
            "duration_seconds": int(duration) if duration else None,
            "description": short_description(args.hypothesis) or args.name[:MAX_DESCRIPTION_CHARS],
            "experiment_md": f"<{len(experiment_md)} chars>",
            "results_md": f"<{len(results_md)} chars>",
            "train_log": f"<{len(train_log)} chars>",
            "metric_points": len(chart),
            "commit_sha": sha,
            "metadata_keys": len(metadata),
        }, indent=2))
        print(f"\nresults.tsv row: {sha}\t{metric:.6f}\t{peak_vram_gb}\t{status}\t{args.name}\t{number}")
        return 0

    workspace = Workspace(env)
    if workspace.missing:
        print("FAIL: missing credentials in .env: " + "; ".join(workspace.missing), file=sys.stderr)
        return 2

    run, created = workspace.ensure_run(args.run_name, REPO_URL, prior_best if prior_best is not None else metric)
    run_id = run["id"]
    if created:
        print(f"Created run '{run['name']}' ({run_id}), baseline_score={run.get('baseline_score')}")

    experiment_id = str(uuid.uuid4())
    workspace.insert_experiment({
        "id": experiment_id,
        "run_id": run_id,
        "experiment_number": number,
        "name": args.name[:MAX_NAME_CHARS],
        "status": status,
        "score": metric,
        "delta": round(delta, 6),
        "duration_seconds": int(duration) if duration else 0,
    })

    if chart:
        smoothed = None
        points_to_write = []
        for point in chart:
            smoothed = point["loss"] if smoothed is None else 0.7 * smoothed + 0.3 * point["loss"]
            points_to_write.append({
                "experiment_id": experiment_id,
                "step": point["step"],
                "train_loss": point["loss"],
                "smoothed_loss": round(smoothed, 6),
            })
        workspace.insert_metric_points(points_to_write)

    workspace.put_artifacts(experiment_id, {
        ARTIFACT_EXPERIMENT: experiment_md,
        ARTIFACT_RESULTS: results_md,
        ARTIFACT_TRAIN_LOG: train_log,
    })

    refresh_run_summary(workspace, run_id, prior_baseline=run.get("baseline_score"))
    print(f"Published experiment #{number} ({experiment_id}) in run '{run['name']}': "
          f"{args.name} score={metric:.6f} delta={delta:+.6f} [{status}]")

    if args.files:
        attach_files(workspace, run_id, summary, status, user_id=run.get("user_id"))

    append_results_tsv(sha, metric, peak_vram_gb, status, args.name, str(number))
    print(f"Recorded {sha}\t{metric:.6f}\t{peak_vram_gb}\t{status}\t{args.name}\t{number} in results.tsv")
    return 0


def refresh_run_summary(workspace, run_id, prior_baseline=None):
    """Recompute the run's counters the site's leaderboard reads."""
    experiments = workspace.experiments_for(run_id)
    kept = [e for e in experiments if e.get("status") == EXPERIMENT_STATUS_KEPT]
    scores = [e["score"] for e in kept if isinstance(e.get("score"), (int, float))]
    baseline = prior_baseline
    if baseline is None:
        baseline = min((e["score"] for e in experiments if isinstance(e.get("score"), (int, float))),
                       default=0.0)
    values = {
        "total_experiments": len(experiments),
        "kept_improvements": len(kept),
        "best_score": min(scores) if scores else baseline,
        "status": "running",
    }
    if baseline is not None:
        values["baseline_score"] = baseline
    workspace.update_run(run_id, values)
    return values


def attach_files(workspace, run_id, summary, status, user_id=None):
    """Tokenizer once per run; the checkpoint on runs we keep.

    Files belong to the run rather than to one experiment, so re-uploading the
    identical tokenizer on every experiment would only burn storage.
    """
    existing = workspace.run_files(run_id)
    have_tokenizer = any(f.get("file_kind") == FILE_KIND_TOKENIZER for f in existing)
    want_model = status == EXPERIMENT_STATUS_KEPT

    for path, file_kind in collect_run_files(summary):
        if file_kind == FILE_KIND_TOKENIZER and have_tokenizer:
            continue
        outcome = workspace.upload_file(run_id, path, file_kind, user_id=user_id)
        if outcome.get("split"):
            parts = outcome.get("parts", [])
            failed = outcome.get("error")
            if failed:
                print(f"  FAILED {file_kind}/{path.name}: {failed}")
            if parts:
                print(f"  attached {file_kind}/{path.name} as {len(parts)} parts "
                      f"({outcome['size_bytes'] / 1024**2:.0f} MB total, "
                      f"{parts[0]['file_name']} ... {parts[-1]['file_name']})")
                print(f"    reassemble with: cat {path.name}.part* > {path.name}")
        elif outcome.get("file_name"):
            print(f"  attached {outcome['file_kind']}/{outcome['file_name']} "
                  f"({outcome['size_bytes'] / 1024 / 1024:.1f} MB)")
        elif outcome.get("skipped"):
            print(f"  skipped {outcome['skipped']}")
        else:
            print(f"  FAILED {file_kind}/{path.name}: {outcome.get('error')}")

    if not want_model:
        print("  (checkpoint not uploaded: experiment was not kept)")


if __name__ == "__main__":
    sys.exit(main())