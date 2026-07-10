# AGENTS.md

How an automated agent should drive BEAK to run remote jobs from the command
line. BEAK submits long-running bioinformatics jobs to a remote server over SSH,
tracks their state locally, and pulls results back to disk. The job model is
asynchronous: **submit returns immediately, then you poll for completion.**

This document describes what works **today**. Rough edges and their roadmap are
listed at the end; where a feature is missing, the workaround here is the
supported path until it lands.

## Environment

BEAK runs from a project-local virtualenv at `.venv/`. There is no activation
step for an agent — invoke the interpreter and CLI by path so every call uses
the right environment regardless of shell state:

```bash
# One-time setup (Python >= 3.8; 3.11 recommended)
python3.11 -m venv .venv
./.venv/bin/pip install -e ".[dev]"

# Thereafter, always invoke by path — never bare `python`/`beak`, which may
# resolve to an unrelated system or conda interpreter:
./.venv/bin/beak doctor
./.venv/bin/python -m pytest tests/ -q
```

Do not rely on `conda activate` or a globally-installed `beak`; the `.venv`
paths above are the supported entry points.

## The loop (machine mode)

The submit commands accept `--json` (structured stdout), `--wait` (block until a
terminal state), `--interval` (poll seconds), and `--dry-run` (validate inputs
and print the submission plan without connecting or submitting — safe to run
offline before committing remote compute). `--json` works either per-command
(`beak search … --json`) or before the subcommand (`beak --json search …`).

```bash
# 1. Preflight — confirm the remote is configured and reachable
beak config show                 # is a host/user set?
beak doctor --json               # {ok, tools, databases, disk, pfam}; exit 0 if ok, 1 if not
# Gate the rest on `.ok` (or on `$? == 0`) before spending remote compute.

# 2. Submit + block + parse — one call, one JSON object on stdout
beak search query.fasta --db uniref90 --name my_search --json --wait --interval 30
# -> {"job_id": "a1b2c3d4", "job_type": "search", "name": "my_search", "status": "COMPLETED"}
# Exit code: 0 if COMPLETED, non-zero (1) if FAILED/CANCELLED/UNKNOWN.

# 3. Fetch — get the on-disk path to results (do NOT scrape stdout tables)
beak results a1b2c3d4     # prints: "✓ Results at: /path/to/results"
```

Prefer `--wait` over hand-rolled polling: it blocks to a terminal state, emits
the final status as JSON, and sets the exit code so you branch on `$?` instead
of parsing. If you'd rather poll yourself (e.g. to do other work meanwhile),
submit with `--json` alone (status `"SUBMITTED"`) and read `~/.beak/jobs.json`
as below.

## Reading job state

Two equivalent machine-readable surfaces:

- `beak status <id> --json` emits `{job_id, name, status, runtime, job_type}`;
  `beak jobs --json` emits a JSON array of `{id, name, type, status, submitted}`.
  Both hit the remote to refresh non-terminal jobs. (Without `--json` these
  render Rich tables built for humans — ANSI color, box-drawing, glyphs.)
- `~/.beak/jobs.json`, a JSON object keyed by the 8-character `job_id`, is the
  same state on disk. Read it directly when you want state **without** a remote
  round-trip (offline, or to avoid an SSH call per poll).

```bash
# State of one job
jq -r '.["a1b2c3d4"].status' ~/.beak/jobs.json          # SUBMITTED | RUNNING | COMPLETED | FAILED | CANCELLED

# The id of the most recently submitted job (robust alternative to parsing submit output)
jq -r 'to_entries | max_by(.value.submitted_at) | .key' ~/.beak/jobs.json
```

Each entry carries at least `job_id`, `job_type`, `name`, `status`, and
`submitted_at` (ISO 8601). Poll on an interval (see latencies below) until
`status` is one of the three terminal values. `FAILED` means inspect the log:

```bash
beak log a1b2c3d4         # remote job log; use to diagnose a FAILED job
```

## Capturing the job id

- Best: submit with `--json` and read `.job_id` from the emitted object.
- Without `--json`, the human submit line is `✓ Submitted <name> → <db>
  (a1b2c3d4)` — the id is the parenthesized 8-hex token — or pass an explicit
  `--name` and look the job up in `jobs.json` by `name` / newest `submitted_at`
  (jq above).

## Getting results

Results are written to disk; do not read them out of terminal output.

- `beak results <id> --json` downloads the artefacts and emits their on-disk
  paths as one object: `{job_id, job_type, results_path, [taxonomy_path]}`.
  Embeddings jobs instead report `{job_id, job_type, results_dir, files}` where
  `files` maps `mean_embeddings` / `per_token_embeddings` / `failed` /
  `taxonomy` to paths. Read those files yourself.
- `beak results <id>` (no flags) is the human form: downloads and prints
  `✓ Results at: <path>`.
- `beak results <id> --parse` prints a **truncated** preview (20 rows) — for
  eyeballing only, never to capture data.
- For programmatic full results, the Python API returns the complete object:
  `from beak.remote import BeakSession; BeakSession().search.get_results(id)`
  yields the full `pandas.DataFrame`.

## Rules for non-interactive operation

- Configure with `beak config set <key> <value>` (e.g. `beak config set
  connection.host my-server.edu`). Do **not** call `beak config init` — it is an
  interactive wizard and will block.
- Never invoke `beak ui` (a full-screen TUI), or the `--watch` / `--follow`
  flags — these are live human displays that never return.
- `beak embeddings` normally prompts for confirmation when the estimated output
  exceeds a size threshold. Passing `--json` bypasses that prompt (and all other
  advisory output), so an agent never blocks on stdin.

## Realistic latencies — poll and time out accordingly

Remote jobs are minutes to hours and consume shared compute. Poll every 30–60 s,
not every few seconds, and set generous timeouts. Order-of-magnitude, for a
~200 aa query against UniRef90:

| Job                                    | Wall time   |
| -------------------------------------- | ----------- |
| `beak search` (default preset)         | 3–5 min     |
| `beak search --preset broad`           | 8–12 min    |
| `beak align`                           | seconds–min |
| `beak taxonomy`                        | 1–5 min     |
| `beak embeddings` (GPU)                | min–tens    |

Benchmark on the actual remote before hard-coding timeouts.

## Shared-server etiquette

The remote is shared. Be a considerate neighbor:

- **Submit expensive jobs one at a time.** Prefer `--wait` (which serializes
  naturally) over firing many jobs at once. Each search/taxonomy job uses up to
  `compute.threads` (default 8) CPUs, so N concurrent jobs ≈ 8N cores plus N
  copies of the database in RAM — a handful of parallel searches and an IQ-TREE
  can saturate a box.
- **Check load before fanning out.** `beak doctor --json` reports a `load`
  object: `{load_1m, load_5m, load_15m, n_cpus, load_per_cpu, mem_total_mb,
  mem_available_mb, gpus:[{util_pct, mem_used_mb, mem_total_mb}]}`. Hold off on
  new submissions when `load_per_cpu` is near or above 1.0, memory is tight, or
  (for embeddings) the GPUs are busy.
- **Cap your own concurrency.** Count RUNNING jobs in `beak jobs --json` (or
  `~/.beak/jobs.json`) and keep simultaneous heavy jobs small — 2–3 is usually
  plenty, one is safest on a busy box.
- **Preview first.** `--dry-run` prints a job's plan (and, for embeddings, its
  output size) without submitting — sanity-check before committing compute.

## Command reference (agent-relevant subset)

```
Preflight   beak config show          # current config
            beak doctor --json        # {ok, tools, databases, disk, pfam}; nonzero exit if not ok

Submit      beak search <fa> --db <alias> [--name N] [--preset default|close|broad|twilight]
            beak taxonomy <fa> --db <alias> [--name N]
            beak align <fa> [-a clustalo|mafft|muscle]
            beak embeddings <fa> [-m MODEL] [--layer N]
            # all four also take: --json  --wait  --interval <sec>  --dry-run

Monitor     beak jobs --json           # JSON array of all jobs
            beak status <id> --json    # JSON status of one job
            beak log <id>              # remote log (human text; FAILED diagnosis)

Fetch       beak results <id> --json   # JSON of downloaded result paths

Projects    beak project init <name> --uniprot <id> | --sequence <fa>
            beak project list
            beak project status <name>

Avoid       beak ui, --watch, --follow, beak config init   # interactive / never-return
```

## Offline vs. remote

Only search, taxonomy, alignment, embeddings, and HMMER need the remote. Project
management, structure feature extraction, conservation, comparative analysis,
and export all run locally with no SSH. An agent doing purely local analysis
does not need `beak config` / `beak doctor` to succeed.

## Exit codes

The submit commands set distinct exit codes so an agent can branch on `$?`
before parsing:

| Code | Meaning                                                          |
| ---- | --------------------------------------------------------------- |
| `0`  | Success (submitted; or, with `--wait`, the job COMPLETED)       |
| `1`  | Job reached a non-COMPLETED terminal state (FAILED/CANCELLED)   |
| `2`  | Usage error (bad/missing arguments)                             |
| `3`  | Remote unreachable (connection/timeout/SSH failure)            |

In `--json` mode a failure also prints `{"error": "...", "exit_code": N}` on
stdout (in addition to setting the exit code), so a stdout-only consumer still
sees the failure. Commands that already emitted a result object (`status
--json`, a failed `--wait`) just set the exit code and print nothing further.

## Known rough edges

Machine output (`--json`) covers the full job loop — submit, `status`, `jobs`,
`results` — plus `doctor` for preflight. Still human-only, by design (no agent
contract needed): `beak log` (free-form remote text) and the `beak project`
commands. Add `--json` there if an agent workflow comes to need it.
