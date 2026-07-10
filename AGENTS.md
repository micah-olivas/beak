# AGENTS.md

How an automated agent should drive BEAK to run remote jobs from the command
line. BEAK submits long-running bioinformatics jobs to a remote server over SSH,
tracks their state locally, and pulls results back to disk. The job model is
asynchronous: **submit returns immediately, then you poll for completion.**

This document describes what works **today**. Rough edges and their roadmap are
listed at the end; where a feature is missing, the workaround here is the
supported path until it lands.

## The loop (machine mode)

The submit commands accept `--json` (structured stdout), `--wait` (block until a
terminal state), and `--interval` (poll seconds). `--json` works either
per-command (`beak search … --json`) or before the subcommand
(`beak --json search …`).

```bash
# 1. Preflight — confirm the remote is configured and reachable
beak config show          # is a host/user set?
beak doctor               # are the required remote tools + databases present?

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

## Read `~/.beak/jobs.json` directly — it is the machine-readable surface

Every submitted job is recorded in `~/.beak/jobs.json`, a JSON object keyed by
the 8-character `job_id`. Parse this file for state rather than scraping the
Rich-formatted output of `beak jobs` / `beak status`, which is built for humans
(ANSI color, box-drawing, glyphs).

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

- `beak results <id>` (no flags) downloads results and prints `✓ Results at:
  <path>`. Capture that path and read the file (FASTA / TSV / parquet) yourself.
- `beak results <id> --parse` prints a **truncated** preview (20 rows). Use it
  only to eyeball completion, never to capture data.
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

## Command reference (agent-relevant subset)

```
Preflight   beak config show          # current config
            beak doctor               # remote tools + database check

Submit      beak search <fa> --db <alias> [--name N] [--preset default|close|broad|twilight]
            beak taxonomy <fa> --db <alias> [--name N]
            beak align <fa> [-a clustalo|mafft|muscle]
            beak embeddings <fa> [-m MODEL] [--layer N]
            # all four also take: --json  --wait  --interval <sec>

Monitor     beak jobs                  # list (human display; prefer jobs.json)
            beak status <id>           # one job (human display; prefer jobs.json)
            beak log <id>              # remote log (use for FAILED diagnosis)

Fetch       beak results <id>          # prints "✓ Results at: <path>"

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

## Known rough edges (roadmap)

Machine output currently covers the **submit** commands. Still human-only:

- **`beak status` / `beak results` have no `--json` mode.** Read
  `~/.beak/jobs.json` for state, and take the result path from the plain
  `beak results <id>` line (or the Python API) for data — as described above.
