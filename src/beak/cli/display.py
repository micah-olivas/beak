"""Rich-based status display for BEAK remote jobs."""

import time

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.text import Text

from .theme import BORDER_STYLE, STATUS_STYLES, STAGE_ICONS, get_console


def _embedding_progress_renderable(done: int, total: int, failed: int) -> Progress:
    """Build a static Rich Progress renderable for an embedding job.

    Used inside a render_status() Group; auto_refresh=False so it doesn't
    try to start its own Live context (the outer watch_status already has
    one). Colors the bar red if any sequences have failed.
    """
    bar_style = "red" if failed else "cyan"
    finished_style = "red" if failed else "green"

    progress = Progress(
        TextColumn("  [bold]Embedding[/bold]"),
        BarColumn(
            bar_width=30,
            style=bar_style,
            complete_style=finished_style,
            finished_style=finished_style,
        ),
        MofNCompleteColumn(),
        TextColumn("[dim]{task.percentage:>3.0f}%[/dim]"),
        auto_refresh=False,
        expand=False,
    )
    progress.add_task("embedding", total=total, completed=done)
    return progress


def render_status(info: dict) -> Group:
    """Build a Rich renderable Group for job status display."""
    status = info.get("status", "UNKNOWN")
    status_style = STATUS_STYLES.get(status, "dim")
    job_id = info.get("job_id", "?")
    name = info.get("name", job_id)
    job_type = info.get("job_type", "")

    parts = []

    # Header panel
    type_label = f"  [dim]{job_type}[/dim]" if job_type else ""
    parts.append(Panel.fit(
        f"[brand]BEAK[/brand]{type_label}  \u00b7  [bold]{name}[/bold]  \u00b7  [{status_style}]{status}[/{status_style}]",
        border_style=BORDER_STYLE,
    ))

    # Details line: runtime, database, preset
    detail_parts = []
    runtime = info.get("runtime")
    if runtime:
        detail_parts.append(f"Runtime: {runtime}")
    database = info.get("database")
    if database:
        detail_parts.append(f"Database: {database}")
    preset = info.get("preset")
    if preset:
        detail_parts.append(f"Preset: {preset}")
    if detail_parts:
        parts.append(Text(f"  {' · '.join(detail_parts)}", style="dim"))

    # Stages
    stages = info.get("stages", [])
    if stages:
        parts.append(Text(""))
        for stage in stages:
            state = stage.get("state", "pending")
            icon = STAGE_ICONS.get(state, STAGE_ICONS["pending"])
            label = stage["label"]

            if state == "done":
                style = ""
            elif state == "active":
                style = "[bold]"
            else:
                style = "[dim]"

            close = "[/bold]" if style == "[bold]" else "[/dim]" if style == "[dim]" else ""
            parts.append(Text.from_markup(f"  {icon}  {style}{label}{close}"))

    # Embedding job progress bar (structured from progress.json)
    emb = info.get("embedding_progress")
    if emb and emb.get("total"):
        done = emb.get("done", 0)
        total = emb.get("total", 0)
        failed = emb.get("failed", 0)
        current = emb.get("current")

        parts.append(Text(""))
        parts.append(_embedding_progress_renderable(done, total, failed))

        # Meta line: failure count + current sequence (if running)
        meta_bits = []
        if failed:
            meta_bits.append(f"[red]{failed} failed[/red]")
        if current and status == "RUNNING":
            meta_bits.append(f"current: [bold]{current}[/bold]")
        if meta_bits:
            parts.append(Text.from_markup("  [dim]" + "  ·  ".join(meta_bits) + "[/dim]"))

        # Most-recent per-sequence error, when we've got one. Truncated
        # to keep the watch view tidy; failed.tsv has the full set.
        last_err = emb.get("last_error")
        if last_err:
            msg = (last_err.get("message") or "")[:120]
            parts.append(Text.from_markup(
                f"  [red]last error[/red] ([dim]{last_err.get('seq_id', '?')}[/dim]): "
                f"{last_err.get('type', '?')}: {msg}"
            ))

    # Last log line
    last_line = (info.get("last_log_line") or "").strip()
    if last_line and status == "RUNNING":
        parts.append(Text(""))
        parts.append(Text.from_markup(f"  [dim]{last_line}[/dim]"))

    # Next step hint
    parts.append(Text(""))
    if status == "COMPLETED":
        parts.append(Text.from_markup(
            f"[bold]Next:[/bold] [cyan]beak results {job_id}[/cyan]"
        ))
    elif status == "FAILED":
        parts.append(Text.from_markup(
            f"[bold]Check log:[/bold] [cyan]beak log {job_id}[/cyan]"
        ))
    parts.append(Text(""))

    return Group(*parts)


def print_status(info: dict, console=None):
    """Print formatted job status to the terminal."""
    console = console or get_console()
    console.print(render_status(info))


def watch_status(mgr, job_id: str, interval: float = 2.0):
    """Live-updating status display using rich.live.Live."""
    console = get_console()

    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                info = mgr.detailed_status(job_id)
                live.update(render_status(info))

                if info.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "UNKNOWN"):
                    break

                time.sleep(interval)
    except KeyboardInterrupt:
        pass
