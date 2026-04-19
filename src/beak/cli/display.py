"""Rich-based status display for BEAK remote jobs."""

import time

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from .theme import BORDER_STYLE, STATUS_STYLES, STAGE_ICONS, get_console


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
