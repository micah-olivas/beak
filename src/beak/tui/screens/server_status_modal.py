"""Detailed server-status modal opened from the top-right pill.

The pill itself fits four numbers (load / cores / GPU / job count)
into one line. This modal expands them into a per-section breakdown
with per-GPU utilization, the actual process table for beak's compute
tools, and the size of `~/beak_jobs/` so the user can spot orphans
without dropping to a shell.

The probe is one chained `&&` shell command per refresh — single SSH
round-trip, sectioned with `=== HEADER ===` sentinels we parse on
the way back. Auto-refreshes every 5 s while the modal is open;
manual `r` also works.
"""

from pathlib import Path
from typing import List, Optional

from fabric import Connection
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static


# One-shot multi-section probe. Each section is wrapped in a sentinel
# so the parser can split deterministically. Every external command is
# `2>/dev/null`'d so a missing tool (no nvidia-smi on a CPU host, no
# ~/beak_jobs yet) leaves an empty section rather than raw stderr.
_PROBE_CMD = r"""
echo "=== HOST ==="
cat /proc/loadavg 2>/dev/null
echo "cores $(nproc 2>/dev/null)"
free -m 2>/dev/null | awk '/^Mem:/{print "mem_mb " $3 " " $2}'
uptime -p 2>/dev/null
echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits 2>/dev/null
echo "=== PROCS ==="
ps -eo pid,user:14,pcpu,pmem,etime,comm 2>/dev/null \
    | awk 'NR==1 || $6 ~ /(mmseqs|clustalo|mafft|muscle|hmmscan|hmmsearch|jackhmmer|python|python3|esm)/' \
    | head -25
echo "=== DISK ==="
du -sh "$HOME/beak_jobs" 2>/dev/null
ls -1t "$HOME/beak_jobs" 2>/dev/null | head -5
"""


class ServerStatusModal(ModalScreen[None]):
    """Auto-refreshing rich server-status modal."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("r", "refresh_now", "Refresh"),
    ]

    DEFAULT_CSS = """
    ServerStatusModal #modal-body {
        width: 96;
        max-height: 90%;
    }
    ServerStatusModal #srv-scroll {
        height: 1fr;
        max-height: 32;
    }
    ServerStatusModal .srv-section {
        margin-top: 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._last_text: str = "[dim]Loading…[/dim]"

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-body"):
            yield Label("[bold]Server status[/bold]", id="modal-title")
            yield Label(
                "[dim]Live snapshot of the configured remote — load, "
                "GPU, beak processes, scratch disk. Refreshes every "
                "5s while open.[/dim]",
            )
            with VerticalScroll(id="srv-scroll"):
                yield Static(self._last_text, id="srv-content")
            with Horizontal(id="modal-buttons"):
                yield Button("Close", id="close-btn", variant="primary")

    def on_mount(self) -> None:
        self._poll()
        self.set_interval(5.0, self._poll)

    def action_close(self) -> None:
        self.dismiss(None)

    def action_refresh_now(self) -> None:
        self._poll()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-btn":
            self.action_close()

    # ---- probe + render ----

    @work(thread=True, exclusive=True, group="srv-detail")
    def _poll(self) -> None:
        conn = self._build_connection()
        if conn is None:
            self.app.call_from_thread(
                self._set_content,
                "[red]No SSH defaults configured.[/red]\n\n"
                "[dim]Run `beak config init` (or hit `s` from the "
                "project list) to configure the remote, then reopen "
                "this modal.[/dim]",
            )
            return

        try:
            result = conn.run(
                _PROBE_CMD, hide=True, warn=True, timeout=12,
            )
            stdout = result.stdout if result and result.ok else ""
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(
                self._set_content,
                f"[red]Probe failed:[/red] [dim]{type(e).__name__}: "
                f"{str(e).splitlines()[0][:200]}[/dim]",
            )
            return
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

        text = self._render_probe(stdout)
        self.app.call_from_thread(self._set_content, text)

    def _set_content(self, text: str) -> None:
        self._last_text = text
        try:
            self.query_one("#srv-content", Static).update(text)
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _build_connection() -> Optional[Connection]:
        """Open a fresh SSH connection from the user's beak config.

        Mirrors `ServerStatusBar._build_connection` — a single probe
        per refresh, torn down immediately after. Self-mutes if SSH
        defaults are missing.
        """
        try:
            from ...config import get_default_connection
            from ...remote.base import RemoteJobManager
            from ...remote.session import _no_stdin_config

            defaults = get_default_connection()
            host = defaults.get("host")
            user = defaults.get("user")
            key_path = defaults.get("key_path")
            if not host or not user:
                return None
            if key_path is None:
                key_path = RemoteJobManager._find_ssh_key(None)
            return Connection(
                host=host,
                user=user,
                connect_timeout=10,
                connect_kwargs={
                    "key_filename": str(Path(key_path).expanduser()),
                },
                config=_no_stdin_config(),
            )
        except Exception:  # noqa: BLE001
            return None

    @classmethod
    def _render_probe(cls, raw: str) -> str:
        if not raw.strip():
            return "[dim]No data returned from probe.[/dim]"

        sections = cls._split_sections(raw)
        out: List[str] = []
        out.append(cls._render_host(sections.get("HOST", "")))
        gpu = cls._render_gpu(sections.get("GPU", ""))
        if gpu:
            out.append(gpu)
        out.append(cls._render_procs(sections.get("PROCS", "")))
        out.append(cls._render_disk(sections.get("DISK", "")))
        return "\n\n".join(out)

    @staticmethod
    def _split_sections(raw: str) -> dict:
        sections: dict = {}
        current = None
        buf: List[str] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith("=== ") and stripped.endswith(" ==="):
                if current is not None:
                    sections[current] = "\n".join(buf).strip()
                current = stripped[4:-4]
                buf = []
            else:
                buf.append(line)
        if current is not None:
            sections[current] = "\n".join(buf).strip()
        return sections

    # ---- per-section renderers ----

    @staticmethod
    def _render_host(s: str) -> str:
        load1 = load5 = load15 = "?"
        cores = "?"
        mem_used = mem_total = None
        uptime = ""

        for line in s.splitlines():
            line = line.strip()
            if not line:
                continue
            # `cat /proc/loadavg` prints e.g. "0.42 0.31 0.28 1/123 4567"
            parts = line.split()
            if len(parts) >= 3 and all(_is_floaty(p) for p in parts[:3]) \
                    and not line.startswith("cores ") \
                    and not line.startswith("mem_mb "):
                load1, load5, load15 = parts[0], parts[1], parts[2]
            elif line.startswith("cores "):
                cores = line.split(None, 1)[1]
            elif line.startswith("mem_mb "):
                bits = line.split()
                if len(bits) >= 3:
                    mem_used, mem_total = bits[1], bits[2]
            elif line.startswith("up "):
                uptime = line

        rows = [
            "[bold]Host[/bold]",
            f"  load   {load1} / {load5} / {load15}   "
            f"[dim]({cores} cores)[/dim]",
        ]
        if mem_used and mem_total:
            try:
                used_gb = int(mem_used) / 1024
                total_gb = int(mem_total) / 1024
                pct = (int(mem_used) / max(int(mem_total), 1)) * 100
                rows.append(
                    f"  memory {used_gb:.1f} / {total_gb:.1f} GB   "
                    f"[dim]({pct:.0f}%)[/dim]"
                )
            except (TypeError, ValueError):
                pass
        if uptime:
            rows.append(f"  [dim]{uptime}[/dim]")
        return "\n".join(rows)

    @staticmethod
    def _render_gpu(s: str) -> str:
        if not s.strip():
            return ""
        rows = ["[bold]GPUs[/bold]"]
        # csv: index, name, utilization, mem_used, mem_total
        for line in s.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                idx = int(parts[0])
                util = int(parts[2])
                mem_used = int(parts[3])
                mem_total = int(parts[4])
            except ValueError:
                continue
            name = parts[1]
            mem_pct = (mem_used / max(mem_total, 1)) * 100
            color = (
                "green" if util < 25
                else "yellow" if util < 75
                else "red"
            )
            bar = _bar(util / 100, 12)
            rows.append(
                f"  [bold]{idx}[/bold] {name[:24]:<24}  "
                f"[{color}]{bar}[/{color}]  "
                f"[bold]{util:>3}%[/bold]   "
                f"[dim]{mem_used:>5}/{mem_total} MiB ({mem_pct:.0f}%)[/dim]"
            )
        return "\n".join(rows)

    @staticmethod
    def _render_procs(s: str) -> str:
        if not s.strip():
            return "[bold]Processes[/bold]\n  [dim]none[/dim]"
        lines = s.splitlines()
        # First line is the ps header from `awk 'NR==1 || ...'`. Strip
        # it and re-emit our own header in a more compact form.
        header_skipped = False
        rows = ["[bold]Processes[/bold]"]
        rows.append(
            "  [dim]" + f"{'PID':>7} {'USER':<14} {'CPU':>5} "
            f"{'MEM':>5} {'TIME':>10}  COMMAND" + "[/dim]"
        )
        proc_count = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if not header_skipped and line.upper().startswith("PID"):
                header_skipped = True
                continue
            parts = line.split(None, 5)
            if len(parts) < 6:
                continue
            pid, user, cpu, mem, etime, comm = parts
            rows.append(
                f"  {pid:>7} {user[:14]:<14} {cpu:>5} {mem:>5} "
                f"{etime:>10}  {comm}"
            )
            proc_count += 1
        if proc_count == 0:
            rows.append("  [dim]none[/dim]")
        return "\n".join(rows)

    @staticmethod
    def _render_disk(s: str) -> str:
        rows = ["[bold]Scratch (~/beak_jobs)[/bold]"]
        if not s.strip():
            rows.append("  [dim]empty[/dim]")
            return "\n".join(rows)
        lines = [ln for ln in s.splitlines() if ln.strip()]
        if not lines:
            rows.append("  [dim]empty[/dim]")
            return "\n".join(rows)
        # First line: `du -sh ~/beak_jobs` → "12.3G   /home/me/beak_jobs"
        first = lines[0]
        size = first.split(None, 1)[0] if first else "?"
        rows.append(f"  total {size}")
        # Remaining lines: most recent job-id dirs.
        recent = lines[1:6]
        if recent:
            rows.append("  [dim]most recent:[/dim]")
            for r in recent:
                rows.append(f"    [dim]{r}[/dim]")
        return "\n".join(rows)


def _is_floaty(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _bar(frac: float, width: int) -> str:
    """Tiny block-drawing bar — used by the GPU utilization column."""
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)
