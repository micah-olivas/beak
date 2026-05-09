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
        """Host load + memory as colored bars instead of plain text.

        Load is plotted relative to core count (`load / cores`) — the
        canonical "is the box overcommitted" ratio. Three rows so the
        1/5/15-minute trend is visible at a glance: a high 1-minute
        with a low 15-minute means the load just spiked, not a
        long-term saturation.
        """
        load1 = load5 = load15 = None
        cores: Optional[int] = None
        mem_used: Optional[int] = None
        mem_total: Optional[int] = None
        uptime = ""

        for line in s.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3 and all(_is_floaty(p) for p in parts[:3]) \
                    and not line.startswith("cores ") \
                    and not line.startswith("mem_mb "):
                try:
                    load1, load5, load15 = (
                        float(parts[0]), float(parts[1]), float(parts[2])
                    )
                except ValueError:
                    pass
            elif line.startswith("cores "):
                tail = line.split(None, 1)[1]
                if tail.isdigit():
                    cores = int(tail)
            elif line.startswith("mem_mb "):
                bits = line.split()
                if len(bits) >= 3 and bits[1].isdigit() and bits[2].isdigit():
                    mem_used, mem_total = int(bits[1]), int(bits[2])
            elif line.startswith("up "):
                uptime = line

        rows = ["[bold]Host[/bold]"]
        if load1 is not None and cores:
            rows.append(
                f"  [dim]load · {cores} cores · "
                f"bar = load ÷ cores[/dim]"
            )
            for label, lv in (("1m ", load1), ("5m ", load5), ("15m", load15)):
                if lv is None:
                    continue
                ratio = lv / max(cores, 1)
                rows.append(
                    f"  {label}   {_bar_colored(ratio, _BAR_W)}  "
                    f"[bold]{lv:5.2f}[/bold]"
                )
        elif load1 is not None:
            rows.append(
                f"  load   {load1:.2f} / {load5 or 0:.2f} / "
                f"{load15 or 0:.2f}   [dim](cores unknown)[/dim]"
            )
        if mem_used is not None and mem_total:
            used_gb = mem_used / 1024
            total_gb = mem_total / 1024
            ratio = mem_used / max(mem_total, 1)
            rows.append(
                f"  mem    {_bar_colored(ratio, _BAR_W)}  "
                f"[bold]{used_gb:5.1f}[/bold] / {total_gb:.1f} GiB  "
                f"[dim]({ratio * 100:.0f}%)[/dim]"
            )
        if uptime:
            rows.append(f"  [dim]{uptime}[/dim]")
        return "\n".join(rows)

    @staticmethod
    def _render_gpu(s: str) -> str:
        if not s.strip():
            return ""
        rows = [
            "[bold]GPUs[/bold]",
            "  [dim]util / mem (GiB)[/dim]",
        ]
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
            name = _compact_gpu_name(parts[1])
            mem_pct = mem_used / max(mem_total, 1)
            # Two narrower bars side-by-side beats one wide bar +
            # a separate mem row. Width 12 each fits the modal's
            # 96 cells once the rest of the row is in (index, name,
            # labels, percentages, mem readout in GiB).
            util_bar = _bar_colored(util / 100, 12)
            mem_bar = _bar_colored(mem_pct, 12)
            mem_used_gib = mem_used / 1024.0
            mem_total_gib = mem_total / 1024.0
            rows.append(
                f"  [bold]{idx}[/bold]  {name:<14}"
                f"  [dim]util[/dim] {util_bar} [bold]{util:>3}%[/bold]"
                f"  [dim]mem[/dim] {mem_bar} "
                f"[bold]{mem_used_gib:4.1f}[/bold]/{mem_total_gib:.0f} GiB"
            )
        return "\n".join(rows)

    @staticmethod
    def _render_procs(s: str) -> str:
        if not s.strip():
            return "[bold]Processes[/bold]\n  [dim]none[/dim]"
        # Parse rows out of `ps` output, sort by CPU% desc, show a
        # per-process CPU bar so the top burner-on-the-box is obvious.
        parsed: List[tuple] = []
        for line in s.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith("PID"):
                continue
            parts = line.split(None, 5)
            if len(parts) < 6:
                continue
            pid, user, cpu, mem, etime, comm = parts
            try:
                cpu_f = float(cpu)
                mem_f = float(mem)
            except ValueError:
                continue
            parsed.append((cpu_f, mem_f, pid, user, etime, comm))

        parsed.sort(reverse=True)
        parsed = parsed[:10]
        if not parsed:
            return "[bold]Processes[/bold]\n  [dim]none[/dim]"

        rows = [
            "[bold]Processes[/bold]  "
            "[dim](top by CPU% — beak compute tools)[/dim]",
        ]
        for cpu_f, mem_f, pid, user, etime, comm in parsed:
            # CPU% can exceed 100 on multi-thread processes; cap the
            # bar at 100% but show the real number on the right.
            bar = _bar_colored(min(cpu_f, 100.0) / 100.0, _BAR_W)
            rows.append(
                f"  {bar}  [bold]{cpu_f:>5.1f}%[/bold]  "
                f"{comm[:14]:<14}  "
                f"[dim]pid {pid:>6}  {user[:10]:<10}  "
                f"mem {mem_f:>4.1f}%  up {etime}[/dim]"
            )
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


# Shared bar width across host/GPU/process panes so the columns of
# numbers right of every bar line up in one visual ruler down the
# modal — hugely helpful for "is this thing maxed?" at a glance.
_BAR_W = 18


def _compact_gpu_name(name: str) -> str:
    """Strip vendor / brand boilerplate from an `nvidia-smi --query`
    GPU name so it fits the status modal's per-GPU row.

    "NVIDIA GeForce GTX 1080 Ti" → "GTX 1080 Ti"
    "NVIDIA A100-SXM4-40GB"      → "A100"
    "NVIDIA H100 80GB HBM3"      → "H100"

    Mirrors the `_compact_gpu_name` helper in the submit-embed modal
    — both panes want the same readable form.
    """
    if not name:
        return "?"
    n = name
    for prefix in ("NVIDIA GeForce ", "NVIDIA ", "GeForce ", "Tesla "):
        if n.startswith(prefix):
            n = n[len(prefix):]
            break
    # Datacenter SKUs encode product + interconnect + memory
    # ("A100-SXM4-40GB"). Keep just the product up to the first dash.
    if "-" in n:
        n = n.split("-", 1)[0]
    # Trim trailing "80GB", "HBM3" tokens.
    out_tokens = []
    for tok in n.split(" "):
        u = tok.upper()
        if u.endswith("GB") or u in ("HBM2", "HBM2E", "HBM3", "PCIE"):
            continue
        out_tokens.append(tok)
    return " ".join(out_tokens).strip() or name


def _bar_colored(frac: float, width: int = _BAR_W) -> str:
    """Block-drawing bar with the canonical green/yellow/red ramp.

    The thresholds match the top-right pill: <50% green (idle / safe),
    <85% yellow (loaded but room), ≥85% red (saturated). Same scheme
    everywhere keeps the modal legible without a legend.
    """
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    bar = "█" * filled + "░" * (width - filled)
    color = (
        "green" if frac < 0.5
        else "yellow" if frac < 0.85
        else "red"
    )
    return f"[{color}]{bar}[/{color}]"
