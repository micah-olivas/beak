"""Top-right server-status pill — load, GPU util, and remote job count.

One SSH round-trip per poll fetches all three numbers; the dot color
summarizes whichever signal is closest to its cap (load/cores or GPU
utilization). Polled on the same 15s cadence as the layers panel.

When SSH defaults aren't configured or the host is unreachable, the
widget self-mutes with a dim hint instead of nagging on every tick.
"""

from pathlib import Path
from typing import Optional

from fabric import Connection
from textual import work
from textual.widgets import Static


_POLL_SECONDS = 15.0

# Single-line remote probe. Each command is `2>/dev/null`'d so a missing
# tool (e.g. nvidia-smi on a CPU-only host) leaves its slot empty rather
# than printing an error into our pipe-delimited output.
_PROBE_CMD = (
    "load=$(awk '{print $1}' /proc/loadavg 2>/dev/null); "
    "cores=$(nproc 2>/dev/null); "
    'gpu=$(nvidia-smi --query-gpu=utilization.gpu '
    '--format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d " "); '
    "jobs=$(ps -eo comm 2>/dev/null | "
    "grep -cE '^(mmseqs|clustalo|mafft|muscle|hmmscan|hmmsearch|jackhmmer)$'); "
    'echo "$load|$cores|$gpu|$jobs"'
)


class ServerStatusBar(Static):
    """Single-line pill in the top-right of the project screen.

    Shows: 1-min loadavg ÷ cores, GPU utilization (when nvidia-smi is
    available), and a count of search/align processes on the remote.
    Green/yellow/red dot tracks whichever signal is closest to its cap.
    """

    DEFAULT_CSS = """
    ServerStatusBar {
        height: 1;
        padding: 0 2 0 0;
        text-align: right;
        color: $text-muted;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("[dim]srv · checking…[/dim]", **kwargs)
        self._unavailable: bool = False

    def on_mount(self) -> None:
        self._poll()
        self.set_interval(_POLL_SECONDS, self._poll)

    def _build_connection(self) -> Optional[Connection]:
        """Open a fresh SSH connection from config defaults.

        Mirrors the connection setup in `RemoteJobManager.__init__` but
        skips the per-job housekeeping (remote dir creation, job DB
        bootstrap) — we only need to run one cheap probe and tear down.
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
                # `_find_ssh_key` is an instance method but doesn't touch
                # `self`; passing None mirrors how `BeakSession.__init__`
                # bootstraps before any RemoteJobManager exists.
                key_path = RemoteJobManager._find_ssh_key(None)

            return Connection(
                host=host,
                user=user,
                connect_timeout=10,
                connect_kwargs={
                    "key_filename": str(Path(key_path).expanduser())
                },
                config=_no_stdin_config(),
            )
        except Exception:
            return None

    @work(thread=True, exclusive=True, group="server-status")
    def _poll(self) -> None:
        if self._unavailable:
            return

        conn = self._build_connection()
        if conn is None:
            # No SSH defaults configured — self-mute permanently. The
            # user can re-launch after `beak config init`.
            self._unavailable = True
            self.app.call_from_thread(
                self.update, "[dim]srv · not configured[/dim]"
            )
            return

        try:
            result = conn.run(_PROBE_CMD, hide=True, warn=True, timeout=8)
            stdout = result.stdout if result and result.ok else ""
        except Exception:
            self.app.call_from_thread(
                self.update,
                "[red]●[/red] [dim]srv · unreachable[/dim]",
            )
            stdout = None
        finally:
            # Mirror the layers-panel SSH-FD discipline: every poll opens
            # a fresh Paramiko transport, so a missed close at this
            # cadence exhausts macOS's 256-FD soft limit in minutes.
            try:
                conn.close()
            except Exception:
                pass

        if stdout is None:
            return

        text = self._format_status(stdout.strip())
        self.app.call_from_thread(self.update, text)

    @staticmethod
    def _format_status(probe_out: str) -> str:
        if not probe_out or "|" not in probe_out:
            return "[dim]srv · no data[/dim]"
        parts = probe_out.split("|")
        if len(parts) < 4:
            return "[dim]srv · no data[/dim]"

        load_s, cores_s, gpu_s, jobs_s = parts[0], parts[1], parts[2], parts[3]
        try:
            load = float(load_s)
        except ValueError:
            load = 0.0
        try:
            cores = max(1, int(cores_s))
        except ValueError:
            cores = 1
        gpu: Optional[int] = None
        if gpu_s.strip():
            try:
                gpu = int(gpu_s)
            except ValueError:
                gpu = None
        try:
            jobs = int(jobs_s)
        except ValueError:
            jobs = 0

        # Pick the worst of (load/cores, gpu/100) for the dot color, so
        # a green host with a saturated GPU still reads as yellow/red.
        load_ratio = load / cores
        worst = max(load_ratio, (gpu / 100.0) if gpu is not None else 0.0)
        if worst < 0.5:
            color = "green"
        elif worst < 0.85:
            color = "yellow"
        else:
            color = "red"

        chunks = [
            f"[{color}]●[/{color}] [dim]srv ·[/dim] "
            f"[bold]load[/bold] {load:.2f}"
        ]
        if gpu is not None:
            chunks.append(f"[bold]gpu[/bold] {gpu}%")
        chunks.append(f"[bold]{jobs}[/bold] jobs")
        return "  ·  ".join(chunks)
