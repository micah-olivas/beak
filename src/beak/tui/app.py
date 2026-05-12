"""Top-level Textual app for beak."""

import os
# Tighten the Esc-key disambiguation timeout BEFORE importing Textual.
# Textual reads `ESCDELAY` from the environment exactly once at module
# import time (see `textual.constants.ESCAPE_DELAY`); the default is
# 100 ms, which is long enough to feel laggy when popping screens with
# Esc on a busy view like the alignment viewer. 25 ms is short enough
# to feel instant but still wide enough to disambiguate a real Esc
# press from the lead byte of an ANSI escape sequence (arrow keys,
# function keys) on every modern terminal — those sequences arrive
# back-to-back within 1–2 ms over a local pty. `setdefault` so the
# user can still override via the standard env var if needed.
os.environ.setdefault("ESCDELAY", "25")

import resource
import threading
from queue import Full, Queue

from textual.app import App
from textual.drivers import _writer_thread as _tx_writer

from .screens.projects import ProjectListScreen


# Module-level signal shared between the WriterThread patch and
# BeakApp's recovery hooks. Set by ``patched_write`` whenever a frame
# is dropped because the writer queue was full; read by BeakApp's
# heartbeat to schedule a full-screen re-sync.
_WRITER_DROPPED = threading.Event()
# Counter for diagnostic log entries — first drop per run is logged
# verbosely; afterwards we summarise.
_WRITER_DROP_COUNT = {"n": 0, "logged_first": False, "lock": threading.Lock()}


def _patch_textual_writer_thread() -> None:
    """Make Textual's stdout writer fail-soft under terminal back-pressure.

    Root cause (captured 2026-05-09 00:24 in ~/.beak/last-crash.log):
    Textual's ``WriterThread`` exposes a bounded ``Queue`` (default
    ``MAX_QUEUED_WRITES = 30``). When the asyncio loop's compositor
    refresh enqueues a frame and the writer thread is blocked inside
    ``os.write`` (because the pty buffer is full — slow terminal,
    flow control, scrollback redraw, …), ``Queue.put`` waits on
    ``not_full`` and the entire main thread parks. The input thread
    keeps reading bytes that nobody dispatches. Symptom: structure
    last frame visible, no keys, no mouse.

    Naïve fix (drop-on-full with a *bigger* queue) backfires: a larger
    queue holds *more* stale rotation frames, so when the user finally
    interacts, their refresh is queued behind ~25 seconds of stale
    animation OR is itself the one dropped. Display stays out of sync
    indefinitely because Textual emits incremental cell diffs — a
    dropped diff means those cells are wrong forever, until something
    triggers a full repaint.

    What works (this patch):

      1. **Smaller queue** (``maxsize`` 64, original was 30, was-256
         in the previous attempt). 64 is enough to absorb a screen-
         pop burst (~one frame per layout pass × the half-dozen
         widgets that mark dirty on transition) without holding so
         much stale animation that user input gets buried.
      2. **Non-blocking ``write``** via ``put_nowait``; on ``Full`` we
         drop the new frame AND set ``_WRITER_DROPPED``.
      3. **Auto-recovery**: ``BeakApp._writer_heartbeat`` polls
         ``_WRITER_DROPPED`` every second from the asyncio loop and,
         when set, writes a screen-clear directly to the driver and
         calls ``self.refresh(layout=True)`` to force a full repaint
         on the next paint pass. This re-syncs the compositor's model
         to the terminal — without it, dropped diffs leave permanent
         visual artefacts (the duplicated Footer / sequence rows the
         user reported on alignment-exit freeze).
      4. **First drop logged** verbosely to ``~/.beak/last-crash.log``;
         subsequent drops are counted and summarised at app exit
         (cheap, doesn't spam the log on a sustained-slow terminal).

    Patching the class (not an instance) means the driver's
    ``WriterThread(...)`` construction inherits the patched ``__init__``
    and ``write`` automatically — no need to thread the patch through
    driver init.
    """
    queue_size = 64
    original_init = _tx_writer.WriterThread.__init__

    def patched_init(self, file):  # type: ignore[no-untyped-def]
        # Run the original constructor first so we inherit any
        # attributes Textual sets up; then resize the queue in place.
        original_init(self, file)
        self._queue = Queue(queue_size)

    def _log_drop() -> None:
        with _WRITER_DROP_COUNT["lock"]:
            _WRITER_DROP_COUNT["n"] += 1
            if _WRITER_DROP_COUNT["logged_first"]:
                return
            _WRITER_DROP_COUNT["logged_first"] = True
        try:
            from datetime import datetime
            from pathlib import Path
            crash_path = Path.home() / ".beak" / "last-crash.log"
            crash_path.parent.mkdir(parents=True, exist_ok=True)
            with crash_path.open("a") as f:
                f.write(
                    f"\n=== {datetime.now().isoformat()} writer-queue "
                    f"saturated · dropping frame, scheduling full repaint "
                    f"===\n"
                )
        except Exception:  # noqa: BLE001
            pass

    def patched_write(self, text):  # type: ignore[no-untyped-def]
        try:
            self._queue.put_nowait(text)
        except Full:
            _log_drop()
            # Signal the asyncio loop to force a full repaint once it
            # has a chance. Without this, the compositor's "what's on
            # screen" model diverges from the terminal permanently —
            # dropped diffs show up as ghost cells / duplicated widgets.
            _WRITER_DROPPED.set()

    _tx_writer.WriterThread.__init__ = patched_init  # type: ignore[method-assign]
    _tx_writer.WriterThread.write = patched_write    # type: ignore[method-assign]
    _tx_writer.MAX_QUEUED_WRITES = queue_size


_patch_textual_writer_thread()

BEAK_BLUE = "#2E86AB"


def _raise_fd_limit() -> None:
    """Bump the FD soft limit toward the hard cap on startup.

    macOS ships with a 256 soft limit. Long-running TUI sessions that
    poll remote jobs can plausibly hit this even without leaks (each
    SSH transport, parquet read, and structure parse takes a few FDs).
    Raising the soft limit to 4096 (or the hard cap, whichever is
    smaller) gives plenty of headroom. Best-effort: silently ignore
    failures because the harness may not allow `setrlimit`.
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(4096, hard) if hard != resource.RLIM_INFINITY else 4096
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ValueError, OSError):
        pass


_raise_fd_limit()


class BeakApp(App):
    """beak TUI — browse long-lived projects and inspect their state."""

    CSS = f"""
    Screen {{
        background: $surface;
    }}

    Header {{
        background: {BEAK_BLUE};
    }}

    DataTable {{
        height: 1fr;
    }}

    .panel {{
        border: round {BEAK_BLUE};
        margin: 1 2;
        padding: 1 2;
    }}

    #detail-body {{
        height: auto;
        width: 100%;
    }}

    #target-panel:hover {{
        background: $boost;
    }}

    /* Explicit percentages — `1fr` was getting clipped to a small width
       when struct-col claimed too much under `auto`, squeezing the layers
       panel and truncating the homologs count + hiding the pills. */
    #info-col {{
        width: 55%;
        height: auto;
    }}

    #layers-panel {{
        min-height: 12;
    }}

    #struct-col {{
        width: 45%;
        height: auto;
    }}

    #struct-controls {{
        height: 1;
        padding: 0 1;
    }}

    /* Tight 5-char label slots — fits "color"/"view" with one
       trailing space and lets all three label+select pairs plus the
       Export button share a single row at narrow aspect ratios. */
    #struct-controls .ctrl-lbl {{
        width: auto;
        padding-right: 1;
        color: $text-muted;
        content-align: right middle;
    }}

    #color-select, #view-select, #bg-select {{
        width: 11;
        margin-right: 1;
    }}

    /* Push Export to the right edge of the structure column. */
    #export-cxc-btn {{
        margin-left: 1;
        min-width: 8;
        dock: right;
    }}

    #struct-meta {{
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }}

    #sequence-view {{
        height: 1fr;
        min-height: 8;
    }}

    /* Every modal gets the centered/panel/right-aligned-button chrome
       defined below. Adding a new modal class to this allowlist is
       the only thing required to make it stylistically consistent
       with the rest — without it, the modal renders top-left with
       no border, no panel background, and no button spacing. */

    SubmitSearchModal, JobStatusModal, LayerDetailModal,
    SubmitEmbedModal, ImportExperimentModal, RemoteSetupModal,
    NewProjectModal, RenameProjectModal, SubmitTaxonomyModal,
    SubmitComparativeModal, FilterLengthModal, RenameSetModal,
    DedupeSetModal, ServerStatusModal, SequenceDetailModal,
    ExportAlignmentModal {{
        align: center middle;
    }}

    SubmitSearchModal #modal-body,
    JobStatusModal #modal-body,
    LayerDetailModal #modal-body,
    SubmitEmbedModal #modal-body,
    RemoteSetupModal #modal-body,
    NewProjectModal #modal-body,
    RenameProjectModal #modal-body,
    SubmitTaxonomyModal #modal-body,
    SubmitComparativeModal #modal-body,
    FilterLengthModal #modal-body,
    RenameSetModal #modal-body,
    DedupeSetModal #modal-body,
    SequenceDetailModal #modal-body,
    ServerStatusModal #modal-body,
    ExportAlignmentModal #modal-body {{
        width: 64;
        height: auto;
        border: thick {BEAK_BLUE};
        background: $panel;
        padding: 1 2;
    }}

    /* Layer modal needs room for the homologs sets table + details.
       Height is fixed at 90% of viewport (rather than `max-height`) so
       cycling rows in the sets table — which changes the per-set
       details length — doesn't make the whole modal expand and contract
       around the cursor. The details panel lives inside a scroller that
       absorbs the variable height. */
    LayerDetailModal #modal-body {{
        width: 140;
        height: 90%;
        /* min-height covers title (~3) + sets table (max 12) +
           set-detail-row min (12) + the now-compact button stack
           (3 rows + margin) + breathing room. The buttons used to
           need 9 rows which forced a min of 38; now they fit in 4,
           so 32 is a comfortable floor that lets the modal render
           cleanly on 40-row terminals. */
        min-height: 32;
    }}
    LayerDetailModal #sets-table {{
        height: auto;
        max-height: 12;
        margin-top: 1;
    }}
    LayerDetailModal #set-detail-row {{
        height: 1fr;
        min-height: 12;
        margin-top: 1;
    }}
    LayerDetailModal #set-details-scroll {{
        width: 60%;
        height: 1fr;
    }}
    SequenceDetailModal #modal-body {{
        width: 95%;
        max-height: 90%;
    }}
    /* Server status modal — wider than the chrome default so the
       per-GPU rows (idx + name + util-bar + util% + mem-bar + mem
       readout in GiB) fit on a single line. The two-bars-per-GPU
       layout is intentionally information-dense; clipping to 64
       cells made it wrap awkwardly. */
    ServerStatusModal #modal-body {{
        width: 96;
        max-height: 90%;
    }}
    /* Structures-layer gallery — canvas fills available space; the
       info card sits below it as a bordered panel. */
    LayerDetailModal _StructureGalleryCanvas {{
        height: 1fr;
        min-height: 20;
    }}
    LayerDetailModal #struct-gallery-card {{
        height: auto;
        border: round $surface-lighten-1;
        padding: 0 1;
        margin-top: 1;
        margin-bottom: 0;
    }}
    LayerDetailModal #struct-gallery-title {{
        height: 1;
        content-align: center middle;
        overflow: hidden hidden;
    }}
    LayerDetailModal #struct-gallery-hint {{
        height: 1;
        content-align: center middle;
        color: $text-muted;
    }}
    LayerDetailModal #length-hist-panel {{
        width: 40%;
        height: 1fr;
        padding: 1 2;
        margin-left: 1;
        border: round $surface-lighten-1;
    }}
    LayerDetailModal .set-details {{
        padding: 0 1;
    }}
    LayerDetailModal .section-label {{
        margin-top: 1;
    }}

    /* Modal-buttons is a Vertical stack of 3 rows so the action set
       fits in the modal's width even when every contextual button is
       visible. Each row is its own Horizontal flowing left-to-right;
       the bottom row uses the flexible spacer to push Search / Close
       to the right edge.

       The buttons are all `compact=True` (1 row tall), so each
       btn-row is 1 row and the stack as a whole is exactly 3 rows
       plus its margin-top. The previous height was 9 (3×3) which
       worked on tall terminals but got clipped at ~43 rows because
       the sibling `1fr` detail row would consume most of the body
       and leave fewer than 9 rows for the stack — exactly two
       button rows would fall off the bottom edge.

       Pinning the height explicitly (vs. `auto`) is still needed
       because the detail row's `1fr` would otherwise squeeze the
       button Vertical to its first child. */
    LayerDetailModal #modal-buttons {{
        height: 3;
        margin-top: 1;
    }}
    LayerDetailModal .btn-row {{
        width: 100%;
        height: 1;
        align: left middle;
    }}
    /* Fixed-width category label on the left of each row. 9 cells is
       enough for "Compute" / "Danger" / "Set" (longest is 7 chars +
       2 trailing for breathing room) and small enough that the
       button area still has full-modal width to lay out. */
    LayerDetailModal .row-label {{
        width: 9;
        height: 1;
        padding-right: 1;
        content-align: right middle;
    }}
    LayerDetailModal .btn-group {{
        width: auto;
        height: 1;
    }}
    /* Uniform spacing between buttons inside a btn-group — each
       button gets 2 cells of right margin, so the visual cadence is
       the same row-to-row even when individual labels are wider.
       The old layout had nested groups with different padding rules,
       which made columns appear to drift between rows. */
    LayerDetailModal .btn-group Button {{
        margin-right: 2;
    }}
    /* Right-anchored cluster (Search / Close on the last row) has
       no trailing margin past the final button — keeps it flush
       with the modal's right edge. */
    LayerDetailModal .right-group Button:last-of-type {{
        margin-right: 0;
    }}
    LayerDetailModal #btn-spacer-grow {{
        width: 1fr;
        height: 1;
    }}

    /* Import modal needs more room for the CSV preview table. */
    ImportExperimentModal #modal-body {{
        width: 96;
        height: auto;
        max-height: 90%;
        border: thick {BEAK_BLUE};
        background: $panel;
        padding: 1 2;
    }}

    SubmitSearchModal #modal-buttons,
    JobStatusModal #modal-buttons,
    LayerDetailModal #modal-buttons,
    SubmitEmbedModal #modal-buttons,
    ImportExperimentModal #modal-buttons,
    RemoteSetupModal #modal-buttons,
    NewProjectModal #modal-buttons,
    RenameProjectModal #modal-buttons,
    SubmitTaxonomyModal #modal-buttons,
    SubmitComparativeModal #modal-buttons,
    FilterLengthModal #modal-buttons,
    RenameSetModal #modal-buttons,
    DedupeSetModal #modal-buttons,
    SequenceDetailModal #modal-buttons,
    ServerStatusModal #modal-buttons {{
        height: 3;
        align: right middle;
        margin-top: 1;
    }}

    SubmitSearchModal Button,
    JobStatusModal Button,
    LayerDetailModal Button,
    SubmitEmbedModal Button,
    ImportExperimentModal Button,
    RemoteSetupModal Button,
    NewProjectModal Button,
    RenameProjectModal Button,
    SubmitTaxonomyModal Button,
    SubmitComparativeModal Button,
    FilterLengthModal Button,
    RenameSetModal Button,
    DedupeSetModal Button,
    SequenceDetailModal Button,
    ServerStatusModal Button {{
        margin-left: 1;
    }}
    """

    TITLE = "beak"
    SUB_TITLE = "projects"

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def on_mount(self) -> None:
        # Project list is always pushed first; if the user has no SSH
        # configured we then layer the setup modal on top so dismissing
        # it lands on a usable screen rather than a blank app.
        self.push_screen(ProjectListScreen())
        from ..config import load_config
        cfg = load_config()
        if not (cfg.get("connection") or {}).get("host"):
            from .screens.remote_setup import RemoteSetupModal
            self.push_screen(RemoteSetupModal(first_run=True))
        # Freeze-debugging hook: when the user runs `kill -USR1 <pid>`
        # we dump every Python thread's stack trace to
        # ~/.beak/last-crash.log. That's the smoking-gun diagnostic for
        # the input-thread-is-dead scenario where no Python exception
        # has fired but keys aren't reaching the app.
        try:
            import signal as _signal
            _signal.signal(_signal.SIGUSR1, self._dump_all_threads)
            # SIGUSR2 — panic exit. The watchdog has shown that during
            # the alignment/PCA-exit "freeze" the asyncio loop is in
            # fact alive (heartbeat keeps ticking), but input/bindings
            # don't dispatch — typically a focus or mouse-capture issue
            # that you can't recover from inside the app. ``kill -USR2
            # <pid>`` from another terminal triggers an immediate
            # ``os._exit`` so the user can recover the terminal without
            # having to ``kill -9``.
            _signal.signal(_signal.SIGUSR2, self._panic_exit)
        except Exception:  # noqa: BLE001
            pass
        # Auto-watchdog (two layers, to cover GIL-bound hangs).
        #
        #   * Python-level daemon thread (`_start_watchdog`) — pings the
        #     asyncio loop and dumps stacks if the loop stops
        #     responding. Catches "loop is alive but not responding"
        #     (e.g. asyncio task spinning, awaitable never completing)
        #     because it does its work from a separate Python thread.
        #
        #   * C-level `faulthandler` (`_arm_faulthandler`) — dumps stacks
        #     after a timeout from a C thread that does NOT need the
        #     GIL. Catches "main thread holds the GIL forever" (e.g.
        #     a C-level regex catastrophic backtrack like the one that
        #     froze the UI on the markup tokenizer). The Python-only
        #     watchdog can't fire in that scenario because daemon
        #     threads can't get the GIL.
        #
        # The startup log entry confirms the new code is actually
        # running — useful when debugging "did I restart?" doubts.
        self._watchdog_log_startup()
        self._arm_faulthandler()
        self._start_watchdog()
        # Writer-drop recovery. When the WriterThread patch drops a
        # frame (pty back-pressure), it sets _WRITER_DROPPED. This
        # 0.5s timer reads the flag from the asyncio loop and forces a
        # full clear + repaint, re-syncing the compositor's model to
        # the terminal so dropped diffs don't leave permanent ghost
        # cells. Keeping the cadence aggressive (twice per second)
        # means a transient slow-down recovers quickly; the timer is
        # otherwise free (one Event.is_set check per tick).
        self.set_interval(0.5, self._writer_heartbeat)

    def _writer_heartbeat(self) -> None:
        """If WriterThread dropped frames, wait for queue to drain
        then force a full re-paint.

        Critical sequencing: the compositor sends incremental cell
        diffs; a dropped diff means those cells stay stale until
        something dirties them again. We need to recover by triggering
        a full repaint — but only *after* the queue has drained,
        otherwise our recovery refresh just adds to the saturation.

        Cheap when no drops have happened: one ``Event.is_set()`` check
        per tick. When drops have happened, we wait for queue depth
        to fall below a low watermark before scheduling the repaint.
        """
        if not _WRITER_DROPPED.is_set():
            return
        # Don't fight the back-pressure: only re-sync once the writer
        # has drained most of what it had. Peeking at queue size
        # touches Textual internals defensively.
        try:
            wq = self._driver._writer_thread._queue
            if wq.qsize() > 3:
                return
        except Exception:  # noqa: BLE001
            pass
        _WRITER_DROPPED.clear()
        # Trust Textual: refresh the full screen layout. Don't write a
        # raw clear-screen escape here — that goes through the same
        # writer queue and could itself be dropped or queued behind
        # the next round of writes, defeating the recovery.
        try:
            self.refresh(layout=True, repaint=True)
        except Exception:  # noqa: BLE001
            pass
        # Back-pressure can desync the mouse-button protocol: a
        # button-down lands but the matching button-up is dropped (or
        # arrives interleaved with stale frames the user hasn't seen
        # yet), leaving a widget stuck in `capture_mouse`. From that
        # point every click anywhere on screen is silently routed to
        # the captured widget. Release any stale capture as part of
        # recovery — the live drag (if any) loses one frame, but the
        # whole UI doesn't lose mouse input until the next process
        # restart.
        try:
            captured = self.mouse_captured
            if captured is not None:
                captured.release_mouse()
        except Exception:  # noqa: BLE001
            pass

    def on_resize(self, event) -> None:
        """Force a full screen clear + repaint on every terminal resize.

        Without this, Textual's partial-update compositor can leave
        ghost rows of old content visible after the terminal grows —
        most obvious as duplicated Footer / SequenceView colorbars
        stacking up at the bottom of the screen on every resize. The
        symptom is purely visual but readily creates the impression
        that the app is wedged. We:

          1. Write `ESC[2J ESC[H` directly to the terminal via the
             driver's writer thread — wipes every cell at the OS
             level, not just Textual's compositor cache.
          2. Tell Textual to do a full layout-and-repaint pass so the
             cleared cells get repopulated with the resized layout.
        """
        try:
            driver = self._driver
            if driver is not None:
                # `Esc[2J` clears the screen; `Esc[H` sends cursor to
                # row=1 col=1 so subsequent draws start at the top.
                driver.write("\x1b[2J\x1b[H")
        except Exception:  # noqa: BLE001
            pass
        try:
            self.refresh(layout=True)
        except Exception:  # noqa: BLE001
            pass

    # Watchdog tunables. The threshold is comfortably above any expected
    # render time (heaviest renders we measured complete in <100ms) and
    # the check cadence is fast enough to land within a couple of
    # seconds of the actual hang while still being cheap (one timestamp
    # comparison per 0.5 s on a daemon thread). Aggressive on purpose
    # so the diagnostic dump happens quickly enough to be useful — and
    # so the user doesn't wonder "did the watchdog fire?" when checking
    # the log immediately after a freeze.
    _WATCHDOG_TIMEOUT_SEC = 5.0
    _WATCHDOG_POLL_SEC = 1.0

    def _arm_faulthandler(self) -> None:
        """Arm `faulthandler.dump_traceback_later` for hard GIL-bound hangs.

        Python's `faulthandler` runs from a C thread, so it can fire
        even when the main thread is holding the GIL forever (e.g.
        catastrophic regex backtrack). We point it at a dedicated log
        file (`~/.beak/last-faulthandler.log`) and re-arm it from the
        asyncio loop's idle ticks via `_watchdog_heartbeat` — every
        time the loop is alive enough to run a callback, the timer
        resets. If the loop genuinely hangs, the timer expires and
        `faulthandler` dumps every thread's stack to disk.
        """
        try:
            import faulthandler as _fh
            from pathlib import Path
            log_path = Path.home() / ".beak" / "last-faulthandler.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # Keep the file open for the lifetime of the app — fault-
            # handler writes via this fd from a C thread, so it must
            # outlive any callable that owns it.
            self._faulthandler_file = log_path.open("a", buffering=1)
            self._faulthandler_file.write(
                f"\n=== {self._iso_now()} faulthandler armed "
                f"(timeout={self._WATCHDOG_TIMEOUT_SEC}s) ===\n"
            )
            _fh.dump_traceback_later(
                self._WATCHDOG_TIMEOUT_SEC,
                repeat=True,
                file=self._faulthandler_file,
            )
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _iso_now() -> str:
        from datetime import datetime
        return datetime.now().isoformat()

    def _watchdog_log_startup(self) -> None:
        """Log a one-line "watchdog active" entry on app start.

        Lets the user verify which process they're running just by
        tailing `~/.beak/last-crash.log`. If the timestamp doesn't
        match the most recent restart, they're still on a stale
        process.
        """
        from datetime import datetime
        from pathlib import Path
        try:
            crash_path = Path.home() / ".beak" / "last-crash.log"
            crash_path.parent.mkdir(parents=True, exist_ok=True)
            with crash_path.open("a") as f:
                f.write(
                    f"\n=== {datetime.now().isoformat()} watchdog active "
                    f"(timeout={self._WATCHDOG_TIMEOUT_SEC}s) ===\n"
                )
        except Exception:  # noqa: BLE001
            pass

    def _start_watchdog(self) -> None:
        """Spawn a daemon thread that auto-dumps stacks if the loop wedges.

        The thread runs an outer loop:
          1. Schedule a tiny coroutine on the asyncio loop that updates
             a shared "last heartbeat" timestamp.
          2. Sleep for `_WATCHDOG_POLL_SEC`.
          3. If the heartbeat is older than `_WATCHDOG_TIMEOUT_SEC`,
             the loop is wedged → dump every thread's stack to
             last-crash.log and reset (so we don't spam the log if the
             hang is permanent).

        Keeps a flag so the dump only fires once per hang event.
        """
        import threading as _th
        import time as _time
        self._watchdog_last_beat = _time.monotonic()
        self._watchdog_dumped = False

        def _watchdog_target() -> None:
            # Continue past transient errors — `call_soon_threadsafe`
            # raises `RuntimeError("Loop is closed")` during startup
            # and shutdown, and we don't want one of those to silently
            # kill the watchdog and leave the app un-monitored.
            from datetime import datetime
            from pathlib import Path
            heartbeat_log = Path.home() / ".beak" / "watchdog-trace.log"
            try:
                heartbeat_log.parent.mkdir(parents=True, exist_ok=True)
            except Exception:  # noqa: BLE001
                pass
            iteration = 0
            while True:
                try:
                    loop = self._loop
                    if loop is not None and loop.is_running():
                        loop.call_soon_threadsafe(self._watchdog_heartbeat)
                except Exception:  # noqa: BLE001
                    pass
                _time.sleep(self._WATCHDOG_POLL_SEC)
                iteration += 1
                try:
                    age = _time.monotonic() - self._watchdog_last_beat
                    # Every 10th iteration (~5s) write a one-line trace
                    # so we can confirm the watchdog is actually running
                    # and see how stale the heartbeat is. Cheap append.
                    if iteration % 10 == 0:
                        try:
                            with heartbeat_log.open("a") as f:
                                f.write(
                                    f"{datetime.now().isoformat()} "
                                    f"iter={iteration} "
                                    f"heartbeat_age={age:.2f}s "
                                    f"loop_running="
                                    f"{loop.is_running() if loop else 'no_loop'}\n"
                                )
                        except Exception:  # noqa: BLE001
                            pass
                    if age > self._WATCHDOG_TIMEOUT_SEC:
                        if not self._watchdog_dumped:
                            self._dump_all_threads(reason=(
                                f"watchdog: loop wedged for {age:.1f}s"
                            ))
                            self._watchdog_dumped = True
                    else:
                        self._watchdog_dumped = False
                except Exception:  # noqa: BLE001
                    pass

        thread = _th.Thread(
            target=_watchdog_target, name="beak-watchdog", daemon=True,
        )
        thread.start()

    def _watchdog_heartbeat(self) -> None:
        """Called from the asyncio loop to mark it alive.

        Updates the Python-watchdog's timestamp and re-arms
        `faulthandler.dump_traceback_later` so the C-level dump only
        fires on a true hang. If we forget to re-arm,
        `faulthandler` would print every `_WATCHDOG_TIMEOUT_SEC`
        seconds on a healthy app.
        """
        import time as _time
        self._watchdog_last_beat = _time.monotonic()
        # Re-arm faulthandler. Cheap: it's a single C call.
        try:
            import faulthandler as _fh
            _fh.dump_traceback_later(
                self._WATCHDOG_TIMEOUT_SEC,
                repeat=True,
                file=self._faulthandler_file,
            )
        except Exception:  # noqa: BLE001
            pass

    def _dump_widget_tree(self, f) -> None:
        """Append the active screen's widget tree to a file.

        Helpful for diagnosing freezes that look like input is being
        consumed by a phantom widget — duplicated Footer/colorbar rows
        after a resize would show up here as multiple Footer widgets
        on the same screen.
        """
        try:
            f.write("\n--- widget tree (active screen) ---\n")
            screen = self.screen
            f.write(f"active screen: {screen!r}\n")
            f.write(f"focused: {screen.focused!r}\n")
            stack = self._screen_stack
            f.write(f"screen stack size: {len(stack)}\n")
            for i, s in enumerate(stack):
                f.write(f"  [{i}] {s!r}\n")
            f.write("\n--- screen DOM walk ---\n")

            def _walk(node, depth=0):
                indent = "  " * depth
                try:
                    cls = type(node).__name__
                    nid = getattr(node, "id", None) or ""
                    f.write(f"{indent}{cls}({nid})\n")
                except Exception:  # noqa: BLE001
                    return
                try:
                    children = list(node.children)
                except Exception:  # noqa: BLE001
                    children = []
                for child in children:
                    _walk(child, depth + 1)

            _walk(screen)

            # Footer count is the smoking gun for the resize-ghost bug.
            try:
                from textual.widgets import Footer
                footer_count = len(list(screen.query(Footer)))
                f.write(f"\nFooter count on active screen: {footer_count}\n")
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # noqa: BLE001
            f.write(f"widget tree dump failed: {e!r}\n")

    def _dump_all_threads(self, signum=None, frame=None, *, reason: str = "SIGUSR1") -> None:
        """SIGUSR1 handler: dump every thread's stack to last-crash.log.

        Useful when the TUI is frozen — `kill -USR1 <pid>` from another
        terminal triggers a full thread-stack dump showing exactly what
        each thread is blocked on. The Textual input thread should
        normally be in `selectors.select(0.1)`; if it's somewhere else
        (or missing entirely) that tells us the freeze cause.
        """
        import sys as _sys
        import threading as _th
        import traceback as _tb
        from datetime import datetime
        from pathlib import Path
        try:
            crash_path = Path.home() / ".beak" / "last-crash.log"
            crash_path.parent.mkdir(parents=True, exist_ok=True)
            with crash_path.open("a") as f:
                f.write(
                    f"\n=== {datetime.now().isoformat()} "
                    f"thread dump ({reason}) ===\n"
                )
                for thread in _th.enumerate():
                    f.write(f"\n--- thread {thread.name} ({thread.ident}) ")
                    f.write(f"daemon={thread.daemon} alive={thread.is_alive()}\n")
                    stack = _sys._current_frames().get(thread.ident)
                    if stack is not None:
                        _tb.print_stack(stack, file=f)
                    else:
                        f.write("(no stack)\n")
                # Also dump the widget tree so we can spot phantom
                # widgets (the resize-ghost case) — input might be
                # going to a Footer that's no longer the active one.
                self._dump_widget_tree(f)
                # And every asyncio task's await chain — this is what
                # finally cracked the May 2026 screen-pop wedge.
                self._dump_async_tasks(f)
        except Exception:  # noqa: BLE001
            pass

    def _dump_async_tasks(self, f) -> None:
        """Append every asyncio task's await chain to ``f``.

        Diagnostic for the screen-pop wedge that bit us in May 2026
        (an attribute-name collision against MessagePump's ``_closing``
        flag silently dropped Prune messages, leaving alignment screen
        pumps blocked at ``Queue.get`` forever — and the App's dispatch
        coroutine wedged awaiting ``screen.remove()``). The fix is now
        in place, but if a similar wedge ever recurs, ``kill -USR1
        <pid>`` writes this dump alongside the thread dump and the
        await-chain frames pinpoint where the loop is parked.

        Walks ``coro.cr_await`` recursively to capture the full chain
        of suspended coroutines (Python's ``Task.get_stack`` only
        returns the topmost suspension frame).
        """
        import asyncio as _asyncio
        try:
            f.write("\n--- asyncio task await-chains ---\n")
            try:
                loop = self._loop
                tasks = list(_asyncio.all_tasks(loop)) if loop else []
            except Exception:  # noqa: BLE001
                tasks = []
            f.write(f"Total asyncio tasks on loop: {len(tasks)}\n")
            for i, task in enumerate(tasks):
                f.write(f"\n--- task[{i}] {task.get_name()!r}"
                        f" done={task.done()}\n")
                try:
                    if task.done():
                        exc = task.exception()
                        if exc is not None:
                            f.write(f"  exception: {exc!r}\n")
                            continue
                    coro = task.get_coro()
                    depth = 0
                    while coro is not None and depth < 20:
                        frame = getattr(coro, "cr_frame", None)
                        if frame is None:
                            break
                        fname = frame.f_code.co_filename
                        f.write(
                            f"  [{depth}] {fname}:{frame.f_lineno}"
                            f" in {frame.f_code.co_name}\n"
                        )
                        next_coro = getattr(coro, "cr_await", None)
                        if next_coro is not None and not hasattr(
                            next_coro, "cr_frame"
                        ):
                            next_coro = getattr(
                                next_coro, "_coro", None
                            ) or getattr(next_coro, "gi_frame", None)
                        if next_coro is None or not hasattr(
                            next_coro, "cr_frame"
                        ):
                            break
                        coro = next_coro
                        depth += 1
                except Exception as e:  # noqa: BLE001
                    f.write(f"  err getting stack: {e!r}\n")
        except Exception:  # noqa: BLE001
            pass

    def action_quit(self) -> None:
        """Exit the app immediately, bypassing thread-pool join.

        Textual's graceful ``app.exit()`` waits for all
        ``@work(thread=True)`` workers to finish. Those workers open SSH
        connections with a 10-second connect_timeout, so pressing ``q``
        while a status poll is in-flight leaves the terminal hanging for
        up to 10 s. ``os._exit`` skips Python's non-daemon thread join
        entirely, matching the SIGUSR2 panic-exit pattern already in
        this file.
        """
        try:
            import faulthandler as _fh
            _fh.cancel_dump_traceback_later()
        except Exception:  # noqa: BLE001
            pass
        try:
            import os as _os
            _os.system("stty sane 2>/dev/null")
        except Exception:  # noqa: BLE001
            pass
        import os as _os
        _os._exit(0)

    def _panic_exit(self, signum=None, frame=None) -> None:
        """SIGUSR2 handler: bail out hard, leave a note about why.

        Lets the user `kill -USR2 <pid>` to escape a wedged TUI cleanly
        — without a SIGKILL that leaves the terminal in raw mode. We
        first try to dump diagnostics so the freeze state is captured
        in the log, then call ``os._exit(1)`` (skips atexit + Textual
        shutdown so we never wait on the writer queue).
        """
        try:
            self._dump_all_threads(reason="SIGUSR2 panic-exit")
        except Exception:  # noqa: BLE001
            pass
        try:
            from datetime import datetime
            from pathlib import Path
            with (Path.home() / ".beak" / "last-crash.log").open("a") as f:
                f.write(
                    f"\n=== {datetime.now().isoformat()} panic-exit "
                    f"via SIGUSR2 ===\n"
                )
        except Exception:  # noqa: BLE001
            pass
        # Try to leave the terminal in cooked mode so the user's shell
        # is usable afterwards. Best-effort.
        try:
            import os as _os
            _os.system("stty sane 2>/dev/null")
        except Exception:  # noqa: BLE001
            pass
        import os as _os
        _os._exit(1)

    def _handle_exception(self, error: Exception) -> None:
        """Dump unhandled exceptions to a log file before Textual swallows them.

        Textual's `_handle_exception` flips `App._closing = True`, which makes
        `post_message` silently drop every subsequent event (including Key
        events) — the app appears to lock up because input is no longer
        delivered, but no traceback is ever shown because the rendered
        traceback is stashed in `_exit_renderables` and only printed after
        the app fully exits. If a message pump is hung mid-cleanup the app
        never reaches that print, so the user just sees a frozen TUI.

        Writing the traceback to `~/.beak/last-crash.log` gives us a way to
        diagnose those silent crashes after the fact.
        """
        self._dump_freeze_log("_handle_exception", error=error)
        super()._handle_exception(error)

    def panic(self, *renderables) -> None:
        """Log panics to disk before they get hidden in `_exit_renderables`."""
        self._dump_freeze_log(
            "panic",
            extra=f"renderables={renderables!r}",
        )
        super().panic(*renderables)

    def _close_messages_no_wait(self) -> None:
        """Log every call so we can see *why* the app entered shutdown mode.

        After this runs, `App._closing = True` and every subsequent
        `post_message` (including Key events) returns False silently —
        which is the freeze the user is hitting.
        """
        self._dump_freeze_log("_close_messages_no_wait")
        super()._close_messages_no_wait()

    def _dump_freeze_log(
        self,
        hook: str,
        *,
        error: Exception | None = None,
        extra: str = "",
    ) -> None:
        """Append a frame to ~/.beak/last-crash.log capturing the call stack.

        We instrument every Textual code path that could silently make
        `App._closing = True` (the root cause of "input goes dead, app
        appears frozen"). Each call appends to the log so we can see the
        exact ordering when the freeze hits.
        """
        import traceback as _tb
        from datetime import datetime
        from pathlib import Path
        try:
            crash_path = Path.home() / ".beak" / "last-crash.log"
            crash_path.parent.mkdir(parents=True, exist_ok=True)
            with crash_path.open("a") as f:
                f.write(f"\n=== {datetime.now().isoformat()} {hook} ===\n")
                if extra:
                    f.write(f"{extra}\n")
                if error is not None:
                    f.write(f"{type(error).__name__}: {error}\n\n")
                    _tb.print_exception(
                        type(error), error, error.__traceback__, file=f,
                    )
                else:
                    f.write("call stack:\n")
                    _tb.print_stack(file=f)
        except Exception:  # noqa: BLE001
            pass
