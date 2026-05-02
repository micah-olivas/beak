"""Top-level Textual app for beak."""

import resource

from textual.app import App

from .screens.projects import ProjectListScreen

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

    #sequence-view {{
        height: 1fr;
        min-height: 8;
    }}

    SubmitSearchModal, JobStatusModal, LayerDetailModal,
    SubmitEmbedModal, ImportExperimentModal, RemoteSetupModal,
    NewProjectModal, RenameProjectModal, SubmitTaxonomyModal {{
        align: center middle;
    }}

    SubmitSearchModal #modal-body,
    JobStatusModal #modal-body,
    LayerDetailModal #modal-body,
    SubmitEmbedModal #modal-body,
    RemoteSetupModal #modal-body,
    NewProjectModal #modal-body,
    RenameProjectModal #modal-body,
    SubmitTaxonomyModal #modal-body {{
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
        width: 120;
        height: 90%;
        min-height: 30;
    }}
    LayerDetailModal #sets-table {{
        height: auto;
        max-height: 12;
        margin-top: 1;
    }}
    LayerDetailModal #set-details-scroll {{
        height: 1fr;
        min-height: 10;
        margin-top: 1;
    }}
    LayerDetailModal .set-details {{
        padding: 0 1;
    }}
    LayerDetailModal .section-label {{
        margin-top: 1;
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
    SubmitTaxonomyModal #modal-buttons {{
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
    SubmitTaxonomyModal Button {{
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
