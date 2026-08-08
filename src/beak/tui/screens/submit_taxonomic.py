"""Modal: cluster the MSA by a taxonomy rank, score per-position clade bias.

Sibling of `submit_comparative.py`. Where that modal splits on a trait
threshold (two groups), this one partitions the alignment by a lineage
rank (N clades) and scores how much each column's residue variation is
explained by clade membership.

The three control families discussed in design surface here:
  · Grouping — which lineage rank defines the clades (`#rank-select`).
  · Rigor    — Henikoff sequence weighting (`#weight-switch`) and the
               permutation-null size (`#perm-input`); with 0 permutations
               the score is the uncertainty coefficient, otherwise a
               per-position z-score. Plus a minimum clade size
               (`#minclade-input`).
  · Output   — score only, or score + per-clade profiles for drill-down
               (`#profiles-switch`).

Build runs synchronously in a worker (seconds even with a permutation
null on hundreds of homologs) and dismisses with the rank name so the
parent screen can flip the structure view to the taxonomic color mode.
"""

from typing import Optional

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Switch

from ...project import BeakProject


class SubmitTaxonomicModal(ModalScreen[Optional[str]]):
    """Pick a lineage rank + rigor controls; cache taxonomic bias scores."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, project: BeakProject) -> None:
        super().__init__()
        self._project = project
        self._busy = False
        self._ranks: list = []

    def compose(self) -> ComposeResult:
        from ..comparative import rank_columns
        self._ranks = rank_columns(self._project)

        with Vertical(id="modal-body"):
            yield Label(
                f"[bold]Taxonomic clustering · {self._project.name}[/bold]",
                id="modal-title",
            )
            yield Label(
                "[dim]Group homologs by lineage; color residues by how much "
                "of their variation tracks clade membership.[/dim]"
            )
            yield Label("")

            if not self._ranks:
                yield Label(
                    "[red]No taxonomy ranks available — pull homologs + "
                    "taxonomy first.[/red]"
                )
                with Horizontal(id="modal-buttons"):
                    yield Button("Close", id="cancel-btn")
                return

            # Prefer a mid-depth rank when present: coarse enough to have
            # several well-populated clades, fine enough to be informative.
            default = next(
                (r for r in ("phylum", "class", "order") if r in self._ranks),
                self._ranks[0],
            )
            yield Label("Rank (groups the alignment into clades)")
            yield Select(
                [(r, r) for r in self._ranks],
                value=default, id="rank-select", allow_blank=False,
            )
            yield Label("", id="rank-stats")

            yield Label("Min sequences per clade (smaller clades dropped)")
            yield Input(value="3", id="minclade-input")

            with Horizontal(classes="switch-row"):
                yield Switch(value=True, id="weight-switch")
                yield Label(
                    "  Henikoff weighting (down-weight redundant sequences)"
                )

            yield Label(
                "Permutations (0 = uncertainty coefficient · "
                ">0 = z-score vs shuffled-label null)"
            )
            yield Input(
                value="0", placeholder="e.g. 200 for a significance z-score",
                id="perm-input",
            )

            with Horizontal(classes="switch-row"):
                yield Switch(value=False, id="profiles-switch")
                yield Label("  Also write per-clade profiles (drill-down)")

            yield Label("", id="status-line")

            with Horizontal(id="modal-buttons"):
                yield Button("Cancel", id="cancel-btn")
                yield Button("Build", id="submit-btn", variant="primary")

    def on_mount(self) -> None:
        if self._ranks:
            self._refresh_stats(self.query_one("#rank-select", Select).value)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "rank-select":
            self._refresh_stats(str(event.value))

    def _current_min_per_clade(self) -> int:
        try:
            return max(1, int(self.query_one("#minclade-input", Input).value.strip()))
        except (ValueError, Exception):  # noqa: BLE001
            return 3

    def _refresh_stats(self, rank: str) -> None:
        from ..comparative import rank_summary
        info = rank_summary(self._project, rank, self._current_min_per_clade())
        stats = self.query_one("#rank-stats", Label)
        if not info or info.get("n", 0) == 0:
            stats.update("[red]no taxonomy values at this rank[/red]")
            return
        nq = info["n_qualifying"]
        top = ", ".join(f"{c} ({n})" for c, n in info["top"][:4])
        colour = "red" if nq < 2 else "dim"
        stats.update(
            f"[{colour}]{info['n']} annotated · {info['n_clades']} clades · "
            f"{nq} clear the size floor[/{colour}]\n[dim]{top}…[/dim]"
        )

    def on_input_changed(self, event: Input.Changed) -> None:
        # Re-evaluate how many clades qualify when the floor changes.
        if event.input.id == "minclade-input" and self._ranks:
            self._refresh_stats(str(self.query_one("#rank-select", Select).value))

    def action_cancel(self) -> None:
        if self._busy:
            return
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "submit-btn" and not self._busy:
            self._do_build()

    @work(thread=True, exclusive=True, group="taxonomic-build")
    def _do_build(self) -> None:
        rank = str(self.query_one("#rank-select", Select).value)
        min_per_clade = self._current_min_per_clade()
        try:
            n_perm = max(0, int(self.query_one("#perm-input", Input).value.strip() or "0"))
        except ValueError:
            self.app.call_from_thread(
                self._set_msg, "[red]Permutations must be a whole number.[/red]"
            )
            return
        use_weights = self.query_one("#weight-switch", Switch).value
        write_profiles = self.query_one("#profiles-switch", Switch).value

        self._busy = True
        msg = "[dim]Building…[/dim]" if n_perm == 0 else \
            f"[dim]Building with {n_perm}-permutation null (this takes a moment)…[/dim]"
        self.app.call_from_thread(self._set_msg, msg)

        try:
            from ..comparative import build_taxonomic
            arr = build_taxonomic(
                self._project, rank,
                min_per_clade=min_per_clade,
                use_weights=use_weights,
                n_permutations=n_perm,
                write_profiles=write_profiles,
            )
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(
                self._set_msg, f"[red]{type(e).__name__}: {e}[/red]"
            )
            self._busy = False
            return

        if arr is None:
            self.app.call_from_thread(
                self._set_msg,
                "[red]Build failed — fewer than two clades cleared the size "
                "floor. Try a coarser rank or a lower minimum.[/red]",
            )
            self._busy = False
            return

        self.app.call_from_thread(self.dismiss, rank)

    def _set_msg(self, msg: str) -> None:
        self.query_one("#status-line", Label).update(msg)
