"""Modal: pick a trait + threshold, build per-target-position enrichment.

The user picks a column from the joined `traits.parquet` and a numeric
split threshold. We run `build_comparative` synchronously (it's a few
hundred ms even on hundreds of homologs) and dismiss with the column
name so the parent screen can flip the structure-view's color mode.
"""

from typing import Optional

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select

from ...project import BeakProject


class SubmitComparativeModal(ModalScreen[Optional[str]]):
    """Pick trait column + threshold; cache differential scores."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, project: BeakProject) -> None:
        super().__init__()
        self._project = project
        self._busy = False
        self._cols: list = []

    def compose(self) -> ComposeResult:
        from ..comparative import trait_columns
        self._cols = trait_columns(self._project)

        with Vertical(id="modal-body"):
            yield Label(
                f"[bold]Differential coloring · {self._project.name}[/bold]",
                id="modal-title",
            )
            yield Label(
                "[dim]Color residues by enrichment in high-vs-low groups.[/dim]"
            )
            yield Label("")

            if not self._cols:
                yield Label(
                    "[red]No traits available — pull homologs + traits first.[/red]"
                )
                with Horizontal(id="modal-buttons"):
                    yield Button("Close", id="cancel-btn")
                return

            # Default to a sensible thermophile-style trait when present.
            default = next(
                (c for c in ("trait_temperature_growth",
                             "trait_temperature_maximum",
                             "trait_ph_growth")
                 if c in self._cols),
                self._cols[0],
            )
            options = [(c.removeprefix("trait_").replace("_", " "), c)
                       for c in self._cols]
            yield Label("Trait")
            yield Select(
                options, value=default, id="trait-select", allow_blank=False,
            )

            yield Label("Threshold (rows >= go in the 'high' group)")
            yield Input(value="", placeholder="loading suggested split…",
                        id="threshold-input")
            yield Label("", id="trait-stats")

            with Horizontal(id="modal-buttons"):
                yield Button("Cancel", id="cancel-btn")
                yield Button("Build", id="submit-btn", variant="primary")
                yield Button("Build + ChimeraX", id="export-btn")

    def on_mount(self) -> None:
        if self._cols:
            self._refresh_stats(self.query_one("#trait-select", Select).value)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "trait-select":
            self._refresh_stats(str(event.value))

    def _refresh_stats(self, column: str) -> None:
        from ..comparative import trait_summary
        info = trait_summary(self._project, column)
        if not info or info.get("kind") == "empty":
            self.query_one("#trait-stats", Label).update(
                "[red]no values for this trait[/red]"
            )
            return
        kind = info["kind"]
        if kind == "binary":
            txt = (
                f"[dim]binary trait · n={info['n']} · "
                f"threshold 0.5 splits true vs false[/dim]"
            )
            self.query_one("#threshold-input", Input).value = "0.5"
        elif kind == "numeric":
            mn, mx, med = info["min"], info["max"], info["median"]
            txt = (
                f"[dim]numeric · n={info['n']} · "
                f"range {mn:.2f}-{mx:.2f} · median {med:.2f}[/dim]"
            )
            self.query_one("#threshold-input", Input).value = f"{med:g}"
        else:  # categorical
            txt = (
                "[red]categorical trait — set the threshold so True maps to "
                "the focal class[/red]"
            )
        self.query_one("#trait-stats", Label).update(txt)

    def action_cancel(self) -> None:
        if self._busy:
            return
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "submit-btn" and not self._busy:
            self._do_build(export=False)
        elif event.button.id == "export-btn" and not self._busy:
            self._do_build(export=True)

    @work(thread=True, exclusive=True, group="comparative-build")
    def _do_build(self, export: bool = False) -> None:
        column = str(self.query_one("#trait-select", Select).value)
        thr_str = self.query_one("#threshold-input", Input).value.strip()
        try:
            threshold = float(thr_str)
        except ValueError:
            self.app.call_from_thread(
                self._set_msg, "[red]Threshold must be numeric.[/red]"
            )
            return

        self._busy = True
        self.app.call_from_thread(
            self._set_msg, "[dim]Building…[/dim]"
        )
        try:
            from ..comparative import build_comparative
            arr = build_comparative(self._project, column, threshold)
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(
                self._set_msg, f"[red]{type(e).__name__}: {e}[/red]"
            )
            self._busy = False
            return

        if arr is None:
            self.app.call_from_thread(
                self._set_msg,
                "[red]Build failed — threshold may yield an empty group, "
                "or alignment / traits aren't ready yet.[/red]",
            )
            self._busy = False
            return

        if export:
            try:
                self._export_to_chimerax(column, threshold, arr)
            except Exception as e:  # noqa: BLE001
                self.app.call_from_thread(
                    self._set_msg, f"[red]Export failed: {e}[/red]"
                )
                self._busy = False
                return

        self.app.call_from_thread(self.dismiss, column)

    def _export_to_chimerax(self, column: str, threshold: float, scores) -> None:
        """Write CIF + .cxc to <project>/exports/. Notifies on completion."""
        from pathlib import Path
        cif = self._project.path / "structures"
        # First .cif under structures/ — naming follows the AF download.
        try:
            cif_path = next(cif.glob("*.cif"))
        except StopIteration:
            raise FileNotFoundError(
                "No structure CIF on disk yet — run AlphaFold fetch first."
            )
        target_seq = self._project.target_sequence() or ""
        if not target_seq:
            raise ValueError("Target sequence is empty")

        out_dir = self._project.path / "exports" / "chimerax"
        nice = column.removeprefix("trait_")
        name = f"{nice}__{threshold:g}".replace(".", "p")
        title = f"{nice.replace('_', ' ')} >= {threshold:g}"

        from ...viz.chimerax import export_chimerax, open_in_chimerax
        cif_out, cxc_out, n_set = export_chimerax(
            cif_path=cif_path,
            scores=scores,
            target_seq=target_seq,
            out_dir=out_dir,
            name=name,
            mode="differential",
            title=title,
        )
        rel = Path(cxc_out).relative_to(self._project.path)
        try:
            open_in_chimerax(cxc_out)
            suffix = "  · opening in ChimeraX…"
        except Exception:
            suffix = "  · ChimeraX not found ($CHIMERAX to set path)"
        self.app.call_from_thread(
            self.app.notify,
            f"ChimeraX export ready · {rel}  ({n_set} residues){suffix}",
            timeout=10,
        )

    def _set_msg(self, msg: str) -> None:
        self.query_one("#trait-stats", Label).update(msg)
