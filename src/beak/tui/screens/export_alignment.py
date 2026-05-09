"""Modal for exporting an alignment in a user-chosen format.

Beak stores alignments on disk as `alignment.fasta` (one per homolog
set). For downstream tools that expect different formats — Clustal
for hand-inspection in Jalview / SeaView, Stockholm for HMMER
profile-builds, PHYLIP for tree-builders, A2M for HHsuite — the user
picks a format here and we convert via `Bio.AlignIO` (or, for A2M,
a small in-house writer since BioPython doesn't ship that codec).

The conversion runs on a worker so the modal isn't held by a 100-MB
parse on the main thread.
"""

from pathlib import Path
from typing import Optional

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select


# (format_id, display_label, file_extension). We don't autoselect by
# extension — the user might want a `.fasta` filename for a Stockholm
# file if they're feeding it to a tool that doesn't care. The two
# fields are independent.
_FORMATS: list[tuple[str, str, str]] = [
    ("fasta", "FASTA (aligned)", ".fasta"),
    ("clustal", "Clustal (.aln)", ".aln"),
    ("stockholm", "Stockholm (.sto)", ".sto"),
    ("phylip-relaxed", "PHYLIP (relaxed)", ".phy"),
    ("a2m", "A2M (HHsuite)", ".a2m"),
]


class ExportAlignmentModal(ModalScreen[Optional[Path]]):
    """Choose a format + output path, write the alignment.

    Dismisses with the output `Path` on success, `None` on cancel.
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, alignment_path: Path, default_stem: str) -> None:
        super().__init__()
        self._alignment_path = Path(alignment_path)
        # `default_stem` is the suggested filename without extension —
        # typically `<project>_<set_name>_alignment` so the user
        # doesn't end up with a generic "alignment.fasta" sitting in
        # ~/Downloads.
        self._default_stem = default_stem
        self._working = False

    def compose(self) -> ComposeResult:
        default_dir = Path.home() / "Downloads"
        # Default to ~/Downloads if it exists, else the user's home.
        # On Linux servers without ~/Downloads we don't want the
        # placeholder to look broken.
        if not default_dir.exists():
            default_dir = Path.home()
        default_path = default_dir / f"{self._default_stem}.fasta"

        with Vertical(id="modal-body"):
            yield Label("[bold]Export alignment[/bold]", id="modal-title")
            yield Label(
                f"[dim]Source: {self._alignment_path}[/dim]"
            )
            yield Label("")

            yield Label("Format")
            yield Select(
                [(label, fmt_id) for (fmt_id, label, _ext) in _FORMATS],
                value="fasta",
                id="format-select",
                allow_blank=False,
            )
            yield Label("Output path")
            yield Input(value=str(default_path), id="path-input")
            yield Label("", id="status-line")

            with Horizontal(id="modal-buttons"):
                yield Button("Cancel", id="cancel-btn")
                yield Button("Export", id="submit-btn", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#path-input", Input).focus()

    def action_cancel(self) -> None:
        if self._working:
            return
        self.dismiss(None)

    def on_select_changed(self, event: Select.Changed) -> None:
        # When the user changes format, swap the path's extension so
        # they don't have to retype it. We only swap when the current
        # extension matches one of *our* known extensions — if the
        # user typed something custom, leave it alone.
        if event.select.id != "format-select":
            return
        try:
            path_input = self.query_one("#path-input", Input)
        except Exception:
            return
        cur = Path(path_input.value)
        new_ext = next(
            (ext for (fmt_id, _label, ext) in _FORMATS
             if fmt_id == str(event.value)),
            None,
        )
        if not new_ext:
            return
        known_exts = {ext for (_id, _l, ext) in _FORMATS}
        if cur.suffix in known_exts:
            path_input.value = str(cur.with_suffix(new_ext))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "submit-btn" and not self._working:
            self._do_export()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "path-input" and not self._working:
            self._do_export()

    @work(thread=True, exclusive=True, group="export-alignment")
    def _do_export(self) -> None:
        # Re-read fields inside the worker so the user can't change
        # them mid-write. A successful export dismisses with the
        # written path; failures leave the modal open with an inline
        # error so the user can fix and retry.
        fmt_id = str(self.query_one("#format-select", Select).value)
        out_str = self.query_one("#path-input", Input).value.strip()

        if not out_str:
            self.app.call_from_thread(
                self._status, "[red]Output path is required.[/red]"
            )
            return
        try:
            out_path = Path(out_str).expanduser().resolve()
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(
                self._status, f"[red]Invalid path: {e}[/red]"
            )
            return
        if out_path.exists() and out_path.is_dir():
            self.app.call_from_thread(
                self._status,
                "[red]Output path is a directory — include a filename.[/red]",
            )
            return

        self._working = True
        self.app.call_from_thread(
            self._status, "[dim]Writing…[/dim]"
        )

        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if fmt_id == "a2m":
                # BioPython doesn't have a native A2M writer; the
                # format's wrinkles (lowercase = insert state, "."
                # = match-state gap) are simple enough to handle
                # here without a dependency. We treat alignment.fasta
                # as fully match-state — no insert state inference —
                # which is the standard "A2M from a regular MSA"
                # interpretation.
                _write_a2m(self._alignment_path, out_path)
            else:
                # Use the cached records when available so a 100 MB
                # FASTA doesn't get re-parsed by BioPython for every
                # export. AlignIO needs a sequence-of-records
                # interface, so we fall back to a thin wrapper.
                from ...alignments.cache import load_alignment_records
                from Bio.Align import MultipleSeqAlignment
                from Bio.AlignIO import write as align_write
                from Bio.Seq import Seq
                from Bio.SeqRecord import SeqRecord

                records = load_alignment_records(self._alignment_path)
                bio_records = [
                    SeqRecord(Seq(seq), id=name, description="")
                    for name, seq in records
                ]
                alignment = MultipleSeqAlignment(bio_records)
                with open(out_path, "w") as f:
                    align_write([alignment], f, fmt_id)
        except Exception as e:  # noqa: BLE001
            self._working = False
            self.app.call_from_thread(
                self._status,
                f"[red]Export failed:[/red] [dim]{type(e).__name__}: "
                f"{str(e).splitlines()[0][:200]}[/dim]",
            )
            return

        self._working = False
        self.app.call_from_thread(self.dismiss, out_path)

    def _status(self, msg: str) -> None:
        try:
            self.query_one("#status-line", Label).update(msg)
        except Exception:
            pass


def _write_a2m(src_fasta: Path, dst: Path) -> None:
    """Write an A2M file from a FASTA alignment.

    A2M = match-state alignment with uppercase letters at match
    columns and `-` for gaps; lowercase + `.` would denote insert
    states. Since beak's `alignment.fasta` is already a fully
    match-state MSA from clustalo / mafft, we just upper-case
    sequence letters and pass `-` through unchanged. Headers stay
    one-line.
    """
    from ...alignments.cache import load_alignment_records

    records = load_alignment_records(src_fasta)
    with open(dst, "w") as f:
        for name, seq in records:
            f.write(f">{name}\n")
            # Letters stay uppercase (match state); '-' kept as gap;
            # '.' (insert-state gap, shouldn't appear in our input)
            # passes through unchanged for safety.
            f.write(seq.upper() + "\n")
