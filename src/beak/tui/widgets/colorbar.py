"""Focusable colorbar with arrow-key midpoint adjustment.

One line tall. Click to focus, ◀/▶ to shift the conservation midpoint.
The current midpoint shows numerically as `mid=50` next to the label;
when focused, an italic "← → adjust" hint appends to the right.

Posts `Colorbar.MidpointChanged` so consumers (sequence + structure
views) can re-render with the new value.
"""

from textual.binding import Binding
from textual.message import Message
from textual.widgets import Static


_BAR_CELLS = 14
_STEP = 5.0
_MIN_MIDPOINT = 5.0
_MAX_MIDPOINT = 95.0


class Colorbar(Static):
    """Inline colorbar legend. Focus + arrows shift conservation midpoint."""

    BINDINGS = [
        Binding("left",  "shift_down", "Shift ◀", show=False),
        Binding("right", "shift_up",   "Shift ▶", show=False),
    ]

    can_focus = True

    DEFAULT_CSS = """
    Colorbar { width: 42; height: 1; padding: 0 1; }
    Colorbar:focus { background: $boost; }
    """

    class MidpointChanged(Message):
        def __init__(self, midpoint: float) -> None:
            super().__init__()
            self.midpoint = midpoint

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._mode: str = "plddt"
        self._midpoint: float = 50.0

    @property
    def midpoint(self) -> float:
        return self._midpoint

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self.refresh()

    def set_midpoint(self, midpoint: float) -> None:
        self._midpoint = max(_MIN_MIDPOINT, min(_MAX_MIDPOINT, midpoint))
        self.refresh()

    def render(self):
        from ..structure import color_for_mode

        cells = []
        for i in range(_BAR_CELLS):
            score = (i / (_BAR_CELLS - 1)) * 100
            color = color_for_mode(score, self._mode, self._midpoint)
            cells.append(f"[{color}]█[/{color}]")

        label = {
            "plddt": "pLDDT",
            "conservation": "conservation",
            "sasa": "SASA",
        }.get(self._mode, self._mode)
        if self._mode == "conservation":
            tag = f"[dim]{label} mid={int(self._midpoint)}[/dim]"
        else:
            tag = f"[dim]{label}[/dim]"

        bar = f"{tag} [dim]0[/dim]{''.join(cells)}[dim]100[/dim]"
        if self.has_focus and self._mode == "conservation":
            bar += "  [italic dim]← → adjust[/italic dim]"
        return bar

    def on_click(self) -> None:
        self.focus()

    def on_focus(self) -> None:
        self.refresh()

    def on_blur(self) -> None:
        self.refresh()

    def action_shift_down(self) -> None:
        if self._mode != "conservation":
            return
        self.set_midpoint(self._midpoint - _STEP)
        self.post_message(self.MidpointChanged(self._midpoint))

    def action_shift_up(self) -> None:
        if self._mode != "conservation":
            return
        self.set_midpoint(self._midpoint + _STEP)
        self.post_message(self.MidpointChanged(self._midpoint))
