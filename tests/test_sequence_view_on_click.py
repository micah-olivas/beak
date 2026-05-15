"""Tests for SequenceView.on_click border-click guard.

The handler must:
  * only act on clicks where the widget receiving the event *is* the
    SequenceView itself — a click that bubbled up from a child (the
    scroll body or footer) keeps the child's coordinates and could
    spuriously match `event.y == 0`.
  * only act on the top border row (`event.y == 0`).
  * step backward on a right-click (button == 3) and forward
    otherwise.

These run as plain unit tests (no Textual app, no project fixture) by
invoking the method directly against a MagicMock self; that keeps the
guard logic verifiable without dragging in the whole UI runtime.
"""

from unittest.mock import MagicMock

from beak.tui.widgets.sequence_view import SequenceView


def _make_event(widget, *, y=0, button=1):
    ev = MagicMock()
    ev.widget = widget
    ev.y = y
    ev.button = button
    return ev


def test_top_border_left_click_cycles_forward():
    self_mock = MagicMock()
    ev = _make_event(widget=self_mock, y=0, button=1)
    SequenceView.on_click(self_mock, ev)
    self_mock.cycle_highlight.assert_called_once_with(backward=False)
    ev.stop.assert_called_once()


def test_top_border_right_click_cycles_backward():
    self_mock = MagicMock()
    ev = _make_event(widget=self_mock, y=0, button=3)
    SequenceView.on_click(self_mock, ev)
    self_mock.cycle_highlight.assert_called_once_with(backward=True)
    ev.stop.assert_called_once()


def test_click_below_border_is_ignored():
    """A click inside the scroll body / footer has y > 0 and must
    not trigger the cycle even if the widget reference is correct."""
    self_mock = MagicMock()
    ev = _make_event(widget=self_mock, y=5, button=1)
    SequenceView.on_click(self_mock, ev)
    self_mock.cycle_highlight.assert_not_called()
    ev.stop.assert_not_called()


def test_bubbled_child_click_is_ignored():
    """When a click on a child widget bubbles up, Textual delivers the
    event to the parent with `event.widget` still pointing at the
    original child. Without the `widget is self` guard, a click at
    `y == 0` of the scroll body (which sits below our border) would
    accidentally trigger residue cycling."""
    self_mock = MagicMock()
    other_widget = MagicMock()
    ev = _make_event(widget=other_widget, y=0, button=1)
    SequenceView.on_click(self_mock, ev)
    self_mock.cycle_highlight.assert_not_called()
    ev.stop.assert_not_called()


def test_event_without_widget_attr_is_ignored():
    """Defensive: if `event.widget` is somehow None (mocked event,
    or a Textual version that doesn't set it), don't cycle."""
    self_mock = MagicMock()
    ev = MagicMock()
    ev.widget = None
    ev.y = 0
    ev.button = 1
    SequenceView.on_click(self_mock, ev)
    self_mock.cycle_highlight.assert_not_called()
