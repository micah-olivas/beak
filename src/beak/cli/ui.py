"""`beak ui` — launch the Textual TUI."""

import click

from .main import main


@main.command('ui')
def ui_command():
    """Open the BEAK TUI to browse projects."""
    from ..tui import BeakApp
    BeakApp().run()
