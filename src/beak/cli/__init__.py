from .main import main

# Import each command module so decorators register them on `main`.
from . import config
from . import jobs
from . import submit
from . import pfam
from . import structures
from . import setup
from . import project
from . import ui


def cli_entry():
    """Console-script entry point (the `beak` command).

    Behaves exactly like `main`, with one addition: in --json mode a Click
    error is rendered as {"error", "exit_code"} on stdout instead of plain
    text on stderr, so an agent parsing stdout still gets structured output
    on failure. Commands that already emitted their own structured object
    (e.g. `status --json`, a failed `--wait`) exit via SystemExit and are
    passed through untouched — no double emit. Human mode is unchanged.
    """
    import sys
    import json as _json
    import click

    want_json = '--json' in sys.argv
    try:
        rv = main.main(standalone_mode=False)
    except click.ClickException as e:
        if want_json:
            click.echo(_json.dumps({'error': e.format_message(),
                                    'exit_code': e.exit_code}))
        else:
            e.show()
        sys.exit(e.exit_code)
    except click.exceptions.Abort:
        if want_json:
            click.echo(_json.dumps({'error': 'Aborted', 'exit_code': 1}))
        else:
            click.echo('Aborted!', err=True)
        sys.exit(1)
    sys.exit(rv if isinstance(rv, int) else 0)


__all__ = ['main', 'cli_entry']
