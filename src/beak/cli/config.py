"""`beak config` commands."""

import click

from .main import main


@main.group()
def config():
    """Manage beak configuration"""
    pass


@config.command('init')
def config_init():
    """Interactive setup of ~/.beak/config.toml"""
    from ..config import save_config, CONFIG_PATH

    click.echo("BEAK Configuration Setup")
    click.echo("=" * 40)

    host = click.prompt("Remote server hostname", type=str)
    user = click.prompt("SSH username", type=str)
    key_path = click.prompt(
        "SSH key path",
        default="~/.ssh/id_ed25519",
        type=str
    )
    remote_job_dir = click.prompt(
        "Remote job directory",
        default="~/beak_jobs",
        type=str
    )

    config_data = {
        'connection': {
            'host': host,
            'user': user,
            'key_path': key_path,
            'remote_job_dir': remote_job_dir,
        },
        'projects': {
            'local_dir': '~/beak_projects',
        }
    }

    save_config(config_data)
    click.echo(f"\n✓ Configuration saved to {CONFIG_PATH}")


@config.command('show')
def config_show():
    """Show current configuration"""
    from ..config import load_config, CONFIG_PATH

    if not CONFIG_PATH.exists():
        click.echo("No configuration found. Run 'beak config init' to set up.")
        return

    config_data = load_config()

    for section, values in config_data.items():
        click.echo(f"[{section}]")
        if isinstance(values, dict):
            for key, val in values.items():
                click.echo(f"  {key} = {val}")
        else:
            click.echo(f"  {values}")
        click.echo()


@config.command('set')
@click.argument('key')
@click.argument('value')
def config_set(key, value):
    """Set a configuration value (e.g., beak config set connection.host myserver)"""
    from ..config import set_config_value

    set_config_value(key, value)
    click.echo(f"✓ Set {key} = {value}")
