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

__all__ = ['main']
