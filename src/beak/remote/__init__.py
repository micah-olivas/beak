"""
Remote computing utilities for SSH connections and remote analysis
"""

from .utils import (
    authenticate,
    get_pw,
    sopen,
    ssend,
    make_temp_dir,
    scp_to_remote,
    scp_from_remote,
    nest,
    search
)

from .interactive import (
    InteractiveRemoteSession,
    connect_remote
)