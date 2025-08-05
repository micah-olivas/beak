"""
Remote computing utilities for SSH connections and remote analysis
"""

from .utils import (
    authenticate,
    get_current_user,
    is_authenticated,
    clear_authentication,
    show_auth_status,
    get_pw,
    sopen,
    ssend,
    make_temp_dir,
    scp_to_remote,
    scp_from_remote,
    nest,
    search,
    status,
    retrieve_results,
    align,
    compute_tree,
    taxonomy,
    check_remote_process,
    list_databases,
    projects,
    debug_projects,
    simple_projects_test
)

from .interactive import (
    InteractiveRemoteSession,
    connect_remote
)