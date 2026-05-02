"""Project system: persistent home for target-centric analyses.

A project pins one target sequence and (over time) accretes homologs,
structures, domains, and experimental measurements joined onto target
positions. Projects live under ~/.beak/projects/<name>/ so they are
discoverable from notebooks without caring about cwd.
"""

from .project import BeakProject, BeakProjectError, PROJECTS_DIR

__all__ = ['BeakProject', 'BeakProjectError', 'PROJECTS_DIR']
