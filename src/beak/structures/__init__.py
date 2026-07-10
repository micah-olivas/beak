"""Per-residue structural feature extraction and coordinate mapping."""

from .features import extract_structure_features
from .mapping import map_target_to_structure, map_alignment_to_target
from .foldseek import (
    FoldseekError,
    download_database,
    foldseek_version,
    parse_foldseek_m8,
    resolve_foldseek_binary,
    run_easy_search,
)

__all__ = [
    'extract_structure_features',
    'map_target_to_structure',
    'map_alignment_to_target',
    'FoldseekError',
    'download_database',
    'foldseek_version',
    'parse_foldseek_m8',
    'resolve_foldseek_binary',
    'run_easy_search',
]
