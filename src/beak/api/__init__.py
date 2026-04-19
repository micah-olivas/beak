"""REST API clients for external bioinformatics services."""

from .uniprot import query_uniprot_by_pfam
from .structures import find_structures, fetch_structures

__all__ = ['query_uniprot_by_pfam', 'find_structures', 'fetch_structures']
