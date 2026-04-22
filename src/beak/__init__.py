"""BEAK - Biophysical and Evolutionary Analysis Kit"""

import importlib as _importlib

from . import remote
from . import datasets
from . import sequence
from . import temperature
from . import alignments
from . import embeddings

# viz is loaded lazily to avoid pulling in heavy matplotlib/seqlogo
# dependencies when only using CLI or remote functionality
def __getattr__(name):
    if name == 'viz':
        return _importlib.import_module('.viz', __name__)
    if name == 'structures':
        return _importlib.import_module('.structures', __name__)
    raise AttributeError(f"module 'beak' has no attribute {name!r}")

__version__ = '0.1.0'

__all__ = ['remote', 'datasets', 'sequence', 'temperature',
           'alignments', 'embeddings', 'viz', 'structures']
