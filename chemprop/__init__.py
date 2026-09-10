"""Chemprop v1 compatibility package.

Subpackages are loaded on first attribute access. Keeping the package root
lightweight is important because every CLI imports it before argument parsing;
eagerly importing training, scikit-learn, plotting, and optional feature
backends previously added seconds of startup time and hundreds of MB of RAM.
"""

from importlib import import_module
from typing import List


__version__ = "1.7.1+kuroki.4"

_LAZY_SUBMODULES = {
    "args",
    "constants",
    "data",
    "features",
    "hyperopt_utils",
    "hyperparameter_optimization",
    "interpret",
    "models",
    "nn_utils",
    "rdkit",
    "sklearn_predict",
    "sklearn_train",
    "spectra_utils",
    "train",
    "uncertainty",
    "utils",
    "web",
}

# The legacy package imported every listed module except ``chemprop.web`` at
# package import time. Keep the Web module available lazily without adding it
# to ``from chemprop import *`` as a new side effect.
__all__ = ["__version__", *sorted(_LAZY_SUBMODULES - {"web"})]


def __getattr__(name: str):
    """Imports historical top-level submodule attributes on demand."""
    if name not in _LAZY_SUBMODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__() -> List[str]:
    return sorted(set(globals()).union(__all__))
