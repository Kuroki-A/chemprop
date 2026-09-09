"""Lazy public API for molecular and graph featurization."""

from importlib import import_module
from typing import Dict


_FEATURE_GENERATOR_EXPORTS = {
    "FEATURE_GENERATOR_METADATA_SCHEMA_VERSION",
    "clear_features_generator_caches",
    "clear_pretrained_transformer_cache",
    "generate_features_batch",
    "get_available_features_generators",
    "get_features_generator",
    "get_features_generator_config",
    "get_features_generators_metadata",
    "get_features_generator_schema",
    "is_builtin_features_generator",
    "morgan_binary_features_generator",
    "morgan_counts_features_generator",
    "rdkit_2d_features_generator",
    "rdkit_2d_normalized_features_generator",
    "register_features_generator",
}

_FEATURIZATION_EXPORTS = {
    "BatchMolGraph",
    "MolGraph",
    "atom_features",
    "bond_features",
    "get_atom_fdim",
    "get_bond_fdim",
    "is_adding_hs",
    "is_explicit_h",
    "is_keeping_atom_map",
    "is_mol",
    "is_reaction",
    "mol2graph",
    "onek_encoding_unk",
    "reaction_mode",
    "reset_featurization_parameters",
    "set_adding_hs",
    "set_explicit_h",
    "set_extra_atom_fdim",
    "set_extra_bond_fdim",
    "set_keeping_atom_map",
    "set_reaction",
}

_UTIL_EXPORTS = {
    "load_features",
    "load_valid_atom_or_bond_features",
    "save_features",
}

_EXPORT_MODULES: Dict[str, str] = {
    **{name: ".features_generators" for name in _FEATURE_GENERATOR_EXPORTS},
    **{name: ".featurization" for name in _FEATURIZATION_EXPORTS},
    **{name: ".utils" for name in _UTIL_EXPORTS},
}

_LAZY_SUBMODULES = {"features_generators", "featurization", "utils"}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is not None:
        value = getattr(import_module(module_name, __name__), name)
    elif name in _LAZY_SUBMODULES:
        value = import_module(f".{name}", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()).union(__all__).union(_LAZY_SUBMODULES))
