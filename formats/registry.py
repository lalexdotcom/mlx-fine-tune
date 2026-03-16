# formats/registry.py
#
# Registry of all supported dataset formats.
# Maps --dataset-format CLI values to their converter modules.

import importlib
from types import ModuleType


# Maps format name → module path
REGISTRY: dict[str, str] = {
    "acon96-v2":       "formats.acon96_v2",
    "allenporter-fc":  "formats.allenporter_fc",
    "allenporter-msg": "formats.allenporter_msg",
}


def load_format(name: str) -> ModuleType:
    """Load a format module by name.
    Raises ValueError with the list of supported formats if name is unknown.
    """
    if name not in REGISTRY:
        supported = ", ".join(sorted(REGISTRY.keys()))
        raise ValueError(
            f"Unknown dataset format: '{name}'\n"
            f"Supported formats: {supported}"
        )
    return importlib.import_module(REGISTRY[name])


def list_formats() -> list[str]:
    """Return sorted list of all supported format names."""
    return sorted(REGISTRY.keys())