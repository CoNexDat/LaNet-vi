"""LaNet-vi: Large scale network visualization using k-core and k-dense decomposition."""

__version__ = "4.0.0"

from lanet_vi.core.network import Network
from lanet_vi.io.config_loader import load_config_from_yaml, save_config_to_yaml
from lanet_vi.models.config import (
    BackgroundColor,
    ColorScheme,
    DecompositionConfig,
    DecompositionType,
    GraphConfig,
    LaNetConfig,
    LayoutConfig,
    VisualizationConfig,
)

__all__ = [
    "Network",
    "LaNetConfig",
    "GraphConfig",
    "DecompositionConfig",
    "VisualizationConfig",
    "LayoutConfig",
    "DecompositionType",
    "ColorScheme",
    "BackgroundColor",
    "load_config_from_yaml",
    "save_config_to_yaml",
]
