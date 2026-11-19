"""LaNet-vi: Large scale network visualization using k-core and k-dense decomposition."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lanet-vi")
except PackageNotFoundError:  # pragma: no cover - only when running from a raw checkout
    __version__ = "0.0.0+unknown"

from lanet_vi.core.network import Network
from lanet_vi.drawing import draw, lanetvi_layout
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
    "__version__",
    "Network",
    "draw",
    "lanetvi_layout",
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
