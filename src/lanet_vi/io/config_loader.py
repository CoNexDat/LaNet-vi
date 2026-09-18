"""Configuration loading from YAML files."""

from pathlib import Path
from typing import Any

import yaml

from lanet_vi.logging_config import get_logger
from lanet_vi.models.config import DEPRECATED_ALIASES, LaNetConfig

logger = get_logger(__name__)


def load_config_from_yaml(file_path: Path | str) -> LaNetConfig:
    """
    Load LaNet-vi configuration from a YAML file.

    Parameters
    ----------
    file_path : Union[Path, str]
        Path to YAML configuration file

    Returns
    -------
    LaNetConfig
        Validated configuration object

    Raises
    ------
    FileNotFoundError
        If the configuration file doesn't exist
    yaml.YAMLError
        If the YAML file is malformed
    pydantic.ValidationError
        If the configuration values are invalid

    Examples
    --------
    >>> config = load_config_from_yaml("config.yaml")
    >>> net = Network(graph, config)
    >>> net.decompose()
    >>> net.visualize("output.png")
    """
    file_path = Path(file_path)

    logger.info(f"Loading configuration from {file_path}")

    if not file_path.exists():
        logger.error(f"Configuration file not found: {file_path}")
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    config_dict = read_config_yaml(file_path)

    # Validate and create config object using Pydantic
    # Pydantic will automatically validate types and constraints
    try:
        config = LaNetConfig(**config_dict)
        logger.info("Configuration validated successfully")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

    return config


def read_config_yaml(file_path: Path | str) -> dict[str, Any]:
    """
    Read a YAML configuration file into a plain dictionary without validating it.

    Used by the CLI to merge the file with explicit command-line flags before a
    single validation pass.

    Parameters
    ----------
    file_path : Union[Path, str]
        Path to the YAML file

    Returns
    -------
    Dict[str, Any]
        Nested dictionary as written in the file (empty if the file is empty)

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the top level of the file is not a mapping
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"Configuration file not found: {file_path}")
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(file_path) as f:
        loaded = yaml.safe_load(f)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{file_path}: top level must be a mapping of sections")

    logger.debug(f"Loaded YAML configuration with keys: {list(loaded.keys())}")
    return loaded


def save_config_to_yaml(config: LaNetConfig, file_path: Path | str) -> None:
    """
    Save LaNet-vi configuration to a YAML file.

    Parameters
    ----------
    config : LaNetConfig
        Configuration object to save
    file_path : Union[Path, str]
        Path where YAML file should be saved

    Examples
    --------
    >>> config = LaNetConfig()
    >>> config.visualization.width = 1920
    >>> save_config_to_yaml(config, "my_config.yaml")
    """
    file_path = Path(file_path)

    logger.info(f"Saving configuration to {file_path}")

    # Convert config to plain JSON-compatible types (enums become their values),
    # so the file can be read back with yaml.safe_load
    config_dict = config.model_dump(mode="json", exclude_none=True)
    # Deprecated aliases stay out of the template: written next to the current field
    # they would be ignored on reload (the alias only applies when the field is absent)
    for alias in DEPRECATED_ALIASES:
        config_dict.get("visualization", {}).pop(alias, None)

    # Save to YAML file
    with open(file_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Configuration saved successfully to {file_path}")
