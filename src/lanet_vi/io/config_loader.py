"""Configuration loading from YAML files."""

from pathlib import Path
from typing import Union

import yaml

from lanet_vi.logging_config import get_logger
from lanet_vi.models.config import LaNetConfig

logger = get_logger(__name__)


def load_config_from_yaml(file_path: Union[Path, str]) -> LaNetConfig:
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

    # Load YAML file
    with open(file_path, "r") as f:
        config_dict = yaml.safe_load(f)

    logger.debug(f"Loaded YAML configuration with keys: {list(config_dict.keys())}")

    # Validate and create config object using Pydantic
    # Pydantic will automatically validate types and constraints
    try:
        config = LaNetConfig(**config_dict)
        logger.info("Configuration validated successfully")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

    return config


def save_config_to_yaml(config: LaNetConfig, file_path: Union[Path, str]) -> None:
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

    # Convert config to dictionary
    config_dict = config.model_dump(exclude_none=True)

    # Save to YAML file
    with open(file_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Configuration saved successfully to {file_path}")
