"""Tests for the package logging setup."""

import logging
from pathlib import Path

from lanet_vi.logging_config import get_logger, setup_logging


def test_get_logger_nests_everything_under_the_package_logger():
    """Module loggers keep their name; foreign names are nested under ``lanet_vi``."""
    assert get_logger("lanet_vi.core.network").name == "lanet_vi.core.network"
    assert get_logger("lanet_vi").name == "lanet_vi"
    assert get_logger("plugin").name == "lanet_vi.plugin"
    assert get_logger("lanet_vi.core.network").parent is not None
    assert get_logger("lanet_vi.core.network").name.split(".")[0] == "lanet_vi"


def test_setup_logging_writes_to_file_and_honors_quiet(tmp_path: Path, capsys):
    """With ``quiet`` only the file handler is installed; messages land in the file."""
    log_file = tmp_path / "lanet.log"
    root = logging.getLogger("lanet_vi")
    try:
        setup_logging(level=logging.INFO, log_file=log_file, quiet=True)
        assert len(root.handlers) == 1
        get_logger("lanet_vi.tests").info("hello from the test")
        for handler in root.handlers:
            handler.flush()
        assert "hello from the test" in log_file.read_text()
        assert "hello from the test" not in capsys.readouterr().out

        setup_logging(level=logging.WARNING)
        assert len(root.handlers) == 1
        assert root.level == logging.WARNING
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers.clear()
