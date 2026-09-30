from __future__ import annotations

import importlib.util
from pathlib import Path


def _driver_module():
    path = Path(__file__).resolve().parents[1] / "scripts/rock/derail_rock_driver.py"
    spec = importlib.util.spec_from_file_location("derail_rock_driver", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_transient_start_error_classification() -> None:
    driver = _driver_module()
    read_error = type("ReadError", (Exception,), {})
    assert driver._transient_start_error(read_error("stream reset"))
    assert driver._transient_start_error(RuntimeError("HTTP 503"))
    assert driver._transient_start_error(TimeoutError("timed out"))


def test_permanent_start_error_is_not_retried() -> None:
    driver = _driver_module()
    assert not driver._transient_start_error(ValueError("invalid image name"))
