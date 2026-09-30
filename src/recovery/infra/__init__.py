from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

INFRA_ROOT = Path(
    os.environ.get("RECOVERY_INFRA_ROOT", Path(__file__).resolve().parents[3] / "infra")
)


def load_vm_script(relative: str) -> ModuleType:
    path = INFRA_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError("infra script not found: %s" % path)
    name = "recovery_infra_" + relative.replace("/", "_").replace(".py", "")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
