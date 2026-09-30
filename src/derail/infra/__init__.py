"""Repository-side access to the VM infrastructure scripts (brief section 1, ``infra/``).

The files under ``<repo>/infra/`` are deployed into the VM and must stay standard-library
only.  Rather than keeping a second copy here, :func:`load_vm_script` imports one of them by
path so offline tests and the replay verifier exercise exactly the deployed code.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

INFRA_ROOT = Path(
    os.environ.get("DERAIL_INFRA_ROOT", Path(__file__).resolve().parents[3] / "infra")
)


def load_vm_script(relative: str) -> ModuleType:
    """Import ``infra/<relative>`` (e.g. ``"triggers/install_triggers.py"``) as a module."""

    path = INFRA_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError("infra script not found: %s" % path)
    name = "derail_infra_" + relative.replace("/", "_").replace(".py", "")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
