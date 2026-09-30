#!/usr/bin/env python3
"""Patch the MyPCBench control API to serve the DERAIL endpoints (brief section 1).

Inserts, right after ``app = Flask(__name__)`` in ``/opt/desktop-seed/server/main.py``:

    import sys as _derail_sys; _derail_sys.path.insert(0, "/opt/derail")
    from control_api_derail import register as _derail_register; _derail_register(app)

Idempotent (a second run is a no-op), keeps a ``main.py.derail.bak`` backup, and verifies
the result still parses.  ``--check`` reports whether the patch is present without writing.

    python3 control_api_patch.py --main /opt/desktop-seed/server/main.py --infra-dir /opt/derail
"""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import sys

MARKER = "# derail-control-api-patch"
FLASK_LINE = re.compile(r"^app\s*=\s*Flask\(__name__\)\s*$", re.MULTILINE)
PATCHER_DEFAULT_OLD = 'data.get("skip_patchers", False)'
PATCHER_DEFAULT_NEW = 'data.get("skip_patchers", True)'


def patched_text(text: str, infra_dir: str) -> str:
    if PATCHER_DEFAULT_OLD in text:
        text = text.replace(PATCHER_DEFAULT_OLD, PATCHER_DEFAULT_NEW, 1)
    if MARKER in text:
        return text
    match = FLASK_LINE.search(text)
    if not match:
        raise SystemExit("could not find 'app = Flask(__name__)' in main.py")
    insertion = (
        "\n%s\nimport sys as _derail_sys; _derail_sys.path.insert(0, %r)\n"
        "import os as _derail_os; "
        "_derail_os.environ.setdefault('PYTHONWARNINGS', 'ignore::DeprecationWarning')\n"
        "from control_api_derail import register as _derail_register; _derail_register(app)\n"
        % (MARKER, infra_dir)
    )
    return text[: match.end()] + insertion + text[match.end() :]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--main", default="/opt/desktop-seed/server/main.py")
    parser.add_argument("--infra-dir", default="/opt/derail")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    with open(args.main, "r", encoding="utf-8") as handle:
        text = handle.read()
    if args.check:
        print("patched" if MARKER in text else "not patched")
        return 0 if MARKER in text else 1
    new_text = patched_text(text, args.infra_dir)
    if new_text == text:
        print("already patched", file=sys.stderr)
        return 0
    ast.parse(new_text)  # refuse to write a file that no longer parses
    shutil.copyfile(args.main, args.main + ".derail.bak")
    with open(args.main, "w", encoding="utf-8") as handle:
        handle.write(new_text)
    print("patched %s (backup at %s.derail.bak)" % (args.main, args.main), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
