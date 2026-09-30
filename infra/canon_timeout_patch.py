#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
from pathlib import Path

OLD = "                timeout=120,\n"
NEW = (
    "                timeout=int(os.environ.get(\n"
    "                    'MYPCBENCH_CANON_PATCH_TIMEOUT', '300'\n"
    "                )),\n"
)


def patched_text(text: str) -> str:
    if "MYPCBENCH_CANON_PATCH_TIMEOUT" in text:
        return text
    if text.count(OLD) != 1:
        raise ValueError("expected exactly one canonical per-patcher timeout")
    result = text.replace(OLD, NEW)
    ast.parse(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--script", default="/opt/mypcbench/scripts/run_all_canon_patches.py"
    )
    args = parser.parse_args()
    path = Path(args.script)
    updated = patched_text(path.read_text())
    if updated != path.read_text():
        path.with_suffix(path.suffix + ".phase5.bak").write_text(path.read_text())
        path.write_text(updated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
