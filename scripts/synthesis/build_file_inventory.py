#!/usr/bin/env python3
"""Build file-inventory/1.0 from a copy of the persona's home tree."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(os.environ.get("DERAIL_REPO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO_ROOT / "src"))

from derail.derived.schema import validate_schema  # noqa: E402
from derail.world.files import FileInventory  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="copy of /home/user")
    parser.add_argument("--home-prefix", default="/home/user")
    parser.add_argument("--include", nargs="*", default=["Documents", "Desktop", "Downloads"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print("reading %s\nwriting %s" % (args.root, args.out), file=sys.stderr)
    inventory = FileInventory.build(args.root, args.home_prefix, args.include)
    record = inventory.to_record()
    validate_schema(record, "file_inventory.schema.json", REPO_ROOT)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    inventory.save(args.out)
    with_text = sum(1 for e in inventory.entries.values() if e.content)
    print("files: %d, with text: %d" % (len(inventory.entries), with_text), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
