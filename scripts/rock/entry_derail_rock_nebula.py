#!/usr/bin/env python3
"""Entry shim that forwards to entry_derail_rock_nebula.sh."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WRAPPER = os.path.join(HERE, "entry_derail_rock_nebula.sh")


def main():
    print(f"[entry-shim] launching: bash {WRAPPER}", flush=True)
    if not os.path.isfile(WRAPPER):
        print(f"[entry-shim] ERROR: {WRAPPER} not found", file=sys.stderr, flush=True)
        return 2
    return subprocess.call(["bash", WRAPPER], cwd=os.path.dirname(os.path.dirname(HERE)))


if __name__ == "__main__":
    sys.exit(main())
