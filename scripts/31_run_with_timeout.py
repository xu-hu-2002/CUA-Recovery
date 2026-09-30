#!/usr/bin/env python3
"""Run one command with a hard timeout and terminate its process group."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys


def run(command: list[str], timeout: int, grace: int) -> int:
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[hard-timeout] exceeded {timeout}s: {' '.join(command)}", file=sys.stderr)
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        return 124


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--grace", type=int, default=30)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.timeout < 1 or args.grace < 1 or not args.command:
        parser.error("positive --timeout/--grace and a command are required")
    return run(args.command, args.timeout, args.grace)


if __name__ == "__main__":
    raise SystemExit(main())
