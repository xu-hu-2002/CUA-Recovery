#!/usr/bin/env python3
"""Compute post-error continuation statistics on the labelled failure corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from derail.longhorizon.precheck import PrecheckConfig, run_precheck


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="precheck YAML config")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root that repository-relative config paths resolve against",
    )
    args = parser.parse_args(list(argv) if argv else None)
    config = PrecheckConfig.from_yaml(args.config, args.repo_root.resolve())
    manifest = run_precheck(
        config,
        repo_root=args.repo_root.resolve(),
        command="scripts/analysis/precheck_post_error_continuation.py --config %s" % args.config,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
