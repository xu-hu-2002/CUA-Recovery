#!/usr/bin/env python3
"""Phase 0 pre-check: post-error continuation statistics on the labelled failure corpus.

Reads the frozen label index and canonical trajectories named in the config, computes
post_error_steps / loop_detected / explicit-terminate rates per agent and paper category, and
writes ``post_error_continuation_report.{md,json}`` plus a stage manifest (manual v0.2 §17).
"""

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
        default=Path(__file__).resolve().parents[1],
        help="repository root that repository-relative config paths resolve against",
    )
    args = parser.parse_args(list(argv) if argv else None)
    config = PrecheckConfig.from_yaml(args.config, args.repo_root.resolve())
    manifest = run_precheck(
        config,
        repo_root=args.repo_root.resolve(),
        command="scripts/precheck_post_error_continuation.py --config %s" % args.config,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
