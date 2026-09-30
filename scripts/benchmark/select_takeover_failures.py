#!/usr/bin/env python3
"""Build an auditable list of valid failure trajectories for takeover."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
SRC = REPOSITORY / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml  # noqa: E402

from derail.derived.layout import atomic_write_json  # noqa: E402
from derail.takeover.selection import select_takeover_failures  # noqa: E402

PROTOCOL_CONFIG = REPOSITORY / "configs" / "benchmark" / "derail_v1.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--human-labels-dir", type=Path, required=True)
    parser.add_argument("--source-agent", required=True)
    parser.add_argument("--annotator-id", default="")
    parser.add_argument("--trajectory-id", default="")
    parser.add_argument(
        "--trajectory-id-file",
        type=Path,
        help="Newline-delimited frozen trajectory IDs shared by every depth/condition.",
    )
    protocol = yaml.safe_load(PROTOCOL_CONFIG.read_text(encoding="utf-8"))
    eligibility = protocol.get("depth_eligibility") or {}
    repair = protocol.get("prefix_repair") or {}
    parser.add_argument("--depths", nargs="*", type=int, default=())
    parser.add_argument(
        "--require-error-explicit",
        action=argparse.BooleanOptionalAction,
        default=bool(eligibility.get("require_error_explicit", True)),
        help="skip failures whose error never becomes explicit (h_e unobserved)",
    )
    parser.add_argument(
        "--repairs-dir",
        type=Path,
        help="repaired-prefix root; default <build-dir>/<prefix_repair.output_layer>",
    )
    parser.add_argument(
        "--require-repaired-prefix",
        action=argparse.BooleanOptionalAction,
        default=bool(repair.get("require_for_takeover", True)),
        help="exclude failures without a human-verified repaired prefix",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--list-file", type=Path, required=True)
    args = parser.parse_args()
    allowlist = ()
    if args.trajectory_id_file:
        allowlist = tuple(
            line.strip()
            for line in args.trajectory_id_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    result = select_takeover_failures(
        build_dir=args.build_dir,
        human_labels_dir=args.human_labels_dir,
        source_agent=args.source_agent,
        annotator_id=args.annotator_id or None,
        trajectory_id_filter=args.trajectory_id or None,
        trajectory_id_allowlist=allowlist,
        depths=args.depths,
        require_error_explicit=args.require_error_explicit,
        repairs_dir=args.repairs_dir
        or args.build_dir / str(repair.get("output_layer", "canonical_repaired")),
        require_repaired_prefix=args.require_repaired_prefix,
    )
    if not result["included"]:
        raise RuntimeError("selection contains no eligible failure trajectories")
    atomic_write_json(args.manifest.resolve(), result)
    rows = "".join(
        "\t".join(
            (
                str(item["trajectory_id"]),
                str(item["canonical_trajectory_uri"]),
                str(item["normalization_report_uri"]),
                str(item["task_config_uri"]),
                str(item["annotation_uri"]),
                str(item["root_cause_action_index"]),
                str(item["last_action_index"]),
                ",".join(str(depth) for depth in item["available_depths"]),
            )
        )
        + "\n"
        for item in result["included"]
    )
    args.list_file.parent.mkdir(parents=True, exist_ok=True)
    args.list_file.write_text(rows, encoding="utf-8")
    print(
        f"selected={result['included_count']} excluded={result['excluded_count']} "
        f"manifest={args.manifest.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
