#!/usr/bin/env python3
"""Root-cause error-type distribution over human failure annotations.

Usage: python examples/error_type_distribution.py <labels_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
from recovery.longhorizon.taxonomy import TAXONOMY_PATH, FailureTaxonomy  # noqa: E402


def primary_types(labels_dir: Path, taxonomy: FailureTaxonomy) -> List[str]:
    primaries = []
    for path in sorted(labels_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or "error_types" not in record:
            continue
        labels, _ = taxonomy.normalize(str(label) for label in record["error_types"])
        primary = taxonomy.primary_paper_type(labels)
        if primary is not None:
            primaries.append(primary)
    return primaries


def print_table(title: str, counts: Counter, keys: List[str], total: int) -> None:
    print("\n%s" % title)
    for key in keys:
        share = 100.0 * counts[key] / total if total else 0.0
        print("  %-30s %5d  %5.1f%%" % (key, counts[key], share))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("labels_dir", type=Path, help="Directory of annotation JSON files")
    parser.add_argument("--taxonomy", type=Path, default=REPOSITORY / TAXONOMY_PATH)
    args = parser.parse_args()
    taxonomy = FailureTaxonomy.from_yaml(args.taxonomy)
    primaries = primary_types(args.labels_dir, taxonomy)
    total = len(primaries)
    print("failures with a primary type: %d" % total)
    categories = Counter(taxonomy.category_of(label) for label in primaries)
    print_table("Category", categories, list(taxonomy.category_priority), total)
    print_table("Error type", Counter(primaries), list(taxonomy.paper_types), total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
