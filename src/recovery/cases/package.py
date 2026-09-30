from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Union


def funnel(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    out = Counter()
    for row in rows:
        out["runs"] += 1
        out["failures"] += int(bool(row.get("failed")))
        out["auto_analyzed"] += int(bool(row.get("analyzed")))
        out["repaired"] += int(bool(row.get("repaired")))
        out["cases"] += int(row.get("cases", 0))
    return dict(out)


def write_bundle(
    out_dir: Union[str, Path],
    cases: Sequence[Mapping[str, Any]],
    removed: Sequence[Mapping[str, Any]],
    per_rollout: Sequence[Mapping[str, Any]],
    depth_grid: Sequence[int],
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    out = Path(out_dir)
    (out / "cases").mkdir(parents=True, exist_ok=True)
    for case in cases:
        (out / "cases" / ("%s.json" % case["case_id"])).write_text(
            json.dumps(case, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    (out / "duplicates.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in removed), encoding="utf-8"
    )
    by_depth = Counter(c["error_depth"] for c in cases)
    by_type = Counter((c["paper_category"], c["reversibility_stratum"]) for c in cases)
    manifest = {
        "schema_version": "case-bundle/1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "depth_grid": list(depth_grid),
        "funnel": funnel(per_rollout),
        "cases": len(cases),
        "duplicates": len(removed),
        "by_depth": {str(k): v for k, v in sorted(by_depth.items())},
        "by_category_stratum": {"%s/%s" % k: v for k, v in sorted(by_type.items())},
        "provenance": dict(provenance),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = ["# Case bundle", "", "| stage | n |", "|---|---|"] + [
        "| %s | %d |" % kv for kv in manifest["funnel"].items()
    ]
    lines += ["", "| depth | cases |", "|---|---|"] + [
        "| %s | %d |" % kv for kv in manifest["by_depth"].items()
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest
