#!/usr/bin/env python3
"""Write a human-review sheet for one Task-IR extraction run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, List, Mapping


def _rows(modules_path: Path) -> List[dict]:
    return [
        json.loads(line)
        for line in modules_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _cell(value: Any) -> str:
    """Markdown table cell: pipes escaped, newlines flattened, never shortened."""

    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _unbound_table(entries: List[Mapping[str, Any]]) -> List[str]:
    if not entries:
        return ["(none)"]
    lines = ["| # | name | app | kind | needed_by | why |", "|---|---|---|---|---|---|"]
    for index, entry in enumerate(entries, start=1):
        record = entry if isinstance(entry, Mapping) else {"name": entry}
        lines.append(
            "| %d | %s | %s | %s | %s | %s |"
            % (
                index,
                _cell(record.get("name")),
                _cell(record.get("app")),
                _cell(record.get("kind")),
                _cell(record.get("needed_by")),
                _cell(record.get("why")),
            )
        )
    return lines


def render(rows: List[dict]) -> str:
    issue_kinds = Counter(
        issue.split(":", 1)[0] for row in rows for issue in row["provenance"]["static_issues"]
    )
    lines = [
        "# Task-IR extraction review sheet",
        "",
        "Modules: %d; static-valid: %d; issue kinds: %s"
        % (
            len(rows),
            sum(1 for row in rows if row["provenance"]["static_valid"]),
            dict(sorted(issue_kinds.items())),
        ),
        "",
        "Reviewer columns are blank on purpose. Fill `entity_precision` and `edge_precision` as",
        "correct/total and `verdict` as accept / fix / reject. Full details per task follow.",
        "",
        "| task | nodes | edges | static valid | issues | unbound "
        "| entity_precision | edge_precision | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        fragment = row["fragment"]
        provenance = row["provenance"]
        lines.append(
            "| %s | %d | %d | %s | %d | %d |  |  |  |"
            % (
                row["source_task_id"],
                len(fragment["nodes"]),
                len(fragment["edges"]),
                "yes" if provenance["static_valid"] else "no",
                len(provenance["static_issues"]),
                len(provenance["unbound_entities"]),
            )
        )
    for row in rows:
        fragment = row["fragment"]
        provenance = row["provenance"]
        lines += [
            "",
            "## %s" % row["source_task_id"],
            "",
            "**Instruction.** %s" % row.get("instruction", ""),
            "",
        ]
        lines += ["**Operation chain.**", ""]
        for node in fragment["nodes"]:
            effects = ", ".join(
                "%s/%s" % (effect.get("effect_type"), effect.get("reversibility_class"))
                for effect in node.get("side_effects", ())
            )
            lines.append(
                "- `%s` %s @ %s: %s%s"
                % (
                    node["node_id"],
                    node.get("op"),
                    node.get("app"),
                    node.get("semantic_goal", ""),
                    (" [effects: %s]" % effects) if effects else "",
                )
            )
        lines += ["", "**Static issues.**", ""]
        lines += ["- %s" % issue for issue in provenance["static_issues"]] or ["(none)"]
        lines += ["", "**Unbound entities.**", ""]
        lines += _unbound_table(provenance["unbound_entities"])
        if provenance.get("extractor_notes"):
            lines += ["", "**Extractor notes.** %s" % provenance["extractor_notes"]]
    lines.append("")
    return "\n".join(lines)


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv else None)
    text = render(_rows(args.modules))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text.splitlines()[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
