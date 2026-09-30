from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

from derail.ir.extract import V1ExtractorConfig, run_extraction_v1
from derail.longhorizon.extraction import LLMClient, LLMResponse, render_prompt

FEEDBACK_MAX_CHARS = 60000


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def collect_problems(run_dir: Path, task_id: str) -> List[str]:
    problems: List[str] = []
    manifest = _load(run_dir / "manifest.json") or {}
    for entry in manifest.get("per_task", []):
        if entry.get("task_id") == task_id:
            problems.extend("static: %s" % e for e in entry.get("static_errors", []))
    task_ir = _load(run_dir / "task_ir" / ("%s.json" % task_id))
    if task_ir:
        for issue in task_ir.get("provenance", {}).get("grounding_issues", []):
            if issue.get("severity") == "error":
                problems.append(
                    "grounding %s at %s: %s"
                    % (issue["code"], issue.get("node_id"), issue.get("detail"))
                )
    checks = run_dir / "rubric_check" / "rubric_checks.jsonl"
    if checks.is_file():
        for line in checks.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("task_id") != task_id or record.get("verdict") == "auto_validated":
                continue
            if record.get("error"):
                problems.append("gold execution: %s" % record["error"])
            if record.get("code") == "FINAL_VERIFIER_FAILED":
                problems.append(
                    "the final_verifier predicate returned false on the gold state produced by "
                    "your own nodes: the verifier and the writes disagree"
                )
            for unmatched in record.get("unmatched", []):
                problems.append(
                    "rubric expectation not reproduced: %s %r (from rubric %s: %s)"
                    % (
                        unmatched["kind"],
                        unmatched["value"],
                        unmatched.get("rubric_index"),
                        unmatched.get("raw"),
                    )
                )
            if record.get("code") == "NO_COMPARABLE_EXPECTATIONS" and record.get("lexical"):
                for criterion in record["lexical"].get("criteria", []):
                    if not criterion["satisfied"]:
                        problems.append(
                            "rubric %s not covered by the IR (missing: %s)"
                            % (
                                criterion["rubric_index"],
                                ", ".join(
                                    k for k in criterion["keywords"] if k not in criterion["hits"]
                                ),
                            )
                        )
    if any(
        p.startswith("static: REASONING_BUDGET_EXHAUSTED") or "contains no JSON object" in p
        for p in problems
    ):
        problems.append(
            "your previous attempt spent its whole budget reasoning and returned nothing: "
            "decide quickly and write the JSON directly"
        )
    if not task_ir and not problems:
        problems.append("the reply contained no JSON object")
    return problems


def failing_tasks(run_dir: Path) -> List[str]:
    manifest = _load(run_dir / "manifest.json") or {}
    validated = set()
    checks = run_dir / "rubric_check" / "rubric_checks.jsonl"
    if checks.is_file():
        for line in checks.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("verdict") == "auto_validated":
                validated.add(record["task_id"])
    return [e["task_id"] for e in manifest.get("per_task", []) if e["task_id"] not in validated]


class FeedbackClient:
    """Wraps the real client so the user prompt of a failing task carries its problems."""

    def __init__(
        self,
        inner: LLMClient,
        template: str,
        previous: Mapping[str, str],
        problems: Mapping[str, Sequence[str]],
        task_of_prompt: Callable[[str], str],
    ):
        self._inner = inner
        self._template = template
        self._previous = previous
        self._problems = problems
        self._task_of_prompt = task_of_prompt

    def complete(self, system: str, user: str) -> LLMResponse:
        task_id = self._task_of_prompt(user)
        problems = self._problems.get(task_id) or ["unknown failure"]
        previous = self._previous.get(task_id, "")[:FEEDBACK_MAX_CHARS]
        prompt = render_prompt(
            self._template,
            {
                "original_prompt": user,
                "problems": "\n".join("- %s" % p for p in problems),
                "previous_json": previous,
            },
        )
        return self._inner.complete(system, prompt)


def run_fix_round(
    tasks: Sequence[Mapping[str, Any]],
    *,
    previous_run_dir: Path,
    output_dir: Path,
    config: V1ExtractorConfig,
    template_path: Path,
    client: LLMClient,
    **extraction_kwargs: Any,
) -> Dict[str, Any]:
    todo = set(failing_tasks(previous_run_dir))
    selected = [t for t in tasks if t["id"] in todo]
    problems = {t["id"]: collect_problems(previous_run_dir, t["id"]) for t in selected}
    previous = {}
    for t in selected:
        ir_path = previous_run_dir / "task_ir" / ("%s.json" % t["id"])
        ir = _load(ir_path)
        if ir:
            ir = {k: v for k, v in ir.items() if k != "provenance"}
            previous[t["id"]] = json.dumps({"task_ir": ir}, ensure_ascii=False, indent=1)
        else:
            reply = previous_run_dir / "replies" / ("%s.txt" % t["id"])
            previous[t["id"]] = reply.read_text(encoding="utf-8") if reply.is_file() else ""
    template = template_path.read_text(encoding="utf-8")

    def task_of_prompt(user: str) -> str:
        for line in user.splitlines():
            if line.startswith("task_id:"):
                return line.split(":", 1)[1].strip()
        return ""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feedback.json").write_text(
        json.dumps(problems, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not selected:
        manifest = {
            "stage": "task_ir_v1_fix_round",
            "counts": {"tasks": 0},
            "previous_run": str(previous_run_dir),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
        )
        return manifest
    wrapped = FeedbackClient(client, template, previous, problems, task_of_prompt)
    manifest = run_extraction_v1(
        selected, config=config, output_dir=output_dir, client=wrapped, **extraction_kwargs
    )
    manifest["previous_run"] = str(previous_run_dir)
    manifest["fed_back"] = {k: len(v) for k, v in problems.items()}
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
