#!/usr/bin/env python3
"""Build Judge evidence in rubric_bundle.json. Usage: python3 scripts/takeover/bundle_prefix.py <dir> [<dir>...]"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

REPOSITORY = Path(__file__).resolve().parents[2]
SRC = REPOSITORY / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recovery.takeover.source_logs import sanitize_visible_response  # noqa: E402

sys.path.insert(0, str(REPOSITORY / "scripts" / "judge"))
from judge_model_registry import load_config  # noqa: E402

TAKEOVER_CONFIG = REPOSITORY / "configs" / "takeover" / "takeover.yaml"
STALE_RESULTS = ("osworld_full_traj_result.json", "rubric_judge_result.json")
CONTINUED = "(same model turn as the previous step)"
PREFIX_DIR = "prefix_replay"
TOOL_MARK = "[tool] "
EVIDENCE_KEY = "recovery_evidence"


def _judge_relative(screenshot: Path, replayed: Path, task: Path) -> str:
    """Record the prefix screenshot path relative to the Judge's working directory."""
    if screenshot == replayed:
        return f"{PREFIX_DIR}/{replayed.name}"
    return str(screenshot.resolve())


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _visible_response(source_record_uri: str, cache: dict[Path, list[str]]) -> str:
    path_text, _, line = source_record_uri.partition("#line=")
    path = Path(path_text)
    if not path.is_file():
        marker = "/artifacts/raw_rollouts/mypcbench/v1/"
        if marker in path_text:
            suffix = path_text.split(marker, 1)[1]
            path = REPOSITORY / "artifacts/raw_rollouts/import_20260908" / suffix
    if path not in cache:
        cache[path] = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(cache[path][int(line) - 1])
    return sanitize_visible_response(record.get("response"))


def _canonical_steps(selection: list[dict], trajectory_id: str) -> dict[int, dict]:
    record = next(r for r in selection if r.get("trajectory_id") == trajectory_id)
    canonical = Path(record["canonical_trajectory_uri"])
    if not canonical.is_file():
        marker = "/artifacts/recovery_builds/"
        uri = str(record["canonical_trajectory_uri"])
        relative = uri.split(marker, 1)[1] if marker in uri else ""
        exact = REPOSITORY / "artifacts/recovery_builds" / relative
        matches = [exact] if exact.is_file() else list(
            (REPOSITORY / "artifacts/recovery_builds").glob(
                f"*/canonical/{trajectory_id}/trajectory.jsonl"
            )
        )
        expected_hash = record.get("source_trajectory_sha256")
        if len(matches) > 1 and expected_hash:
            hash_matches = [
                path for path in matches
                if hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
            ]
            if hash_matches:
                matches = hash_matches
        if len(matches) > 1:
            content_hashes = {
                hashlib.sha256(path.read_bytes()).hexdigest() for path in matches
            }
            if len(content_hashes) == 1:
                matches = [sorted(matches)[0]]
        if len(matches) != 1:
            raise RuntimeError(f"{trajectory_id}: canonical trajectory has {len(matches)} local matches")
        canonical = matches[0]
    rows = {}
    for line in canonical.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[int(row["action_index_global"])] = row
    return rows


PREFIX_ARTIFACTS = ("native_history.json", "prefix_replay_log.json", "takeover_manifest.json")


def has_prefix_artifacts(task: Path) -> bool:
    return all((task / name).is_file() for name in PREFIX_ARTIFACTS)


def rewrite_task(task: Path, selection: list[dict], cache: dict) -> str:
    bundle_path = task / "rubric_bundle.json"
    if not bundle_path.is_file():
        return "no-bundle"
    if not has_prefix_artifacts(task):
        return "no-prefix-artifacts"
    bundle = _load_json(bundle_path)
    artifacts = bundle.setdefault("artifacts", {})
    if artifacts.get("takeover_prefix"):
        return "skipped"

    history = _load_json(task / "native_history.json")
    replay = _load_json(task / "prefix_replay_log.json")
    manifest = _load_json(task / "takeover_manifest.json")
    indices = [int(i) for i in history["action_indices"]]
    if [int(r["action_index_global"]) for r in replay] != indices:
        raise RuntimeError(f"{task}: prefix_replay_log does not match native_history action_indices")
    canonical = _canonical_steps(selection, manifest["human_annotation"]["trajectory_id"])

    prefix_steps = []
    last_turn = None
    for idx, entry in zip(indices, replay):
        row = canonical[idx]
        screenshot = Path(entry["observation_after_uri"])
        replayed = task / PREFIX_DIR / screenshot.name
        if not screenshot.is_file():
            screenshot = replayed
        if not screenshot.is_file():
            raise RuntimeError(f"{task}: missing prefix screenshot {screenshot}")
        turn = row.get("turn_index")
        message = _visible_response(row["source_record_uri"], cache) if turn != last_turn else CONTINUED
        last_turn = turn
        if row.get("tool_result") and message in ("", CONTINUED):
            message = "\n".join(filter(None, (
                "" if message == CONTINUED else message,
                _tool_line({"type": "tool_call", "action": row.get("action")}),
                TOOL_MARK + str(row["tool_result"]),
            )))
        prefix_steps.append(
            {
                "step_num": f"prefix_{idx}",
                "raw_response_text": message,
                "parsed_action_obj": {"action": row["action_summary"], "done": False, "reward": 0.0},
                "screenshot": _judge_relative(screenshot, replayed, task),
            }
        )

    post_only = task / "rubric_bundle.post_takeover_only.json"
    if not post_only.exists():
        post_only.write_text(bundle_path.read_text(encoding="utf-8"), encoding="utf-8")
    post_steps = artifacts.get("steps") or []
    artifacts["steps"] = prefix_steps + post_steps
    artifacts.setdefault("counts", {})["steps"] = len(artifacts["steps"])
    artifacts["takeover_prefix"] = {
        "n_prefix_steps": len(prefix_steps),
        "n_post_takeover_steps": len(post_steps),
        "prefix_end_action_index": manifest["prefix_end_action_index"],
        "depth": manifest["depth"],
        "screenshots": "prefix_replay",
    }
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    for name in STALE_RESULTS:
        stale = task / name
        if stale.is_file():
            stale.replace(task / name.replace(".json", ".post_takeover_only.json"))
    return "rewritten"


def _tool_line(item: dict) -> str:
    return TOOL_MARK + json.dumps(item, ensure_ascii=False, sort_keys=True)


def _blocks(message: dict) -> list[dict]:
    content = message.get("content")
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _text_only(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(b.get("text") or "") for b in content
                         if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _claude_tool_text(messages: list[dict], turn: int) -> str:
    """Non-GUI tool calls of one assistant turn plus their results, images dropped."""
    positions = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    position = positions[turn]
    uses = [b for b in _blocks(messages[position])
            if b.get("type") == "tool_use" and b.get("name") != "computer"]
    if not uses:
        return ""
    ids = {b.get("id") for b in uses}
    results = [b for b in (_blocks(messages[position + 1]) if position + 1 < len(messages) else [])
               if b.get("type") == "tool_result" and b.get("tool_use_id") in ids]
    lines = [_tool_line({"type": "tool_use", "id": b.get("id"), "name": b.get("name"),
                         "input": b.get("input")}) for b in uses]
    lines += [_tool_line({"type": "tool_result", "tool_use_id": b.get("tool_use_id"),
                          "content": _text_only(b.get("content"))}) for b in results]
    return "\n".join(lines)


def _tool_text(row: dict, file_index: int, messages: list[dict] | None) -> str:
    metadata = row.get("agent_metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("tool_messages"), list):
        return "\n".join(_tool_line(m) for m in metadata["tool_messages"] if isinstance(m, dict))
    return _claude_tool_text(messages, file_index) if messages is not None else ""


def _final_state(task: Path, probe_file: str, max_chars: int) -> dict | None:
    """s_T from the last line of the task's state probe file (recovery.rollout.state_probe)."""
    path = task / probe_file
    lines = [line for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()] if path.is_file() else []
    if not lines:
        return None
    last = json.loads(lines[-1])
    outputs = last.get("probe_output") if isinstance(last.get("probe_output"), dict) else {}
    parts = [f"[{name}]\n{str(output).rstrip()}" for name, output in sorted(outputs.items())]
    fingerprint = last.get("state_fingerprint")
    if isinstance(fingerprint, dict) and fingerprint.get("sha256"):
        parts.append(f"State fingerprint (sha256): {fingerprint['sha256']}")
    text = "\n".join(parts)
    if not text:
        return None
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated to {max_chars} characters]"
    return {"source": probe_file, "traj_index": last.get("traj_index"), "text": text}


def enrich_task(task: Path, max_chars: int, probe_file: str) -> str:
    """Fill empty tool-round payloads and attach s_T; mark the bundle so reruns skip it."""
    bundle_path = task / "rubric_bundle.json"
    if not bundle_path.is_file():
        return "no-bundle"
    bundle = _load_json(bundle_path)
    artifacts = bundle.setdefault("artifacts", {})
    if EVIDENCE_KEY in artifacts:
        return "evidence-present"
    traj = task / "traj.jsonl"
    rows = [json.loads(line) for line in traj.read_text(encoding="utf-8").splitlines()
            if line.strip()] if traj.is_file() else []
    messages_path = task / "messages.json"
    messages = _load_json(messages_path) if messages_path.is_file() else None
    if not isinstance(messages, list) or \
            sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant") != len(rows):
        messages = None

    order = sorted(range(len(rows)), key=lambda i: int(rows[i].get("step_num") or 0))
    post = [s for s in artifacts.get("steps") or [] if not str(s.get("step_num")).startswith("prefix_")]
    filled = []
    if len(post) == len(order):
        for step, index in zip(post, order):
            action = (step.get("parsed_action_obj") or {}).get("action")
            if action != "TOOL_CALL" or str(step.get("raw_response_text") or "").strip():
                continue
            if str(step.get("step_num")) != str(rows[index].get("step_num")):
                continue
            text = _tool_text(rows[index], index, messages)
            if text:
                step["raw_response_text"] = text
                filled.append(step.get("step_num"))
    final_state = _final_state(task, probe_file, max_chars)
    artifacts[EVIDENCE_KEY] = {"tool_payloads_filled": filled,
                               "final_state": bool(final_state)}
    if final_state:
        artifacts["final_state"] = final_state
    if filled or final_state:
        backup = task / "rubric_bundle.pre_evidence.json"
        if not backup.exists():
            backup.write_text(bundle_path.read_text(encoding="utf-8"), encoding="utf-8")
        for name in STALE_RESULTS:
            if (task / name).is_file():
                (task / name).replace(task / name.replace(".json", ".pre_evidence.json"))
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    return "enriched" if filled or final_state else "unchanged"


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    cache: dict = {}
    max_chars = int(load_config()["final_state"]["max_chars"])
    takeover = yaml.safe_load(TAKEOVER_CONFIG.read_text(encoding="utf-8"))
    environment = Path(str(takeover["environment_config"]))
    environment = environment if environment.is_absolute() else REPOSITORY / environment
    probe_file = str(yaml.safe_load(environment.read_text(encoding="utf-8"))["state_probe_file"])
    for cell in map(Path, argv):
        tasks = sorted(p for p in cell.iterdir() if p.is_dir() and not p.name.startswith("_"))
        manifest = next(
            (
                parent / "selection_manifest.json"
                for parent in (cell.parent, *cell.parents)
                if (parent / "selection_manifest.json").is_file()
            ),
            cell.parent.parent / "selection_manifest.json",
        )
        counts: dict[str, int] = {}
        if any(has_prefix_artifacts(task) for task in tasks):
            if not manifest.is_file():
                raise SystemExit(f"{cell} has replayed prefixes but no {manifest}")
            selection = _load_json(manifest)["included"]
            for task in tasks:
                status = rewrite_task(task, selection, cache)
                counts[status] = counts.get(status, 0) + 1
        for task in tasks:
            status = enrich_task(task, max_chars, probe_file)
            counts[status] = counts.get(status, 0) + 1
        print(f"[bundle-evidence] {cell}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
