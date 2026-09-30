#!/usr/bin/env python3
"""Resolve the single DERAIL judge and its fail-closed runtime settings."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import yaml


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPOSITORY / "configs/judges/routify_model_registry.json"
DEFAULT_CONFIG = REPOSITORY / "configs/judges/default.yaml"
AGENT_CONFIGS = REPOSITORY / "configs/agents"
FIELDS = ("protocol", "max_images", "admission", "concurrency", "timeout_seconds")


def load_config(path: Path | None = None) -> dict:
    path = Path(path or os.environ.get("DERAIL_JUDGE_CONFIG") or DEFAULT_CONFIG)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not str(config.get("model") or "").strip():
        raise ValueError(f"judge config has no model: {path}")
    return config


def configured_model(config: dict | None = None) -> str:
    return str((config or load_config())["model"]).strip()


def require_single_judge(model: str | None, config: dict | None = None) -> str:
    """Return the judge to use; a model other than the configured one is an error."""
    expected = configured_model(config)
    if not model or model == expected:
        return expected
    if os.environ.get("ALLOW_CONFIG_OVERRIDE") == "1":
        return model
    raise ValueError(
        f"judge model {model!r} differs from configs/judges/default.yaml ({expected!r}); "
        "the formal path uses one judge (set ALLOW_CONFIG_OVERRIDE=1 only for an experiment)"
    )


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower().rsplit("/", 1)[-1])


def agent_names(agent_id: str) -> set[str]:
    """agent_id plus the model its config names, for the same-source check."""
    names = {agent_id}
    path = AGENT_CONFIGS / f"{agent_id}.yaml"
    if path.is_file():
        agent = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        names |= {str(agent[key]) for key in ("model", "served_model_name") if agent.get(key)}
    return names


def same_source(judge: str, names: set[str], config: dict | None = None) -> bool:
    config = config or load_config()
    judge_n = _norm(judge)
    groups = [{_norm(m) for m in group} for group in config.get("same_source_groups") or []]
    for name in map(_norm, names):
        if not name:
            continue
        if name == judge_n or name.endswith(judge_n) or judge_n.endswith(name):
            return True
        if any(judge_n in group and name in group for group in groups):
            return True
    return False


def evaluated_agents(result_dir: Path) -> set[str]:
    """Agents whose actions a takeover result dir contains (source prefix + target)."""
    names: set[str] = set()
    for task in [result_dir, *result_dir.iterdir()] if result_dir.is_dir() else []:
        for manifest in ("takeover_manifest.json", "takeover_launch.json"):
            path = task / manifest
            if not path.is_file():
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            for key in ("source_agent", "target_agent"):
                if record.get(key):
                    names |= agent_names(str(record[key]))
            if record.get("target_model"):
                names.add(str(record["target_model"]))
    return names


def require_independent(judge: str, names: set[str], config: dict | None = None) -> None:
    if same_source(judge, names, config):
        raise ValueError(f"judge {judge!r} shares a source with an evaluated agent {sorted(names)}")


def resolve(registry_path: Path, model: str) -> dict[str, object]:
    document = json.loads(registry_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("models"), dict):
        raise ValueError("unsupported Judge model registry schema")
    entry = document["models"].get(model)
    if not isinstance(entry, dict):
        raise ValueError(f"Judge model is not registered: {model}")
    missing = [field for field in FIELDS if field not in entry]
    if missing:
        raise ValueError(f"Judge model registry entry {model} is missing: {', '.join(missing)}")
    allow_candidate = os.environ.get("DERAIL_JUDGE_ALLOW_CANDIDATE") == "1"
    if entry["admission"] != "admitted" and not (allow_candidate and entry["admission"] == "candidate"):
        raise ValueError(
            f"Judge model is not admitted: {model} (state={entry['admission']})"
        )
    for field in ("max_images", "concurrency", "timeout_seconds"):
        if not isinstance(entry[field], int) or entry[field] <= 0:
            raise ValueError(f"Judge model registry entry {model} has invalid {field}")
    if entry["protocol"] != "openai_chat_compatible":
        raise ValueError(f"Judge runner does not support protocol: {entry['protocol']}")
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--agent", action="append", default=[],
                        help="Evaluated agent_id; repeat for several")
    parser.add_argument("--result-dir", action="append", default=[], type=Path,
                        help="Takeover result dir whose manifests name the evaluated agents")
    parser.add_argument("--print-model", action="store_true",
                        help="Print the resolved judge model instead of registry fields")
    args = parser.parse_args()
    try:
        config = load_config()
        model = require_single_judge(args.model, config)
        names = set().union(*(agent_names(a) for a in args.agent), set())
        for result_dir in args.result_dir:
            names |= evaluated_agents(result_dir)
        require_independent(model, names, config)
        if args.print_model:
            print(model)
            return 0
        entry = resolve(args.registry, model)
    except (OSError, json.JSONDecodeError, yaml.YAMLError, ValueError) as error:
        parser.exit(2, f"FATAL: {error}\n")
    print("\t".join(str(entry[field]) for field in FIELDS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
