"""Takeover protocol settings and the prefix / replayed-state rules they drive."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import yaml

from derail.canonical.actions import action_to_dict
from derail.canonical.mypcbench import load_canonical_jsonl
from derail.canonical.trajectory import CanonicalStep
from derail.derived.layout import sha256_file
from derail.replay.verification import ReplayVerificationError, StateFingerprint
from derail.takeover.source_logs import read_source_row

REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPOSITORY / "configs" / "takeover" / "takeover.yaml"
_SCHEMA_VERSION = "takeover-protocol/1.0"
_CHOICES = {
    ("prefix", "source"): {"repaired", "original"},
    ("prefix", "on_missing_repaired"): {"error", "skip", "fallback_original"},
    ("replay_verification", "on_mismatch"): {"reject", "warn"},
    ("replay_verification", "on_missing_expected"): {"reject", "warn"},
}


class ProtocolExclusion(RuntimeError):
    """The protocol forbids running this episode; the reason is recorded, not retried."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__("%s: %s" % (reason, detail))
        self.reason = reason
        self.detail = detail


def load_takeover_config(path: Optional[Path] = None) -> Dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if config.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported takeover config: %s" % config_path)
    for (section, key), allowed in _CHOICES.items():
        if config[section][key] not in allowed:
            raise ValueError("%s.%s must be one of %s" % (section, key, sorted(allowed)))
    unknown = set(config["conditions"]) - set(config["allowed_conditions"])
    if unknown:
        raise ValueError("conditions not in allowed_conditions: %s" % sorted(unknown))
    environment = Path(str(config["environment_config"]))
    config["environment_config"] = str(
        (environment if environment.is_absolute() else REPOSITORY / environment).resolve()
    )
    config["config_uri"] = str(config_path.resolve())
    config["config_sha256"] = sha256_file(config_path)
    return config


def _same_step(left: CanonicalStep, right: CanonicalStep) -> bool:
    return (
        action_to_dict(left.action) == action_to_dict(right.action)
        and left.source_record_uri == right.source_record_uri
    )


def load_takeover_steps(
    canonical_path: Path,
    trajectory_id: str,
    root_cause_action_index: int,
    prefix_config: Mapping[str, Any],
    repaired_dir: Optional[Path] = None,
) -> Tuple[Tuple[CanonicalStep, ...], Dict[str, Any]]:
    """Return the steps to replay/inject and where they came from."""

    original = load_canonical_jsonl(canonical_path)
    info: Dict[str, Any] = {
        "prefix_source": "original",
        "prefix_uri": str(canonical_path.resolve()),
        "prefix_sha256": sha256_file(canonical_path),
    }
    if prefix_config["source"] == "original":
        return original, info
    base = Path(repaired_dir) if repaired_dir else Path(str(prefix_config["repaired_dir"]))
    if not base.is_absolute():
        base = canonical_path.resolve().parents[2] / base
    path = base / trajectory_id / str(prefix_config["trajectory_filename"])
    if not path.is_file():
        policy = prefix_config["on_missing_repaired"]
        if policy == "fallback_original":
            print("WARN: repaired prefix missing, using the unrepaired prefix: %s" % path,
                  file=sys.stderr)
            info["prefix_source"] = "original_fallback"
            return original, info
        if policy == "skip":
            raise ProtocolExclusion("repaired_prefix_missing", str(path))
        raise FileNotFoundError("repaired prefix is missing: %s" % path)
    repaired = load_canonical_jsonl(path, allow_gaps=True)
    by_index = {step.action_index_global: step for step in repaired}
    tail = [step for step in original if step.action_index_global >= root_cause_action_index]
    if not tail or any(
        step.action_index_global not in by_index
        or not _same_step(step, by_index[step.action_index_global])
        for step in tail
    ) or repaired[-1].action_index_global != original[-1].action_index_global:
        raise ValueError(
            "repaired prefix changes the root cause or later steps: %s" % path
        )
    return repaired, {
        "prefix_source": "repaired",
        "prefix_uri": str(path.resolve()),
        "prefix_sha256": sha256_file(path),
        "removed_action_indices": sorted(
            {step.action_index_global for step in original} - set(by_index)
        ),
        "repaired_action_indices": [step.action_index_global for step in repaired if step.repaired],
    }


def takeover_prefix(
    steps: Sequence[CanonicalStep], root_cause_action_index: int, depth: int
) -> Tuple[CanonicalStep, ...]:
    """Steps ``<= root + depth``; every index from the root cause on must be present."""

    prefix_end = root_cause_action_index + depth
    indices = [step.action_index_global for step in steps]
    if indices != sorted(set(indices)):
        raise ValueError("canonical action indices must be strictly increasing")
    if not set(range(root_cause_action_index, prefix_end + 1)).issubset(indices):
        raise ValueError(
            f"takeover depth {depth} is unavailable: root={root_cause_action_index}, "
            f"last_action_index={indices[-1] if indices else -1}"
        )
    return tuple(step for step in steps if step.action_index_global <= prefix_end)


def recorded_state_fingerprint(
    steps: Sequence[CanonicalStep], prefix_end: int, probe_file: str
) -> Tuple[Optional[StateFingerprint], str]:
    """Fingerprint the source rollout recorded after action ``prefix_end``."""

    by_index = {step.action_index_global: step for step in steps}
    step = by_index[prefix_end]
    following = by_index.get(prefix_end + 1)
    if not step.source_record_uri:
        return None, "source_record_uri_missing"
    if following is not None and following.source_record_uri == step.source_record_uri:
        return None, "takeover_point_inside_recorded_turn"
    path, line_number, _row = read_source_row(step)
    raw = None
    probes = path.parent / probe_file
    if probes.is_file():
        for line in probes.read_text(encoding="utf-8").splitlines():
            record = json.loads(line) if line.strip() else {}
            if record.get("traj_index") == line_number - 1:
                raw = record.get("state_fingerprint")
    if raw is None:
        return None, "source_state_fingerprint_missing"
    try:
        return StateFingerprint.from_dict(raw), ""
    except (ReplayVerificationError, AttributeError, TypeError) as exc:
        return None, "source_state_fingerprint_invalid: %s" % exc
