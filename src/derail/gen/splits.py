"""Train/test split of the composed workflows (paper section 3, "Constructing erroneous states").

The split is by workflow: every rollout, takeover state and recovery trajectory derived from a
workflow belongs to that workflow's split.  ``split_workflows`` is deterministic in ``seed``
(workflows are ranked by ``sha256(seed|id)``, so input order does not matter); with
``source_disjoint`` the test workflows' source tasks are also kept out of training (the paper
does not require it; workflows that would share a source task across the split are
``excluded``).  ``scripts/freeze_source_splits.py`` freezes the manifest; downstream code asks
``split_of(item_id, load_workflow_splits(path))``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

SPLITS_VERSION = "workflow-splits/1.0"


def _rank(seed: int, key: str) -> str:
    return hashlib.sha256(("%s|%s" % (seed, key)).encode("utf-8")).hexdigest()


def split_workflows(
    workflows: Mapping[str, Sequence[str]],
    seed: int,
    test_size: Optional[int] = None,
    test_fraction: Optional[float] = None,
    source_disjoint: bool = False,
) -> Dict[str, Any]:
    """``workflows`` maps a workflow id to its source task ids.  Returns ``{"splits": {id:
    train|test|excluded}, "counts", "source_tasks"}``; ``test_size`` wins over
    ``test_fraction``."""

    ids = sorted(workflows, key=lambda w: _rank(seed, w))
    if test_size is None:
        if test_fraction is None:
            raise ValueError("need test_size or test_fraction")
        test_size = int(round(float(test_fraction) * len(ids)))
    sources = {w: set(workflows[w]) for w in ids}
    if not source_disjoint:
        test = ids[:test_size]
        train = ids[test_size:]
    else:
        # Grow a set of held-out source tasks until enough workflows lie entirely inside it.
        held: set = set()
        inside: List[str] = []
        for task in sorted({t for s in sources.values() for t in s}, key=lambda t: _rank(seed, t)):
            held.add(task)
            inside = [w for w in ids if sources[w] <= held]
            if len(inside) >= test_size:
                break
        test = inside[:test_size]
        test_sources = set().union(*(sources[w] for w in test)) if test else set()
        train = [w for w in ids if w not in test and not sources[w] & test_sources]
    splits = {w: "excluded" for w in ids}
    splits.update({w: "train" for w in train})
    splits.update({w: "test" for w in test})
    by_split = {
        name: sorted(set().union(*(sources[w] for w in ids if splits[w] == name)))
        for name in ("train", "test")
    }
    return {
        "splits": dict(sorted(splits.items())),
        "counts": {
            name: sum(1 for v in splits.values() if v == name)
            for name in ("train", "test", "excluded")
        },
        "source_tasks": {
            **by_split,
            "shared": sorted(set(by_split["train"]) & set(by_split["test"])),
        },
    }


def load_workflow_splits(path: Union[str, Path]) -> Dict[str, Any]:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("schema_version") != SPLITS_VERSION:
        raise ValueError("unsupported workflow splits %r" % record.get("schema_version"))
    return record


def split_of(item_id: str, splits: Mapping[str, Any]) -> Optional[str]:
    """Split (``train`` / ``test`` / ``excluded``) of a workflow or of anything derived from
    it: an id that is the workflow id, carries a ``#<variant>`` suffix (``gen-...#v0``) or
    otherwise starts with the workflow id.  None when the id belongs to no split workflow."""

    table = splits.get("splits", splits)
    key = str(item_id).split("#", 1)[0]
    if key in table:
        return table[key]
    owner = max((w for w in table if str(item_id).startswith(w)), key=len, default=None)
    return table[owner] if owner else None
