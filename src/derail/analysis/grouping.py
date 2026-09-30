"""常用结果分组；后续图表生成应复用同一分组逻辑。"""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Tuple

from derail.evaluation.metrics import EpisodeResult


def group_by_agent_and_depth(
    results: Iterable[EpisodeResult],
) -> Dict[Tuple[str, int], List[EpisodeResult]]:
    grouped: DefaultDict[Tuple[str, int], List[EpisodeResult]] = defaultdict(list)
    for result in results:
        grouped[(result.agent_id, result.depth)].append(result)
    return dict(grouped)
