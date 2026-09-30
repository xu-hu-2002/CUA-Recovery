from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import yaml

from derail.longhorizon.taxonomy import FailureTaxonomy


@dataclass(frozen=True)
class TypingRules:
    rules: Mapping[str, Mapping[str, Any]]
    residual_threshold: float
    taxonomy: FailureTaxonomy

    @classmethod
    def from_yaml(cls, path: Union[str, Path], taxonomy: FailureTaxonomy) -> "TypingRules":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if raw.get("schema_version") != "typing-rules/1.0":
            raise ValueError("unsupported typing rules %r" % raw.get("schema_version"))
        rules = {str(k): dict(v) for k, v in (raw.get("rules") or {}).items()}
        for pattern, rule in rules.items():
            if rule["paper_type"] not in taxonomy.paper_types:
                raise ValueError(
                    "typing rule %s names unknown paper type %s" % (pattern, rule["paper_type"])
                )
        return cls(
            rules=rules,
            residual_threshold=float(raw.get("residual_threshold", 0.7)),
            taxonomy=taxonomy,
        )

    def classify(
        self, evidence_pattern: Optional[str], outcome: Mapping[str, Any]
    ) -> Tuple[Optional[str], Optional[str], Optional[str], float, List[Dict[str, Any]]]:
        pattern = evidence_pattern
        if pattern is None:
            if outcome.get("budget_exhausted"):
                pattern = "budget_exhausted_no_root_cause"
            elif outcome.get("declared_complete") and outcome.get("final_verifier") is False:
                pattern = "premature_completion_claim"
        rule = self.rules.get(pattern or "")
        if rule is None:
            return None, None, None, 0.0, []
        paper_type = str(rule["paper_type"])
        category = self.taxonomy.category_of(paper_type)
        return (
            paper_type,
            category,
            self.taxonomy.group_of(paper_type),
            float(rule["confidence"]),
            [],
        )
