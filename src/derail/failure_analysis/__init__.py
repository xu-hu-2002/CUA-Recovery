"""Automatic failure analysis (execution doc v1.2 section 9; brief ``analysis/``).

The brief names this package ``analysis/``; ``derail.analysis`` already holds the paper's
result analysis, so the lineage-based failure analysis lives here (DECISIONS D-027).

- ``detectors``      state / parameter / omission root-cause detectors over a
                      ``rollout-trace/1.0`` and its ``gold-lineage/1.0``.
- ``typing_rules``   evidence pattern -> paper type (``configs/synthesis/typing_rules_v1.yaml``).
- ``horizon``        earliest identifiable point, action and semantic horizons, censoring.
- ``run``            ``analyze_failure(...) -> failure-analysis/1.0``.
"""
