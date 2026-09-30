"""Training-set construction (execution doc v1.2 section 10A; brief ``train/``).

``build_samples`` turns traces + failure analyses + gold lineages into ``training-sample/1.0``
records: base (success steps), detection (evidence step, program-verified references),
verification (before an irreversible write of an unexposed error), negative (consistent
steps).  Recovery samples need the base model's own rollouts from the truncation state and
are produced VM/GPU-side (``scripts/recovery_gen_v1.py``); the builder emits their
truncation points.
"""
