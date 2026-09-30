"""Rollout harness layer (execution doc v1.2 sections 4.4, 8; brief ``harness/``).

- ``control_client``  the DERAIL endpoints of the VM control API (cursor, changelog,
                       observations, digest) over the standard library.
- ``trace_builder``   turns per-step ledgers into ``rollout-trace/1.0`` records.
- ``env_wrapper``     wraps a MyPCBench ``env`` so the official runner loop is unchanged:
                       cursor before every action, three ledgers after it.
- ``replay_runner``   replays a repaired prefix + d steps and compares ledgers.

Everything is testable with fakes (``tests/test_harness_rollout.py``); the VM is only needed
by ``scripts/run_rollout_v1.py``.
"""
