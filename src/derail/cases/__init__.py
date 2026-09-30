"""Case construction (execution doc v1.2 section 10; brief ``cases/``).

- ``repair``       automatic prefix repair: state-neutral segment elimination (10.1).
- ``instantiate``  depth instantiation, reversibility stratum, dedup (10.3) reusing the
                    v0.2 helpers in ``derail.longhorizon.cases``; emits ``derail-case/1.0``
                    with the section 10.4 fields from the failure analysis and the static
                    profile.
- ``package``      release bundle: cases, funnel, manifest.

Replay verification (10.2) is ``derail.harness.replay_runner`` (VM side).
"""
