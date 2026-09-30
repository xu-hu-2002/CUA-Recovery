"""Detectability model (execution doc v1.2 section 6; brief ``detect/``).

- ``latent_static``  static latent horizon per node: propagate the contamination set of a
                      node along data dependencies and find the first downstream node whose
                      required reads intersect it (section 2.2 / 6.2, fact-level, no OSM).
- ``mutations``      the mutation library (section 6.1): which mutation types apply to a
                      produce type; static ``mutation/1.0`` records.
- ``profile``        task-level ``latent-profile/1.0`` and bucket membership (section 6.6).

Dynamic latent horizon (mutations executed through the gold interpreter, section 6.3) is
deferred (brief section 5: static first).
"""
