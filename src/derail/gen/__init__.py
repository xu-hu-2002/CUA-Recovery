"""Task generation (execution doc v1.2 section 7; brief ``gen/``).

- ``verifiers``  compile milestone / final verifiers from an IR, generate mutation values
                  from the world, run the mutation test and the dynamic latent horizon
                  (sections 6.3, 7.5) through the gold interpreter's value overrides.
- ``compat``     schema-derived typed compatibility between IR output and input ports (7.1).
- ``graft``      the five grafting operators and the latent-horizon-targeted beam search (7.2).
- ``hazards``    world-side hazard base rates and persona diffs (7.3); seeding is VM-side.
- ``realize``    instruction realization and round-trip prompts (7.6); LLM calls gated.
"""
