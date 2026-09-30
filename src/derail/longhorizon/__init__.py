"""Long-horizon task generation and DERAIL case construction (manual v0.2).

This package adds the v0.2 increments on top of two existing layers and deliberately does not
re-implement them:

* ``derail.synthesis`` -- the v0.1 symbolic core (Task-IR validation, typed compatibility,
  skeleton mining, empirical/failure-informed skeleton weights, beam-search composition).
  ``derail.longhorizon`` consumes its fragments and generation records unchanged.
* ``derail.construction`` / ``derail.replay`` / ``derail.annotation`` -- action-indexed prefix
  repair, replay verification and immutable human annotations.  Section 14A of the manual is
  built from these primitives instead of a second copy.

Module map (manual section in brackets):

``ontology``               operations, R0-R3 classes, effect types, commit scopes   [2.1, 5.1]
``taxonomy``               paper error types -> coarse group, primary-type rule    [9.2]
``effects``                side-effect validation and reversibility metrics        [2.1, 11.3]
``carry``                  lineage analysis and the DECORATIVE_CARRY hard filter    [8.4]
``complexity``             v0.2 intrinsic long-horizon metrics                      [2]
``gates``                  composable hard filters with stable rejection codes      [16.1]
``reversibility_sampling`` ``empirical_stratified`` allocation and coverage gaps    [10.1]
``continuation``           post-error continuation statistics per failed rollout    [14A.1]
``cases``                  depth instantiation, stratum, dedup, funnel, yields      [14A.4-6]
``precheck``               Phase 0 pre-check report over the existing corpus        [17]
``pipeline``               v0.2 post-filter over ``derail.synthesis`` results       [10, 16]

Every module is importable on its own; nothing here reads the filesystem except ``precheck``
and the loaders that take an explicit path.
"""
