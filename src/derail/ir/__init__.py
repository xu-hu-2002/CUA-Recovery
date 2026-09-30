"""Task IR layer of the v1.2 pipeline (execution doc v1.2 sections 2.1, 5; brief section 1).

- ``model``             load / validate ``task-ir/1.0`` and expose the fragment view that the
                         v0.2 DAG utilities (``derail.longhorizon.dag``) consume unchanged.
- ``expr``              restricted expression evaluator used by ``derivation.kind == "expr"``
                         and by derived verifiers.
- ``gold_interpreter``  execute a Task IR against isolated database copies and emit
                         ``gold-lineage/1.0``.

``extract`` (LLM extraction), ``validate`` (entity existence, OSM reachability) and
``rubric_check`` are build step 3.
"""
