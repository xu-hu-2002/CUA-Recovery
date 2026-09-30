"""World layer of the v1.2 pipeline (execution doc v1.2 sections 2.1, 4; brief section 1).

- ``facts``          canonical world facts ``(table, column, entity, value)`` and the registry
                      of equivalent sources for the same fact.
- ``state``          deterministic digests and row snapshots of a set of SQLite databases.
- ``sqlite_fixture`` build a database from a seed file (DDL + rows); used by the miniworld and,
                      later, by hazard variants that re-seed a world.

``schema_graph`` and ``osm_crawl`` from the brief are later build steps and need the VM.
"""
