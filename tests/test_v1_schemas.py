"""Validator tests for the v1.0 schemas."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import jsonschema

from derail.derived.schema import validate_schema

REPOSITORY = Path(__file__).resolve().parents[1]
SHA = "0" * 64

FACT = {
    "table": "hoolicalendar.events",
    "column": "start_at",
    "entity": "events:530",
    "value": "2026-04-07T11:00:00",
}
PORT = {
    "port_id": "p",
    "type": "Date",
    "grounding": "hoolicalendar.events.start_at",
    "source": "world",
    "cardinality": "one",
}
NODE = {
    "node_id": "n1",
    "op": "retrieve",
    "app": "hoolicalendar",
    "route_hint": ["/event/530"],
    "reads": [{"table": "hoolicalendar.events", "column": "start_at", "entity_ref": "events:530"}],
    "produces": [
        {
            "name": "start",
            "type": "Date",
            "derivation": {
                "kind": "sql",
                "app": "hoolicalendar",
                "query": "SELECT start_at FROM events WHERE id = 530",
                "returns": "scalar",
            },
        }
    ],
    "writes": [],
    "reversibility_class": "R0",
    "side_effects": [],
    "inputs": [],
    "outputs": [dict(PORT, port_id="start", grounding="derived:n1:start")],
    "verifier": None,
}
CHANGELOG_ROW = {
    "schema_version": "changelog-row/1.0",
    "db": "hoolicalendar",
    "seq": 1,
    "ts": 1.0,
    "tbl": "events",
    "rowid": 530,
    "op": "UPDATE",
    "old_json": "{}",
    "new_json": "{}",
    "action_index": 3,
}
OBSERVATION = {
    "schema_version": "observation-event/1.0",
    "action_index": 2,
    "source": "gui",
    "facts": [FACT],
}
STEP = {
    "action_index": 0,
    "action": {"type": "click"},
    "params": [{"name": "x", "value": 1}],
    "screenshot_sha256": SHA,
    "a11y_sha256": None,
    "delta": [],
    "observations": [],
    "pages": [],
}

EXAMPLES = {
    "task_ir.schema.json": {
        "schema_version": "task-ir/1.0",
        "task_id": "mw-001",
        "world_id": "miniworld",
        "nodes": [NODE],
        "edges": [],
        "provenance": {},
    },
    "gold_lineage.schema.json": {
        "schema_version": "gold-lineage/1.0",
        "task_id": "mw-001",
        "world_id": "miniworld",
        "initial_state_sha256": SHA,
        "node_order": ["n1"],
        "values": [{"node_id": "n1", "name": "start", "type": "Date", "value": "2026-04-07"}],
        "state_timeline": [{"node_id": "n1", "state_sha256": SHA, "changed": False}],
        "writes_gold": [],
        "verifier_results": [],
        "resolved_reads": [],
        "reads_undeclared": [],
        "interpreter_version": "gold-interpreter/1.1",
        "provenance": {},
    },
    "observation_event.schema.json": OBSERVATION,
    "changelog_row.schema.json": CHANGELOG_ROW,
    "rollout_trace.schema.json": {
        "schema_version": "rollout-trace/1.0",
        "rollout_id": "r1",
        "task_id": "mw-001",
        "world_id": "miniworld",
        "agent": "test",
        "seed": 0,
        "sampling_round": 1,
        "step_budget": 100,
        "steps": [STEP],
        "outcome": {
            "final_verifier": False,
            "declared_complete": True,
            "declared_infeasible": False,
            "budget_exhausted": False,
        },
        "modality": {"primary": "gui", "counts": {"gui": 1}},
    },
    "failure_analysis.schema.json": {
        "schema_version": "failure-analysis/1.0",
        "rollout_id": "r1",
        "task_id": "mw-001",
        "agent": "test",
        "detectors": {
            "state": [{"action_index": 3, "evidence": [], "confidence": 0.9}],
            "parameter": [],
            "omission": [],
        },
        "root_cause_action_index": 3,
        "root_cause_detector": "state",
        "paper_type": "wrong_subgoal",
        "paper_category": "planning",
        "group": "long_horizon",
        "type_confidence": 0.9,
        "earliest_identifiable_action_index": 5,
        "action_horizon": 2,
        "semantic_horizon": 1,
        "horizon_censored": False,
        "analysis_path": "auto",
        "residual_code": None,
        "multi_root_cause": False,
        "analysis_version": "v1",
    },
    "prefix_repair.schema.json": {
        "schema_version": "prefix-repair/1.0",
        "rollout_id": "r1",
        "task_id": "mw-001",
        "root_cause_action_index": 3,
        "removed_segments": [
            {
                "start_action_index": 1,
                "end_action_index": 2,
                "net_delta_empty": True,
                "page_before": "/a",
                "page_after": "/a",
            }
        ],
        "non_neutral_segments": [],
        "repaired_prefix_action_indices": [0, 3],
        "status": "repaired",
        "repair_version": "v1",
    },
    "replay_verification.schema.json": {
        "schema_version": "replay-verification/1.0",
        "rollout_id": "r1",
        "task_id": "mw-001",
        "world_id": "miniworld",
        "root_cause_action_index": 3,
        "repaired_prefix_ref": "pr1",
        "required_attempts": 3,
        "depths": [
            {
                "depth": 0,
                "replay_end_action_index": 3,
                "passed": True,
                "code": None,
                "attempts": [
                    {
                        "attempt_index": 1,
                        "changelog_match": True,
                        "page_sequence_match": True,
                        "state_sha256": SHA,
                    }
                ],
            }
        ],
        "execution_mode": "dry_run",
        "snapshot_sha256": SHA,
        "verification_version": "v1",
    },
    "derail_case.schema.json": {
        "schema_version": "derail-case/1.0",
        "case_id": "c1",
        "task_id": "mw-001",
        "source_rollout_id": "r1",
        "source_agent": "test",
        "verification_path": "auto",
        "error_depth": 0,
        "root_cause_action_index": 3,
        "paper_type": "wrong_subgoal",
        "paper_category": "planning",
        "group": "long_horizon",
        "reversibility_stratum": "reversible",
        "high_consequence": False,
        "post_error_steps_available": 4,
        "root_action_signature": "click:events:531",
        "status": "candidate",
        "merged_from": [],
        "provenance": {},
        "world_id": "miniworld",
        "predicted_latent_horizon_semantic": 2,
        "predicted_observability_class": "required_later",
        "measured_semantic_horizon": 1,
        "measured_action_horizon": 2,
        "horizon_censored": False,
        "prefix_modality": "gui",
        "analysis_path": "auto",
    },
    "training_sample.schema.json": {
        "schema_version": "training-sample/1.0",
        "sample_id": "s1",
        "sample_kind": "detection",
        "source": "real_rollout",
        "rollout_id": "r1",
        "task_id": "mw-001",
        "agent": "test",
        "truncation_action_index": 5,
        "input": {
            "instruction": "Move it",
            "history": [
                {"action_index": 0, "action": {"type": "click"}, "observation_ref": "obs:0"}
            ],
            "observation_ref": "obs:5",
        },
        "target": {
            "ledger": [
                {
                    "name": "start",
                    "value": "2026-04-07",
                    "source_action_index": 1,
                    "source_ref": "obs:1",
                }
            ],
            "check": "inconsistent",
            "thought": "The page shows Dwight.",
            "action": {"type": "click"},
        },
        "loss_weights": {"input": 0, "action": 1, "thought": 0.5, "ledger": 0.5, "check": 3.2},
        "hint_level": "none",
        "teacher": "none",
        "modality": "gui",
        "split": "train",
        "cluster_id": "dundies",
    },
    "mutation.schema.json": {
        "schema_version": "mutation/1.0",
        "task_id": "mw-001",
        "node_id": "n1",
        "mutation_type": "date_plus_minus_day",
        "mutated_value_ref": "2026-04-08T11:00:00",
        "latent_horizon_static": 2,
        "latent_horizon_dynamic": None,
        "observability_class": "required_later",
        "static_class": "cross_app",
        "cross_app": True,
        "visible_node_id": "n3",
        "visible_pages": [],
        "contradiction_visible": False,
        "silent": False,
        "verifier_rejects": None,
        "mode": "static",
    },
    "latent_profile.schema.json": {
        "schema_version": "latent-profile/1.0",
        "task_id": "mw-001",
        "mode": "static",
        "node_count": 4,
        "class_counts": {"required_next": 1, "required_later": 2},
        "semantic_horizon_max": 3,
        "semantic_horizon_median": 2.0,
        "cross_app_or_verifier_only_ratio": 0.5,
        "verifier_only_before_first_r3": 0,
        "bucket": "2-3",
        "nodes_in_target_bucket": 2,
        "structural": {
            "dependency_depth": 3,
            "independent_component_ratio": 0.0,
            "decorative_carry_violations": 0,
        },
        "profile_version": "v1",
    },
    "hazard_injection.schema.json": {
        "schema_version": "hazard-injection/1.0",
        "injection_id": "h1",
        "base_world_id": "miniworld",
        "variant_world_id": "miniworld-h1",
        "task_id": "mw-001",
        "hazard_type": "identity_alias",
        "target_edge": {"from_node_id": "n1", "to_node_id": "n2"},
        "target_paper_type": "wrong_subgoal",
        "target_cell": {},
        "persona_diff": [{"op": "add", "path": "/contacts/-", "value": {"name": "Jim Halpert"}}],
        "profile_before": None,
        "profile_after": None,
        "gold_unique": True,
        "seeder_consistent": True,
        "status": "accepted",
        "hazard_provenance": {"base_rate_before": 1, "base_rate_after": 2},
    },
    "osm.schema.json": {
        "schema_version": "osm/1.0",
        "app": "hoolicalendar",
        "routes": [
            {
                "route_pattern": "/event/:id",
                "fields": [{"table": "events", "column": "start_at", "hidden": False}],
            }
        ],
        "spot_check": {"status": "pending", "checked_routes": []},
        "built_from": {},
    },
}

BREAKERS = {
    "task_ir.schema.json": lambda r: r["nodes"][0].__setitem__("op", "browse"),
    "gold_lineage.schema.json": lambda r: r.__setitem__("initial_state_sha256", "abc"),
    "observation_event.schema.json": lambda r: r.__setitem__("source", "screen"),
    "changelog_row.schema.json": lambda r: r.__setitem__("op", "UPSERT"),
    "rollout_trace.schema.json": lambda r: r["steps"][0].pop("delta"),
    "failure_analysis.schema.json": lambda r: r.__setitem__("analysis_path", "guessed"),
    "prefix_repair.schema.json": lambda r: r.__setitem__("status", "fixed_by_hand"),
    "replay_verification.schema.json": lambda r: r["depths"][0].pop("attempts"),
    "derail_case.schema.json": lambda r: r.pop("prefix_modality"),
    "training_sample.schema.json": lambda r: r["target"].__setitem__("check", "maybe"),
    "mutation.schema.json": lambda r: r.__setitem__("mutation_type", "random"),
    "latent_profile.schema.json": lambda r: r.__setitem__("cross_app_or_verifier_only_ratio", 2),
    "hazard_injection.schema.json": lambda r: r.__setitem__("hazard_type", "trap"),
    "osm.schema.json": lambda r: r["routes"][0]["fields"][0].pop("hidden"),
}


class V1SchemaTests(unittest.TestCase):
    def test_every_new_schema_has_an_example_and_a_breaker(self):
        self.assertEqual(set(EXAMPLES), set(BREAKERS))

    def test_examples_validate(self):
        for name, record in EXAMPLES.items():
            with self.subTest(schema=name):
                validate_schema(copy.deepcopy(record), name, REPOSITORY)

    def test_broken_examples_are_rejected(self):
        for name, record in EXAMPLES.items():
            with self.subTest(schema=name):
                broken = copy.deepcopy(record)
                BREAKERS[name](broken)
                with self.assertRaises(jsonschema.ValidationError):
                    validate_schema(broken, name, REPOSITORY)

    def test_derail_case_0_1_records_still_validate(self):
        record = {
            k: v
            for k, v in EXAMPLES["derail_case.schema.json"].items()
            if k
            not in {
                "world_id",
                "predicted_latent_horizon_semantic",
                "predicted_observability_class",
                "measured_semantic_horizon",
                "measured_action_horizon",
                "horizon_censored",
                "prefix_modality",
                "analysis_path",
            }
        }
        record["schema_version"] = "derail-case/0.1"
        validate_schema(record, "derail_case.schema.json", REPOSITORY)

    def test_derail_case_1_0_requires_the_new_fields(self):
        record = copy.deepcopy(EXAMPLES["derail_case.schema.json"])
        record.pop("measured_action_horizon")
        with self.assertRaises(jsonschema.ValidationError) as ctx:
            validate_schema(record, "derail_case.schema.json", REPOSITORY)
        self.assertIn("branch v1_0", str(ctx.exception))
        self.assertIn("measured_action_horizon", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
