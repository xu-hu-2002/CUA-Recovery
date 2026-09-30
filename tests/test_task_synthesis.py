"""Tests for verifier-first long-horizon task synthesis."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import yaml

from derail.derived.schema import validate_schema
from derail.synthesis.compatibility import TypeSystem, build_compatibility_edges
from derail.synthesis.graph import SynthesisValidationError, compute_complexity
from derail.synthesis.pipeline import SynthesisConfig, SynthesisPipeline
from derail.synthesis.sampling import compute_skeleton_weights
from derail.synthesis.skeletons import (
    canonical_structural_signature,
    extract_observed_skeletons,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _port(port_id: str, value_type: str, grounding: str, source: str = "world"):
    return {
        "port_id": port_id,
        "type": value_type,
        "grounding": grounding,
        "source": source,
        "cardinality": "one",
    }


def _module(
    module_id: str,
    task_id: str,
    op: str,
    app: str,
    inputs,
    outputs,
    *,
    identity="person:michael",
):
    node_id = "n1"
    public_inputs = [
        {
            "node_id": node_id,
            "port_id": port["port_id"],
            "type": port["type"],
            "grounding": port["grounding"],
            "cardinality": port["cardinality"],
            "required": True,
            "dependency_kind": "data_dependency",
        }
        for port in inputs
    ]
    public_outputs = [
        {
            "node_id": node_id,
            "port_id": port["port_id"],
            "type": port["type"],
            "grounding": port["grounding"],
            "cardinality": port["cardinality"],
        }
        for port in outputs
    ]
    return {
        "schema_version": "grounded-task-module/0.1",
        "module_id": module_id,
        "source_task_id": task_id,
        "source_split": "source-development",
        "environment_id": "mypcbench-michael-scott",
        "snapshot_id": "mypcbench-base-v1",
        "world_schema_version": "mypcbench-world/0.1",
        "world_subgraph_id": "world-%s" % module_id,
        "instruction": "source instruction for %s" % task_id,
        "fragment": {
            "schema_version": "task-fragment/0.1",
            "fragment_id": "fragment-%s" % module_id,
            "nodes": [
                {
                    "node_id": node_id,
                    "op": op,
                    "app": app,
                    "semantic_goal": "%s a grounded value" % op,
                    "inputs": inputs,
                    "outputs": outputs,
                    "side_effects": [],
                    "verifier": {
                        "verifier_id": "verify-%s" % module_id,
                        "observability": "environment_state",
                    },
                    "critical": True,
                }
            ],
            "edges": [],
        },
        "interface": {
            "produces": public_outputs,
            "accepts": public_inputs,
            "identity_scope": [identity],
            "time_scope": {"timezone": "America/Chicago"},
            "reads": [port["grounding"] for port in inputs],
            "side_effects": [],
        },
        "provenance": {
            "source_uri": "fixture://%s" % task_id,
            "source_sha256": "a" * 64,
            "review_status": "human_verified",
        },
    }


def _fixtures():
    owner = _port("owner", "PersonRef", "person:michael")
    date = _port("date", "Date", "calendar:event:1.date")
    option = _port("option", "TravelOptionRef", "travel:derived:selected")
    selected = _port("selected", "TravelOptionRef", "report:derived:selected")
    modules = [
        _module("module-a", "task-a", "retrieve", "calendar", [owner], [date]),
        _module(
            "module-b",
            "task-b",
            "filter",
            "travel",
            [_port("query_date", "Date", "travel:query.date")],
            [option],
        ),
        _module(
            "module-c",
            "task-c",
            "compare",
            "calc",
            [
                _port("constraint_date", "Date", "calc:input.date"),
                _port("candidate", "TravelOptionRef", "calc:input.option"),
            ],
            [selected],
        ),
    ]
    skeleton = {
        "schema_version": "skeleton/0.1",
        "skeleton_id": "skeleton-retrieve-filter-compare",
        "kind": "observed",
        "motif_family": "retrieve-use-decide",
        "source_task_ids": ["task-a", "task-b", "task-c"],
        "support_task_count": 3,
        "empirical_probability": 1.0,
        "complexity_prior": 1.0,
        "anchor_ops": ["retrieve"],
        "composition_rules": [
            {
                "rule_id": "r-date-filter",
                "producer_op": "retrieve",
                "consumer_op": "filter",
                "value_type": "Date",
                "support_task_count": 3,
                "approved": False,
                "probability": 0.5,
            },
            {
                "rule_id": "r-date-compare",
                "producer_op": "retrieve",
                "consumer_op": "compare",
                "value_type": "Date",
                "support_task_count": 3,
                "approved": False,
                "probability": 0.4,
            },
            {
                "rule_id": "r-option-compare",
                "producer_op": "filter",
                "consumer_op": "compare",
                "value_type": "TravelOptionRef",
                "support_task_count": 3,
                "approved": False,
                "probability": 0.3,
            },
        ],
        "review_status": "human_approved",
    }
    config = {
        "schema_version": "synthesis-config/0.1",
        "generation_version": "test-v0.1",
        "target_count": 1,
        "target_environment": "mypcbench-michael-scott",
        "target_snapshot": "mypcbench-base-v1",
        "source_split": "source-development",
        "sampling_mode": "empirical_only",
        "seed": 7,
        "max_attempts": 10,
        "beam_width": 16,
        "min_modules": 3,
        "max_modules": 3,
        "dependency_depth": [3, 3],
        "cross_app_dependencies": [3, 3],
        "max_independent_component_ratio": 0.0,
        "required_delayed_reuse": True,
        "max_irreversible_actions": 0,
        "min_verifier_coverage": 1.0,
        "failure_reweighting": {
            "alpha": 1.0,
            "beta": 1.0,
            "min_exposures": 10,
            "adjustment_range": [0.5, 2.0],
        },
        "type_system": {"parents": {}, "converters": []},
    }
    return modules, skeleton, config


class CompatibilityTests(unittest.TestCase):
    def test_identity_conflict_is_audited(self) -> None:
        modules, _, _ = _fixtures()
        conflicting = copy.deepcopy(modules[1])
        conflicting["interface"]["identity_scope"] = ["person:dwight"]
        edges = build_compatibility_edges([modules[0], conflicting], TypeSystem())
        self.assertEqual(len(edges), 1)
        self.assertIn("identity_scope", edges[0]["conflicts"])

    def test_type_hierarchy_supports_subtyping(self) -> None:
        type_system = TypeSystem(parents={"AirportCity": ("CityRef",)})
        unified = type_system.unify("AirportCity", "CityRef")
        self.assertEqual(unified, ("CityRef", None, 0.97))


class SynthesisPipelineTests(unittest.TestCase):
    def test_pipeline_builds_dependency_carry_candidate(self) -> None:
        modules, skeleton, config_raw = _fixtures()
        for module in modules:
            validate_schema(module, "grounded_task_module.schema.json", REPOSITORY)
        validate_schema(skeleton, "empirical_skeleton.schema.json", REPOSITORY)
        validate_schema(config_raw, "synthesis_config.schema.json", REPOSITORY)
        result = SynthesisPipeline(
            modules, [skeleton], SynthesisConfig.from_dict(config_raw)
        ).run()
        self.assertEqual(len(result.accepted), 1)
        candidate = result.accepted[0]
        self.assertEqual(candidate["complexity"]["dependency_depth"], 3)
        self.assertEqual(candidate["complexity"]["cross_app_dependency_count"], 3)
        self.assertEqual(candidate["complexity"]["max_information_carry_distance"], 2)
        self.assertEqual(candidate["complexity"]["max_fan_in"], 2)
        self.assertIsNone(candidate["instruction"])
        self.assertFalse(candidate["release_eligible"])
        validate_schema(candidate, "generation_record.schema.json", REPOSITORY)

    def test_cycle_is_rejected_by_complexity_validation(self) -> None:
        modules, skeleton, config_raw = _fixtures()
        pipeline = SynthesisPipeline(
            modules, [skeleton], SynthesisConfig.from_dict(config_raw)
        )
        result = pipeline.run()
        fragment = copy.deepcopy(result.accepted[0]["task_ir"])
        fragment["edges"].append(
            {
                "edge_id": "cycle",
                "from": {"node_id": "module-c::n1", "port_id": "selected"},
                "to": {"node_id": "module-a::n1", "port_id": "owner"},
                "kind": "data_dependency",
            }
        )
        with self.assertRaisesRegex(SynthesisValidationError, "not a DAG"):
            compute_complexity(fragment)

    def test_canonical_signature_ignores_node_ids_and_apps(self) -> None:
        modules, skeleton, config_raw = _fixtures()
        result = SynthesisPipeline(
            modules, [skeleton], SynthesisConfig.from_dict(config_raw)
        ).run()
        fragment = result.accepted[0]["task_ir"]
        renamed = copy.deepcopy(fragment)
        mapping = {
            node["node_id"]: "renamed-%d" % index
            for index, node in enumerate(reversed(renamed["nodes"]), start=1)
        }
        renamed["nodes"].reverse()
        for node in renamed["nodes"]:
            node["node_id"] = mapping[node["node_id"]]
            node["app"] = "different-app-name"
            for port in node["inputs"]:
                if port.get("bound_from"):
                    port["bound_from"]["node_id"] = mapping[
                        port["bound_from"]["node_id"]
                    ]
        for edge in renamed["edges"]:
            edge["edge_id"] = "renamed-%s" % edge["edge_id"]
            edge["from"]["node_id"] = mapping[edge["from"]["node_id"]]
            edge["to"]["node_id"] = mapping[edge["to"]["node_id"]]
        self.assertEqual(
            canonical_structural_signature(fragment)["sha256"],
            canonical_structural_signature(renamed)["sha256"],
        )

    def test_observed_skeletons_are_review_gated(self) -> None:
        modules, skeleton, config_raw = _fixtures()
        candidate = SynthesisPipeline(
            modules, [skeleton], SynthesisConfig.from_dict(config_raw)
        ).run().accepted[0]
        fragment = candidate["task_ir"]
        observed_module = {
            "schema_version": "grounded-task-module/0.1",
            "module_id": "observed-composite",
            "source_task_id": "observed-task",
            "source_split": "source-development",
            "environment_id": "mypcbench-michael-scott",
            "snapshot_id": "mypcbench-base-v1",
            "world_schema_version": "mypcbench-world/0.1",
            "world_subgraph_id": "world-observed",
            "fragment": fragment,
            "interface": {
                "produces": [
                    {
                        "node_id": "module-c::n1",
                        "port_id": "selected",
                        "type": "TravelOptionRef",
                        "grounding": "report:derived:selected",
                        "cardinality": "one",
                    }
                ],
                "accepts": [],
                "identity_scope": ["person:michael"],
                "time_scope": {"timezone": "America/Chicago"},
                "reads": [],
                "side_effects": [],
            },
            "provenance": {
                "source_uri": "fixture://observed-task",
                "source_sha256": "b" * 64,
                "review_status": "human_verified",
            },
        }
        extracted = extract_observed_skeletons([observed_module])
        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0]["review_status"], "needs_review")
        self.assertEqual(len(extracted[0]["composition_rules"]), 3)
        validate_schema(extracted[0], "empirical_skeleton.schema.json", REPOSITORY)

    def test_failure_weights_require_human_adjudication(self) -> None:
        _, skeleton, _ = _fixtures()
        with self.assertRaisesRegex(ValueError, "human-adjudicated"):
            compute_skeleton_weights(
                [skeleton],
                [
                    {
                        "structure_id": skeleton["skeleton_id"],
                        "family": skeleton["motif_family"],
                        "model_id": "agent-a",
                        "exposures": 20,
                        "attributed_failures": 10,
                        "review_status": "llm_only",
                    }
                ],
                enabled=True,
            )


class SynthesisCliTests(unittest.TestCase):
    def test_cli_writes_auditable_stage_outputs(self) -> None:
        modules, skeleton, config = _fixtures()
        script_path = REPOSITORY / "scripts" / "synthesis" / "synthesize_long_horizon_tasks.py"
        spec = importlib.util.spec_from_file_location("derail_task_synthesis_cli", script_path)
        script = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(script)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            modules_path = root / "modules.jsonl"
            skeletons_path = root / "skeletons.jsonl"
            config_path = root / "config.yaml"
            output_dir = root / "out"
            modules_path.write_text(
                "".join(json.dumps(module) + "\n" for module in modules), encoding="utf-8"
            )
            skeletons_path.write_text(json.dumps(skeleton) + "\n", encoding="utf-8")
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                status = script.main(
                    [
                        "--modules",
                        str(modules_path),
                        "--skeletons",
                        str(skeletons_path),
                        "--config",
                        str(config_path),
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            self.assertEqual(status, 0)
            manifest = json.loads((output_dir / "stage_manifest.json").read_text())
            self.assertEqual(manifest["status"], "symbolic_candidates_ready")
            self.assertEqual(manifest["counts"]["accepted_symbolic_candidates"], 1)
            self.assertFalse(manifest["release_eligible"])
            self.assertTrue((output_dir / "compatibility_edges.jsonl").is_file())
            with self.assertRaisesRegex(RuntimeError, "non-empty"):
                with redirect_stdout(io.StringIO()):
                    script.main(
                        [
                            "--modules",
                            str(modules_path),
                            "--skeletons",
                            str(skeletons_path),
                            "--config",
                            str(config_path),
                            "--output-dir",
                            str(output_dir),
                        ]
                    )


if __name__ == "__main__":
    unittest.main()
