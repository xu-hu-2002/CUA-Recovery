"""Tests for derail.longhorizon: ontology, effects, carry, complexity, gates, sampling."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from derail.derived.schema import validate_schema
from derail.longhorizon.carry import carry_distances, decorative_carry_violations
from derail.longhorizon.complexity import compute_complexity_v02
from derail.longhorizon.dag import DagIndex
from derail.longhorizon.effects import (
    EffectError,
    effect_safety_violations,
    high_consequence_prerequisite_depth,
    irreversible_action_depth,
    reversibility_profile,
    validate_node_reversibility,
    validate_side_effect,
)
from derail.longhorizon.gates import GateConfig, run_hard_gates, verifier_coverage
from derail.longhorizon.ontology import Ontology, OntologyError
from derail.longhorizon.pipeline import apply_v02_gates
from derail.longhorizon.reversibility_sampling import (
    ReversibilitySamplingConfig,
    allocate_targets,
    empirical_class_distribution,
    stratified_pick,
)
from derail.longhorizon.taxonomy import FailureTaxonomy, TaxonomyError

REPOSITORY = Path(__file__).resolve().parents[1]
ONTOLOGY = Ontology.from_yaml(REPOSITORY / "configs/synthesis/ontology_v0.2.yaml")
TAXONOMY = FailureTaxonomy.from_yaml(REPOSITORY / "configs/synthesis/failure_taxonomy_v0.1.yaml")


def _effect(effect_id="fx1", effect_type="send_message", reversibility="R3", **overrides):
    effect = {
        "effect_id": effect_id,
        "effect_type": effect_type,
        "target_ref": "sandbox_mail:thread:42",
        "reversibility_class": reversibility,
        "commit_scope": "external_sandbox",
        "compensation_available": False,
        "compensating_effect_type": None,
        "checkpoint_required": True,
        "checkpoint_id": "ckpt_1",
        "restore_verifier_id": "verify_restore_1",
    }
    effect.update(overrides)
    return effect


def _node(node_id, op, app, inputs=(), outputs=(), effects=(), reversibility="R0", critical=True):
    return {
        "node_id": node_id,
        "op": op,
        "app": app,
        "inputs": [
            {"port_id": port, "type": "Value", "grounding": "world:%s" % port, "source": "upstream"}
            for port in inputs
        ],
        "outputs": [
            {"port_id": port, "type": "Value", "grounding": "derived:%s:%s" % (node_id, port)}
            for port in outputs
        ],
        "side_effects": list(effects),
        "reversibility_class": reversibility,
        "verifier": {"verifier_id": "verify_%s" % node_id, "observability": "environment_state"},
        "critical": critical,
    }


def _edge(edge_id, source, source_port, target, target_port, kind="data_dependency"):
    return {
        "edge_id": edge_id,
        "from": {"node_id": source, "port_id": source_port},
        "to": {"node_id": target, "port_id": target_port},
        "kind": kind,
    }


def _chain_fragment(with_second_lineage: bool):
    decide_inputs = ("total", "date_copy") + (("budget",) if with_second_lineage else ())
    nodes = [
        _node("r1", "retrieve", "calendar", outputs=("date", "date_copy")),
        _node("f2", "filter", "travel", inputs=("date",), outputs=("options",)),
        _node("a3", "aggregate", "travel", inputs=("options",), outputs=("total",)),
        _node("d4", "decide", "travel", inputs=decide_inputs, outputs=("choice",)),
        _node(
            "w5",
            "communicate",
            "mail",
            inputs=("choice",),
            outputs=("sent",),
            effects=(_effect(),),
            reversibility="R3",
        ),
    ]
    edges = [
        _edge("e1", "r1", "date", "f2", "date"),
        _edge("e2", "f2", "options", "a3", "options"),
        _edge("e3", "a3", "total", "d4", "total"),
        _edge("e4", "d4", "choice", "w5", "choice"),
        _edge("e5", "r1", "date_copy", "d4", "date_copy"),
    ]
    if with_second_lineage:
        nodes += [
            _node("r6", "retrieve", "bank", outputs=("balance",)),
            _node("c7", "compare", "bank", inputs=("balance",), outputs=("budget",)),
        ]
        edges += [
            _edge("e6", "r6", "balance", "c7", "balance"),
            _edge("e7", "c7", "budget", "d4", "budget"),
        ]
    return {
        "schema_version": "task-fragment/0.1",
        "fragment_id": "chain",
        "nodes": nodes,
        "edges": edges,
    }


class OntologyTests(unittest.TestCase):
    def test_loads_and_orders_classes(self):
        self.assertEqual(ONTOLOGY.read_only_class, "R0")
        self.assertEqual(ONTOLOGY.max_class(("R1", "R3", "R2")), "R3")
        self.assertEqual(ONTOLOGY.default_class_for("send_message"), "R3")

    def test_rejects_operation_drift(self):
        import yaml

        raw = yaml.safe_load((REPOSITORY / "configs/synthesis/ontology_v0.2.yaml").read_text())
        raw["operations"].append("teleport")
        with self.assertRaises(OntologyError):
            Ontology.from_dict(raw)


class TaxonomyTests(unittest.TestCase):
    def test_group_is_derived_from_paper_type(self):
        self.assertEqual(TAXONOMY.group_of("grounding_failure"), "local")
        self.assertEqual(TAXONOMY.group_of("progress_misperception"), "long_horizon")
        self.assertEqual(TAXONOMY.group_of("hit_budget_limit"), "termination")

    def test_normalize_renames_and_drops(self):
        kept, dropped = TAXONOMY.normalize(["wrong_subgoal", "gui_workflow_bypass", "scope_error"])
        self.assertEqual(kept, ("misunderstand_task_objective", "scope_error"))
        self.assertEqual(dropped, ("gui_workflow_bypass",))
        with self.assertRaises(TaxonomyError):
            TAXONOMY.normalize(["made_up_label"])

    def test_primary_type_follows_configured_priority(self):
        self.assertEqual(
            TAXONOMY.primary_paper_type(["detail_misperception", "premature_completion"]),
            "premature_completion",
        )
        self.assertEqual(
            TAXONOMY.primary_paper_type(["detail_misperception", "grounding_failure"]),
            "grounding_failure",
        )
        self.assertEqual(
            TAXONOMY.primary_paper_type(
                ["premature_completion", "detail_misperception", "scope_error"]
            ),
            "scope_error",
        )
        self.assertEqual(
            TAXONOMY.primary_paper_type(["ineffective_action", "grounding_failure"]),
            "grounding_failure",
        )
        self.assertIsNone(TAXONOMY.primary_paper_type([]))


class EffectTests(unittest.TestCase):
    def test_valid_effect_passes_schema_and_validator(self):
        effect = _effect()
        validate_schema(effect, "side_effect.schema.json", REPOSITORY)
        validate_side_effect(effect, ONTOLOGY)

    def test_compensation_requires_type(self):
        with self.assertRaises(EffectError):
            validate_side_effect(_effect(compensation_available=True), ONTOLOGY)
        validate_side_effect(
            _effect(compensation_available=True, compensating_effect_type="cancel_order"), ONTOLOGY
        )

    def test_node_class_must_match_effects(self):
        node = _node("w", "communicate", "mail", effects=(_effect(),), reversibility="R1")
        with self.assertRaises(EffectError):
            validate_node_reversibility(node, ONTOLOGY)
        read_only = _node("r", "retrieve", "mail", effects=(_effect(),), reversibility="R3")
        with self.assertRaises(EffectError):
            validate_node_reversibility(read_only, ONTOLOGY)

    def test_safety_violations(self):
        node = _node(
            "w",
            "communicate",
            "mail",
            effects=(_effect(checkpoint_id=None, restore_verifier_id=None),),
            reversibility="R3",
        )
        reasons = [item["reason"] for item in effect_safety_violations(node, ONTOLOGY)]
        self.assertEqual(len(reasons), 1)
        self.assertIn("checkpoint_id", reasons[0])
        unflagged = _node(
            "w",
            "communicate",
            "mail",
            effects=(_effect(checkpoint_required=False),),
            reversibility="R3",
        )
        self.assertIn(
            "checkpoint_required", effect_safety_violations(unflagged, ONTOLOGY)[0]["reason"]
        )

    def test_depth_metrics(self):
        fragment = _chain_fragment(True)
        dag = DagIndex.from_fragment(fragment)
        self.assertEqual(irreversible_action_depth(dag, ONTOLOGY), 3)
        self.assertEqual(
            high_consequence_prerequisite_depth(dag, ONTOLOGY), 5
        )
        self.assertEqual(
            reversibility_profile(fragment, ONTOLOGY), {"R0": 6, "R1": 0, "R2": 0, "R3": 1}
        )


class CarryTests(unittest.TestCase):
    def test_decorative_carry_without_second_lineage(self):
        dag = DagIndex.from_fragment(_chain_fragment(False))
        self.assertEqual(carry_distances(dag)["e5"], 3)
        violations = decorative_carry_violations(dag, threshold=3)
        self.assertEqual([item["edge_id"] for item in violations], ["e5"])

    def test_second_lineage_justifies_carry(self):
        dag = DagIndex.from_fragment(_chain_fragment(True))
        self.assertEqual(decorative_carry_violations(dag, threshold=3), [])

    def test_shallow_second_input_does_not_justify(self):
        fragment = _chain_fragment(True)
        fragment["nodes"] = [node for node in fragment["nodes"] if node["node_id"] != "c7"]
        fragment["edges"] = [
            edge for edge in fragment["edges"] if edge["edge_id"] not in {"e6", "e7"}
        ]
        fragment["nodes"][-1]["outputs"] = [
            {"port_id": "budget", "type": "Value", "grounding": "derived:r6:budget"}
        ]
        fragment["edges"].append(_edge("e8", "r6", "budget", "d4", "budget"))
        dag = DagIndex.from_fragment(fragment)
        self.assertEqual(len(decorative_carry_violations(dag, threshold=3)), 1)


class ComplexityAndGateTests(unittest.TestCase):
    def _config(self, **overrides):
        raw = {
            "dependency_depth": [3, 9],
            "cross_app_dependencies": [1, 5],
            "max_independent_component_ratio": 0.25,
            "required_delayed_reuse": True,
            "carry_threshold": 3,
            "max_irreversible_actions": 1,
            "allowed_irreversible_actions": ["send_message"],
            "min_verifier_coverage": 1.0,
        }
        raw.update(overrides)
        return GateConfig.from_dict(raw)

    def test_metrics_and_gates_pass_for_justified_carry(self):
        fragment = _chain_fragment(True)
        metrics = compute_complexity_v02(fragment, ONTOLOGY)
        self.assertEqual(metrics["dependency_depth"], 5)
        self.assertEqual(metrics["irreversible_action_count"], 1)
        self.assertEqual(metrics["decorative_carry_violations"], [])
        self.assertEqual(run_hard_gates(fragment, metrics, self._config(), ONTOLOGY), [])

    def test_gates_reject_decorative_carry_and_disallowed_effect(self):
        fragment = _chain_fragment(False)
        metrics = compute_complexity_v02(fragment, ONTOLOGY)
        codes = [item.code for item in run_hard_gates(fragment, metrics, self._config(), ONTOLOGY)]
        self.assertIn("DECORATIVE_CARRY", codes)
        restricted = self._config(allowed_irreversible_actions=["create_event"])
        codes = [
            item.code
            for item in run_hard_gates(
                _chain_fragment(True),
                compute_complexity_v02(_chain_fragment(True), ONTOLOGY),
                restricted,
                ONTOLOGY,
            )
        ]
        self.assertEqual(codes, ["GLOBAL_CONSTRAINT_UNSAT"])

    def test_verifier_coverage_ignores_unobservable(self):
        fragment = _chain_fragment(True)
        fragment["nodes"][0]["verifier"]["observability"] = "unobservable"
        self.assertLess(verifier_coverage(fragment), 1.0)

    def test_apply_v02_gates_splits_candidates(self):
        good = {
            "schema_version": "generation-record/0.1",
            "candidate_id": "gen_good",
            "status": "symbolic_accepted",
            "anchor_module_id": None,
            "source_skeleton_ids": ["sk"],
            "task_ir": _chain_fragment(True),
            "validation": {},
            "provenance": {"candidate_fingerprint": "a" * 64, "llm_calls": []},
        }
        bad = copy.deepcopy(good)
        bad["candidate_id"] = "gen_bad"
        bad["task_ir"] = _chain_fragment(False)
        legacy = copy.deepcopy(good)
        legacy["candidate_id"] = "gen_legacy"
        for node in legacy["task_ir"]["nodes"]:
            node.pop("reversibility_class")
        result = apply_v02_gates([good, bad, legacy], gate_config=self._config(), ontology=ONTOLOGY)
        self.assertEqual([item["candidate_id"] for item in result.accepted], ["gen_good"])
        self.assertEqual(result.accepted[0]["reversibility_class"], "R3")
        codes = {
            item["provenance"]["source_candidate_id"]: item["rejection_code"]
            for item in result.rejected
        }
        self.assertEqual(codes["gen_bad"], "DECORATIVE_CARRY")
        self.assertEqual(codes["gen_legacy"], "INVALID_GROUNDING")
        for record in result.rejected:
            validate_schema(record, "generation_record.schema.json", REPOSITORY)


class ReversibilitySamplingTests(unittest.TestCase):
    def _modules(self):
        return [
            {"module_id": "m%d" % index, "fragment": _chain_fragment(True)} for index in range(3)
        ] + [
            {
                "module_id": "r%d" % index,
                "fragment": {
                    "nodes": [_node("r", "retrieve", "mail", outputs=("x",))],
                    "edges": [],
                },
            }
            for index in range(7)
        ]

    def test_distribution_and_allocation(self):
        distribution = empirical_class_distribution(self._modules(), ONTOLOGY)
        self.assertEqual(distribution, {"R0": 7, "R1": 0, "R2": 0, "R3": 3})
        config = ReversibilitySamplingConfig(
            min_pilot_per_available_class=2, allowed_irreversible_effect_types=("send_message",)
        )
        allocation = allocate_targets(20, distribution, config, ONTOLOGY)
        self.assertEqual(allocation.targets, {"R0": 14, "R3": 6})
        gaps = {gap["class"]: gap["reason"] for gap in allocation.coverage_gaps}
        self.assertEqual(gaps, {"R1": "absent_in_source", "R2": "absent_in_source"})

    def test_missing_sandbox_support_is_a_gap_not_an_invention(self):
        distribution = {"R0": 7, "R3": 3}
        config = ReversibilitySamplingConfig(classes=("R0", "R3"), min_pilot_per_available_class=1)
        allocation = allocate_targets(10, distribution, config, ONTOLOGY)
        self.assertEqual(allocation.targets, {"R0": 10})
        self.assertEqual(allocation.coverage_gaps[0]["reason"], "no_sandbox_support")

    def test_pilot_floor_and_deterministic_pick(self):
        distribution = {"R0": 95, "R1": 5}
        config = ReversibilitySamplingConfig(classes=("R0", "R1"), min_pilot_per_available_class=10)
        allocation = allocate_targets(40, distribution, config, ONTOLOGY)
        self.assertEqual(allocation.targets, {"R0": 30, "R1": 10})
        pools = {"R0": list(range(100)), "R1": list(range(5))}
        first = stratified_pick(pools, allocation, seed=7)
        second = stratified_pick(pools, allocation, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first["R1"]), 5)


if __name__ == "__main__":
    unittest.main()


class DepthGridConsistencyTests(unittest.TestCase):
    def test_synthesis_and_precheck_grids_match_benchmark_config(self):
        import yaml

        from derail.derived.layout import DEPTH_GRID

        benchmark = yaml.safe_load((REPOSITORY / "configs/benchmark/derail_v1.yaml").read_text())
        for path in (
            "configs/synthesis/mypcbench_v0.2.yaml",
            "configs/synthesis/precheck_v0.2.yaml",
        ):
            raw = yaml.safe_load((REPOSITORY / path).read_text())
            self.assertEqual(list(raw["depth_grid"]), list(benchmark["depths"]), path)
        self.assertEqual(list(DEPTH_GRID), list(benchmark["depths"]), "derail.derived.layout")
