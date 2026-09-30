"""Tests for splits, world graph, milestone rubric, failure annotations and IR extraction."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from derail.derived.schema import validate_schema
from derail.longhorizon.annotations import build_failure_annotation
from derail.longhorizon.continuation import compute_continuation
from derail.longhorizon.extraction import (
    ApprovalRequired,
    ExtractionParseError,
    ExtractorConfig,
    LLMResponse,
    OpenAICompatibleClient,
    parse_extraction_response,
    render_prompt,
    run_extraction,
    to_grounded_module,
)
from derail.longhorizon.ontology import Ontology
from derail.longhorizon.rubric import (
    derive_milestones,
    earliest_identifiable_hint,
    score_milestones,
)
from derail.longhorizon.splits import (
    SPLIT_NAMES,
    SplitConfig,
    assign_splits,
    build_split_record,
    cluster_key,
    enrich_tasks,
    select_dev_sample,
)
from derail.longhorizon.taxonomy import FailureTaxonomy
from derail.longhorizon.world import AppAliases, WorldError, build_world_graph, subgraph_for_apps

REPOSITORY = Path(__file__).resolve().parents[1]
ONTOLOGY = Ontology.from_yaml(REPOSITORY / "configs/synthesis/ontology_v0.2.yaml")
TAXONOMY = FailureTaxonomy.from_yaml(REPOSITORY / "configs/synthesis/failure_taxonomy_v0.1.yaml")


def _port(port_id, grounding, source="upstream"):
    return {"port_id": port_id, "type": "Value", "grounding": grounding, "source": source}


def _fragment():
    return {
        "schema_version": "task-fragment/0.1",
        "fragment_id": "tf_test",
        "nodes": [
            {
                "node_id": "r1",
                "op": "retrieve",
                "app": "hoolicalendar",
                "inputs": [_port("owner", "person:michael_scott", "world")],
                "outputs": [_port("date", "derived:r1:date")],
                "side_effects": [],
                "reversibility_class": "R0",
                "critical": True,
                "verifier": {"verifier_id": "v1", "observability": "environment_state"},
            },
            {
                "node_id": "f2",
                "op": "filter",
                "app": "dinoco_airlines",
                "inputs": [_port("date", "derived:r1:date")],
                "outputs": [_port("flight", "derived:f2:flight")],
                "side_effects": [],
                "reversibility_class": "R0",
                "critical": True,
                "verifier": {"verifier_id": "v2", "observability": "derived_from_observable"},
            },
            {
                "node_id": "w3",
                "op": "communicate",
                "app": "hoolimail",
                "inputs": [_port("flight", "derived:f2:flight")],
                "outputs": [_port("sent", "derived:w3:sent")],
                "side_effects": [
                    {
                        "effect_id": "fx1",
                        "effect_type": "send_message",
                        "target_ref": "hoolimail:draft",
                        "reversibility_class": "R3",
                        "commit_scope": "external_sandbox",
                        "compensation_available": False,
                        "compensating_effect_type": None,
                        "checkpoint_required": True,
                        "checkpoint_id": "c1",
                        "restore_verifier_id": "rv1",
                    }
                ],
                "reversibility_class": "R3",
                "critical": True,
                "verifier": {"verifier_id": "v3", "observability": "environment_state"},
            },
        ],
        "edges": [
            {
                "edge_id": "e1",
                "from": {"node_id": "r1", "port_id": "date"},
                "to": {"node_id": "f2", "port_id": "date"},
                "kind": "data_dependency",
            },
            {
                "edge_id": "e2",
                "from": {"node_id": "f2", "port_id": "flight"},
                "to": {"node_id": "w3", "port_id": "flight"},
                "kind": "data_dependency",
            },
        ],
    }


class SplitTests(unittest.TestCase):
    def _tasks(self):
        tasks = []
        for index in range(60):
            apps = (
                ["HooliMail", "Gringotts"]
                if index % 3 == 0
                else ["Files"]
                if index % 3 == 1
                else ["BatBucks", "LibreOffice"]
            )
            tasks.append(
                {
                    "id": "task-%03d" % index,
                    "category": "aggregation" if index % 2 else "retrieval",
                    "apps_involved": apps,
                }
            )
        return tasks

    def test_clusters_never_straddle_and_splits_are_deterministic(self):
        tasks = self._tasks()
        config = SplitConfig(seed=3)
        assignment = assign_splits(tasks, config)
        self.assertEqual(assignment, assign_splits(tasks, config))
        by_cluster = {}
        for task in tasks:
            by_cluster.setdefault(cluster_key(task, config.cluster_fields), set()).add(
                assignment[task["id"]]
            )
        self.assertTrue(all(len(splits) == 1 for splits in by_cluster.values()))
        self.assertEqual(set(assignment.values()) <= set(SPLIT_NAMES), True)

    def test_dev_sample_only_from_development_and_stratified(self):
        tasks = enrich_tasks(
            self._tasks(),
            task_types={},
            task_horizons={},
            failure_task_ids={"task-000", "task-003"},
        )
        config = SplitConfig(seed=3, dev_sample_size=4)
        assignment = assign_splits(tasks, config)
        sample = select_dev_sample(tasks, assignment, config)
        self.assertEqual(len(sample), 4)
        self.assertTrue(all(assignment[task_id] == "source-development" for task_id in sample))
        record = build_split_record(tasks, config, source_sha256="a" * 64)
        self.assertEqual(record["dev_sample"]["task_ids"], sample)
        self.assertEqual(sum(record["split_counts"].values()), 60)


class WorldTests(unittest.TestCase):
    def _sources(self):
        return {
            "environment_id": "env",
            "snapshot_id": "snap",
            "world_schema_version": "w/0.1",
            "app_aliases": {
                "hoolimail": ["HooliMail", "mail"],
                "gringotts": ["Gringotts", "vaultbank"],
                "hoolicalendar": ["HooliCalendar"],
            },
            "persona": {
                "root_entity": {
                    "path": "identity",
                    "entity_type": "Person",
                    "id_fields": ["name"],
                    "surfaces": ["hoolimail"],
                },
                "sections": [
                    {
                        "path": "contacts",
                        "entity_type": "Person",
                        "id_fields": ["name"],
                        "surfaces_field": "apps_present_in",
                        "relation": {"type": "contact_of", "to_root": True},
                    },
                    {
                        "path": "trips",
                        "entity_type": "Trip",
                        "id_fields": ["destination", "dates"],
                        "surfaces": ["hoolicalendar"],
                        "children": [
                            {
                                "path": "flight",
                                "entity_type": "Flight",
                                "id_fields": ["number"],
                                "surfaces": ["gringotts"],
                                "mutable": True,
                                "relation_type": "flight_of",
                            }
                        ],
                    },
                    {
                        "path": "financial",
                        "entity_type": "BankProfile",
                        "singleton": True,
                        "surfaces": ["gringotts"],
                        "mutable": True,
                    },
                    {
                        "path": "planted_dependencies",
                        "entity_type": "PlantedDependency",
                        "id_fields": ["id"],
                        "nested_attributes": ["affected"],
                        "surfaces_field": "affected.app",
                    },
                ],
            },
            "variables": {
                "entity_type": "ReferenceValue",
                "prefix_surfaces": {"GRINGOTTS_": ["gringotts"]},
                "default_surfaces": [],
            },
        }

    def _persona(self):
        return {
            "identity": {"name": "Michael Scott", "age": 46},
            "contacts": [
                {"name": "Jim Halpert", "apps_present_in": ["mail", "vaultbank"], "phone": "1"}
            ],
            "trips": [
                {
                    "destination": "NYC",
                    "dates": ["2026-04-17", "2026-04-20"],
                    "flight": {"number": "DN6769", "seat": "12A"},
                }
            ],
            "financial": {"checking_balance": 3210, "recurring_charges": [{"name": "x"}]},
            "planted_dependencies": [
                {
                    "id": "cascade",
                    "trigger": "cancel",
                    "affected": [
                        {"app": "vaultbank", "item": "hold"},
                        {"app": "mail", "item": "confirmation"},
                    ],
                }
            ],
        }

    def test_build_and_subgraph(self):
        world = build_world_graph(
            self._persona(),
            {"GRINGOTTS_CHECKING_BALANCE": 3210, "OTHER": "y"},
            self._sources(),
            provenance={
                "persona_uri": "p",
                "persona_sha256": "a" * 64,
                "variables_uri": "v",
                "variables_sha256": "b" * 64,
                "sources_config_sha256": "c" * 64,
            },
        )
        validate_schema(world, "world_graph.schema.json", REPOSITORY)
        ids = {entity["entity_id"] for entity in world["entities"]}
        self.assertIn("person:michael_scott", ids)
        self.assertIn("flight:dn6769", ids)
        self.assertIn("trip:nyc_2026-04-17_2026-04-20", ids)
        self.assertIn("referencevalue:gringotts_checking_balance", ids)
        self.assertTrue(all("." not in entity_id for entity_id in ids))
        relation_types = {rel["type"] for rel in world["relations"]}
        self.assertEqual(relation_types, {"contact_of", "flight_of"})
        bank = subgraph_for_apps(world, ["gringotts"])
        self.assertEqual(
            {e["entity_type"] for e in bank["entities"]},
            {"Person", "Flight", "BankProfile", "ReferenceValue", "PlantedDependency"},
        )
        profile = next(e for e in world["entities"] if e["entity_type"] == "BankProfile")
        self.assertEqual(profile["attributes"], {"checking_balance": 3210})
        cascade = next(e for e in world["entities"] if e["entity_type"] == "PlantedDependency")
        self.assertEqual(cascade["observation_surfaces"], ["gringotts", "hoolimail"])
        self.assertEqual(len(cascade["attributes"]["affected"]), 2)

    def test_unknown_app_spelling_is_an_error(self):
        aliases = AppAliases.from_config({"hoolimail": ["HooliMail"]})
        self.assertEqual(aliases.resolve("hoolimail"), "hoolimail")
        with self.assertRaises(WorldError):
            aliases.resolve("Notepad")


class RubricTests(unittest.TestCase):
    def test_milestones_scoring_and_hint(self):
        fragment = _fragment()
        rubric = derive_milestones(fragment, ONTOLOGY)
        self.assertEqual([m["milestone_id"] for m in rubric["milestones"]], ["m1", "m2", "m3"])
        self.assertEqual(rubric["milestones"][2]["depends_on"], ["m2"])
        self.assertEqual(rubric["milestones"][2]["weight"], 2.0)
        score = score_milestones(rubric, {"m1": True, "m2": False, "m3": True})
        self.assertAlmostEqual(score.milestone_completion, 3 / 4)
        self.assertAlmostEqual(score.dependency_consistent_completion, 1 / 4)
        self.assertEqual(score.first_failed_dependency, "m2")
        self.assertEqual(score.critical_error_count, 1)
        self.assertEqual(score.completion_curve, (0.25, 0.25, 0.25))
        hint = earliest_identifiable_hint(rubric, {"m1": True, "m2": False, "m3": True}, fragment)
        self.assertEqual(hint["earliest_identifiable_milestone_id"], "m2")
        self.assertEqual(hint["lineage_node_ids"], ["r1"])
        self.assertFalse(hint["is_root_cause"])
        none = earliest_identifiable_hint(rubric, {"m1": True, "m2": True, "m3": True}, fragment)
        self.assertIsNone(none["earliest_identifiable_milestone_id"])

    def test_unobservable_milestone_is_never_credited(self):
        fragment = _fragment()
        fragment["nodes"][0]["verifier"]["observability"] = "unobservable"
        rubric = derive_milestones(fragment, ONTOLOGY)
        score = score_milestones(rubric, {"m1": True, "m2": True, "m3": True})
        self.assertEqual(score.unscored_milestones, ("m1",))
        self.assertAlmostEqual(score.dependency_consistent_completion, 0.0)


class AnnotationTests(unittest.TestCase):
    def test_record_validates_and_derives_group(self):
        actions = [{"kind": "click", "x_px": 1, "y_px": 1}] * 6 + [
            {"kind": "terminate", "status": "success"}
        ]
        stats = compute_continuation(actions, 2)
        record = build_failure_annotation(
            annotation_version="v",
            task_id="t",
            model_id="m",
            rollout_id="r",
            raw_error_types=["detail_misperception", "wrong_subgoal"],
            reversibility="reversible",
            stats=stats,
            taxonomy=TAXONOMY,
            identifiable_at_action_index=5,
        )
        validate_schema(record, "failure_annotation.schema.json", REPOSITORY)
        self.assertEqual(record["paper_type"], "misunderstand_task_objective")
        self.assertEqual(record["group"], "long_horizon")
        self.assertEqual(record["action_horizon"], 3)
        self.assertFalse(record["horizon_censored"])
        self.assertEqual(record["secondary_causes"], ["detail_misperception"])
        censored = build_failure_annotation(
            annotation_version="v",
            task_id="t",
            model_id="m",
            rollout_id="r",
            raw_error_types=["scope_error"],
            reversibility="irreversible",
            stats=stats,
            taxonomy=TAXONOMY,
        )
        self.assertTrue(censored["horizon_censored"])
        self.assertIsNone(censored["action_horizon"])


class _FakeClient:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def complete(self, system: str, user: str) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text=self.reply, model="fake", usage={"total_tokens": 1}, request_sha256="d" * 64
        )


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.config = ExtractorConfig.from_yaml(
            REPOSITORY / "configs/synthesis/task_ir_extractor.yaml", REPOSITORY
        )
        self.world = {
            "environment_id": "env",
            "snapshot_id": "snap",
            "world_schema_version": "w/0.1",
            "entities": [
                {
                    "entity_id": "person:michael_scott",
                    "entity_type": "Person",
                    "attributes": {"name": "Michael"},
                    "observation_surfaces": ["hoolimail", "hoolicalendar"],
                    "mutable": False,
                    "source_path": "identity",
                    "id_confidence": "seed_derived",
                },
                {
                    "entity_id": "flight:dn6769",
                    "entity_type": "Flight",
                    "attributes": {"number": "DN6769"},
                    "observation_surfaces": ["dinoco_airlines"],
                    "mutable": True,
                    "source_path": "trips[0].flight",
                    "id_confidence": "seed_derived",
                },
            ],
            "relations": [],
        }
        self.task = {
            "id": "retrieval-f001",
            "category": "retrieval",
            "instruction": "Email Jim my NYC flight number.",
            "apps_involved": ["HooliCalendar", "Dinoco Airlines", "HooliMail"],
            "grading": {"rubrics": [{"criterion": "c", "weight": 1.0}]},
        }
        self.aliases = AppAliases.from_config(
            {
                "hoolimail": ["HooliMail"],
                "hoolicalendar": ["HooliCalendar"],
                "dinoco_airlines": ["Dinoco Airlines"],
            }
        )

    def _reply(self):
        fragment = _fragment()
        interface = {
            "produces": [
                {
                    "node_id": "f2",
                    "port_id": "flight",
                    "type": "Value",
                    "grounding": "derived:f2:flight",
                    "cardinality": "one",
                }
            ],
            "accepts": [
                {
                    "node_id": "r1",
                    "port_id": "owner",
                    "type": "Value",
                    "grounding": "person:michael_scott",
                    "cardinality": "one",
                    "required": True,
                    "dependency_kind": "data_dependency",
                }
            ],
            "identity_scope": ["person:michael_scott"],
            "time_scope": {"timezone": "America/New_York"},
            "reads": ["person:michael_scott"],
            "side_effects": [],
        }
        return "Here you go:\n```json\n%s\n```" % json.dumps(
            {"fragment": fragment, "interface": interface, "unbound_entities": [], "notes": "ok"}
        )

    def test_render_and_parse(self):
        self.assertEqual(render_prompt("a {{x}} b", {"x": "1"}), "a 1 b")
        with self.assertRaises(KeyError):
            render_prompt("{{missing}}", {})
        parsed = parse_extraction_response(self._reply())
        self.assertIn("fragment", parsed)
        with self.assertRaises(ExtractionParseError):
            parse_extraction_response("no json here")
        with self.assertRaises(ExtractionParseError):
            parse_extraction_response('{"fragment": {}}')

    def test_module_static_checks_and_grounding(self):
        parsed = parse_extraction_response(self._reply())
        subgraph = subgraph_for_apps(self.world, ["hoolimail", "hoolicalendar", "dinoco_airlines"])
        extracted = to_grounded_module(
            parsed,
            task=self.task,
            source_split="source-development",
            world=self.world,
            world_subgraph=subgraph,
            ontology=ONTOLOGY,
            source_uri="u",
            source_sha256="a" * 64,
            llm_call={"model": "fake"},
        )
        self.assertTrue(extracted.static_valid, extracted.issues)
        validate_schema(extracted.module, "grounded_task_module.schema.json", REPOSITORY)
        self.assertEqual(extracted.module["provenance"]["review_status"], "needs_review")
        parsed["fragment"]["nodes"][0]["inputs"][0]["grounding"] = "person:nobody"
        bad = to_grounded_module(
            parsed,
            task=self.task,
            source_split="source-development",
            world=self.world,
            world_subgraph=subgraph,
            ontology=ONTOLOGY,
            source_uri="u",
            source_sha256="a" * 64,
            llm_call={},
        )
        self.assertFalse(bad.static_valid)
        self.assertTrue(any("person:nobody" in issue for issue in bad.issues))

    def test_run_extraction_dry_run_and_fake_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = run_extraction(
                [self.task],
                world=self.world,
                aliases=self.aliases,
                ontology=ONTOLOGY,
                config=self.config,
                splits={},
                output_dir=out / "dry",
                source_uri="u",
                source_sha256="a" * 64,
            )
            self.assertEqual(manifest["mode"], "dry_run_prompts_only")
            prompt = (out / "dry" / "prompts" / "retrieval-f001.md").read_text()
            self.assertIn("flight:dn6769", prompt)
            self.assertNotIn("{{", prompt)
            client = _FakeClient(self._reply())
            manifest = run_extraction(
                [self.task],
                world=self.world,
                aliases=self.aliases,
                ontology=ONTOLOGY,
                config=self.config,
                splits={"retrieval-f001": "source-pilot"},
                output_dir=out / "live",
                source_uri="u",
                source_sha256="a" * 64,
                client=client,
            )
            self.assertEqual(client.calls, 1)
            self.assertEqual(manifest["counts"]["static_valid_modules"], 1)
            module = json.loads((out / "live" / "modules.jsonl").read_text().splitlines()[0])
            self.assertEqual(module["source_split"], "source-pilot")
            self.assertEqual(module["provenance"]["llm_calls"][0]["model"], "fake")

    def test_synonym_keys_are_normalised_and_replies_reused(self):
        parsed = parse_extraction_response(self._reply())
        for node in parsed["fragment"]["nodes"]:
            node["operation"] = node.pop("op")
            node["application"] = node.pop("app")
        subgraph = subgraph_for_apps(self.world, ["hoolimail", "hoolicalendar", "dinoco_airlines"])
        extracted = to_grounded_module(
            parsed,
            task=self.task,
            source_split="source-development",
            world=self.world,
            world_subgraph=subgraph,
            ontology=ONTOLOGY,
            source_uri="u",
            source_sha256="a" * 64,
            llm_call={},
        )
        self.assertTrue(extracted.static_valid, extracted.issues)
        del parsed["fragment"]["nodes"][1]["application"]
        missing = to_grounded_module(
            parsed,
            task=self.task,
            source_split="source-development",
            world=self.world,
            world_subgraph=subgraph,
            ontology=ONTOLOGY,
            source_uri="u",
            source_sha256="a" * 64,
            llm_call={},
        )
        self.assertEqual(
            [i for i in missing.issues if i.startswith("APP_UNSPECIFIED")],
            ["APP_UNSPECIFIED: node f2 has no application"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            replies = Path(tmp) / "old"
            replies.mkdir()
            (replies / "retrieval-f001.txt").write_text(self._reply())
            manifest = run_extraction(
                [self.task],
                world=self.world,
                aliases=self.aliases,
                ontology=ONTOLOGY,
                config=self.config,
                splits={},
                output_dir=Path(tmp) / "re",
                source_uri="u",
                source_sha256="a" * 64,
                saved_replies_dir=replies,
            )
            self.assertEqual(manifest["mode"], "reparse_saved_replies")
            self.assertEqual(manifest["counts"]["modules"], 1)
            module = json.loads((Path(tmp) / "re" / "modules.jsonl").read_text().splitlines()[0])
            self.assertTrue(module["provenance"]["llm_calls"][0]["reused_saved_reply"])

    def test_unbound_entities_are_normalised_to_objects(self):
        from derail.longhorizon.extraction import normalize_unbound

        rows = normalize_unbound(
            ["free text", {"name": "inbox", "app": "hoolimail", "kind": "record"}]
        )
        self.assertEqual(rows[0]["name"], "free text")
        self.assertIsNone(rows[0]["app"])
        self.assertEqual(rows[1]["app"], "hoolimail")
        self.assertEqual(set(rows[1]), {"name", "app", "kind", "needed_by", "why"})

    def test_concurrent_workers_keep_task_order(self):
        tasks = [dict(self.task, id="retrieval-f00%d" % index) for index in range(1, 5)]
        client = _FakeClient(self._reply())
        with tempfile.TemporaryDirectory() as tmp:
            manifest = run_extraction(
                tasks,
                world=self.world,
                aliases=self.aliases,
                ontology=ONTOLOGY,
                config=self.config,
                splits={},
                output_dir=Path(tmp),
                source_uri="u",
                source_sha256="a" * 64,
                client=client,
                workers=3,
            )
            self.assertEqual(client.calls, 4)
            self.assertEqual(manifest["workers"], 3)
            seen = []
            run_extraction(
                tasks,
                world=self.world,
                aliases=self.aliases,
                ontology=ONTOLOGY,
                config=self.config,
                splits={},
                output_dir=Path(tmp) / "p",
                source_uri="u",
                source_sha256="a" * 64,
                client=client,
                workers=2,
                progress=lambda done, total, task_id, seconds: seen.append((done, total)),
            )
            self.assertEqual(seen, [(1, 4), (2, 4), (3, 4), (4, 4)])
            ids = [
                json.loads(line)["source_task_id"]
                for line in (Path(tmp) / "modules.jsonl").read_text().splitlines()
            ]
            self.assertEqual(ids, [task["id"] for task in tasks])

    def test_client_refuses_without_approval(self):
        with mock.patch.dict(
            os.environ, {self.config.approval_env: "", self.config.purpose_env: ""}
        ):
            with self.assertRaises(ApprovalRequired):
                OpenAICompatibleClient(self.config, "task_ir_extraction")
        with mock.patch.dict(
            os.environ,
            {self.config.approval_env: "1", self.config.purpose_env: "trajectory_rubric_judge"},
        ):
            with self.assertRaises(ApprovalRequired):
                OpenAICompatibleClient(self.config, "task_ir_extraction")


if __name__ == "__main__":
    unittest.main()


class TypeRegistryTests(unittest.TestCase):
    def setUp(self):
        from derail.longhorizon.types import ValueTypeRegistry

        self.registry = ValueTypeRegistry.from_yaml(
            REPOSITORY / "configs/synthesis/value_types_v0.yaml"
        )

    def test_aliases_and_collections(self):
        self.assertEqual(self.registry.resolve("Money").canonical, "MoneyAmount")
        self.assertEqual(self.registry.resolve("MonetaryAmount").canonical, "MoneyAmount")
        self.assertEqual(self.registry.resolve("Person").canonical, "PersonRef")
        collection = self.registry.resolve("OrderHistoryRecord[]")
        self.assertEqual((collection.canonical, collection.collection), ("OrderRef", True))
        self.assertEqual(
            self.registry.resolve("MarketPositionCollection").canonical, "MarketPositionRef"
        )
        self.assertEqual(self.registry.resolve("List<Text>").canonical, "Text")
        unknown = self.registry.resolve("ReplyLatencyBucket")
        self.assertFalse(unknown.registered)
        self.assertEqual(unknown.canonical, "ReplyLatencyBucket")
        self.assertEqual(self.registry.resolve_cardinality("five", collection=False), "many")
        self.assertEqual(self.registry.resolve_cardinality("1", collection=False), "one")
        self.assertEqual(self.registry.resolve_cardinality("one", collection=True), "many")

    def test_type_system_config_matches_synthesis_config(self):
        import yaml

        raw = yaml.safe_load((REPOSITORY / "configs/synthesis/mypcbench_v0.2.yaml").read_text())
        self.assertEqual(raw["type_system"], self.registry.type_system_config())

    def test_normalise_module_reports_unregistered(self):
        from derail.longhorizon.types import normalize_module_types

        module = {
            "fragment": {
                "nodes": [
                    {
                        "node_id": "n1",
                        "inputs": [{"port_id": "a", "type": "Money", "cardinality": "1"}],
                        "outputs": [{"port_id": "b", "type": "WeirdThing[]"}],
                    }
                ]
            },
            "interface": {
                "produces": [{"port_id": "b", "type": "WeirdThing[]", "cardinality": "0..*"}],
                "accepts": [],
            },
        }
        result, report = normalize_module_types(module, self.registry)
        out = result["fragment"]["nodes"][0]["outputs"][0]
        self.assertEqual(
            (out["type"], out["cardinality"], out["original_type"]),
            ("WeirdThing", "many", "WeirdThing[]"),
        )
        self.assertEqual(result["fragment"]["nodes"][0]["inputs"][0]["type"], "MoneyAmount")
        self.assertEqual(list(report["unregistered_types"]), ["WeirdThing"])
        self.assertEqual(report["ports_changed"], 3)


class ReviewTests(unittest.TestCase):
    def test_apply_review_accepts_and_projects_effects(self):
        from derail.longhorizon.review import ReviewVerdicts, apply_review
        from derail.longhorizon.types import ValueTypeRegistry

        registry = ValueTypeRegistry.from_yaml(REPOSITORY / "configs/synthesis/value_types_v0.yaml")
        fragment = _fragment()
        module = {
            "schema_version": "grounded-task-module/0.1",
            "module_id": "gm_t",
            "source_task_id": "t",
            "source_split": "source-development",
            "environment_id": "e",
            "snapshot_id": "s",
            "world_schema_version": "w",
            "world_subgraph_id": "wg_t",
            "instruction": "i",
            "fragment": fragment,
            "interface": {
                "produces": [
                    {
                        "node_id": "f2",
                        "port_id": "flight",
                        "type": "Value",
                        "grounding": "derived:f2:flight",
                        "cardinality": "one",
                    }
                ],
                "accepts": [],
                "identity_scope": [],
                "time_scope": {},
                "reads": [],
                "side_effects": [],
            },
            "provenance": {
                "source_uri": "u",
                "source_sha256": "a" * 64,
                "review_status": "needs_review",
                "static_valid": True,
            },
        }
        rejected = dict(module, source_task_id="r", provenance=dict(module["provenance"]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.yaml"
            path.write_text(
                "\n".join(
                    [
                        "schema_version: task-ir-review/0.1",
                        "reviewer: x",
                        "reviewed_at: '2026-09-03'",
                        "method: per_module",
                        "verdicts:",
                        "  t: {verdict: accept, entity_precision: 0.97}",
                        "  r: {verdict: reject}",
                        "",
                    ]
                )
            )
            verdicts = ReviewVerdicts.from_yaml(path)
        accepted, report = apply_review(
            {"t": module, "r": rejected}, verdicts, registry=registry, ontology=ONTOLOGY
        )
        self.assertEqual(report["accepted"], ["t"])
        self.assertEqual(report["skipped"], {"r": "reject"})
        got = accepted[0]
        self.assertEqual(got["provenance"]["review_status"], "human_verified")
        self.assertEqual(got["provenance"]["review"]["entity_precision"], 0.97)
        self.assertEqual(got["interface"]["side_effects"][0]["effect_type"], "send")
        self.assertTrue(got["interface"]["side_effects"][0]["irreversible"])
        validate_schema(got, "grounded_task_module.schema.json", REPOSITORY)
        self.assertIn("Value", report["unregistered_types"])
