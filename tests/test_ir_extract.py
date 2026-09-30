"""Task IR v1 extraction on the miniworld with a fake model: prompt rendering, reply
normalisation, static + grounding validation, gold execution and the rubric check."""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path

from derail.ir.extract import (
    V1ExtractorConfig,
    normalize_task_ir,
    parse_task_ir_reply,
    run_extraction_v1,
)
from derail.ir.gold_interpreter import GoldInterpreter, InterpreterConfig, WorldCopy
from derail.ir.model import load_task_ir
from derail.ir.rubric_check import (
    RubricCheckConfig,
    check,
    context_values,
    expand_variables,
    extract_expectations,
)
from derail.ir.validate import literal_allowed, validate_grounding
from derail.longhorizon.extraction import LLMResponse
from derail.longhorizon.ontology import Ontology
from derail.longhorizon.types import ValueTypeRegistry
from derail.longhorizon.world import AppAliases, load_sources
from derail.world.schema_graph import SchemaGraph, SchemaGraphConfig, build_schema_graph
from derail.world.sqlite_fixture import build_world

REPOSITORY = Path(__file__).resolve().parents[1]
MINIWORLD = REPOSITORY / "tests" / "miniworld"
SEEDS = {
    "hoolicalendar": MINIWORLD / "hoolicalendar.seed.json",
    "workbuzz": MINIWORLD / "workbuzz.seed.json",
}
TASK = {
    "id": "mw-002",
    "category": "long_horizon",
    "apps_involved": ["HooliCalendar", "HooliWork"],
    "instruction": "Look up the date of THE DUNDIES on my calendar and send Jim Halpert a direct "
    'message on HooliWork saying "Dundies is on <date>, save the date."',
    "grading": {
        "rubrics": [
            {
                "criterion": "Agent sends Jim Halpert (jim.halpert@dundermifflin.com) "
                "a HooliWork DM "
                "with the Dundies date July 2, 2026",
                "type": "llm_judge",
                "weight": 0.6,
            },
            {"criterion": 'The message says "save the date"', "type": "llm_judge", "weight": 0.4},
        ]
    },
}


class FakeClient:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        return LLMResponse(
            text=self.reply, model="fake", usage={"total_tokens": 1}, request_sha256="0" * 64
        )


def _model_style_reply() -> str:
    """The mw-002 IR as a model would plausibly write it: fenced, synonyms, defaults omitted."""

    ir = json.loads((MINIWORLD / "task_ir" / "mw-002.json").read_text())
    nodes = copy.deepcopy(ir["nodes"])
    for node in nodes:
        node.pop("route_hint", None)
        node.pop("critical", None)
        node["operation"] = node.pop("op")
        if node["node_id"] == "n4":
            node["app"] = "HooliWork"
            node.pop("outputs")
        if node["node_id"] == "n5":
            node.pop("side_effects")
            node.pop("reversibility_class")
        if node["node_id"] in ("n1", "n2", "n3"):
            node.pop("reversibility_class")
            node.pop("side_effects")
    body = {
        "task_ir": {"nodes": nodes, "edges": ir["edges"], "final_verifier": ir["final_verifier"]},
        "unbound_entities": [],
        "notes": "nothing ambiguous",
    }
    return "Here is the IR.\n```json\n%s\n```\n" % json.dumps(body, indent=1)


class ExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("DERAIL_TMP_ROOT"))
        root = Path(cls.tmp.name)
        cls.db_dir = root / "db"
        cls.dbs = build_world(SEEDS, cls.db_dir)
        graph = build_schema_graph(
            cls.dbs,
            SchemaGraphConfig.from_yaml(REPOSITORY / "configs/synthesis/schema_graph_v1.yaml"),
            "miniworld",
        )
        cls.graph = SchemaGraph.from_record(graph)
        base = V1ExtractorConfig.from_yaml(
            REPOSITORY / "configs/synthesis/task_ir_v1_extractor.yaml", REPOSITORY
        )
        cls.config = dataclasses.replace(
            base, world_id="miniworld", reference_time="2026-04-06T09:00:00"
        )
        sources = load_sources(base.base.world_sources_config)
        cls.aliases = AppAliases.from_config(sources["app_aliases"])
        cls.ontology = Ontology.from_yaml(base.base.ontology_config)
        cls.registry = ValueTypeRegistry.from_yaml(base.base.value_types_config)
        cls.rubric_config = RubricCheckConfig.from_yaml(
            REPOSITORY / "configs/synthesis/rubric_check_v1.yaml"
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _run(self, client, out_name):
        out = Path(self.tmp.name) / out_name
        manifest = run_extraction_v1(
            [TASK],
            config=self.config,
            schema_graph=self.graph,
            database_dir=self.db_dir,
            ontology=self.ontology,
            aliases=self.aliases,
            registry=self.registry,
            output_dir=out,
            repository=REPOSITORY,
            client=client,
        )
        return out, manifest

    def test_dry_run_renders_schema_prompt_only(self):
        out, manifest = self._run(None, "dry")
        self.assertEqual(manifest["mode"], "dry_run_prompts_only")
        prompt = (out / "prompts" / "mw-002.md").read_text()
        self.assertIn("## workbuzz", prompt)
        self.assertIn("- events (26 rows)", prompt)
        self.assertIn("sample:", prompt)
        self.assertIn("dm_id -> workbuzz.dms.id (explicit)", prompt)
        self.assertIn('reference_time (the world\'s "now", ISO 8601): 2026-04-06T09:00:00', prompt)
        self.assertFalse((out / "task_ir").exists())

    def test_model_reply_becomes_valid_task_ir_that_executes(self):
        client = FakeClient(_model_style_reply())
        out, manifest = self._run(client, "run")
        self.assertEqual(client.calls, 1)
        self.assertEqual(manifest["counts"]["static_valid"], 1, manifest["per_task"])
        self.assertEqual(manifest["counts"]["grounding_clean"], 1, manifest["per_task"])
        task_ir = load_task_ir(out / "task_ir" / "mw-002.json", REPOSITORY)
        n4 = [n for n in task_ir["nodes"] if n["node_id"] == "n4"][0]
        self.assertEqual(n4["app"], "workbuzz")
        self.assertEqual([p["port_id"] for p in n4["outputs"]], ["dm_id"])
        n5 = [n for n in task_ir["nodes"] if n["node_id"] == "n5"][0]
        self.assertEqual(n5["reversibility_class"], "R3")
        self.assertEqual(n5["side_effects"][0]["effect_type"], "send_message")
        self.assertIsNone(n5["side_effects"][0]["compensating_effect_type"])
        self.assertEqual(task_ir["provenance"]["review_status"], "needs_review")
        with WorldCopy.open("miniworld", self.dbs, Path(self.tmp.name) / "copy") as world:
            gold = GoldInterpreter(
                InterpreterConfig.from_yaml(
                    REPOSITORY / "configs/synthesis/gold_interpreter_v1.yaml"
                )
            ).run(task_ir, world, REPOSITORY)
        self.assertIs(gold["final_verifier_passed"], True)
        # reuse the saved reply: no model call, same result
        reparse = Path(self.tmp.name) / "reparse"
        manifest2 = run_extraction_v1(
            [TASK],
            config=self.config,
            schema_graph=self.graph,
            database_dir=self.db_dir,
            ontology=self.ontology,
            aliases=self.aliases,
            registry=self.registry,
            output_dir=reparse,
            repository=REPOSITORY,
            saved_replies_dir=out / "replies",
        )
        self.assertEqual(manifest2["mode"], "reparse_saved_replies")
        self.assertEqual(manifest2["counts"]["static_valid"], 1)
        self.gold = gold

    def test_rubric_expectations_and_check(self):
        variables = {
            "MICHAEL_EMAIL": "michael.scott@dundermifflin.com",
            "DUNDIES_VENUE": "Chili's",
            "X": 12.5,
        }
        expectations = extract_expectations(TASK, variables, self.rubric_config)
        kinds = {(e["kind"], e["value"]) for e in expectations}
        self.assertIn(("date", "2026-07-02"), kinds)
        self.assertIn(("email", "jim.halpert@dundermifflin.com"), kinds)
        self.assertIn(("text", "save the date"), kinds)
        task_ir = load_task_ir(MINIWORLD / "task_ir" / "mw-002.json", REPOSITORY)
        with WorldCopy.open("miniworld", self.dbs, Path(self.tmp.name) / "copy2") as world:
            gold = GoldInterpreter().run(task_ir, world, REPOSITORY)
            extra = context_values(task_ir, gold, world.connections)
        # Jim's e-mail is not a gold value (the DM is resolved by id) but sits in the dms row.
        self.assertTrue(any("jim.halpert" in str(v) for v in extra))
        self.assertEqual(
            check("mw-002", gold, expectations, self.rubric_config)["verdict"], "needs_review"
        )
        result = check("mw-002", gold, expectations, self.rubric_config, extra)
        self.assertEqual(result["verdict"], "auto_validated", result["unmatched"])
        comparable = expectations
        wrong = comparable + [
            {
                "kind": "date",
                "value": "2026-07-03",
                "source": "rubric",
                "rubric_index": 0,
                "weight": 1.0,
                "raw": "July 3",
            }
        ]
        result = check("mw-002", gold, wrong, self.rubric_config, extra)
        self.assertEqual(
            (result["verdict"], result["code"]), ("needs_review", "IR_RUBRIC_MISMATCH")
        )
        self.assertEqual(
            check("mw-002", gold, [], self.rubric_config)["code"], "NO_COMPARABLE_EXPECTATIONS"
        )

    def test_grounding_validator_flags_world_mismatches(self):
        task_ir = load_task_ir(MINIWORLD / "task_ir" / "mw-001.json", REPOSITORY)
        clean = [
            i
            for i in validate_grounding(
                task_ir, self.graph, self.db_dir, allowed_literals=self.config.persona_literals
            )
            if i["severity"] == "error"
        ]
        self.assertEqual(clean, [])
        broken = copy.deepcopy(task_ir)
        broken["nodes"][1]["reads"].append(
            {"table": "hoolicalendar.nope", "column": "x", "entity_ref": "nope:*"}
        )
        broken["nodes"][1]["reads"].append(
            {"table": "hoolicalendar.events", "column": "start_at", "entity_ref": "events:999999"}
        )
        broken["nodes"][1]["produces"][0]["derivation"]["query"] = (
            "SELECT start_at FROM eventz WHERE id = :event_id"
        )
        codes = sorted(
            i["code"]
            for i in validate_grounding(
                broken, self.graph, self.db_dir, allowed_literals=self.config.persona_literals
            )
            if i["severity"] == "error"
        )
        self.assertEqual(codes, ["ENTITY_MISSING", "SQL_COMPILE_ERROR", "TABLE_UNKNOWN"])

    def test_parse_reply_accepts_bare_nodes_and_rejects_prose(self):
        parsed = parse_task_ir_reply('{"nodes": [], "edges": []}')
        self.assertIn("task_ir", parsed)
        with self.assertRaises(Exception):
            parse_task_ir_reply("no json here")
        raw = {
            "task_ir": {
                "nodes": [
                    {
                        "node_id": "n1",
                        "operation": "retrieve",
                        "app": "HooliCalendar",
                        "produces": [{"name": "x", "type": "Text", "query": "SELECT 1"}],
                    }
                ],
                "edges": [],
            }
        }
        normalized = normalize_task_ir(raw, TASK, self.config, self.ontology, self.aliases)
        node = normalized["nodes"][0]
        self.assertEqual(
            (node["op"], node["app"], node["reversibility_class"]),
            ("retrieve", "hoolicalendar", "R0"),
        )
        self.assertEqual(
            node["produces"][0]["derivation"],
            {"kind": "sql", "query": "SELECT 1", "returns": "scalar"},
        )
        self.assertEqual(node["outputs"][0]["grounding"], "derived:n1:x")


if __name__ == "__main__":
    unittest.main()


class LiteralAndVariableTests(unittest.TestCase):
    def test_literal_spellings(self):
        # An ISO date literal is accepted when the instruction says "July 2"; times likewise.
        self.assertTrue(literal_allowed("2026-07-02", "meet on July 2 at 3pm"))
        self.assertTrue(literal_allowed("15:00", "meet on July 2 at 3pm"))
        self.assertFalse(literal_allowed("16:00", "meet on July 2 at 3pm"))
        self.assertTrue(literal_allowed("Jim Halpert", "DM jim halpert now"))
        self.assertFalse(literal_allowed("jim.halpert@dundermifflin.com", "DM Jim Halpert now"))
        self.assertTrue(
            literal_allowed(
                "michael.scott@dundermifflin.com", "anything", ["michael.scott@dundermifflin.com"]
            )
        )

    def test_literals_must_come_from_the_instruction(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("DERAIL_TMP_ROOT")) as tmp:
            dbs = build_world(SEEDS, Path(tmp) / "db")
            graph = SchemaGraph.from_record(
                build_schema_graph(
                    dbs,
                    SchemaGraphConfig.from_yaml(
                        REPOSITORY / "configs/synthesis/schema_graph_v1.yaml"
                    ),
                    "miniworld",
                )
            )
            task_ir = load_task_ir(MINIWORLD / "task_ir" / "mw-002.json", REPOSITORY)
            persona = ("Michael Scott", "michael.scott@dundermifflin.com")
            clean = [
                i["code"]
                for i in validate_grounding(task_ir, graph, Path(tmp) / "db", persona)
                if i["severity"] == "error"
            ]
            self.assertEqual(clean, [])
            leaked = copy.deepcopy(task_ir)
            n4 = [n for n in leaked["nodes"] if n["node_id"] == "n4"][0]
            n4["inputs"][0]["literal"] = "jim.halpert@dundermifflin.com"  # a world fact
            n1 = [n for n in leaked["nodes"] if n["node_id"] == "n1"][0]
            n1["inputs"].append(
                {
                    "port_id": "attachment",
                    "type": "Text",
                    "grounding": "unbound:invite_pdf",
                    "source": "world",
                }
            )
            codes = sorted(
                i["code"]
                for i in validate_grounding(leaked, graph, Path(tmp) / "db", persona)
                if i["severity"] == "error"
            )
            self.assertEqual(codes, ["LITERAL_NOT_IN_INSTRUCTION", "UNBOUND_INPUT"])

    def test_today_variables_expand_from_reference_time(self):
        expanded = expand_variables(
            {"A": "${TODAY+15}", "B": "${TODAY-1}", "C": "${TODAY}", "D": 3}, "2026-07-19T05:00:01Z"
        )
        self.assertEqual(
            expanded, {"A": "2026-08-03", "B": "2026-07-18", "C": "2026-07-19", "D": 3}
        )
