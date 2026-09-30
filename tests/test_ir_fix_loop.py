"""Structural repairs, cycle naming, lexical rubric judging and the fix loop (fake client)."""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path

from derail.ir.extract import V1ExtractorConfig, repair_edges
from derail.ir.fix_loop import FeedbackClient, collect_problems, failing_tasks, run_fix_round
from derail.ir.model import find_cycle
from derail.ir.rubric_check import RubricCheckConfig, check
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


class RepairTests(unittest.TestCase):
    def test_self_edge_and_literal_edge_are_repaired(self):
        body = {
            "nodes": [
                {
                    "node_id": "n1",
                    "inputs": [{"port_id": "owner", "literal": "x"}, {"port_id": "loop"}],
                    "produces": [{"name": "loop"}],
                    "outputs": [{"port_id": "loop"}],
                },
                {
                    "node_id": "n2",
                    "inputs": [{"port_id": "owner", "source": "upstream"}],
                    "produces": [],
                },
            ],
            "edges": [
                {
                    "edge_id": "self",
                    "from": {"node_id": "n1", "port_id": "loop"},
                    "to": {"node_id": "n1", "port_id": "loop"},
                },
                {
                    "edge_id": "lit",
                    "from": {"node_id": "n1", "port_id": "owner"},
                    "to": {"node_id": "n2", "port_id": "owner"},
                },
            ],
        }
        repairs = repair_edges(body)
        self.assertEqual([r["repair"] for r in repairs], ["dropped_self_edge", "literal_copied"])
        self.assertEqual(body["edges"], [])
        self.assertEqual(body["nodes"][1]["inputs"][0]["literal"], "x")
        self.assertEqual([p["port_id"] for p in body["nodes"][0]["inputs"]], ["owner"])

    def test_upstream_input_edge_is_rewired(self):
        body = {
            "nodes": [
                {
                    "node_id": "n1",
                    "inputs": [],
                    "produces": [{"name": "v"}],
                    "outputs": [{"port_id": "v"}],
                },
                {
                    "node_id": "n2",
                    "inputs": [{"port_id": "v"}],
                    "produces": [{"name": "w"}],
                    "outputs": [{"port_id": "w"}],
                },
                {"node_id": "n3", "inputs": [{"port_id": "v"}], "produces": []},
            ],
            "edges": [
                {
                    "edge_id": "a",
                    "from": {"node_id": "n1", "port_id": "v"},
                    "to": {"node_id": "n2", "port_id": "v"},
                },
                {
                    "edge_id": "b",
                    "from": {"node_id": "n2", "port_id": "v"},
                    "to": {"node_id": "n3", "port_id": "v"},
                },
            ],
        }
        repairs = repair_edges(body)
        self.assertEqual(repairs[0]["repair"], "rewired_to_producer")
        self.assertEqual(body["edges"][1]["from"], {"node_id": "n1", "port_id": "v"})

    def test_find_cycle_names_edges(self):
        ir = {
            "nodes": [{"node_id": "a"}, {"node_id": "b"}, {"node_id": "c"}],
            "edges": [
                {"edge_id": "e1", "from": {"node_id": "a"}, "to": {"node_id": "b"}},
                {"edge_id": "e2", "from": {"node_id": "b"}, "to": {"node_id": "c"}},
                {"edge_id": "e3", "from": {"node_id": "c"}, "to": {"node_id": "b"}},
            ],
        }
        self.assertEqual(sorted(find_cycle(ir)), ["e2", "e3"])
        ir["edges"].pop()
        self.assertEqual(find_cycle(ir), [])


class LexicalTests(unittest.TestCase):
    def setUp(self):
        self.config = RubricCheckConfig.from_yaml(
            REPOSITORY / "configs/synthesis/rubric_check_v1.yaml"
        )

    def test_wording_match_validates_when_no_literal_expectations(self):
        task = {
            "grading": {
                "rubrics": [
                    {"criterion": "Agent finds the channel with the most messages", "weight": 0.5},
                    {"criterion": "Agent creates a calendar event for the review", "weight": 0.5},
                ]
            }
        }
        task_ir = {
            "nodes": [
                {
                    "semantic_goal": "Pick the channel with the most messages",
                    "produces": [{"name": "top_channel"}],
                },
                {"semantic_goal": "Create the calendar event", "produces": [{"name": "event_id"}]},
            ],
            "provenance": {"notes": "review event"},
        }
        gold = {
            "values": [{"node_id": "n1", "name": "x", "value": "Channel review: #general"}],
            "writes_gold": [],
            "final_verifier_passed": True,
        }
        result = check("t", gold, [], self.config, task=task, task_ir=task_ir)
        self.assertEqual((result["verdict"], result["code"]), ("auto_validated", "LEXICAL_MATCH"))
        self.assertGreaterEqual(result["lexical"]["satisfied_share"], 0.7)
        off = {
            "grading": {
                "rubrics": [
                    {
                        "criterion": "Agent cancels the Jamaica flight and refunds the hotel",
                        "weight": 1.0,
                    }
                ]
            }
        }
        self.assertEqual(
            check("t", gold, [], self.config, task=off, task_ir=task_ir)["verdict"], "needs_review"
        )
        self.assertEqual(
            check(
                "t",
                dict(gold, final_verifier_passed=False),
                [],
                self.config,
                task=task,
                task_ir=task_ir,
            )["code"],
            "NO_COMPARABLE_EXPECTATIONS",
        )

    def test_literal_threshold_is_lenient(self):
        gold = {
            "values": [{"node_id": "n1", "name": "d", "value": "2026-07-02"}],
            "writes_gold": [],
            "final_verifier_passed": True,
        }
        expectations = [
            {
                "kind": "date",
                "value": "2026-07-02",
                "source": "rubric",
                "rubric_index": 0,
                "weight": 1.0,
                "raw": "",
            }
        ] * 4 + [
            {
                "kind": "date",
                "value": "2026-07-09",
                "source": "rubric",
                "rubric_index": 1,
                "weight": 1.0,
                "raw": "",
            }
        ]
        self.assertEqual(check("t", gold, expectations, self.config)["verdict"], "auto_validated")


class FixLoopTests(unittest.TestCase):
    def test_problems_are_collected_and_fed_back(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("DERAIL_TMP_ROOT")) as tmp:
            run = Path(tmp) / "run"
            (run / "task_ir").mkdir(parents=True)
            (run / "rubric_check").mkdir()
            (run / "replies").mkdir()
            (run / "manifest.json").write_text(
                json.dumps(
                    {
                        "per_task": [
                            {
                                "task_id": "t1",
                                "static_errors": ["STATIC: NOT_A_DAG: edges e2 -> e3 form a cycle"],
                            },
                            {"task_id": "t2", "static_errors": []},
                            {"task_id": "t3", "static_errors": []},
                        ]
                    }
                )
            )
            (run / "task_ir" / "t1.json").write_text(
                json.dumps(
                    {
                        "nodes": [],
                        "provenance": {
                            "grounding_issues": [
                                {
                                    "code": "LITERAL_NOT_IN_INSTRUCTION",
                                    "severity": "error",
                                    "node_id": "n4",
                                    "detail": "x",
                                }
                            ]
                        },
                    }
                )
            )
            (run / "rubric_check" / "rubric_checks.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "task_id": "t1",
                                "verdict": "needs_review",
                                "code": "GOLD_EXECUTION_FAILED",
                                "error": "boom",
                            }
                        ),
                        json.dumps({"task_id": "t2", "verdict": "auto_validated", "code": None}),
                        json.dumps(
                            {
                                "task_id": "t3",
                                "verdict": "needs_review",
                                "code": "IR_RUBRIC_MISMATCH",
                                "unmatched": [
                                    {
                                        "kind": "date",
                                        "value": "2026-07-02",
                                        "rubric_index": 0,
                                        "raw": "July 2",
                                    }
                                ],
                            }
                        ),
                    ]
                )
                + "\n"
            )
            self.assertEqual(failing_tasks(run), ["t1", "t3"])
            problems = collect_problems(run, "t1")
            self.assertTrue(any("NOT_A_DAG" in p for p in problems))
            self.assertTrue(any("LITERAL_NOT_IN_INSTRUCTION" in p for p in problems))
            self.assertTrue(any("boom" in p for p in problems))
            self.assertTrue(any("2026-07-02" in p for p in collect_problems(run, "t3")))

            captured = {}

            class Inner:
                def complete(self, system, user):
                    captured["user"] = user
                    return LLMResponse(text="{}", model="fake", usage={}, request_sha256="")

            client = FeedbackClient(
                Inner(),
                (REPOSITORY / "prompts/synthesis/task_ir_v1_fix_user.txt").read_text(),
                {"t1": '{"task_ir": {}}'},
                {"t1": problems},
                lambda user: "t1",
            )
            client.complete("sys", "task_id: t1\ninstruction: x")
            self.assertIn("- static: STATIC: NOT_A_DAG", captured["user"])
            self.assertIn('{"task_ir": {}}', captured["user"])
            self.assertTrue(captured["user"].startswith("task_id: t1"))

    def test_fix_round_only_retries_failing_tasks(self):
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
            base = V1ExtractorConfig.from_yaml(
                REPOSITORY / "configs/synthesis/task_ir_v1_extractor.yaml", REPOSITORY
            )
            config = dataclasses.replace(
                base,
                world_id="miniworld",
                reference_time="2026-04-06T09:00:00",
                persona_literals=("Michael Scott", "michael.scott@dundermifflin.com"),
            )
            sources = load_sources(base.base.world_sources_config)
            previous = Path(tmp) / "prev"
            (previous / "rubric_check").mkdir(parents=True)
            (previous / "manifest.json").write_text(
                json.dumps(
                    {
                        "per_task": [
                            {"task_id": "mw-001", "static_errors": []},
                            {"task_id": "mw-002", "static_errors": ["PARSE: no JSON"]},
                        ]
                    }
                )
            )
            (previous / "rubric_check" / "rubric_checks.jsonl").write_text(
                json.dumps({"task_id": "mw-001", "verdict": "auto_validated", "code": None}) + "\n"
            )
            reply = json.dumps(
                {"task_ir": json.loads((MINIWORLD / "task_ir" / "mw-002.json").read_text())}
            )
            calls = []

            class Client:
                def complete(self, system, user):
                    calls.append(user)
                    return LLMResponse(text=reply, model="fake", usage={}, request_sha256="")

            tasks = [
                {
                    "id": "mw-001",
                    "instruction": "a",
                    "apps_involved": ["HooliCalendar"],
                    "grading": {"rubrics": []},
                },
                {
                    "id": "mw-002",
                    "instruction": "Look up the date of THE DUNDIES and DM Jim Halpert",
                    "apps_involved": ["HooliCalendar", "HooliWork"],
                    "grading": {"rubrics": []},
                },
            ]
            manifest = run_fix_round(
                tasks,
                previous_run_dir=previous,
                output_dir=Path(tmp) / "fix",
                config=config,
                template_path=REPOSITORY / "prompts/synthesis/task_ir_v1_fix_user.txt",
                client=Client(),
                schema_graph=graph,
                database_dir=Path(tmp) / "db",
                ontology=Ontology.from_yaml(base.base.ontology_config),
                aliases=AppAliases.from_config(sources["app_aliases"]),
                registry=ValueTypeRegistry.from_yaml(base.base.value_types_config),
                repository=REPOSITORY,
            )
            self.assertEqual(len(calls), 1)
            self.assertIn("no JSON", calls[0])
            self.assertEqual(manifest["counts"]["static_valid"], 1)
            self.assertEqual(manifest["fed_back"], {"mw-002": 1})


if __name__ == "__main__":
    unittest.main()
