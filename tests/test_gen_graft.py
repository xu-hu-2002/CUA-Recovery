"""Grafting operators, beam search, hazards and realization on the miniworld."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from derail.gen.compat import build_compat_index, literal_input_ports, output_ports
from derail.gen.graft import SearchConfig, beam_search, conditionalize, graft, rename_ir
from derail.gen.hazards import HazardConfig, base_rates, injectable, injection_record, render_patch
from derail.gen.realize import (
    RealizationConfig,
    build_realization_fields,
    graph_summary,
    hidden_values,
    parse_realization_reply,
    round_trip_compare,
    style_stats_from_instructions,
)
from derail.ir.model import load_task_ir, validate_task_ir
from derail.longhorizon.types import ValueTypeRegistry
from derail.world.sqlite_fixture import build_world

REPOSITORY = Path(__file__).resolve().parents[1]
MINIWORLD = REPOSITORY / "tests" / "miniworld"
SEEDS = {
    "hoolicalendar": MINIWORLD / "hoolicalendar.seed.json",
    "workbuzz": MINIWORLD / "workbuzz.seed.json",
}


def _ir(task):
    return load_task_ir(MINIWORLD / "task_ir" / (task + ".json"), REPOSITORY)


class GraftTests(unittest.TestCase):
    def setUp(self):
        self.registry = ValueTypeRegistry.from_yaml(
            REPOSITORY / "configs/synthesis/value_types_v0.yaml"
        )

    def test_rename_keeps_references_consistent(self):
        renamed = rename_ir(_ir("mw-001"), "x_")
        node_ids = {n["node_id"] for n in renamed["nodes"]}
        self.assertIn("x_n4", node_ids)
        n5 = next(n for n in renamed["nodes"] if n["node_id"] == "x_n5")
        self.assertIn('V["x_n3"]', n5["verifier"]["predicate"])
        self.assertEqual(n5["reads"][0]["entity_ref"], "derived:x_n1:event_id")
        self.assertTrue(all(e["from"]["node_id"].startswith("x_") for e in renamed["edges"]))
        validate_task_ir(renamed, REPOSITORY)

    def test_compat_index_finds_text_edges_and_graft_validates(self):
        a, b = _ir("mw-003"), _ir("mw-002")
        edges = build_compat_index([a, b], self.registry)
        text_edges = [
            e
            for e in edges
            if e["from"]["task_id"] == "mw-003"
            and e["to"]["task_id"] == "mw-002"
            and e["to"]["port_id"] == "title"
        ]
        self.assertTrue(text_edges, edges)
        composed = graft(a, b, text_edges[0])
        validate_task_ir(composed, REPOSITORY)
        self.assertEqual(len(composed["nodes"]), len(a["nodes"]) + len(b["nodes"]))
        binding = composed["provenance"]["composition"]["binding"]
        self.assertTrue(
            binding["from"]["node_id"].startswith("a_")
            and binding["to"]["node_id"].startswith("b_")
        )
        target = next(n for n in composed["nodes"] if n["node_id"] == binding["to"]["node_id"])
        port = next(p for p in target["inputs"] if p["port_id"] == binding["to"]["port_id"])
        self.assertNotIn("literal", port)
        self.assertEqual(port["source"], "upstream")
        self.assertIn(
            composed["provenance"]["composition"]["operator"],
            ("typed_grafting", "motif_repetition", "fan_in_extension", "delayed_reuse"),
        )
        self.assertEqual(len(output_ports(a)) + len(literal_input_ports(b)) > 0, True)

    def test_conditionalize_adds_control_gate(self):
        a = _ir("mw-002")
        decide = next(n for n in a["nodes"] if n["op"] == "decide")
        decide["produces"].append(
            {"name": "go", "type": "Boolean", "derivation": {"kind": "literal", "value": True}}
        )
        decide["outputs"].append(
            {
                "port_id": "go",
                "type": "Boolean",
                "grounding": "derived:%s:go" % decide["node_id"],
                "source": "upstream",
                "cardinality": "one",
            }
        )
        composed = conditionalize(a, _ir("mw-001"), decide["node_id"], "go")
        validate_task_ir(composed, REPOSITORY)
        control = [e for e in composed["edges"] if e["kind"] == "control_dependency"]
        self.assertEqual(len(control), 1)
        self.assertEqual(control[0]["to"]["node_id"], "b_n4")

    def test_beam_search_scores_and_gates(self):
        config = SearchConfig(beam_width=3, max_grafts=1, depth_range=(3, 12))
        results = beam_search([_ir("mw-001"), _ir("mw-002"), _ir("mw-003")], self.registry, config)
        self.assertTrue(results)
        best = results[0]
        self.assertGreater(best.score, 0)
        self.assertEqual(best.rejections, [])
        self.assertIn("composition", best.task_ir["provenance"])
        none = beam_search(
            [_ir("mw-001"), _ir("mw-002"), _ir("mw-003")],
            self.registry,
            config,
            executable=lambda ir: False,
        )
        self.assertEqual(none, [])


class HazardTests(unittest.TestCase):
    def test_base_rates_patch_and_record(self):
        config = HazardConfig.from_yaml(REPOSITORY / "configs/synthesis/hazards_v1.yaml")
        with tempfile.TemporaryDirectory(dir=os.environ.get("DERAIL_TMP_ROOT")) as tmp:
            build_world(SEEDS, Path(tmp))
            rates = base_rates(config, tmp)
        self.assertIn("identity_alias", rates)
        self.assertTrue(all("count" in u for u in rates["stale_version"]["units"]))
        patch = render_patch(
            config,
            "identity_alias",
            {
                "name": "Jim Halpert",
                "alias_email": "jim.halpert@warehouse.example",
                "app": "workbuzz",
            },
        )
        self.assertEqual(patch[0]["value"]["name"], "Jim Halpert")
        record = injection_record(
            config,
            "identity_alias",
            "mw-002",
            "miniworld",
            {"from_node_id": "n4", "to_node_id": "n5"},
            {"name": "Jim Halpert", "alias_email": "x@y.z", "app": "workbuzz"},
            rates,
        )
        from derail.derived.schema import validate_schema

        validate_schema(record, "hazard_injection.schema.json", REPOSITORY)
        self.assertEqual(
            record["status"], "pending" if injectable("identity_alias", rates) else "BASE_RATE_ZERO"
        )
        self.assertEqual(record["target_paper_type"], "misunderstand_task_objective")


class RealizationTests(unittest.TestCase):
    def setUp(self):
        self.config = RealizationConfig.from_yaml(
            REPOSITORY / "configs/synthesis/realization_v1.yaml", REPOSITORY
        )

    def test_summary_hides_gold_and_round_trip_flags_leaks(self):
        task_ir = _ir("mw-002")
        gold = {
            "values": [
                {"node_id": "n2", "name": "start_at", "value": "2026-07-02T18:00:00"},
                {
                    "node_id": "n3",
                    "name": "message",
                    "value": "Dundies is on 2026-07-02, save the date.",
                },
                {"node_id": "n1", "name": "event_id", "value": 545},
            ]
        }
        hidden = hidden_values(task_ir, gold)
        self.assertIn("2026-07-02T18:00:00", hidden)
        self.assertNotIn("THE DUNDIES", hidden)
        fields = build_realization_fields(
            task_ir, gold, style_stats_from_instructions(["Move my 1:1 with Jim to Tuesday."])
        )
        self.assertIn("communicate on workbuzz", fields["graph_summary"])
        self.assertIn("2026-07-02T18:00:00", fields["hidden_values"])
        good = (
            "Check when THE DUNDIES is on my calendar and DM Jim Halpert on HooliWork: "
            "Dundies is on <date>, save the date."
        )
        persona = ("Michael Scott", "michael.scott@dundermifflin.com")
        verdict = round_trip_compare(task_ir, task_ir, gold, good, self.config, persona)
        self.assertEqual(verdict["verdict"], "accepted", verdict)
        leaked = good.replace("<date>", "2026-07-02")
        self.assertIn(
            "2026-07-02T18:00:00",
            round_trip_compare(task_ir, task_ir, gold, leaked, self.config, persona)["gold_leaks"],
        )
        missing = "Tell Jim about the party."
        self.assertEqual(
            round_trip_compare(task_ir, None, gold, missing, self.config, persona)["verdict"],
            "re_realize",
        )
        parsed = parse_realization_reply('```json\n{"instruction": "x", "literals_used": []}\n```')
        self.assertEqual(parsed["instruction"], "x")
        summary = graph_summary(task_ir)
        self.assertIn("send_message(R3)", summary)
        self.assertNotIn("2026-07-02", summary)


if __name__ == "__main__":
    unittest.main()


class RebindTests(unittest.TestCase):
    def test_rebind_predicate_forms(self):
        from derail.gen.graft import rebind_predicate

        bindings = [
            ("b_n3__message", "Dundies is on 2026-07-02, save the date."),
            ("b_n2__start_at", "2026-07-02T18:00:00"),
            ("b_n4__dm_id", 1),
            ("b_n9__flag", True),
        ]
        predicate = (
            "SELECT count(*) > 0 FROM dm_messages WHERE dm_id = 1 AND id = 1 "
            "AND content = 'Dundies is on 2026-07-02, save the date.' "
            "AND content LIKE '%2026-07-02%' AND sender = 'x@y'"
        )
        out, used = rebind_predicate(predicate, bindings)
        self.assertIn("dm_id = :b_n4__dm_id", out)
        self.assertIn("id = 1 ", out)  # a column not named like a produce keeps its literal
        self.assertIn("content = :b_n3__message", out)
        self.assertIn("'%' || substr(:b_n2__start_at, 1, 10) || '%'", out)
        self.assertIn("sender = 'x@y'", out)
        self.assertEqual(sorted(set(used)), ["b_n2__start_at", "b_n3__message", "b_n4__dm_id"])

    def test_graft_with_gold_rebinds_and_drops_stale_verifiers(self):
        from derail.gen.graft import graft

        a, b = _ir("mw-003"), _ir("mw-002")
        registry = ValueTypeRegistry.from_yaml(REPOSITORY / "configs/synthesis/value_types_v0.yaml")
        edge = next(
            e
            for e in build_compat_index([a, b], registry)
            if e["from"]["task_id"] == "mw-003" and e["to"]["port_id"] == "title"
        )
        gold_b = {
            "values": [
                {"node_id": "n1", "name": "event_id", "value": 545},
                {"node_id": "n2", "name": "start_at", "value": "2026-07-02T18:00:00"},
                {"node_id": "n3", "name": "dundies_date", "value": "2026-07-02"},
                {"node_id": "n4", "name": "dm_id", "value": 1},
            ]
        }
        composed = graft(a, b, edge, gold_a={"values": []}, gold_b=gold_b)
        validate_task_ir(composed, REPOSITORY)
        final = composed["final_verifier"]
        self.assertEqual(final["kind"], "all_of")
        self.assertEqual(len(final["parts"]), 2)
        self.assertIn(":b_n3__dundies_date", final["parts"][1]["predicate"])
        note = composed["provenance"]["composition"]["verifier_rebinding"]
        self.assertIn("b_v-n1", note["refrozen_node_verifiers"])  # expected 545 no longer holds
        self.assertEqual(note["dropped_alternatives"], 1)  # '%July 2%' would accept stale text
        b_n1 = next(n for n in composed["nodes"] if n["node_id"] == "b_n1")
        self.assertEqual(b_n1["verifier"]["expected_ref"], "b_n1__event_id")
        from derail.gen.graft import freeze_verifiers

        frozen = freeze_verifiers(
            composed,
            {
                "initial_state_sha256": "x",
                "values": [
                    {"node_id": "b_n1", "name": "event_id", "value": 777},
                    {"node_id": "b_n3", "name": "dundies_date", "value": "2026-07-09"},
                ],
            },
        )
        validate_task_ir(frozen, REPOSITORY)
        self.assertIn("'2026-07-09'", frozen["final_verifier"]["parts"][1]["predicate"])
        self.assertNotIn(":b_n3__", frozen["final_verifier"]["parts"][1]["predicate"])
        b_n1 = next(n for n in frozen["nodes"] if n["node_id"] == "b_n1")
        self.assertEqual(b_n1["verifier"]["expected"], 777)
        self.assertNotIn("expected_ref", b_n1["verifier"])
        self.assertTrue(frozen["provenance"]["composition"]["verifier_rebinding"]["frozen"])


class GenericTypeTests(unittest.TestCase):
    def test_generic_scalar_types_need_cell_evidence(self):
        import yaml

        from derail.gen.compat import CompatConfig, build_compat_index

        sampling = yaml.safe_load(
            (REPOSITORY / "configs/synthesis/sampling_v1.yaml").read_text(encoding="utf-8")
        )
        config = SearchConfig.from_sampling(sampling).compat
        self.assertIn("Integer", config.generic_types)
        registry = ValueTypeRegistry.from_yaml(REPOSITORY / "configs/synthesis/value_types_v0.yaml")
        irs = [_ir("mw-001"), _ir("mw-002"), _ir("mw-003"), _ir("mw-004")]
        loose = build_compat_index(irs, registry, None, CompatConfig())
        strict = build_compat_index(irs, registry, None, config)
        self.assertTrue(any(e["from"]["type"] == "Text" for e in loose))
        self.assertFalse(any(e["from"]["type"] in config.generic_types for e in strict))
        self.assertLess(len(strict), len(loose))


class CellGroundingTests(unittest.TestCase):
    def test_sql_cells_and_persona_exclusion(self):
        from derail.gen.compat import (
            CompatConfig,
            literal_input_ports,
            output_ports,
            sql_param_cell,
            sql_scalar_cell,
        )

        self.assertEqual(
            sql_scalar_cell("SELECT e.title AS name FROM events e WHERE id = :id", "hoolicalendar"),
            ("hoolicalendar.events", "title"),
        )
        self.assertIsNone(sql_scalar_cell("SELECT COUNT(*) FROM events", "hoolicalendar"))
        self.assertIsNone(sql_scalar_cell("SELECT title, start_at FROM events", "hoolicalendar"))
        self.assertEqual(
            sql_param_cell(
                "SELECT id FROM members WHERE lower(display_name) LIKE :jim_name",
                "workbuzz",
                "jim_name",
            ),
            ("workbuzz.members", "display_name"),
        )
        self.assertEqual(
            sql_param_cell(
                "SELECT id FROM bookings WHERE user_email = :user_email AND "
                "(lower(location) LIKE '%' || lower(:city) || '%' OR lower(address) LIKE :x)",
                "cheskepdia",
                "city",
            ),
            ("cheskepdia.bookings", "location"),
        )
        ir = {
            "task_id": "t",
            "nodes": [
                {
                    "node_id": "n1",
                    "app": "workbuzz",
                    "inputs": [
                        {
                            "port_id": "user_email",
                            "type": "EmailAddress",
                            "literal": "michael.scott@dundermifflin.com",
                        },
                        {"port_id": "jim_name", "type": "Text", "literal": "Jim"},
                    ],
                    "produces": [
                        {
                            "name": "member_name",
                            "type": "Text",
                            "derivation": {
                                "kind": "sql",
                                "query": (
                                    "SELECT display_name FROM members "
                                    "WHERE display_name = :jim_name"
                                ),
                            },
                        }
                    ],
                    "outputs": [{"port_id": "member_name", "type": "Text"}],
                    "reads": [],
                }
            ],
        }
        targets = literal_input_ports(ir, ("michael.scott@dundermifflin.com",))
        self.assertEqual([t.port_id for t in targets], ["jim_name"])
        self.assertEqual(targets[0].cell, ("workbuzz.members", "display_name"))
        self.assertEqual(output_ports(ir)[0].cell, ("workbuzz.members", "display_name"))
        self.assertEqual(CompatConfig().excluded_literals, ())


class SeedFreezeTests(unittest.TestCase):
    def test_upstream_references_freeze_and_sinks_stay_live(self):
        from derail.gen.graft import freeze_verifiers

        ir = _ir("mw-002")
        ir = json.loads(json.dumps(ir))
        ir["final_verifier"] = {
            "verifier_id": "final",
            "kind": "derived",
            "predicate": (
                'V["n5"]["message_id"] > 0 and '
                'V["n3"]["message"] == fmt("Dundies is on {}", V["n2"]["start_at"])'
            ),
            "observability": "derived_from_observable",
        }
        ir["nodes"][4]["verifier"] = {
            "verifier_id": "v-n5",
            "kind": "sql",
            "predicate": (
                "SELECT content = :message AND dm_id = :dm_id "
                "FROM dm_messages WHERE id = :message_id"
            ),
            "observability": "environment_state",
        }
        gold = {
            "initial_state_sha256": "x",
            "values": [
                {"node_id": "n2", "name": "start_at", "value": "2026-07-02T18:00:00"},
                {"node_id": "n3", "name": "message", "value": "Dundies is on 2026-07-02T18:00:00"},
                {"node_id": "n4", "name": "dm_id", "value": 1},
                {"node_id": "n5", "name": "message_id", "value": 9},
            ],
        }
        frozen = freeze_verifiers(ir, gold)
        final = frozen["final_verifier"]["predicate"]
        self.assertIn("'2026-07-02T18:00:00'", final)  # upstream fact frozen
        self.assertIn('V["n5"]["message_id"]', final)  # sink output stays live
        self.assertNotIn('V["n2"]', final)
        n5 = frozen["nodes"][4]["verifier"]["predicate"]
        self.assertIn("dm_id = 1", n5)  # upstream parameter frozen
        self.assertIn("content = 'Dundies is on 2026-07-02T18:00:00'", n5)
        self.assertIn(":message_id", n5)  # the verified node's own value stays live
        self.assertEqual(frozen["provenance"]["verifier_freeze"]["sink_nodes"], ["n5"])


class HiddenValueTests(unittest.TestCase):
    def test_public_names_years_and_short_words_are_not_hidden(self):
        ir = {
            "task_id": "t",
            "instruction": "Check the Q2 team morale initiative deadlines.",
            "nodes": [{"node_id": "n1", "inputs": [{"port_id": "p", "literal": "NYC"}]}],
        }
        gold = {
            "values": [
                {"node_id": "n1", "name": "project", "value": "Team Morale Initiative Q2"},
                {"node_id": "n1", "name": "hotel", "value": "The Greenwich Hotel"},
                {"node_id": "n1", "name": "year", "value": 2026},
                {"node_id": "n1", "name": "dept", "value": "sales"},
                {"node_id": "n1", "name": "when", "value": "7:30am"},
                {"node_id": "n1", "name": "total", "value": 12140},
            ]
        }
        self.assertEqual(hidden_values(ir, gold), ["The Greenwich Hotel", "7:30am", "12140"])
        self.assertEqual(
            hidden_values(ir, gold, ["I stay at The Greenwich Hotel"]), ["7:30am", "12140"]
        )


class CompositionPaperTests(unittest.TestCase):
    def _with_decide(self):
        a = _ir("mw-002")
        decide = next(n for n in a["nodes"] if n["op"] == "decide")
        decide["produces"].append(
            {"name": "go", "type": "Boolean", "derivation": {"kind": "literal", "value": True}}
        )
        return a, decide["node_id"]

    def test_conditionalize_concatenates_both_final_verifiers(self):
        a, decide = self._with_decide()
        b = _ir("mw-001")
        composed = conditionalize(a, b, decide, "go")
        final = composed["final_verifier"]
        self.assertEqual(final["kind"], "all_of")
        self.assertEqual(len(final["parts"]), 2)
        self.assertEqual(final["parts"][0], rename_ir(a, "a_")["final_verifier"])
        receiver = conditionalize(a, b, decide, "go", final_verifier="receiver")
        self.assertEqual(receiver["final_verifier"], rename_ir(b, "b_")["final_verifier"])
        prefixes = composed["provenance"]["composition"]["source_prefixes"]
        self.assertEqual(
            prefixes,
            [
                {"task_id": "mw-002", "node_prefix": "a_"},
                {"task_id": "mw-001", "node_prefix": "b_"},
            ],
        )

    def test_rename_recurses_into_all_of_and_qualified_params(self):
        ir = _ir("mw-001")
        ir["final_verifier"] = {
            "verifier_id": "f",
            "kind": "all_of",
            "parts": [
                {"kind": "derived", "predicate": 'V["n1"]["event_id"] > 0'},
                {"kind": "sql", "predicate": "SELECT 1 WHERE x = :n2__start_at"},
            ],
        }
        final = rename_ir(ir, "a_")["final_verifier"]
        self.assertNotIn("predicate", final)
        self.assertIn('V["a_n1"]', final["parts"][0]["predicate"])
        self.assertIn(":a_n2__start_at", final["parts"][1]["predicate"])

    def test_compat_requires_an_identified_column(self):
        from derail.gen.compat import CompatConfig, PortRef, port_compatibility

        registry = ValueTypeRegistry.from_yaml(REPOSITORY / "configs/synthesis/value_types_v0.yaml")
        cell = ("hoolicalendar.events", "title")
        src = PortRef("a", "n1", "title", "EventTitle", "one", None, "hoolicalendar")
        dst = PortRef("b", "n1", "title", "EventTitle", "one", cell, "hoolicalendar")
        self.assertIsNone(port_compatibility(src, dst, registry, None, CompatConfig()))
        loose = CompatConfig(require_cell=False)
        self.assertIsNotNone(port_compatibility(src, dst, registry, None, loose))

    def test_composed_rubric_updates_expected_values_to_composed_gold(self):
        from derail.gen.graft import compose_rubric
        from derail.ir.rubric_check import RubricCheckConfig

        config = RubricCheckConfig.from_yaml(REPOSITORY / "configs/synthesis/rubric_check_v1.yaml")
        source_b = {
            "task_id": "B",
            "nodes": [
                {"node_id": "n1", "inputs": [{"port_id": "title", "literal": "Budget Review"}]},
                {"node_id": "n2", "inputs": []},
            ],
        }
        composed = {
            "task_id": "gen-x",
            "nodes": [
                {"node_id": "a_n1", "inputs": []},
                {
                    "node_id": "b_n1",
                    "inputs": [{"port_id": "title", "grounding": "derived:a_n1:title"}],
                },
                {"node_id": "b_n2", "inputs": []},
            ],
            "provenance": {
                "composition": {
                    "source_task_ids": ["A", "B"],
                    "source_prefixes": [
                        {"task_id": "A", "node_prefix": "a_"},
                        {"task_id": "B", "node_prefix": "b_"},
                    ],
                }
            },
        }
        gold = {
            "values": [
                {"node_id": "a_n1", "name": "title", "value": "Q3 Planning"},
                {"node_id": "b_n2", "name": "total", "value": 95.5},
            ]
        }
        sources = {
            "A": {"task_ir": None, "gold": None, "rubrics": [{"criterion": "Find", "weight": 0.4}]},
            "B": {
                "task_ir": source_b,
                "gold": {"values": [{"node_id": "n2", "name": "total", "value": 120.0}]},
                "rubrics": [
                    {"criterion": 'Opens "Budget Review" and reports $120.00', "weight": 0.6}
                ],
            },
        }
        rubric = compose_rubric(composed, gold, sources, config, 4)
        self.assertEqual(rubric["source_task_ids"], ["A", "B"])
        self.assertEqual([i["weight"] for i in rubric["rubrics"]], [0.4, 0.6])
        item = rubric["rubrics"][1]
        self.assertEqual(item["criterion"], 'Opens "Q3 Planning" and reports $95.50')
        self.assertTrue(item["updated"])
        self.assertEqual({e["value"] for e in item["expected"]}, {"Q3 Planning", 95.5})
        self.assertFalse(rubric["rubrics"][0]["updated"])
