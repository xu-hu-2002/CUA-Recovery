"""Tests for post-error continuation, case instantiation, dedup and the precheck report."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from derail.derived.schema import validate_schema
from derail.longhorizon.cases import (
    DUPLICATE_CASE,
    INSUFFICIENT_POST_ERROR_STEPS,
    CaseCandidate,
    FailureSource,
    FunnelRow,
    case_yield,
    dedup_cases,
    depth_availability,
    funnel_by_agent,
    instantiate_depths,
    reversibility_stratum,
)
from derail.longhorizon.continuation import LoopConfig, action_signature, compute_continuation
from derail.longhorizon.ontology import Ontology
from derail.longhorizon.precheck import (
    PrecheckConfig,
    build_report,
    read_failure_rows,
    render_markdown,
    run_precheck,
)
from derail.longhorizon.taxonomy import FailureTaxonomy

REPOSITORY = Path(__file__).resolve().parents[1]
ONTOLOGY = Ontology.from_yaml(REPOSITORY / "configs/synthesis/ontology_v0.2.yaml")


def _click(x, y):
    return {
        "kind": "click",
        "x_px": x,
        "y_px": y,
        "button": "left",
        "target": "",
        "frame_width": 1280,
        "frame_height": 800,
    }


def _terminate(status="success"):
    return {"kind": "terminate", "status": status, "answer": ""}


class ContinuationTests(unittest.TestCase):
    def test_signature_quantizes_pixels_and_drops_frame(self):
        self.assertEqual(action_signature(_click(100, 50)), action_signature(_click(105, 60)))
        self.assertNotEqual(action_signature(_click(100, 50)), action_signature(_click(300, 50)))

    def test_loop_and_termination(self):
        actions = [
            _click(10, 10),
            _click(20, 20),
            _click(500, 500),
            _click(500, 502),
            {"kind": "wait", "seconds": 1},
            _click(501, 499),
            _click(30, 30),
            _terminate(),
        ]
        stats = compute_continuation(
            actions, 1, loop=LoopConfig(min_repeats=3), early_stop_window=2
        )
        self.assertEqual(stats.post_error_steps, 6)
        self.assertTrue(stats.terminated_explicitly)
        self.assertEqual(stats.terminate_status, "success")
        self.assertTrue(stats.loop_detected)
        self.assertEqual(stats.loop_repeat_count, 3)
        self.assertEqual(stats.loop_start_offset, 1)
        self.assertEqual(stats.loop_established_offset, 4)
        self.assertFalse(stats.early_stop)

    def test_budget_exhaustion_and_early_stop(self):
        actions = [_click(1, 1)] * 3 + [_click(99, 99)]
        stats = compute_continuation(actions, 2, loop=LoopConfig(min_repeats=3))
        self.assertFalse(stats.terminated_explicitly)
        self.assertTrue(stats.early_stop)
        self.assertFalse(stats.loop_detected)
        with self.assertRaises(ValueError):
            compute_continuation(actions, 4)


class CaseTests(unittest.TestCase):
    def _source(
        self,
        rollout="roll_a",
        agent="agent_a",
        root=10,
        steps=9,
        paper_type="scope_error",
        root_action=None,
    ):
        return FailureSource(
            task_id="aggregation-f001",
            source_rollout_id=rollout,
            source_agent=agent,
            root_cause_action_index=root,
            root_action=root_action or _click(100, 100),
            paper_type=paper_type,
            paper_category="planning",
            group="long_horizon",
            reversibility_stratum="reversible",
            post_error_steps_available=steps,
        )

    def test_depth_availability_never_pads(self):
        self.assertEqual(
            depth_availability(9, (0, 2, 8, 20)),
            {0: "available", 2: "available", 8: "available", 20: INSUFFICIENT_POST_ERROR_STEPS},
        )

    def test_stratum_from_effects(self):
        self.assertEqual(
            reversibility_stratum([{"reversibility_class": "R1"}], ONTOLOGY), ("reversible", False)
        )
        self.assertEqual(
            reversibility_stratum([{"reversibility_class": "R2"}], ONTOLOGY), ("reversible", True)
        )
        self.assertEqual(
            reversibility_stratum(
                [{"reversibility_class": "R2"}, {"reversibility_class": "R3"}], ONTOLOGY
            ),
            ("irreversible", True),
        )

    def test_instantiate_and_records_validate(self):
        cases, skipped = instantiate_depths(self._source(), (0, 2, 8, 20))
        self.assertEqual([case.error_depth for case in cases], [0, 2, 8])
        self.assertEqual(skipped, {20: INSUFFICIENT_POST_ERROR_STEPS})
        for case in cases:
            validate_schema(case.to_record(), "derail_case.schema.json", REPOSITORY)
        with self.assertRaises(ValueError):
            CaseCandidate(**{**cases[2].__dict__, "post_error_steps_available": 1})

    def test_dedup_merges_across_agents_and_keeps_strongest(self):
        a, _ = instantiate_depths(
            self._source(rollout="roll_a", agent="agent_a", root=10, steps=9), (0, 2)
        )
        b, _ = instantiate_depths(
            self._source(rollout="roll_b", agent="agent_b", root=11, steps=30), (0, 2)
        )
        c, _ = instantiate_depths(
            self._source(rollout="roll_c", agent="agent_c", root=13, steps=30), (0, 2)
        )
        far, _ = instantiate_depths(
            self._source(
                rollout="roll_d", agent="agent_d", root=11, steps=5, root_action=_click(900, 700)
            ),
            (0, 2),
        )
        kept, removed = dedup_cases(a + b + c + far)
        kept_ids = sorted(case.case_id for case in kept)
        self.assertEqual(len(kept), 6)
        self.assertTrue(all(case.status == DUPLICATE_CASE for case in removed))
        keeper = next(
            case for case in kept if case.source_rollout_id == "roll_b" and case.error_depth == 0
        )
        self.assertEqual(keeper.merged_from, ("derail_roll_a_scope_error_d0",))
        self.assertNotIn("derail_roll_a_scope_error_d0", kept_ids)

    def test_funnel_and_yield(self):
        rows = [
            FunnelRow("x", failed=True, typed=True, repaired=True, verified=True, case_count=3),
            FunnelRow("x", failed=True, typed=False),
            FunnelRow("x"),
            FunnelRow("y", failed=True, typed=True, repaired=False),
        ]
        self.assertEqual(
            funnel_by_agent(rows)["x"],
            {"runs": 3, "failed": 2, "typed": 1, "repaired": 1, "verified": 1, "cases": 3},
        )
        self.assertEqual(funnel_by_agent(rows)["y"]["repaired"], 0)
        cases, _ = instantiate_depths(self._source(), (0, 2, 8))
        self.assertEqual(case_yield(cases, (0, 2, 8)), {"planning": {0: 1, 2: 1, 8: 1}})


class PrecheckTests(unittest.TestCase):
    def _write_corpus(self, root: Path):
        actions_a = [_click(10, 10)] * 12 + [_click(700, 600)] * 4 + [_terminate()]
        actions_b = [_click(10, 10)] * 25
        rows = []
        for traj, model, annotator, actions, root_index, labels, state in (
            (
                "agentA-r1-vm0-aggregation-f001",
                "agent_a",
                "ann1",
                actions_a,
                10,
                "premature_completion|scope_error",
                "failure",
            ),
            (
                "agentB-r1-vm0-aggregation-f001",
                "agent_b",
                "ann2",
                actions_b,
                11,
                "scope_error",
                "failure",
            ),
            ("agentA-r1-vm0-aggregation-f002", "agent_a", "ann1", actions_a, 0, "", "perfect_pass"),
            ("agentB-r1-vm0-aggregation-f002", "agent_b", "ann2", actions_b, 0, "", "n/a"),
        ):
            traj_dir = root / "canonical" / traj
            traj_dir.mkdir(parents=True)
            with (traj_dir / "trajectory.jsonl").open("w") as handle:
                for index, action in enumerate(actions):
                    handle.write(
                        json.dumps({"action_index_global": index, "action": action}) + "\n"
                    )
            task_id = traj.split("-", 3)[3]
            (traj_dir / "task_config.json").write_text(
                json.dumps({"id": task_id, "category": "aggregation"})
            )
            annotation_path = ""
            if state == "failure":
                annotation_path = root / "human_labels" / ("%s__%s.json" % (traj, annotator))
                annotation_path.parent.mkdir(exist_ok=True)
                annotation_path.write_text(
                    json.dumps(
                        {
                            "reversibility": "reversible",
                            "identifiable_at_action_index": root_index + 3,
                            "root_cause_action_index": root_index,
                        }
                    )
                )
            rows.append(
                {
                    "trajectory_id": traj,
                    "model": model,
                    "annotator": annotator,
                    "state": state,
                    "error_types": labels,
                    "root_cause_action_index": str(root_index) if state == "failure" else "",
                    "trajectory_path": str(traj_dir / "trajectory.jsonl"),
                    "task_config_path": str(traj_dir / "task_config.json"),
                    "annotation_path": str(annotation_path),
                    "rationale": "a, b, c",
                }
            )
        with (root / "labels.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        bucket = root / "tasks" / "aggregation"
        bucket.mkdir(parents=True)
        (bucket / "aggregation.rubrics.json").write_text(
            json.dumps([{"id": "aggregation-f001", "horizon": "H3"}])
        )
        (root / "precheck.yaml").write_text(
            "\n".join(
                [
                    "labels_csv: labels.csv",
                    "human_labels_dir: human_labels",
                    "taxonomy_config: %s"
                    % (REPOSITORY / "configs/synthesis/failure_taxonomy_v0.1.yaml"),
                    "output_dir: out",
                    "depth_grid: [0, 2, 8, 20]",
                    "comparison_depth_grids: {alt: [0, 2, 5]}",
                    "loop_detection: {min_repeats: 3, click_cell_px: 40}",
                    "early_stop_window_steps: 5",
                    "task_horizon_dir: tasks",
                ]
            )
        )

    def test_end_to_end_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_corpus(root)
            config = PrecheckConfig.from_yaml(root / "precheck.yaml", root)
            self.assertEqual(len(read_failure_rows(config.labels_csv)), 2)
            manifest = run_precheck(config, repo_root=root, command="test")
            self.assertEqual(manifest["counts"]["records"], 2)
            report = json.loads((root / "out" / "post_error_continuation_report.json").read_text())
            overall = report["overall"]
            self.assertEqual(overall["n"], 2)
            self.assertEqual(overall["depth_candidates"]["primary"]["20"]["candidates"], 0)
            self.assertEqual(overall["depth_candidates"]["primary"]["8"]["candidates"], 1)
            self.assertEqual(overall["loop_detected"], 2)
            self.assertEqual(overall["terminated_explicitly"], 1)
            self.assertEqual(report["primary_type_counts"], {"scope_error": 2})
            self.assertEqual(report["case_level"]["primary"]["dedup_removed"], 2)
            records = [
                json.loads(line)
                for line in (root / "out" / "failure_continuation_records.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(records[0]["task_horizon_bucket"], "H3")
            markdown = (root / "out" / "post_error_continuation_report.md").read_text()
            self.assertIn("## By agent", markdown)
            self.assertEqual(render_markdown(report), markdown)
            taxonomy = FailureTaxonomy.from_yaml(config.taxonomy_config)
            self.assertEqual(build_report([], config, taxonomy)["overall"]["n"], 0)


if __name__ == "__main__":
    unittest.main()
