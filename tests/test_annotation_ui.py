"""Annotation UI service: provenance filling, derived horizon, immutability, path guards."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
# The UI derives the audited prefix from the deepest depth in DEPTH_GRID that still fits the
# trajectory, so the fixture has to be long enough for a non-degenerate window.  With the root
# cause at action 1 and the grid's shallowest non-zero point at 5, that needs at least 7 actions.
_FIXTURE_ACTIONS = 8
TRAJ_SHA = "c" * 64
AFTER_SHA = "d" * 64
BUNDLE_SHA = "e" * 64


def _load_service_module():
    """The UI server is an unnumbered dev script, so import it by path rather than by package."""

    path = REPOSITORY / "scripts" / "serve_annotation_ui.py"
    spec = importlib.util.spec_from_file_location("derail_serve_annotation_ui", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


UI = _load_service_module()


def _action(index: int, after_uri: str) -> dict:
    return {
        "step_id": index,
        "action_index_global": index,
        "turn_index": index,
        "action_index_within_turn": 0,
        "observation_before_sha256": "" if index == 0 else AFTER_SHA,
        "observation_before_uri": "" if index == 0 else after_uri,
        "action": {
            "kind": "click",
            "x_px": 10,
            "y_px": 20,
            "button": "left",
            "frame_width": 1280,
            "frame_height": 800,
        },
        "action_summary": "Click at (10, 20).",
        "observation_after_sha256": AFTER_SHA,
        "observation_after_uri": after_uri,
        "tool_result": "ok",
        "source_agent": "gpt_5_5",
        "source_step_id": index,
        "source_action_timestamp": "20260807@02340%d000000" % index,
        "source_record_uri": "/raw/traj.jsonl#line=%d" % (index + 1),
        "repaired": False,
        "repair_patch_id": "",
    }


class AnnotationUIServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.collection = root / "raw"
        (self.collection / "task").mkdir(parents=True)
        (self.collection / "_tasks").mkdir()
        (self.collection / "_tasks" / "batch.json").write_text(
            json.dumps(
                [
                    {
                        "id": "task-a",
                        "category": "retrieval",
                        "instruction": "Find the requested document.",
                        "grading": {
                            "type": "llm_judge",
                            "rubrics": [
                                {
                                    "rubric_id": "R1",
                                    "requirement": "The requested document is opened.",
                                    "weight": 1.0,
                                }
                            ],
                        },
                    }
                ]
            ),
            encoding="utf-8",
        )
        self.screenshot = self.collection / "task" / "step_1.png"
        self.screenshot.write_bytes(b"\x89PNG\r\n\x1a\n")
        self.outside = root / "outside.png"
        self.outside.write_bytes(b"\x89PNG\r\n\x1a\n")
        self.unreferenced = self.collection / "task" / "unreferenced.png"
        self.unreferenced.write_bytes(b"\x89PNG\r\n\x1a\n")

        self.build_dir = root / "builds" / "b1"
        (self.build_dir / "canonical" / "traj-a").mkdir(parents=True)
        (self.build_dir / "build_manifest.json").write_text(
            json.dumps(
                {
                    "build_id": "b1",
                    "source_collection": {"collection_id": "c1", "uri": str(self.collection)},
                    "taxonomy": {
                        "version": "draft",
                        "mode": "open_coding",
                        "seed_labels": ["grounding_failure", "wrong_subgoal"],
                        "uri": "",
                        "sha256": "",
                    },
                }
            ),
            encoding="utf-8",
        )
        actions = [_action(i, str(self.screenshot)) for i in range(_FIXTURE_ACTIONS)]
        for index, action in enumerate(actions, 1):
            action["source_record_uri"] = str(self.collection / "task-a" / "traj.jsonl") + (
                "#line=%d" % index
            )
        (self.build_dir / "canonical" / "traj-a" / "annotation_task.json").write_text(
            json.dumps(
                {
                    "trajectory_id": "traj-a",
                    "source_trajectory_sha256": TRAJ_SHA,
                    "normalization_gate_passed": True,
                    "actions": actions,
                }
            ),
            encoding="utf-8",
        )
        (self.build_dir / "canonical" / "traj-a" / "normalization_report.json").write_text(
            json.dumps({"task_provenance": {"rubric_bundle_sha256": BUNDLE_SHA}}),
            encoding="utf-8",
        )
        self.out_dir = root / "annotations"
        self.out_dir.mkdir()
        self.service = UI.AnnotationService(self.build_dir, self.out_dir, REPOSITORY)

    def _payload(self, **overrides) -> dict:
        payload = {
            "trajectory_id": "traj-a",
            "annotator_id": "zhang",
            "root_cause_action_index": 1,
            "identifiable_at_action_index": 3,
            "error_types": ["grounding_failure"],
            "reversibility": "reversible",
            "rationale": "点到了旁边的控件。",
            "cleaning_review_complete": True,
            "cleaning_drop_candidates": [],
        }
        payload.update(overrides)
        return payload

    def test_submit_derives_horizon_and_fills_provenance(self) -> None:
        target = self.service.submit(self._payload(), allow_resubmit=False)
        record = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(record["error_horizon_actions"], 2)
        self.assertEqual(record["identifiable_at_action_index"], 3)
        self.assertEqual(record["source_trajectory_sha256"], TRAJ_SHA)
        self.assertEqual(record["taxonomy_version"], "draft")
        self.assertEqual(record["annotator_role"], "human")
        self.assertEqual(record["annotation_id"], "traj-a__zhang")
        self.assertEqual(target.name, "traj-a__zhang.json")
        proposal = json.loads(
            (self.out_dir / "cleaning_proposals" / "traj-a__zhang.json").read_text(
                encoding="utf-8"
            )
        )
        # Root cause at 1 plus the deepest grid point that fits the 8-action fixture (5).
        self.assertEqual(proposal["audit_end_action_index"], 6)
        self.assertEqual(proposal["audited_action_indices"], [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(proposal["drop_candidates"], [])

    def test_raw_truncation_is_a_hint_not_a_forced_human_label(self) -> None:
        source_path = self.collection / "task-a" / "traj.jsonl"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(
            "".join(
                json.dumps(
                    {
                        "step_num": index + 1,
                        "action": "pyautogui.click(10, 20)",
                        "response": "",
                        "done": False,
                    }
                )
                + "\n"
                for index in range(_FIXTURE_ACTIONS)
            ),
            encoding="utf-8",
        )

        target = self.service.submit(self._payload(), allow_resubmit=False)
        record = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(record["error_types"], ["grounding_failure"])
        self.assertEqual(
            self.service.task("traj-a")["derived_error_types"],
            ["hit_budget_limit"],
        )

    def test_terminal_predict_crash_does_not_add_budget_label(self) -> None:
        source_path = self.collection / "task-a" / "traj.jsonl"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "step_num": index + 1,
                "action": "PREDICT_CRASH" if index == _FIXTURE_ACTIONS - 1 else "pyautogui.click(10, 20)",
                "response": "harness error" if index == _FIXTURE_ACTIONS - 1 else "",
                "done": index == _FIXTURE_ACTIONS - 1,
            }
            for index in range(_FIXTURE_ACTIONS)
        ]
        source_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

        target = self.service.submit(self._payload(), allow_resubmit=False)
        record = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(record["error_types"], ["grounding_failure"])
        self.assertEqual(self.service.task("traj-a")["derived_error_types"], [])

    def test_human_budget_label_is_preserved_for_bash_budget_abort(self) -> None:
        source_path = self.collection / "task-a" / "traj.jsonl"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "step_num": index + 1,
                "action": "FAIL" if index == _FIXTURE_ACTIONS - 1 else "pyautogui.click(10, 20)",
                "response": '{"abort":{"type":"BASH_BUDGET_ABORT"}}' if index == _FIXTURE_ACTIONS - 1 else "",
                "done": index == _FIXTURE_ACTIONS - 1,
            }
            for index in range(_FIXTURE_ACTIONS)
        ]
        source_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

        payload = self._payload()
        payload["error_types"].append("hit_budget_limit")
        target = self.service.submit(payload, allow_resubmit=False)
        record = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(
            record["error_types"], ["grounding_failure", "hit_budget_limit"]
        )
        self.assertEqual(
            self.service.task("traj-a")["derived_error_types"],
            ["hit_budget_limit"],
        )

    def test_submit_review_saves_success_without_failure_annotation(self) -> None:
        result = self.service.submit_review(
            {
                "trajectory_id": "traj-a",
                "annotator_id": "zhang",
                "rubric_scores": {"R1": 1},
                "task_success": True,
            },
            allow_resubmit=False,
        )
        self.assertIsNone(result["annotation_path"])
        self.assertIsNone(result["cleaning_proposal_path"])
        review = json.loads(result["rubric_score_path"].read_text(encoding="utf-8"))
        self.assertEqual(review["scores"], {"R1": 1})
        self.assertTrue(review["task_success"])
        self.assertEqual(self.service.config()["trajectories"][0]["annotated_by"], ["zhang"])

    def test_submit_review_saves_scores_and_failure_annotation(self) -> None:
        result = self.service.submit_review(
            self._payload(rubric_scores={"R1": 0}, task_success=False),
            allow_resubmit=False,
        )
        self.assertTrue(result["annotation_path"].is_file())
        self.assertTrue(result["cleaning_proposal_path"].is_file())
        review = json.loads(result["rubric_score_path"].read_text(encoding="utf-8"))
        self.assertEqual(review["scores"], {"R1": 0})
        self.assertFalse(review["task_success"])

    def test_server_draft_round_trip_accepts_incomplete_work(self) -> None:
        target = self.service.save_draft(
            {
                "trajectory_id": "traj-a",
                "annotator_id": "zhang",
                "draft": {
                    "root": None,
                    "types": [],
                    "rationale": "还在检查",
                    "cleanReviewed": False,
                },
                "rubric_scores": {},
            }
        )
        self.assertEqual(target.name, "traj-a__zhang.json")
        self.assertEqual(target.parent.name, "drafts")
        recovered = self.service.draft("traj-a", "zhang")
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered["draft"]["rationale"], "还在检查")
        self.assertEqual(recovered["rubric_scores"], {})
        self.assertTrue(recovered["saved_at"].endswith("+00:00"))

    def test_final_submission_supersedes_server_draft(self) -> None:
        self.service.save_draft(
            {
                "trajectory_id": "traj-a",
                "annotator_id": "zhang",
                "draft": {"rationale": "unfinished"},
                "rubric_scores": {"R1": 0},
            }
        )
        self.service.submit_review(
            self._payload(rubric_scores={"R1": 0}, task_success=False),
            allow_resubmit=False,
        )
        self.assertIsNone(self.service.draft("traj-a", "zhang"))

    def test_submission_is_empty_before_anything_is_submitted(self) -> None:
        self.assertEqual(
            self.service.submission("traj-a", "zhang"),
            {
                "rubric_scores": None,
                "annotation": None,
                "cleaning_proposal": None,
                "rollout_flag": None,
            },
        )

    def test_submission_reads_back_a_finished_review(self) -> None:
        self.service.submit_review(
            self._payload(rubric_scores={"R1": 0}, task_success=False),
            allow_resubmit=False,
        )
        submission = self.service.submission("traj-a", "zhang")
        self.assertEqual(submission["annotation"]["root_cause_action_index"], 1)
        self.assertEqual(submission["annotation"]["identifiable_at_action_index"], 3)
        self.assertEqual(submission["annotation"]["error_types"], ["grounding_failure"])
        self.assertEqual(submission["rubric_scores"]["scores"], {"R1": 0})
        self.assertFalse(submission["rubric_scores"]["task_success"])
        self.assertTrue(submission["cleaning_proposal"]["review_complete"])
        self.assertIsNone(submission["rollout_flag"])

    def test_submission_reads_back_a_wrong_rollout_flag(self) -> None:
        self.service.submit_review(
            {"trajectory_id": "traj-a", "annotator_id": "zhang", "wrong_rollout": True},
            allow_resubmit=False,
        )
        submission = self.service.submission("traj-a", "zhang")
        self.assertEqual(submission["rollout_flag"]["rollout_status"], "needs_rerun")
        self.assertIsNone(submission["annotation"])
        self.assertIsNone(submission["rubric_scores"])

    def test_submission_only_returns_the_requesting_annotator(self) -> None:
        self.service.submit_review(
            self._payload(rubric_scores={"R1": 0}, task_success=False),
            allow_resubmit=False,
        )
        other = self.service.submission("traj-a", "li")
        self.assertEqual(set(other.values()), {None})

    def test_reading_a_submission_never_touches_the_label_tree(self) -> None:
        """Reading back a submission must leave every annotator's file byte-identical."""

        self.service.submit_review(
            self._payload(rubric_scores={"R1": 0}, task_success=False),
            allow_resubmit=False,
        )

        def snapshot() -> dict:
            return {
                str(path.relative_to(self.out_dir)): path.read_bytes()
                for path in sorted(self.out_dir.rglob("*"))
                if path.is_file()
            }

        before = snapshot()
        self.assertTrue(before)
        self.service.submission("traj-a", "zhang")
        self.service.submission("traj-a", "li")
        self.assertEqual(snapshot(), before)

    def test_clear_review_archives_only_current_annotator_and_task(self) -> None:
        review_id = "traj-a__zhang"
        other_id = "traj-a__li"
        targets = [
            self.out_dir / (review_id + ".json"),
            self.out_dir / "drafts" / (review_id + ".json"),
            self.out_dir / "rubric_scores" / (review_id + ".json"),
            self.out_dir / "cleaning_proposals" / (review_id + ".json"),
            self.out_dir / "rollout_flags" / (review_id + ".json"),
        ]
        other = self.out_dir / "rollout_flags" / (other_id + ".json")
        for path in [*targets, other]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")

        result = self.service.clear_review(
            {"trajectory_id": "traj-a", "annotator_id": "zhang"}
        )

        self.assertEqual(len(result["cleared"]), 5)
        self.assertTrue(result["archive_path"])
        self.assertTrue(all(not path.exists() for path in targets))
        self.assertTrue(other.is_file())
        archived = Path(result["archive_path"])
        self.assertEqual(len(list(archived.rglob("*.json"))), 5)

    def test_clear_review_is_idempotent_when_nothing_is_saved(self) -> None:
        result = self.service.clear_review(
            {"trajectory_id": "traj-a", "annotator_id": "zhang"}
        )
        self.assertEqual(result, {"cleared": [], "archive_path": None})

    def test_export_results_are_compact_and_include_weighted_score(self) -> None:
        batch_path = self.collection / "_tasks" / "batch.json"
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        batch[0]["grading"]["rubrics"] = [
            {"rubric_id": "R1", "requirement": "First criterion.", "weight": 0.25},
            {"rubric_id": "R2", "requirement": "Second criterion.", "weight": 0.75},
        ]
        batch_path.write_text(json.dumps(batch), encoding="utf-8")
        self.service.submit_review(
            self._payload(rubric_scores={"R1": 1, "R2": 0}, task_success=False),
            allow_resubmit=False,
        )
        records = self.service.export_results("zhang")
        self.assertEqual(len(records), 1)
        self.assertEqual(
            set(records[0]),
            {
                "trajectory_id",
                "annotator_id",
                "scores",
                "weighted_rubric_score",
                "task_score",
                "source_trajectory_sha256",
                "rubric_bundle_sha256",
                "failure_annotation",
                "rollout_status",
            },
        )
        self.assertEqual(records[0]["trajectory_id"], "traj-a")
        self.assertEqual(records[0]["annotator_id"], "zhang")
        self.assertEqual(records[0]["scores"], {"R1": 1, "R2": 0})
        self.assertEqual(records[0]["weighted_rubric_score"], 0.25)
        self.assertEqual(records[0]["task_score"], 0)
        self.assertEqual(records[0]["source_trajectory_sha256"], TRAJ_SHA)
        self.assertEqual(records[0]["rubric_bundle_sha256"], BUNDLE_SHA)
        self.assertEqual(records[0]["rollout_status"], "valid")
        self.assertEqual(
            records[0]["failure_annotation"]["root_cause_action_index"], 1
        )

    def test_export_results_jsonl_writes_local_file(self) -> None:
        self.service.submit_review(
            self._payload(rubric_scores={"R1": 0}, task_success=False),
            allow_resubmit=False,
        )

        result = self.service.export_results_jsonl("zhang")

        target = result["path"]
        self.assertEqual(result["record_count"], 1)
        self.assertEqual(target.parent, self.out_dir / "exports")
        self.assertTrue(target.name.startswith("b1_human_label_results_zhang_"))
        rows = [
            json.loads(line)
            for line in target.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["trajectory_id"], "traj-a")
        self.assertEqual(rows[0]["failure_annotation"]["annotator_id"], "zhang")

    def test_export_results_ignores_submissions_outside_active_builds(self) -> None:
        self.service.submit_review(
            self._payload(rubric_scores={"R1": 0}, task_success=False),
            allow_resubmit=False,
        )
        review_path = self.out_dir / "rubric_scores" / "traj-a__zhang.json"
        stale_review = json.loads(review_path.read_text(encoding="utf-8"))
        stale_review["trajectory_id"] = "retired-traj"
        stale_review["review_id"] = "retired-traj__zhang"
        (review_path.parent / "retired-traj__zhang.json").write_text(
            json.dumps(stale_review), encoding="utf-8"
        )

        records = self.service.export_results(
            "zhang", active_trajectory_ids={"traj-a"}
        )

        self.assertEqual([record["trajectory_id"] for record in records], ["traj-a"])

    def test_compact_success_export_has_binary_and_weighted_scores(self) -> None:
        self.service.submit_review(
            {
                "trajectory_id": "traj-a",
                "annotator_id": "zhang",
                "rubric_scores": {"R1": 1},
                "task_success": True,
            },
            allow_resubmit=False,
        )
        record = self.service.export_results("zhang")[0]
        self.assertEqual(record["weighted_rubric_score"], 1.0)
        self.assertEqual(record["task_score"], 1)
        self.assertIsNone(record["failure_annotation"])
        self.assertEqual(record["rollout_status"], "valid")

    def test_wrong_rollout_requires_no_scores_and_is_exported_for_rerun(self) -> None:
        task_path = self.build_dir / "canonical" / "traj-a" / "annotation_task.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task["normalization_gate_passed"] = False
        task_path.write_text(json.dumps(task), encoding="utf-8")

        result = self.service.submit_review(
            {
                "trajectory_id": "traj-a",
                "annotator_id": "zhang",
                "wrong_rollout": True,
            },
            allow_resubmit=False,
        )
        self.assertIsNone(result["rubric_score_path"])
        self.assertIsNone(result["annotation_path"])
        flag = json.loads(result["rollout_flag_path"].read_text(encoding="utf-8"))
        self.assertEqual(flag["rollout_status"], "needs_rerun")
        self.assertEqual(flag["reason"], "wrong_rollout")

        config = self.service.config()["trajectories"][0]
        self.assertEqual(config["annotated_by"], ["zhang"])
        self.assertEqual(config["rerun_requested_by"], ["zhang"])
        record = self.service.export_results("zhang")[0]
        self.assertEqual(record["rollout_status"], "needs_rerun")
        self.assertIsNone(record["scores"])
        self.assertIsNone(record["weighted_rubric_score"])
        self.assertIsNone(record["task_score"])

    def test_latest_submission_replaces_review_with_wrong_rollout_and_back(self) -> None:
        review_result = self.service.submit_review(
            {
                "trajectory_id": "traj-a",
                "annotator_id": "zhang",
                "rubric_scores": {"R1": 1},
                "task_success": True,
            },
            allow_resubmit=False,
        )
        flag_result = self.service.submit_review(
            {
                "trajectory_id": "traj-a",
                "annotator_id": "zhang",
                "wrong_rollout": True,
            },
            allow_resubmit=False,
        )
        self.assertFalse(review_result["rubric_score_path"].exists())
        self.assertTrue(flag_result["rollout_flag_path"].is_file())
        records = self.service.export_results("zhang")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["rollout_status"], "needs_rerun")

        latest = self.service.submit_review(
            {
                "trajectory_id": "traj-a",
                "annotator_id": "zhang",
                "rubric_scores": {"R1": 1},
                "task_success": True,
            },
            allow_resubmit=False,
        )
        self.assertFalse(flag_result["rollout_flag_path"].exists())
        self.assertTrue(latest["rubric_score_path"].is_file())
        self.assertEqual(self.service.export_results("zhang")[0]["rollout_status"], "valid")

    def test_latest_success_removes_stale_failure_and_cleaning_files(self) -> None:
        failed = self.service.submit_review(
            self._payload(rubric_scores={"R1": 0}, task_success=False),
            allow_resubmit=False,
        )
        annotation_path = failed["annotation_path"]
        cleaning_path = failed["cleaning_proposal_path"]
        self.assertTrue(annotation_path.is_file())
        self.assertTrue(cleaning_path.is_file())

        latest = self.service.submit_review(
            {
                "trajectory_id": "traj-a",
                "annotator_id": "zhang",
                "rubric_scores": {"R1": 1},
                "task_success": True,
            },
            allow_resubmit=False,
        )
        self.assertFalse(annotation_path.exists())
        self.assertFalse(cleaning_path.exists())
        review = json.loads(latest["rubric_score_path"].read_text(encoding="utf-8"))
        self.assertEqual(review["scores"], {"R1": 1})
        self.assertTrue(review["task_success"])
        self.assertIn("submitted_at", review)

    def test_invalid_resubmission_keeps_previous_valid_review(self) -> None:
        result = self.service.submit_review(
            {
                "trajectory_id": "traj-a",
                "annotator_id": "zhang",
                "rubric_scores": {"R1": 1},
                "task_success": True,
            },
            allow_resubmit=False,
        )
        before = result["rubric_score_path"].read_bytes()
        with self.assertRaises(UI.UIError):
            self.service.submit_review(
                {
                    "trajectory_id": "traj-a",
                    "annotator_id": "zhang",
                    "rubric_scores": {},
                    "task_success": False,
                },
                allow_resubmit=False,
            )
        self.assertEqual(result["rubric_score_path"].read_bytes(), before)

    def test_third_annotator_adjudicates_two_reviews_and_protocol_files_surface(self) -> None:
        for annotator, identifiable in (("zhang", 3), ("li", 4)):
            self.service.submit_review(
                self._payload(
                    annotator_id=annotator,
                    identifiable_at_action_index=identifiable,
                    rubric_scores={"R1": 0},
                    task_success=False,
                ),
                allow_resubmit=False,
            )
        adjudication = {
            "trajectory_id": "traj-a",
            "annotator_id": "wang",
            "task_success": False,
            "resolution_rationale": "The error is visible at action 4.",
            "failure_adjudication": {
                "input_annotation_ids": ["traj-a__li", "traj-a__zhang"],
                "root_cause_action_index": 1,
                "error_horizon_actions": 3,
                "identifiable_at_action_index": 4,
                "error_types": ["grounding_failure"],
                "reversibility": "reversible",
                "disagreement_fields": ["identifiable_at_action_index"],
                "evidence_refs": ["action:4"],
                "taxonomy_version": "draft",
            },
        }
        with self.assertRaises(UI.UIError):
            self.service.submit_adjudication({**adjudication, "annotator_id": "zhang"})
        record = json.loads(self.service.submit_adjudication(adjudication).read_text("utf-8"))
        self.assertEqual(record["input_review_ids"], ["traj-a__li", "traj-a__zhang"])
        self.assertEqual(record["failure_adjudication"]["identifiable_at_action_index"], 4)

        (self.out_dir / "protocol").mkdir()
        (self.out_dir / "protocol" / "double_annotation_plan.json").write_text(
            json.dumps({"trajectories": [{"trajectory_id": "traj-a"}]}), encoding="utf-8"
        )
        (self.out_dir / "auto_analysis").mkdir()
        (self.out_dir / "auto_analysis" / "traj-a.json").write_text(
            json.dumps({"trajectory_id": "traj-a", "review_route": "human_verify"}),
            encoding="utf-8",
        )
        listed = self.service.config()["trajectories"][0]
        self.assertTrue(listed["double_annotation"])
        self.assertEqual(listed["adjudicated_by"], ["wang"])
        self.assertEqual(self.service.task("traj-a")["auto_proposal"]["review_route"], "human_verify")

    def test_missing_horizon_keeps_both_indices_null(self) -> None:
        target = self.service.submit(
            self._payload(annotator_id="li", identifiable_at_action_index=None),
            allow_resubmit=False,
        )
        record = json.loads(target.read_text(encoding="utf-8"))
        self.assertIsNone(record["error_horizon_actions"])
        self.assertIsNone(record["identifiable_at_action_index"])

    def test_identifiable_before_root_is_rejected(self) -> None:
        with self.assertRaises(Exception):
            self.service.submit(
                self._payload(root_cause_action_index=3, identifiable_at_action_index=1),
                allow_resubmit=False,
            )

    def test_cleaning_proposal_pins_old_action_and_rejects_root_drop(self) -> None:
        target = self.service.submit(
            self._payload(
                annotator_id="li",
                cleaning_drop_candidates=[
                    {
                        "action_index_global": 2,
                        "recovery_action_index": 3,
                        "reason": "Recovered at the next action.",
                    }
                ],
            ),
            allow_resubmit=False,
        )
        proposal = json.loads(
            (target.parent / "cleaning_proposals" / target.name).read_text(encoding="utf-8")
        )
        candidate = proposal["drop_candidates"][0]
        self.assertEqual(candidate["action_index_global"], 2)
        self.assertEqual(candidate["recovery_action_index"], 3)
        self.assertEqual(candidate["old_action"], _action(2, str(self.screenshot))["action"])
        self.assertTrue(candidate["self_recovered"])
        self.assertFalse(candidate["persistent_state_effect"])
        self.assertFalse(candidate["causal_to_root_or_task"])

        with self.assertRaises(UI.UIError):
            self.service.submit(
                self._payload(
                    annotator_id="wang",
                    cleaning_drop_candidates=[
                        {
                            "action_index_global": 1,
                            "recovery_action_index": 2,
                            "reason": "Cannot remove the root.",
                        }
                    ],
                ),
                allow_resubmit=False,
            )

    def test_cleaning_review_is_required(self) -> None:
        with self.assertRaises(UI.UIError):
            self.service.submit(
                self._payload(annotator_id="li", cleaning_review_complete=False),
                allow_resubmit=False,
            )

    def test_cleaning_recovery_must_be_retained_inside_the_audited_prefix(self) -> None:
        invalid_recoveries = (2, 4)
        for offset, recovery in enumerate(invalid_recoveries):
            with self.subTest(recovery=recovery), self.assertRaises(UI.UIError):
                self.service.submit(
                    self._payload(
                        annotator_id="reviewer%d" % offset,
                        cleaning_drop_candidates=[
                            {
                                "action_index_global": 2,
                                "recovery_action_index": recovery,
                                "reason": "Invalid recovery boundary.",
                            }
                        ],
                    ),
                    allow_resubmit=False,
                )
        target = self.service.submit(
            self._payload(
                annotator_id="rootrecovery",
                cleaning_drop_candidates=[
                    {
                        "action_index_global": 0,
                        "recovery_action_index": 1,
                        "reason": "The retained root action is the recovery boundary.",
                    }
                ],
            ),
            allow_resubmit=False,
        )
        proposal = json.loads(
            (target.parent / "cleaning_proposals" / target.name).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(proposal["drop_candidates"][0]["recovery_action_index"], 1)

    def test_cleaning_range_must_be_contiguous_and_keep_recovery(self) -> None:
        with self.assertRaises(UI.UIError):
            self.service.submit(
                self._payload(
                    annotator_id="gap",
                    cleaning_drop_candidates=[
                        {
                            "action_index_global": 0,
                            "recovery_action_index": 3,
                            "reason": "This leaves an unexplained gap.",
                        },
                        {
                            "action_index_global": 2,
                            "recovery_action_index": 3,
                            "reason": "This leaves an unexplained gap.",
                        },
                    ],
                ),
                allow_resubmit=False,
            )

    def test_root_beyond_trajectory_is_rejected(self) -> None:
        from derail.annotation.records import AnnotationError

        with self.assertRaises(AnnotationError):
            self.service.submit(
                self._payload(root_cause_action_index=9, identifiable_at_action_index=None),
                allow_resubmit=False,
            )

    def test_existing_annotation_is_immutable_unless_resubmit(self) -> None:
        self.service.submit(self._payload(), allow_resubmit=False)
        with self.assertRaises(UI.ConflictError):
            self.service.submit(self._payload(rationale="改一下"), allow_resubmit=False)
        target = self.service.submit(self._payload(rationale="改一下"), allow_resubmit=True)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["rationale"], "改一下")

    def test_annotator_and_trajectory_ids_reject_path_components(self) -> None:
        for field in ("annotator_id", "trajectory_id"):
            with self.assertRaises(UI.UIError):
                self.service.submit(self._payload(**{field: "../etc"}), allow_resubmit=False)

    def test_task_rewrites_observation_uris_to_image_urls(self) -> None:
        task = self.service.task("traj-a")
        self.assertEqual(task["actions"][0]["observation_before_url"], "")
        self.assertTrue(task["actions"][0]["observation_after_url"].startswith("api/image?p="))
        self.assertTrue(task["actions"][1]["observation_before_url"].startswith("api/image?p="))
        self.assertEqual(task["task_instruction"], "Find the requested document.")
        self.assertEqual(
            task["task_rubrics"],
            [
                {
                    "rubric_id": "R1",
                    "requirement": "The requested document is opened.",
                    "weight": 1.0,
                }
            ],
        )

    def test_task_includes_exact_source_traj_action_and_response_for_each_step(self) -> None:
        source_path = self.collection / "task-a" / "traj.jsonl"
        source_path.parent.mkdir(parents=True)
        records = [
            {
                "step_num": index + 1,
                "action_timestamp": "20260807@02340%d000000" % index,
                "action": "pyautogui.click(%d, 20)" % (10 + index),
                "response": "model response for step %d" % (index + 1),
                "reward": 0.0,
                "done": index == _FIXTURE_ACTIONS - 1,
                "info": {"source_index": index},
                "agent_metadata": None,
                "screenshot_file": "step_%d.png" % (index + 1),
            }
            for index in range(_FIXTURE_ACTIONS)
        ]
        source_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

        task = self.service.task("traj-a")
        for index, action in enumerate(task["actions"]):
            self.assertEqual(action["source_traj_record"], records[index])
            self.assertNotIn("source_traj_error", action)

    def test_image_route_is_confined_to_collection_and_build(self) -> None:
        body, mime = self.service.image(str(self.screenshot))
        self.assertEqual(mime, "image/png")
        self.assertTrue(body.startswith(b"\x89PNG"))
        with self.assertRaises(UI.UIError):
            self.service.image(str(self.outside))
        with self.assertRaises(UI.UIError):
            self.service.image(str(self.unreferenced))
        with self.assertRaises(UI.UIError):
            self.service.image(str(self.collection / "task" / ".." / ".." / "outside.png"))
        with self.assertRaises(UI.UIError):
            self.service.image("relative.png")

    def test_config_reports_progress_and_open_coded_labels(self) -> None:
        self.service.submit(self._payload(error_types=["grounding_failure", "menu_drift"]),
                            allow_resubmit=False)
        config = self.service.config()
        self.assertEqual(config["known_extra_labels"], ["menu_drift"])
        self.assertEqual(config["open_coded_label_groups"]["others"], ["menu_drift"])
        self.assertEqual(config["trajectories"][0]["annotated_by"], ["zhang"])
        self.assertEqual(config["trajectories"][0]["action_count"], _FIXTURE_ACTIONS)
        self.assertTrue(config["trajectories"][0]["normalization_gate_passed"])
        self.assertEqual(config["trajectories"][0]["agent_id"], "gpt_5_5")
        self.assertEqual(config["trajectories"][0]["task_id"], "task-a")
        self.assertEqual(config["trajectories"][0]["task_category"], "retrieval")
        self.assertEqual(
            config["trajectories"][0]["task_instruction"],
            "Find the requested document.",
        )
        # No readable snapshot pinned, so the grouped view must degrade to the flat seed list.
        self.assertEqual(list(config["taxonomy"]["groups"]), ["seed_labels"])

    def test_open_coded_label_registry_is_shared_and_categorized(self) -> None:
        created = self.service.add_open_coded_label(
            {
                "annotator_id": "zhang",
                "label": "Missed required comparison",
                "category": "planning",
                "description_zh": "遗漏了任务明确要求进行的比较或对照分析。",
            }
        )
        self.assertEqual(
            created,
            {
                "label": "missed_required_comparison",
                "category": "planning",
                "description_zh": "遗漏了任务明确要求进行的比较或对照分析。",
            },
        )
        registry_path = self.out_dir / "taxonomy" / "open_coded_labels.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        metadata = registry["labels"]["missed_required_comparison"]
        self.assertEqual(metadata["category"], "planning")
        self.assertEqual(metadata["created_by"], "zhang")
        self.assertEqual(
            metadata["description_zh"],
            "遗漏了任务明确要求进行的比较或对照分析。",
        )

        # A newly constructed service represents another annotator/browser and reads the same
        # out-dir registry rather than relying on local browser state.
        other_service = UI.AnnotationService(self.build_dir, self.out_dir, REPOSITORY)
        config = other_service.config()
        self.assertIn(
            "missed_required_comparison",
            config["open_coded_label_groups"]["planning"],
        )
        self.assertIn("missed_required_comparison", config["known_extra_labels"])
        self.assertEqual(
            config["open_coded_label_descriptions"]["missed_required_comparison"],
            "遗漏了任务明确要求进行的比较或对照分析。",
        )

    def test_open_coded_label_can_be_recategorized_and_soft_deleted(self) -> None:
        self.service.add_open_coded_label(
            {
                "annotator_id": "zhang",
                "label": "menu drift",
                "category": "perception",
                "description_zh": "误判菜单展开后目标选项所在的位置。",
            }
        )
        annotation = self.service.submit(
            self._payload(
                annotator_id="li",
                error_types=["grounding_failure", "menu_drift"],
            ),
            allow_resubmit=False,
        )
        original_annotation = annotation.read_bytes()

        updated = self.service.update_open_coded_label(
            {
                "annotator_id": "wang",
                "label": "menu_drift",
                "category": "execution",
            }
        )
        self.assertEqual(updated["category"], "execution")
        self.assertIn(
            "menu_drift", self.service.open_coded_label_groups()["execution"]
        )
        registry_path = self.out_dir / "taxonomy" / "open_coded_labels.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        metadata = registry["labels"]["menu_drift"]
        self.assertEqual(metadata["updated_by"], "wang")
        self.assertTrue(metadata["updated_at"].endswith("+00:00"))

        deleted = self.service.delete_open_coded_label(
            {"annotator_id": "wang", "label": "menu_drift"}
        )
        self.assertTrue(deleted["deleted"])
        groups = self.service.open_coded_label_groups()
        self.assertNotIn("menu_drift", [label for labels in groups.values() for label in labels])
        self.assertNotIn("menu_drift", self.service.open_coded_label_descriptions())
        self.assertEqual(annotation.read_bytes(), original_annotation)
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        self.assertNotIn("menu_drift", registry["labels"])
        self.assertEqual(
            registry["deleted_labels"]["menu_drift"]["last_category"],
            "execution",
        )

    def test_seed_error_type_category_and_description_are_shared_overrides(self) -> None:
        updated = self.service.update_open_coded_label(
            {
                "annotator_id": "zhang",
                "label": "grounding_failure",
                "category": "planning",
                "description_zh": "模型知道目标，但界面定位或操作落点不准确。",
            }
        )
        self.assertEqual(updated["category"], "planning")
        self.assertEqual(
            updated["description_zh"],
            "模型知道目标，但界面定位或操作落点不准确。",
        )
        self.assertNotIn(
            "grounding_failure",
            [
                label
                for labels in self.service.open_coded_label_groups().values()
                for label in labels
            ],
        )
        self.assertNotIn(
            "grounding_failure", self.service.open_coded_label_descriptions()
        )
        self.assertEqual(
            self.service.error_type_overrides()["grounding_failure"],
            {
                "category": "planning",
                "description_zh": "模型知道目标，但界面定位或操作落点不准确。",
            },
        )
        self.assertEqual(
            self.service.config()["error_type_overrides"]["grounding_failure"][
                "category"
            ],
            "planning",
        )
        with self.assertRaises(UI.UIError):
            self.service.delete_open_coded_label(
                {"annotator_id": "zhang", "label": "grounding_failure"}
            )

    def test_open_coded_label_registration_is_idempotent_but_category_is_stable(self) -> None:
        payload = {
            "annotator_id": "zhang",
            "label": "menu drift",
            "category": "perception",
            "description_zh": "误判菜单展开后目标选项所在的位置。",
        }
        first = self.service.add_open_coded_label(payload)
        second = self.service.add_open_coded_label(
            {**payload, "annotator_id": "li"}
        )
        self.assertEqual(first, second)
        with self.assertRaises(UI.ConflictError):
            self.service.add_open_coded_label(
                {**payload, "annotator_id": "li", "category": "execution"}
            )
        with self.assertRaises(UI.ConflictError):
            self.service.add_open_coded_label(
                {
                    **payload,
                    "annotator_id": "li",
                    "description_zh": "这是另一个不同的中文定义。",
                }
            )

    def test_open_coded_label_requires_supported_category(self) -> None:
        with self.assertRaises(UI.UIError):
            self.service.add_open_coded_label(
                {
                    "annotator_id": "zhang",
                    "label": "menu drift",
                    "category": "uncategorized",
                    "description_zh": "误判菜单展开后目标选项所在的位置。",
                }
            )

    def test_open_coded_label_requires_chinese_description(self) -> None:
        for description in ("", "Wrong menu state"):
            with self.subTest(description=description), self.assertRaises(UI.UIError):
                self.service.add_open_coded_label(
                    {
                        "annotator_id": "zhang",
                        "label": "menu drift",
                        "category": "perception",
                        "description_zh": description,
                    }
                )

    def test_multi_build_service_combines_agents_and_routes_trajectories(self) -> None:
        second_build = self.build_dir.parent / "b2"
        trajectory_dir = second_build / "canonical" / "traj-b"
        trajectory_dir.mkdir(parents=True)
        manifest = json.loads(
            (self.build_dir / "build_manifest.json").read_text(encoding="utf-8")
        )
        manifest["build_id"] = "b2"
        (second_build / "build_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        actions = [_action(index, str(self.screenshot)) for index in range(_FIXTURE_ACTIONS)]
        for action in actions:
            action["source_agent"] = "evocua_32b"
        (trajectory_dir / "annotation_task.json").write_text(
            json.dumps(
                {
                    "trajectory_id": "traj-b",
                    "source_trajectory_sha256": "e" * 64,
                    "normalization_gate_passed": True,
                    "task_rubrics": [
                        {
                            "rubric_id": "R1",
                            "requirement": "The requested document is opened.",
                            "weight": 1.0,
                        }
                    ],
                    "actions": actions,
                }
            ),
            encoding="utf-8",
        )
        second_service = UI.AnnotationService(
            second_build, self.out_dir, REPOSITORY
        )
        service = UI.MultiBuildAnnotationService([self.service, second_service])

        config = service.config()
        self.assertEqual(config["build_ids"], ["b1", "b2"])
        self.assertEqual(len(config["trajectories"]), 2)
        self.assertEqual(
            {item["agent_id"] for item in config["trajectories"]},
            {"gpt_5_5", "evocua_32b"},
        )
        self.assertEqual(service.task("traj-b")["trajectory_id"], "traj-b")
        self.assertTrue(service.image(str(self.screenshot))[0].startswith(b"\x89PNG"))
        draft = service.save_draft(
            {
                "trajectory_id": "traj-b",
                "annotator_id": "zhang",
                "draft": {"rationale": "checking second build"},
                "rubric_scores": {},
            }
        )
        self.assertEqual(draft.name, "traj-b__zhang.json")
        self.assertEqual(
            service.draft("traj-b", "zhang")["draft"]["rationale"],
            "checking second build",
        )

    def test_failed_normalization_gate_blocks_submission(self) -> None:
        path = self.build_dir / "canonical" / "traj-a" / "annotation_task.json"
        task = json.loads(path.read_text(encoding="utf-8"))
        task["normalization_gate_passed"] = False
        path.write_text(json.dumps(task), encoding="utf-8")
        with self.assertRaises(UI.UIError):
            self.service.submit(self._payload(), allow_resubmit=False)


class TaxonomyGroupingTest(unittest.TestCase):
    def test_shipped_taxonomy_groups_match_the_flattened_seed_labels(self) -> None:
        from derail.derived.layout import sha256_file

        path = REPOSITORY / "prompts" / "annotation" / "taxonomy.yaml"
        seed = sorted(
            line.strip()[2:].strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("- ")
        )
        groups = UI._taxonomy_groups(
            {"seed_labels": seed, "uri": str(path), "sha256": sha256_file(path)}
        )
        self.assertEqual(list(groups), ["planning", "perception", "execution", "termination"])
        self.assertEqual(sorted(l for v in groups.values() for l in v), seed)
        self.assertIn("fabricate_data", groups["planning"])
        self.assertNotIn("wrong_subgoal", groups["planning"])

    def test_changed_snapshot_falls_back_to_the_flat_seed_list(self) -> None:
        path = REPOSITORY / "prompts" / "annotation" / "taxonomy.yaml"
        groups = UI._taxonomy_groups(
            {"seed_labels": ["a", "b"], "uri": str(path), "sha256": "0" * 64}
        )
        self.assertEqual(groups, {"seed_labels": ["a", "b"]})


if __name__ == "__main__":
    unittest.main()
