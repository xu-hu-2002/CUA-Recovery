"""Synthetic end-to-end proof for the DERAIL derived-data workflow."""

import base64
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from derail.adapters.base import HistoryStep
from derail.adapters.holo31 import Holo31Adapter
from derail.annotation.labels import Reversibility
from derail.annotation.records import Adjudication, HumanAnnotation
from derail.annotation.taxonomy import error_types_outside_seed, summarize_open_codes
from derail.canonical.actions import ClickAction, ScrollAction, SequenceAction, TerminateAction
from derail.canonical.mypcbench import (
    NormalizationError,
    PyAutoGUINormalizer,
    _editor_replay_command,
    _is_assistant_only_tool_call,
    load_canonical_jsonl,
    normalize_task_directory,
)
from derail.construction.cases import build_case_plan
from derail.construction.repair import PrefixAudit, PrefixRepairError, RepairPatch
from derail.derived.layout import (
    DerivedBuild,
    atomic_write_jsonl,
    sha256_file,
    tree_fingerprint,
)
from derail.evaluation.records import (
    ErrorAwarenessJudgment,
    EvaluationEpisode,
    EvaluationRecordError,
    PostErrorOutcome,
)
from derail.evaluation.takeover import TakeoverTurn, run_takeover_episode
from derail.replay.verification import (
    DeterministicSyntheticBackend,
    ReplayPlan,
    ReplayVerification,
    StateFingerprint,
    execute_replay_plan,
)
from derail.takeover.history import build_native_history_artifact


def _write_minimal_png(path: Path, width: int = 1280, height: int = 800) -> None:
    # The normalizer only reads the PNG signature and IHDR dimensions; pixel bytes stay raw-only.
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
    )


class DerivedWorkflowTests(unittest.TestCase):
    def test_qwen35_terminal_tool_call_row_is_not_a_missing_shell_call(self) -> None:
        terminal = {
            "action": "TOOL_CALL",
            "response": "Reasoning complete.</think>\n\nTask completed successfully.",
        }
        malformed = {"action": "TOOL_CALL", "response": "<tool_call>broken</tool_call>"}
        self.assertTrue(_is_assistant_only_tool_call(terminal, "qwen3_5_35b_a3b"))
        self.assertTrue(_is_assistant_only_tool_call(malformed, "qwen3_5_35b_a3b"))
        self.assertFalse(_is_assistant_only_tool_call(terminal, "gpt_5_5"))

    def test_anthropic_editor_replay_is_exact_and_bounded(self) -> None:
        create = _editor_replay_command(
            {"command": "create", "path": "/tmp/a.txt", "file_text": "hello ' world"}
        )
        replace = _editor_replay_command(
            {
                "command": "str_replace",
                "path": "/tmp/a.txt",
                "old_str": "hello",
                "new_str": "goodbye",
            }
        )
        self.assertIn("write_bytes", create)
        self.assertIn("s.count(o)==1", replace)
        self.assertEqual(
            _editor_replay_command({"command": "view", "path": "/tmp/a.txt"}),
            "test -e /tmp/a.txt",
        )

    def _make_raw(self, root: Path) -> tuple[Path, Path]:
        collection = root / "raw" / "collection-synthetic"
        task = collection / "agent" / "repeat_1" / "vm0" / "task-1"
        task.mkdir(parents=True)
        task_config = {
            "id": "task-1",
            "instruction": "Complete the synthetic desktop task.",
            "grading": {
                "type": "llm_judge",
                "rubrics": [{"rubric_id": "r1", "requirement": "done", "weight": 1.0}],
            },
        }
        batch_dir = task.parent / "_tasks"
        batch_dir.mkdir()
        (batch_dir / "batch.json").write_text(json.dumps([task_config]), encoding="utf-8")
        (task / "rubric_bundle.json").write_text(
            json.dumps({"task": task_config}), encoding="utf-8"
        )
        (collection / "collection_manifest.json").write_text(
            json.dumps({"collection_id": "collection-synthetic", "max_steps": 100}),
            encoding="utf-8",
        )
        rows = []
        for action_index in range(24):
            # action 0 and 1 share one model turn, proving turn != executed action.
            turn = 1 if action_index < 2 else action_index
            screenshot = "step_%d.png" % action_index
            _write_minimal_png(task / screenshot)
            rows.append(
                {
                    "step_num": turn,
                    "action_timestamp": "t%02d" % action_index,
                    "action": "pyautogui.click(%d, 100)" % (10 + action_index),
                    "response": "raw response remains in raw tree",
                    "reward": 0,
                    "done": False,
                    "info": {},
                    "screenshot_file": screenshot,
                }
            )
        (task / "traj.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return collection, task

    def test_tool_rows_are_annotation_complete_without_becoming_replay_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _collection, task = self._make_raw(root)
            actions = (
                "TOOL_CALL",
                "pyautogui.sleep(0.1)",
                "pyautogui.moveTo(864, 511); pyautogui.mouseDown()",
                "DONE",
            )
            rows = []
            for index, action in enumerate(actions, start=1):
                screenshot = "claude_step_%d.png" % index
                _write_minimal_png(task / screenshot)
                rows.append(
                    {
                        "step_num": index,
                        "action_timestamp": "claude-%02d" % index,
                        "action": action,
                        "response": "tool transcript" if action == "TOOL_CALL" else "",
                        "reward": 0,
                        "done": action == "DONE",
                        "info": {},
                        "screenshot_file": screenshot,
                    }
                )
            (task / "traj.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            output = root / "canonical-tool-rows"
            report = normalize_task_directory(
                task,
                output,
                trajectory_id="claude-tool-rows",
                source_agent="claude",
                wait_seconds=1.0,
            )
            self.assertFalse(report["normalization_complete"])
            self.assertTrue(report["annotation_complete"])
            self.assertFalse(report["case_eligible"])
            self.assertEqual(report["source_record_count"], 4)
            self.assertEqual(report["canonical_action_count"], 4)
            self.assertEqual(
                [issue["code"] for issue in report["issues"]],
                ["non_desktop_tool_call"],
            )
            steps = load_canonical_jsonl(output / "trajectory.jsonl")
            self.assertEqual(
                [step.action.kind for step in steps],
                ["no_op", "wait", "sequence", "terminate"],
            )

    def test_single_task_object_is_valid_provenance_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _collection, task = self._make_raw(root)
            task_config = root / "task_config.json"
            task_config.write_text(
                json.dumps({"id": task.name, "instruction": "Complete the task."}),
                encoding="utf-8",
            )

            report = normalize_task_directory(
                task,
                root / "canonical-single-task",
                trajectory_id="single-task-provenance",
                source_agent="test-agent",
                wait_seconds=1.0,
                task_config_source=task_config,
            )

            self.assertTrue(report["normalization_complete"])
            self.assertEqual(report["task_provenance"]["task_id"], task.name)

    def test_openai_bash_rows_use_exact_messages_json_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _collection, task = self._make_raw(root)
            response = {
                "content": "",
                "tool_calls": [
                    {
                        "id": "bash_0",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps({"command": "printf exact"}),
                        },
                    }
                ],
            }
            row = {
                "step_num": 1,
                "action_timestamp": "bash-01",
                "action": "TOOL_CALL",
                "response": json.dumps(response),
                "reward": 0,
                "done": False,
                "info": {},
                "screenshot_file": "",
            }
            (task / "traj.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            exact_result = "exit code: 0\nstdout:\nexact\nstderr:\n(empty)"
            (task / "messages.json").write_text(
                json.dumps(
                    [
                        {"role": "tool", "tool_call_id": "hotkey_3", "content": "first"},
                        {"role": "tool", "tool_call_id": "hotkey_3", "content": "second"},
                        {
                            "role": "assistant",
                            "content": "(tool call)",
                            "tool_calls": response["tool_calls"],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "bash_0",
                            "content": exact_result,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            output = root / "canonical-openai-bash"
            report = normalize_task_directory(
                task,
                output,
                trajectory_id="kimi-openai-bash",
                source_agent="kimi_k3",
                wait_seconds=1.0,
            )

            self.assertTrue(report["normalization_complete"])
            step = load_canonical_jsonl(output / "trajectory.jsonl")[0]
            self.assertEqual(step.action.kind, "shell")
            self.assertEqual(step.action.commands, ("printf exact",))
            self.assertEqual(
                json.loads(step.tool_result)["shell_results"][0]["result"], exact_result
            )

    def test_anthropic_bash_row_uses_native_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _collection, task = self._make_raw(root)
            row = {
                "step_num": 1,
                "action_timestamp": "bash-01",
                "action": "TOOL_CALL",
                "response": "I will inspect the file.",
                "reward": 0,
                "done": False,
                "info": {"kind": "tool_call_round"},
                "screenshot_file": "",
            }
            (task / "traj.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (task / "messages.json").write_text(
                json.dumps(
                    [
                        {"role": "user", "content": [{"type": "text", "text": "task"}]},
                        {
                            "role": "assistant",
                            "content": [{
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "bash",
                                "input": {"command": "printf exact"},
                            }],
                        },
                        {
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "exact",
                                "is_error": False,
                            }],
                        },
                    ]
                ),
                encoding="utf-8",
            )

            output = root / "canonical-anthropic-bash"
            report = normalize_task_directory(
                task,
                output,
                trajectory_id="claude-anthropic-bash",
                source_agent="claude_opus_4_8",
                wait_seconds=1.0,
            )

            self.assertTrue(report["normalization_complete"])
            step = load_canonical_jsonl(output / "trajectory.jsonl")[0]
            self.assertEqual(step.action.commands, ("printf exact",))
            result = json.loads(step.tool_result)["shell_results"][0]["result"]
            self.assertEqual(result, {"content": "exact", "is_error": False})

    def test_qwen35_xml_command_is_canonical_shell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _collection, task = self._make_raw(root)
            response = (
                "Action: inspect\n<tool_call><function=computer_use>"
                "<parameter=command>printf exact</parameter></function></tool_call>"
            )
            row = {
                "step_num": 1,
                "action_timestamp": "shell-01",
                "action": "TOOL_CALL",
                "response": response,
                "reward": 0,
                "done": False,
                "info": {"kind": "tool_call_round"},
                "screenshot_file": "",
            }
            (task / "traj.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            output = root / "canonical-qwen35-shell"
            report = normalize_task_directory(
                task,
                output,
                trajectory_id="qwen35-shell",
                source_agent="qwen3_5_35b_a3b",
                wait_seconds=1.0,
            )
            self.assertTrue(report["normalization_complete"])
            step = load_canonical_jsonl(output / "trajectory.jsonl")[0]
            self.assertEqual(step.action.commands, ("printf exact",))

    def test_normalize_annotate_repair_depth_and_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collection, task = self._make_raw(root)
            raw_before = tree_fingerprint(collection)
            build = DerivedBuild(
                root=root / "derived",
                build_id="build-synthetic",
                collection_root=collection,
                collection_id="collection-synthetic",
            )
            manifest = build.initialize(
                collection_manifest_path=collection / "collection_manifest.json",
                git_commit="synthetic-commit",
                builder_config={"depths": [0, 5, 10, 15, 20, 25], "wait_seconds": 1.0},
                created_at="2026-08-03T00:00:00Z",
                taxonomy={
                    "version": "draft-test",
                    "frozen": False,
                    "mode": "open_coding",
                    "uri": "artifact://taxonomy",
                    "sha256": "0" * 64,
                    "labels": ["wrong_subgoal"],
                    "seed_labels": ["wrong_subgoal"],
                },
            )
            self.assertEqual(manifest["depths"], [0, 5, 10, 15, 20, 25])
            canonical_dir = build.path / "canonical" / "trajectory-1"
            report = normalize_task_directory(
                task,
                canonical_dir,
                trajectory_id="trajectory-1",
                source_agent="synthetic-agent",
                wait_seconds=1.0,
            )
            self.assertTrue(report["normalization_complete"])
            steps = load_canonical_jsonl(canonical_dir / "trajectory.jsonl")
            self.assertEqual(len(steps), 24)
            self.assertEqual((steps[0].turn_index, steps[1].turn_index), (0, 0))
            self.assertEqual(
                (steps[0].action_index_within_turn, steps[1].action_index_within_turn), (0, 1)
            )

            source_sha = report["source_trajectory_sha256"]
            annotations = tuple(
                HumanAnnotation(
                    annotation_id="ann-%d" % index,
                    trajectory_id="trajectory-1",
                    source_trajectory_sha256=source_sha,
                    annotator_id="human-%d" % index,
                    root_cause_action_index=3,
                    error_horizon_actions=20,
                    identifiable_at_action_index=23,
                    error_types=("wrong_subgoal",),
                    reversibility=Reversibility.REVERSIBLE,
                    rationale="The first policy-induced deviation occurs at action 3.",
                    taxonomy_version="draft-test",
                )
                for index in (1, 2)
            )
            adjudication = Adjudication(
                adjudication_id="adj-1",
                trajectory_id="trajectory-1",
                input_annotation_ids=("ann-1", "ann-2"),
                adjudicator_id="human-3",
                root_cause_action_index=3,
                error_horizon_actions=20,
                identifiable_at_action_index=23,
                error_types=("wrong_subgoal",),
                reversibility=Reversibility.REVERSIBLE,
                resolution_rationale="Both independent labels agree.",
                evidence_refs=("observation:action-3",),
                taxonomy_version="draft-test",
            )
            adjudication.validate_inputs(annotations)
            patch = RepairPatch(
                patch_id="repair-1",
                case_id="case-1",
                step_id=1,
                root_cause_step=3,
                old_action=steps[1].action,
                new_action=ClickAction(kind="click", x_px=99, y_px=100),
                reason="Remove unrelated pre-root click error.",
                annotator_id="human-3",
                evidence=("observation:action-1",),
            )
            drop_patch = RepairPatch(
                patch_id="drop-2",
                case_id="case-1",
                step_id=2,
                root_cause_step=3,
                old_action=steps[2].action,
                operation="drop",
                reason="The agent recovered from this redundant detour at the next source step.",
                annotator_id="human-3",
                self_recovered=True,
                persistent_state_effect=False,
                causal_to_root_or_task=False,
                evidence=("observation:action-2",),
            )
            prefix_audit = PrefixAudit(
                audit_id="prefix-audit-1",
                case_id="case-1",
                source_trajectory_sha256=source_sha,
                root_cause_action_index=3,
                audit_end_action_index=23,
                audited_prefix_action_indices=tuple(range(24)),
                unrelated_error_action_indices=(1, 2),
                repair_patch_ids=("repair-1", "drop-2"),
                reviewer_ids=("human-1", "human-2"),
                evidence_refs=("observation:action-0-2",),
                rationale="Every pre-root action was checked; action 1 required replacement.",
                approved=True,
            )
            plan = build_case_plan(
                case_id="case-1",
                trajectory_id="trajectory-1",
                source_trajectory_sha256=source_sha,
                steps=steps,
                adjudication=adjudication,
                prefix_audit=prefix_audit,
                patches=(patch, drop_patch),
            )
            # 24 source steps with the root cause at 3, so the deepest grid point (25) does not
            # fit and is reported as unavailable instead of instantiated.
            self.assertEqual(plan.available_depths, (0, 5, 10, 15, 20))
            self.assertEqual(plan.unavailable_depth_reasons["25"], "source suffix shorter than root + depth")
            self.assertEqual(
                [item.replay_end_action_index for item in plan.instances], [3, 8, 13, 18, 23]
            )
            self.assertNotIn(2, [step.action_index_global for step in plan.repaired_steps])
            self.assertEqual(
                plan.instances[1].executed_action_indices, (0, 1, 3, 4, 5, 6, 7, 8)
            )
            # Every old state observation from the earliest repaired action onward is invalidated.
            self.assertTrue(plan.repaired_steps[0].observation_after_sha256)
            self.assertEqual(plan.repaired_steps[1].observation_after_sha256, "")
            self.assertEqual(plan.repaired_steps[2].observation_before_sha256, "")
            self.assertEqual(plan.repaired_steps[2].tool_result, "pending_replay")
            self.assertEqual(plan.repaired_steps[2].action, steps[3].action)

            instance = plan.instances[-1]
            repaired_path = build.path / "repairs" / "case-1" / "trajectory.jsonl"
            atomic_write_jsonl(repaired_path, (step.to_dict() for step in plan.repaired_steps))
            state_probe_path = build.path / "cases" / "case-1" / "state_probes.json"
            state_probe_path.parent.mkdir(parents=True, exist_ok=True)
            state_probe_path.write_text(json.dumps({"db": "sha256sum /tmp/db"}), encoding="utf-8")
            task_provenance = report["task_provenance"]
            first_plan = ReplayPlan(
                attempt_id="synthetic-first",
                build_id="build-synthetic",
                case_id="case-1",
                instance=instance,
                snapshot_uri="synthetic://snapshot",
                snapshot_sha256="a" * 64,
                canonical_trajectory_uri=str(repaired_path),
                canonical_trajectory_sha256=sha256_file(repaired_path),
                task_id="task-1",
                instruction_sha256=task_provenance["instruction_sha256"],
                task_config_uri=task_provenance["task_config_uri"],
                task_config_sha256=task_provenance["task_config_sha256"],
                state_probe_config_uri=str(state_probe_path),
                state_probe_config_sha256=sha256_file(state_probe_path),
            )
            first = execute_replay_plan(
                first_plan, plan.repaired_steps, DeterministicSyntheticBackend()
            )
            second_plan = ReplayPlan(
                attempt_id="synthetic-second",
                build_id="build-synthetic",
                case_id="case-1",
                instance=instance,
                snapshot_uri="synthetic://snapshot",
                snapshot_sha256="a" * 64,
                canonical_trajectory_uri=str(repaired_path),
                canonical_trajectory_sha256=sha256_file(repaired_path),
                expected_state_sha256=first.state_fingerprint.sha256,
                task_id="task-1",
                instruction_sha256=task_provenance["instruction_sha256"],
                task_config_uri=task_provenance["task_config_uri"],
                task_config_sha256=task_provenance["task_config_sha256"],
                state_probe_config_uri=str(state_probe_path),
                state_probe_config_sha256=sha256_file(state_probe_path),
            )
            second = execute_replay_plan(
                second_plan, plan.repaired_steps, DeterministicSyntheticBackend()
            )
            self.assertTrue(second.automated_checks["state_fingerprint_match"])
            self.assertFalse(second.accepted_for_release)
            self.assertIn(
                "dry_run_or_synthetic_replay_is_not_release_evidence",
                second.rejection_reasons,
            )
            self.assertTrue(build.verify_raw_unchanged())
            self.assertEqual(raw_before, tree_fingerprint(collection))

    def test_open_coding_accepts_and_summarizes_human_discovered_types(self) -> None:
        taxonomy = {
            "version": "draft-test",
            "frozen": False,
            "mode": "open_coding",
            "labels": ["wrong_subgoal"],
            "seed_labels": ["wrong_subgoal"],
        }
        self.assertEqual(
            error_types_outside_seed(("human_discovered_type",), taxonomy),
            ("human_discovered_type",),
        )
        adjudications = (
            Adjudication(
                adjudication_id="adj-open-1",
                trajectory_id="trajectory-1",
                input_annotation_ids=("ann-1", "ann-2"),
                adjudicator_id="human-3",
                root_cause_action_index=3,
                error_horizon_actions=2,
                identifiable_at_action_index=5,
                error_types=("human_discovered_type", "wrong_subgoal"),
                reversibility=Reversibility.REVERSIBLE,
                resolution_rationale="Human adjudication retained both observed codes.",
                evidence_refs=("observation:action-3",),
                taxonomy_version="draft-test",
            ),
        )
        summary = summarize_open_codes(adjudications, taxonomy["seed_labels"])
        self.assertEqual(summary["adjudicated_case_count"], 1)
        self.assertEqual(summary["multi_label_case_count"], 1)
        self.assertEqual(
            summary["outside_seed_error_type_counts"], {"human_discovered_type": 1}
        )
        self.assertEqual(
            summary["error_horizon_actions_by_type"]["human_discovered_type"]["median"],
            2,
        )

        frozen = {**taxonomy, "frozen": True, "mode": "frozen"}
        with self.assertRaises(ValueError):
            error_types_outside_seed(("human_discovered_type",), frozen)

    def test_parser_handles_source_action_groups_and_rejects_code(self) -> None:
        parser = PyAutoGUINormalizer()
        scroll = parser.normalize(
            "pyautogui.moveTo(541, 421)\npyautogui.scroll(-10)", wait_seconds=1.0
        )
        self.assertIsInstance(scroll, ScrollAction)
        self.assertEqual((scroll.x_px, scroll.y_px, scroll.delta_y), (541, 421, -10))
        sequence = parser.normalize(
            "pyautogui.press('a')\npyautogui.press('b')", wait_seconds=1.0
        )
        self.assertIsInstance(sequence, SequenceAction)
        self.assertEqual(len(sequence.actions), 2)
        hotkey = parser.normalize("pyautogui.hotkey(['ctrl', 's'])", wait_seconds=1.0)
        self.assertEqual(hotkey.keys, ("ctrl", "s"))
        long_hotkey = parser.normalize(
            "pyautogui.hotkey('ctrl', 'z', 'ctrl', 'z', 'ctrl', 'z')",
            wait_seconds=1.0,
        )
        self.assertEqual(len(long_hotkey.keys), 6)
        key_down = parser.normalize("pyautogui.keyDown('shift')", wait_seconds=1.0)
        key_up = parser.normalize("pyautogui.keyUp('shift')", wait_seconds=1.0)
        self.assertEqual((key_down.kind, key_down.key), ("key_down", "shift"))
        self.assertEqual((key_up.kind, key_up.key), ("key_up", "shift"))
        no_actions = parser.normalize("NO_ACTIONS", wait_seconds=1.0)
        self.assertEqual((no_actions.kind, no_actions.reason), ("no_op", "NO_ACTIONS"))
        zero_scroll = parser.normalize("pyautogui.scroll(0)", wait_seconds=1.0)
        self.assertEqual((zero_scroll.kind, zero_scroll.reason), ("no_op", "scroll(0)"))
        crash = parser.normalize("PREDICT_CRASH", wait_seconds=1.0)
        self.assertEqual((crash.kind, crash.status, crash.answer), ("terminate", "failure", "PREDICT_CRASH"))
        semicolon = parser.normalize(
            "pyautogui.hotkey('ctrl', 'a'); pyautogui.write('text')", wait_seconds=1.0
        )
        self.assertIsInstance(semicolon, SequenceAction)
        double = parser.normalize("pyautogui.click(10, 20, 2)", wait_seconds=1.0)
        self.assertEqual(double.kind, "double_click")
        repeated = parser.normalize("pyautogui.press('a', 3)", wait_seconds=1.0)
        self.assertIsInstance(repeated, SequenceAction)
        self.assertEqual(len(repeated.actions), 3)
        space = parser.normalize("pyautogui.press(' ')", wait_seconds=1.0)
        self.assertEqual(space.keys, ("space",))
        triple = parser.normalize("pyautogui.tripleClick(10, 20)", wait_seconds=1.0)
        self.assertIsInstance(triple, SequenceAction)
        self.assertEqual(len(triple.actions), 3)
        self.assertTrue(all(action.kind == "click" for action in triple.actions))
        encoded = base64.b64encode("alpha\tbeta\nnext".encode("utf-8")).decode("ascii")
        macro = parser.normalize(
            """import base64, time, pyautogui
_text = base64.b64decode('%s').decode('utf-8')
_text = _text.replace('\\r\\n', '\\n').replace('\\r', '\\n')
for _line_index, _line in enumerate(_text.split('\\n')):
    for _part_index, _part in enumerate(_line.split('\\t')):
        if _part:
            pyautogui.typewrite(_part, interval=0.01)
        if _part_index < len(_line.split('\\t')) - 1:
            pyautogui.press('tab')
    if _line_index < len(_text.split('\\n')) - 1:
        pyautogui.press('enter')""" % encoded,
            wait_seconds=1.0,
        )
        self.assertIsInstance(macro, SequenceAction)
        self.assertEqual(
            [(action.kind, getattr(action, "text", None), getattr(action, "keys", None))
             for action in macro.actions],
            [
                ("type", "alpha", None),
                ("hotkey", None, ("tab",)),
                ("type", "beta", None),
                ("hotkey", None, ("enter",)),
                ("type", "next", None),
            ],
        )
        tool_call = parser.normalize("TOOL_CALL", wait_seconds=1.0)
        self.assertEqual(
            (tool_call.kind, tool_call.reason),
            ("no_op", "non-desktop tool call"),
        )
        typed = parser.normalize("pyautogui.typewrite('x', interval=0.5)", wait_seconds=1.0)
        self.assertEqual((typed.kind, typed.text, typed.interval_s), ("type", "x", 0.5))
        waited = parser.normalize("import time\ntime.sleep(0.1)", wait_seconds=1.0)
        self.assertEqual((waited.kind, waited.seconds), ("wait", 0.1))
        pyautogui_waited = parser.normalize("pyautogui.sleep(2)", wait_seconds=1.0)
        self.assertEqual((pyautogui_waited.kind, pyautogui_waited.seconds), ("wait", 2.0))
        mouse_transition = parser.normalize(
            "pyautogui.moveTo(864, 511); pyautogui.mouseDown()",
            wait_seconds=1.0,
        )
        self.assertIsInstance(mouse_transition, SequenceAction)
        self.assertEqual(
            [(action.kind, getattr(action, "button", None)) for action in mouse_transition.actions],
            [("move", None), ("mouse_down", "left")],
        )
        mouse_up = parser.normalize(
            "pyautogui.mouseUp(button='right')", wait_seconds=1.0
        )
        self.assertEqual((mouse_up.kind, mouse_up.button), ("mouse_up", "right"))
        grouped = parser.normalize(
            "import pyautogui\npyautogui.typewrite('abc', interval=0.03)\n"
            "pyautogui.press('enter')",
            wait_seconds=1.0,
        )
        self.assertIsInstance(grouped, SequenceAction)
        self.assertEqual(len(grouped.actions), 2)
        horizontal = parser.normalize(
            "pyautogui.hscroll(-762, x=1100, y=730)", wait_seconds=1.0
        )
        self.assertEqual(
            (horizontal.kind, horizontal.delta_x, horizontal.x_px, horizontal.y_px),
            ("horizontal_scroll", -762, 1100, 730),
        )
        with self.assertRaises(NormalizationError):
            parser.normalize("pyautogui.moveTo(10, 20, 1.5)", wait_seconds=1.0)
        with self.assertRaises(NormalizationError):
            parser.normalize("__import__('os').system('id')", wait_seconds=1.0)
        with self.assertRaises(NormalizationError):
            parser.normalize("pyautogui.sleep(-1)", wait_seconds=1.0)
        with self.assertRaises(NormalizationError):
            parser.normalize("pyautogui.sleep(seconds())", wait_seconds=1.0)
        with self.assertRaises(NormalizationError):
            parser.normalize(
                "import base64\nbase64.b64decode('YQ==')",
                wait_seconds=1.0,
            )
        with self.assertRaises(NormalizationError):
            parser.normalize("import os\nos.system('id')", wait_seconds=1.0)

    def test_evidence_booleans_and_pesr_threshold_are_not_coerced(self) -> None:
        with self.assertRaises((PrefixRepairError, ValueError)):
            PrefixAudit.from_dict(
                {
                    "audit_id": "audit",
                    "case_id": "case",
                    "source_trajectory_sha256": "a" * 64,
                    "root_cause_action_index": 0,
                    "audit_end_action_index": 0,
                    "audited_prefix_action_indices": [],
                    "unrelated_error_action_indices": [],
                    "repair_patch_ids": [],
                    "reviewer_ids": ["h1", "h2"],
                    "evidence_refs": ["e"],
                    "rationale": "reviewed",
                    "approved": "false",
                }
            )
        with self.assertRaises(EvaluationRecordError):
            PostErrorOutcome(
                recovered=True,
                post_takeover_action_count=1,
                termination_reason="done",
                rubric_bundle_uri="u",
                rubric_bundle_sha256="a" * 64,
                rubric_judge_model="j",
                rubric_judge_snapshot="s",
                rubric_judge_prompt_sha256="b" * 64,
                raw_judgment_uri="r",
                raw_judgment_sha256="c" * 64,
                score=0.5,
                success_threshold=0.5,
                per_rubric_results=({"rubric_id": "r1", "score": 0.5},),
            )

    def test_evaluation_evidence_and_budget_are_explicit(self) -> None:
        ear = ErrorAwarenessJudgment(
            verdict=True,
            judge_model="judge",
            judge_snapshot="judge-snapshot",
            judge_prompt_sha256="a" * 64,
            takeover_output_uri="artifact://takeover",
            takeover_output_sha256="b" * 64,
            raw_judgment_uri="artifact://ear",
            raw_judgment_sha256="c" * 64,
            rationale="The output explicitly identifies the erroneous state.",
        )
        pesr = PostErrorOutcome(
            recovered=True,
            post_takeover_action_count=12,
            termination_reason="agent_done",
            rubric_bundle_uri="artifact://rubric",
            rubric_bundle_sha256="d" * 64,
            rubric_judge_model="rubric-judge",
            rubric_judge_snapshot="rubric-snapshot",
            rubric_judge_prompt_sha256="e" * 64,
            raw_judgment_uri="artifact://pesr",
            raw_judgment_sha256="f" * 64,
            score=1.0,
            success_threshold=1.0,
            per_rubric_results=({"rubric_id": "r1", "score": 1.0},),
        )
        episode = EvaluationEpisode(
            run_id="run-1",
            build_id="build-1",
            instance_id="case-1-d5",
            case_id="case-1",
            agent_id="agent-1",
            agent_model_revision="revision-1",
            agent_prompt_sha256="1" * 64,
            adapter_sha256="2" * 64,
            depth=5,
            repeat_id=1,
            valid=True,
            replay_verification_id="vm-replay-1",
            ear=ear,
            pesr=pesr,
        )
        self.assertTrue(episode.to_dict()["ear"]["verdict"])
        with self.assertRaises(EvaluationRecordError):
            PostErrorOutcome(
                recovered=False,
                post_takeover_action_count=51,
                termination_reason="budget",
                rubric_bundle_uri="u",
                rubric_bundle_sha256="h",
                rubric_judge_model="j",
                rubric_judge_snapshot="s",
                rubric_judge_prompt_sha256="p",
                raw_judgment_uri="u2",
                raw_judgment_sha256="h2",
                score=0,
                success_threshold=1,
                per_rubric_results=({"rubric_id": "r1"},),
            )

    def test_native_history_preserves_recorded_reasoning_and_requires_probe(self) -> None:
        steps = (
            HistoryStep(
                step_id=0,
                turn_index=0,
                action_index_within_turn=0,
                observation_image_url="artifact://observation-0",
                observation_sha256="a" * 64,
                action=ClickAction(kind="click", x_px=100, y_px=100),
                trajectory_log=json.dumps({"reasoning_content": "inspect save button"}),
            ),
            HistoryStep(
                step_id=2,
                turn_index=0,
                action_index_within_turn=1,
                observation_image_url="artifact://observation-1",
                observation_sha256="b" * 64,
                action=ClickAction(kind="click", x_px=110, y_px=100),
            ),
        )
        artifact = build_native_history_artifact(
            Holo31Adapter(instruction="Complete the synthetic task"),
            steps,
            renderer_version="holo31-v1",
            strip_reasoning=False,
        )
        self.assertEqual(artifact.action_indices, (0, 2))
        self.assertFalse(artifact.release_eligible)
        serialized = json.dumps(artifact.to_dict())
        self.assertIn("reasoning_content", serialized)
        self.assertEqual(artifact.to_dict()["reasoning_policy"], "preserved_as_recorded")

    def test_takeover_judges_ear_before_first_action(self) -> None:
        class Environment:
            def __init__(self):
                self.executed = []

            def observe(self):
                return {"executed": len(self.executed)}

            def execute(self, action):
                self.executed.append(action)

            def assert_replay_binding(self, verification):
                self.bound_attempt = verification.attempt_id

        class Agent:
            def __init__(self):
                self.turn = 0

            def begin(self, instruction, native_history):
                self.instruction = instruction
                self.native_history = native_history

            def predict(self, observation):
                self.turn += 1
                if self.turn == 1:
                    return TakeoverTurn(
                        public_output="The previous action selected the wrong item.",
                        actions=(ClickAction(kind="click", x_px=10, y_px=10),),
                    )
                return TakeoverTurn(
                    public_output="Recovered.",
                    actions=(TerminateAction(kind="terminate", status="success"),),
                )

        environment = Environment()

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        evidence_dir = Path(temporary.name)
        first_output = "The previous action selected the wrong item."
        output_path = evidence_dir / "takeover.txt"
        output_path.write_text(first_output, encoding="utf-8")
        ear_path = evidence_dir / "ear.json"
        ear_path.write_text("{}", encoding="utf-8")
        rubric_path = evidence_dir / "rubric.json"
        rubric_path.write_text("{}", encoding="utf-8")
        pesr_path = evidence_dir / "pesr.json"
        pesr_path.write_text("{}", encoding="utf-8")
        observation_path = evidence_dir / "observation.png"
        observation_path.write_bytes(b"test-observation")
        probe_path = evidence_dir / "probe.json"
        probe_path.write_text("{}", encoding="utf-8")

        class EARJudge:
            def judge(self, public_output, takeover_observation):
                self.executed_when_judged = len(environment.executed)
                return ErrorAwarenessJudgment(
                    verdict=True,
                    judge_model="judge",
                    judge_snapshot="judge-v1",
                    judge_prompt_sha256="a" * 64,
                    takeover_output_uri=str(output_path),
                    takeover_output_sha256=sha256_file(output_path),
                    raw_judgment_uri=str(ear_path),
                    raw_judgment_sha256=sha256_file(ear_path),
                    rationale="Recognizes the error.",
                )

        class Grader:
            def grade(self, action_count, termination_reason):
                return PostErrorOutcome(
                    recovered=True,
                    post_takeover_action_count=action_count,
                    termination_reason=termination_reason,
                    rubric_bundle_uri=str(rubric_path),
                    rubric_bundle_sha256=sha256_file(rubric_path),
                    rubric_judge_model="rubric-judge",
                    rubric_judge_snapshot="rubric-v1",
                    rubric_judge_prompt_sha256="e" * 64,
                    raw_judgment_uri=str(pesr_path),
                    raw_judgment_sha256=sha256_file(pesr_path),
                    score=1.0,
                    success_threshold=1.0,
                    per_rubric_results=({"rubric_id": "r1", "score": 1.0},),
                )

        fingerprint = StateFingerprint(components={"db": "a" * 64})
        checks = {
            "snapshot_restored": True,
            "action_count_match": True,
            "action_indices_match": True,
            "action_hashes_present": True,
            "expected_state_present": True,
            "state_fingerprint_match": True,
            "non_screenshot_state_present": True,
            "replay_observations_present": True,
            "task_provenance_verified": True,
        }
        verification = ReplayVerification(
            attempt_id="vm-replay-1",
            build_id="build-1",
            case_id="case-1",
            instance_id="case-1-d0",
            task_id="task-1",
            root_cause_action_index=0,
            depth=0,
            replay_end_action_index=0,
            canonical_actions_sha256="1" * 64,
            canonical_trajectory_sha256="2" * 64,
            snapshot_sha256="3" * 64,
            instruction_sha256=hashlib.sha256(
                b"Recover and finish the task."
            ).hexdigest(),
            task_config_sha256="4" * 64,
            state_probe_config_sha256="5" * 64,
            vm_session_id="vm-session-1",
            execution_mode="vm",
            snapshot_restored=True,
            executed_action_indices=(0,),
            per_action_log=(
                {
                    "action_index_global": 0,
                    "action_sha256": "6" * 64,
                    "result": "ok",
                    "observation_before_uri": str(observation_path),
                    "observation_before_sha256": sha256_file(observation_path),
                    "observation_after_uri": str(observation_path),
                    "observation_after_sha256": sha256_file(observation_path),
                },
            ),
            state_fingerprint=fingerprint,
            expected_state_sha256=fingerprint.sha256,
            automated_checks=checks,
            accepted_for_release=True,
            reviewer_ids=("human-reviewer",),
            rejection_reasons=(),
        )
        malformed = verification.to_dict()
        malformed["accepted_for_release"] = "false"
        with self.assertRaises(ValueError):
            ReplayVerification.from_dict(malformed)
        with self.assertRaises(ValueError):
            replace(verification, expected_state_sha256="0" * 64)
        history_step = HistoryStep(
            step_id=0,
            observation_image_url=str(observation_path),
            observation_sha256=sha256_file(observation_path),
            action=ClickAction(kind="click", x_px=10, y_px=10),
        )
        history = build_native_history_artifact(
            Holo31Adapter(instruction="Recover and finish the task."),
            (history_step,),
            renderer_version="holo31-v1",
            conformance_probe_sha256=sha256_file(probe_path),
            conformance_passed=True,
        )
        history = replace(
            history,
            replay_verification_id=verification.attempt_id,
            instance_id=verification.instance_id,
            replay_observation_sha256s=(sha256_file(observation_path),),
            conformance_probe_uri=str(probe_path),
        )
        ear_judge = EARJudge()
        trace = run_takeover_episode(
            agent=Agent(),
            environment=environment,
            ear_judge=ear_judge,
            pesr_grader=Grader(),
            instruction="Recover and finish the task.",
            native_history=history,
            replay_verification=verification,
            run_id="run-1",
            build_id="build-1",
            instance_id="case-1-d0",
            case_id="case-1",
            agent_id="holo_3_1_35b_a3b",
            agent_model_revision="rev-1",
            agent_prompt_sha256="1" * 64,
            adapter_sha256="2" * 64,
            depth=0,
            repeat_id=1,
        )
        self.assertEqual(ear_judge.executed_when_judged, 0)
        self.assertEqual(len(trace.executed_actions), 2)
        self.assertEqual(len(environment.executed), 1)


if __name__ == "__main__":
    unittest.main()
