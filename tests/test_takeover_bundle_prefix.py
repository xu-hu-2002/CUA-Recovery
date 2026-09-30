"""Direct regression tests for full-prefix Rubric Judge bundle construction."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "takeover" / "bundle_prefix.py"
SPEC = importlib.util.spec_from_file_location("takeover_bundle_prefix", SCRIPT)
assert SPEC and SPEC.loader
bundle_prefix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle_prefix)


class TakeoverBundlePrefixTests(unittest.TestCase):
    def _fixture(self, root: Path, responses: list[str] | None = None):
        responses = responses or ["Action: click Save", "Action: click Send"]
        source = root / "source.jsonl"
        source.write_text(
            "".join(json.dumps({"response": response}) + "\n" for response in responses),
            encoding="utf-8",
        )
        canonical = root / "canonical.jsonl"
        canonical.write_text(
            "".join(
                json.dumps(
                    {
                        "action_index_global": index,
                        "turn_index": index,
                        "action_summary": f"action-{index}",
                        "source_record_uri": f"{source}#line={index + 1}",
                    }
                )
                + "\n"
                for index in range(len(responses))
            ),
            encoding="utf-8",
        )
        task = root / "experiment" / "depth_5" / "unaware" / "task-1"
        task.mkdir(parents=True)
        screenshots = []
        for index in range(len(responses)):
            screenshot = root / f"prefix-{index}.png"
            screenshot.write_bytes(b"png")
            screenshots.append(screenshot)
        (task / "native_history.json").write_text(
            json.dumps({"action_indices": list(range(len(responses)))}), encoding="utf-8"
        )
        (task / "prefix_replay_log.json").write_text(
            json.dumps(
                [
                    {"action_index_global": index, "observation_after_uri": str(screenshot)}
                    for index, screenshot in enumerate(screenshots)
                ]
            ),
            encoding="utf-8",
        )
        (task / "takeover_manifest.json").write_text(
            json.dumps(
                {
                    "human_annotation": {"trajectory_id": "trajectory-1"},
                    "prefix_end_action_index": len(responses) - 1,
                    "depth": 5,
                }
            ),
            encoding="utf-8",
        )
        original = {
            "artifacts": {
                "steps": [{"step_num": 0, "raw_response_text": "post takeover"}],
                "counts": {"steps": 1},
            }
        }
        (task / "rubric_bundle.json").write_text(json.dumps(original), encoding="utf-8")
        selection = [{"trajectory_id": "trajectory-1", "canonical_trajectory_uri": str(canonical)}]
        return task, selection, screenshots, original

    def test_orders_prefix_before_post_and_sanitizes_four_response_styles(self):
        responses = [
            "Action: click Save",
            "secret plan</think>\nAction: click Send",
            "## Thought:\nprivate\n\n## Action:\nClick Done",
            json.dumps(
                {
                    "content": "private chain of thought",
                    "tool_calls": [{"name": "computer", "arguments": {"action": "click"}}],
                }
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            task, selection, _, _ = self._fixture(Path(directory), responses)
            self.assertEqual(bundle_prefix.rewrite_task(task, selection, {}), "rewritten")
            steps = json.loads((task / "rubric_bundle.json").read_text())["artifacts"]["steps"]
            self.assertEqual([step["step_num"] for step in steps], ["prefix_0", "prefix_1", "prefix_2", "prefix_3", 0])
            rendered = "\n".join(step["raw_response_text"] for step in steps[:-1])
            self.assertIn("Action: click Save", rendered)
            self.assertIn("Action: click Send", rendered)
            self.assertIn("Click Done", rendered)
            self.assertIn("tool_calls", rendered)
            self.assertNotIn("secret plan", rendered)
            self.assertNotIn("private chain of thought", rendered)
            self.assertNotIn("## Thought", rendered)

    def test_preserves_stale_results_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            task, selection, _, original = self._fixture(Path(directory))
            for name in bundle_prefix.STALE_RESULTS:
                (task / name).write_text('{"old": true}', encoding="utf-8")
            self.assertEqual(bundle_prefix.rewrite_task(task, selection, {}), "rewritten")
            self.assertEqual(
                json.loads((task / "rubric_bundle.post_takeover_only.json").read_text()), original
            )
            for name in bundle_prefix.STALE_RESULTS:
                archived = task / name.replace(".json", ".post_takeover_only.json")
                self.assertEqual(json.loads(archived.read_text()), {"old": True})
                self.assertFalse((task / name).exists())
            rewritten = (task / "rubric_bundle.json").read_bytes()
            self.assertEqual(bundle_prefix.rewrite_task(task, selection, {}), "skipped")
            self.assertEqual((task / "rubric_bundle.json").read_bytes(), rewritten)

    def test_missing_prefix_artifacts_skips_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            bundle = task / "rubric_bundle.json"
            bundle.write_text('{"artifacts":{"steps":[]}}', encoding="utf-8")
            before = bundle.read_bytes()
            self.assertEqual(bundle_prefix.rewrite_task(task, [], {}), "no-prefix-artifacts")
            self.assertEqual(bundle.read_bytes(), before)

    def test_index_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            task, selection, _, _ = self._fixture(Path(directory))
            replay = json.loads((task / "prefix_replay_log.json").read_text())
            replay[0]["action_index_global"] = 99
            (task / "prefix_replay_log.json").write_text(json.dumps(replay), encoding="utf-8")
            before = (task / "rubric_bundle.json").read_bytes()
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                bundle_prefix.rewrite_task(task, selection, {})
            self.assertEqual((task / "rubric_bundle.json").read_bytes(), before)
            self.assertFalse((task / "rubric_bundle.post_takeover_only.json").exists())

    def test_missing_screenshot_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            task, selection, screenshots, _ = self._fixture(Path(directory))
            screenshots[0].unlink()
            before = (task / "rubric_bundle.json").read_bytes()
            with self.assertRaisesRegex(RuntimeError, "missing prefix screenshot"):
                bundle_prefix.rewrite_task(task, selection, {})
            self.assertEqual((task / "rubric_bundle.json").read_bytes(), before)
            self.assertFalse((task / "rubric_bundle.post_takeover_only.json").exists())

    def test_identical_duplicate_canonical_files_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory_id = "trajectory-1"
            content = '{"action_index_global": 0}\n'
            for version in ("v2", "v3"):
                canonical = (
                    root / "artifacts" / "derail_builds" / version
                    / "canonical" / trajectory_id / "trajectory.jsonl"
                )
                canonical.parent.mkdir(parents=True)
                canonical.write_text(content, encoding="utf-8")
            selection = [{
                "trajectory_id": trajectory_id,
                "canonical_trajectory_uri": "/missing/trajectory.jsonl",
                "source_trajectory_sha256": "historical-semantic-hash",
            }]
            original_repository = bundle_prefix.REPOSITORY
            bundle_prefix.REPOSITORY = root
            try:
                rows = bundle_prefix._canonical_steps(selection, trajectory_id)
            finally:
                bundle_prefix.REPOSITORY = original_repository
            self.assertEqual(rows, {0: {"action_index_global": 0}})

    def test_replayed_screenshot_path_resolves_from_the_judge_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            task, selection, screenshots, _ = self._fixture(Path(directory))
            replayed = task / "prefix_replay"
            replayed.mkdir()
            log = []
            for index, screenshot in enumerate(screenshots):
                copy = replayed / screenshot.name
                copy.write_bytes(b"png")
                log.append({
                    "action_index_global": index,
                    "observation_after_uri": f"/tmp/dead-vm-run/prefix_replay/{screenshot.name}",
                })
                screenshot.unlink()
            (task / "prefix_replay_log.json").write_text(json.dumps(log), encoding="utf-8")

            relative_task = Path(os.path.relpath(task))
            self.assertEqual(bundle_prefix.rewrite_task(relative_task, selection, {}), "rewritten")

            steps = json.loads((task / "rubric_bundle.json").read_text(encoding="utf-8"))["artifacts"]["steps"]
            for step in steps[: len(log)]:
                recorded = Path(step["screenshot"])
                self.assertFalse(recorded.is_absolute(), step["screenshot"])
                self.assertTrue((task / recorded).is_file(), step["screenshot"])


class BundleEvidenceTests(unittest.TestCase):
    def test_claude_tool_round_and_final_state_are_filled(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task"
            task.mkdir()
            rows = [
                {"step_num": 1, "action": "TOOL_CALL", "response": ""},
                {"step_num": 2, "action": "pyautogui.click(1, 2)", "response": "done"},
            ]
            (task / "traj.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (task / "state_probes.jsonl").write_text(json.dumps(
                {"traj_index": 1, "probe_output": {"orders": "orders.status=shipped"},
                 "state_fingerprint": {"sha256": "f" * 64}}) + "\n")
            (task / "messages.json").write_text(json.dumps([
                {"role": "user", "content": [{"type": "text", "text": "task"}]},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "ls"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                              "content": [{"type": "text", "text": "a.txt"},
                                                          {"type": "image"}]}]},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "computer"}]},
            ]))
            steps = [{"step_num": r["step_num"], "raw_response_text": r["response"],
                      "parsed_action_obj": {"action": r["action"]}} for r in rows]
            (task / "rubric_bundle.json").write_text(json.dumps({"artifacts": {"steps": steps}}))
            (task / "rubric_judge_result.json").write_text("{}")

            self.assertEqual(bundle_prefix.enrich_task(task, 1000, "state_probes.jsonl"), "enriched")
            bundle = json.loads((task / "rubric_bundle.json").read_text())
            first = bundle["artifacts"]["steps"][0]["raw_response_text"]
            self.assertIn('"command": "ls"', first)
            self.assertIn("a.txt", first)
            self.assertEqual(bundle["artifacts"]["steps"][1]["raw_response_text"], "done")
            self.assertIn("orders.status=shipped", bundle["artifacts"]["final_state"]["text"])
            self.assertTrue((task / "rubric_judge_result.pre_evidence.json").is_file())
            self.assertEqual(bundle_prefix.enrich_task(task, 1000, "state_probes.jsonl"), "evidence-present")

            wrapper_spec = importlib.util.spec_from_file_location(
                "full_traj_judge", Path(__file__).parents[1] / "scripts" / "judge" / "full_traj_judge.py")
            wrapper = importlib.util.module_from_spec(wrapper_spec)
            wrapper_spec.loader.exec_module(wrapper)
            section = wrapper.final_state_section(bundle)
            text = wrapper.with_section("Trajectory\n\nScreenshots attached below: 1", section)
            self.assertLess(text.index("orders.status=shipped"), text.index("Screenshots attached"))

    def test_prefix_tool_step_takes_output_from_canonical_row(self):
        with tempfile.TemporaryDirectory() as directory:
            task, selection, _, _ = TakeoverBundlePrefixTests()._fixture(Path(directory), ["", ""])
            canonical = Path(selection[0]["canonical_trajectory_uri"])
            rows = [json.loads(line) for line in canonical.read_text().splitlines()]
            rows[0]["action"] = {"kind": "shell", "commands": ["cat notes.txt"]}
            rows[0]["tool_result"] = json.dumps({"shell_results": [{"result": "meeting at 3pm"}]})
            canonical.write_text("".join(json.dumps(r) + "\n" for r in rows))
            self.assertEqual(bundle_prefix.rewrite_task(task, selection, {}), "rewritten")
            steps = json.loads((task / "rubric_bundle.json").read_text())["artifacts"]["steps"]
            self.assertIn("cat notes.txt", steps[0]["raw_response_text"])
            self.assertIn("meeting at 3pm", steps[0]["raw_response_text"])


if __name__ == "__main__":
    unittest.main()
