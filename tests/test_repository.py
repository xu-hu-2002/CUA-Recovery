import json
import subprocess
import unittest

from derail.cli import REQUIRED_PATHS, repository_root


class RepositoryTests(unittest.TestCase):
    def test_required_paths_exist(self) -> None:
        root = repository_root()
        missing = [path for path in REQUIRED_PATHS if not (root / path).exists()]
        self.assertEqual(missing, [])

    def test_all_json_files_parse(self) -> None:
        root = repository_root()
        json_files = list((root / "schemas").glob("*.json")) + list(
            (root / "benchmark").rglob("*.json")
        )
        self.assertTrue(json_files)
        for path in json_files:
            with self.subTest(path=path):
                json.loads(path.read_text(encoding="utf-8"))

    def test_rollout_patch_preserves_tool_messages(self) -> None:
        root = repository_root()
        patch = (
            root / "patches/mypcbench_caf9c754_message_trajectory_agents.patch"
        ).read_text(encoding="utf-8")
        self.assertIn('"tool_messages": trajectory_tool_messages', patch)
        self.assertIn('"type": "shell_call_output"', patch)
        self.assertIn('f"computer_call {call_text[:1000]}"', patch)
        self.assertIn('f"function_call {fn_name}({raw_args})', patch)
        self.assertIn('if not resp_str.strip() and actions:', patch)
        self.assertIn('"response": resp_str', patch)
        self.assertIn("last_trajectory_tool_messages", patch)
        self.assertIn("def _trajectory_agent_metadata(agent):", patch)
        self.assertIn('metadata["tool_messages"] = copy.deepcopy(tool_messages)', patch)
        self.assertIn('"agent_metadata": agent_metadata', patch)
        self.assertIn('+            actions.append("FAIL")', patch)
        self.assertIn('+            self.agent_metadata["termination_source"] = "text_only_no_tool"', patch)
        self.assertNotIn('+            step_summary_parts.append("DONE(text-only-fallback)")', patch)

    def test_judge_prompt_is_message_trajectory_first(self) -> None:
        root = repository_root()
        prompt = (
            root / "prompts/judges/mypcbench_full_traj_system.txt"
        ).read_text(encoding="utf-8")
        judge_patch = (
            root / "patches/mypcbench_caf9c754_message_first_judge.patch"
        ).read_text(encoding="utf-8")
        self.assertIn("message/action trajectory as the PRIMARY evidence", prompt)
        self.assertIn("screenshots only as SECONDARY corroborating evidence", prompt)
        self.assertIn('Message: {raw[:4000]', judge_patch)
        self.assertIn("but no recorded message", judge_patch)
        self.assertIn("Primary Message/Action Trajectory", judge_patch)

    def test_local_secret_files_are_not_tracked(self) -> None:
        root = repository_root()
        for name in (".env", "env.yaml"):
            result = subprocess.run(
                ["git", "ls-files", "--error-unmatch", name],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0, "%s 禁止进入 Git index" % name)

    SHORT_LIVED_SHELL_ENTRYPOINTS = frozenset({"setup_third_party.sh", "run_takeover_judge.sh", "run.sh"})

    def test_project_shell_entrypoints_are_tmux_managed(self) -> None:
        scripts = sorted(
            path for path in (repository_root() / "scripts").rglob("*.sh")
            if path.parent.name != "lib"
        )
        self.assertTrue(scripts)
        for path in scripts:
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                if path.name not in self.SHORT_LIVED_SHELL_ENTRYPOINTS:
                    self.assertIn("tmux", source)
                result = subprocess.run(
                    ["bash", "-n", str(path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
