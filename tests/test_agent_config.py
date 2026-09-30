import pathlib
import re
import shutil
import tempfile
import unittest
from unittest import mock

from derail.mypcbench import agent_config
from derail.mypcbench.agent_config import (
    AGENT_ID_BY_TYPE,
    SCAFFOLDS,
    UPSTREAM_RUNNER,
    AgentConfigError,
    agent_id_for_type,
    all_agent_ids,
    config_dir,
    load_agent_config,
    load_config,
)

GOLDEN_LIVE = {
    "qwen3_6_27b": {
        "coordinate_protocol": "absolute_pixels",
        "system_prompt_file": "qwen3_6_27b_mypcbench_system.txt",
        "temperature": 0.0,
        "max_tokens": 2048,
        "max_images_in_context": 20,
        "history_turns": 100,
        "image_fold_size": 10,
        "previous_action_log": True,
        "tool_choice": "auto",
        "schema_repair_attempts": 2,
        "multi_tool_policy": "one_interaction_plus_collapsed_waits",
        "alternating_action_repeat_limit": 0,
        "stalled_state_step_limit": 0,
        "enable_bash": False,
    },
    "qwen3_8_27b": {
        "coordinate_protocol": "absolute_pixels",
        "system_prompt_file": "qwen3_8_27b_mypcbench_system.txt",
        "enable_shell": True,
        "temperature": 0.0,
        "max_tokens": 2048,
        "max_images_in_context": 20,
        "history_turns": 100,
        "image_fold_size": 10,
        "previous_action_log": True,
        "tool_choice": "auto",
        "schema_repair_attempts": 2,
        "multi_tool_policy": "one_interaction_plus_collapsed_waits",
        "alternating_action_repeat_limit": 0,
        "stalled_state_step_limit": 0,
        "enable_bash": False,
    },
    "holo_3_1_35b_a3b": {
        "coordinate_protocol": "normalized_0_1000",
        "system_prompt_file": "holo_3_1_35b_a3b_mypcbench_system.txt",
        "temperature": 0.0,
        "max_tokens": 2048,
        "max_images_in_context": 20,
        "history_turns": 100,
        "image_fold_size": 10,
        "previous_action_log": True,
        "tool_choice": "auto",
        "schema_repair_attempts": 2,
        "multi_tool_policy": "execute_all_calls_in_order",
        "alternating_action_repeat_limit": 0,
        "stalled_state_step_limit": 0,
        "enable_bash": False,
    },
    "kimi_k3": {
        "coordinate_protocol": "absolute_pixels",
        "system_prompt_file": "kimi_k3_mypcbench_system.txt",
        "temperature": None,
        "max_tokens": 4096,
        "max_images_in_context": 20,
        "history_turns": 100,
        "image_fold_size": 10,
        "previous_action_log": True,
        "tool_choice": "auto",
        "schema_repair_attempts": 2,
        "multi_tool_policy": "one_interaction_plus_collapsed_waits",
        "alternating_action_repeat_limit": 0,
        "stalled_state_step_limit": 0,
        "enable_bash": False,
    },
    "kimi_k3_cuabash": {
        "coordinate_protocol": "absolute_pixels",
        "system_prompt_file": "kimi_k3_cuabash_mypcbench_system.txt",
        "temperature": None,
        "max_tokens": 4096,
        "max_images_in_context": 20,
        "history_turns": 100,
        "image_fold_size": 10,
        "previous_action_log": True,
        "tool_choice": "auto",
        "schema_repair_attempts": 2,
        "multi_tool_policy": "one_interaction_plus_collapsed_waits",
        "alternating_action_repeat_limit": 0,
        "stalled_state_step_limit": 0,
        "enable_bash": True,
    },
    "evocua_32b": {
        "prompt_style": "S2",
        "coordinate_type": "relative",
        "resize_factor": 32,
        "max_history_turns": 20,
        "max_tokens": 4096,
        "top_p": 0.9,
        "temperature": 0.01,
        "step_budget_policy": "runner_authoritative_upstream_guard_plus_one",
        "upstream_max_steps_fallback": 50,
    },
    "opencua_72b": {
        "history_type": "action_history",
        "coordinate_type": "qwen25",
        "cot_level": "l2",
        "max_image_history_length": 20,
        "use_old_sys_prompt": False,
        "max_tokens": 1500,
        "top_p": 0.9,
        "temperature": 0.0,
        "step_budget_policy": "runner_authoritative_upstream_guard_plus_one",
        "upstream_max_steps_fallback": 100,
    },
}


class GoldenConfigTests(unittest.TestCase):
    def test_live_values_match_golden_snapshot(self):
        for agent_id, expected in GOLDEN_LIVE.items():
            with self.subTest(agent_id=agent_id):
                config = load_agent_config(agent_id)
                self.assertEqual(
                    dict(config.live),
                    expected,
                    f"{agent_id}.yaml 的 live 取值变了。这些值现在会真的进模型；"
                    "确认是有意改动后同步更新 GOLDEN_LIVE",
                )

    def test_every_factory_agent_has_a_config(self):
        for agent_type, agent_id in AGENT_ID_BY_TYPE.items():
            with self.subTest(agent_type=agent_type):
                self.assertEqual(load_agent_config(agent_id).agent_id, agent_id)

    def test_referenced_system_prompts_exist(self):
        prompts = config_dir().parents[1] / "prompts" / "agents"
        for agent_id in (
            "qwen3_6_27b",
            "qwen3_8_27b",
            "holo_3_1_35b_a3b",
            "kimi_k3",
            "kimi_k3_cuabash",
        ):
            with self.subTest(agent_id=agent_id):
                name = load_agent_config(agent_id)["system_prompt_file"]
                self.assertTrue((prompts / name).is_file(), f"缺少冻结 prompt：{name}")

    def test_qwen38_explicitly_enables_hybrid_shell(self):
        qwen38 = load_agent_config("qwen3_8_27b")
        qwen36 = load_agent_config("qwen3_6_27b")
        self.assertTrue(qwen38["enable_shell"])
        self.assertNotIn("enable_shell", qwen36.live)
        self.assertNotEqual(qwen38["system_prompt_file"], qwen36["system_prompt_file"])

    def test_kimik3_cuabash_differs_only_in_bash_surface(self):
        gui = dict(load_agent_config("kimi_k3").live)
        bash = dict(load_agent_config("kimi_k3_cuabash").live)
        self.assertEqual(
            {k: v for k, v in bash.items() if k not in ("enable_bash", "system_prompt_file")},
            {k: v for k, v in gui.items() if k not in ("enable_bash", "system_prompt_file")},
        )
        self.assertEqual(gui["enable_bash"], False)
        self.assertEqual(bash["enable_bash"], True)


class AgentIdResolutionTests(unittest.TestCase):
    def test_unknown_agent_type_is_rejected(self):
        with self.assertRaises(ValueError):
            agent_id_for_type("derail_nonexistent")

    def test_env_agent_id_is_only_a_cross_check(self):
        with mock.patch.dict("os.environ", {"DERAIL_AGENT_ID": "evocua_32b"}):
            self.assertEqual(agent_id_for_type("derail_evocua"), "evocua_32b")
        with mock.patch.dict("os.environ", {"DERAIL_AGENT_ID": "opencua_72b"}):
            with self.assertRaisesRegex(AgentConfigError, "漂移"):
                agent_id_for_type("derail_evocua")

    def test_missing_env_agent_id_still_resolves(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(agent_id_for_type("derail_holo31"), "holo_3_1_35b_a3b")


class ScaffoldUniformityTests(unittest.TestCase):
    def test_every_config_declares_a_known_scaffold(self):
        for agent_id in all_agent_ids():
            with self.subTest(agent_id=agent_id):
                config = load_config(agent_id)
                self.assertIn(config.scaffold, SCAFFOLDS)

    def test_every_config_declares_its_runner_agent_type(self):
        for agent_type, agent_id in AGENT_ID_BY_TYPE.items():
            with self.subTest(agent_id=agent_id):
                self.assertEqual(load_config(agent_id).document["agent_type"], agent_type)

    def test_runner_owned_configs_refuse_to_load_as_execution_config(self):
        for agent_id in all_agent_ids():
            if load_config(agent_id).scaffold != UPSTREAM_RUNNER:
                continue
            with self.subTest(agent_id=agent_id):
                with self.assertRaisesRegex(AgentConfigError, "内置 agent"):
                    load_agent_config(agent_id)

    def test_only_runner_owned_configs_have_empty_live(self):
        for agent_id in all_agent_ids():
            config = load_config(agent_id)
            with self.subTest(agent_id=agent_id):
                if config.scaffold == UPSTREAM_RUNNER:
                    self.assertEqual(dict(config.live), {})
                else:
                    self.assertTrue(config.live)

    def test_concurrency_field_matches_serving_mode(self):
        for agent_id in all_agent_ids():
            document = load_config(agent_id).document
            with self.subTest(agent_id=agent_id):
                self.assertNotEqual(
                    "tensor_parallel_size" in document,
                    "num_vms" in document,
                    "两者必须恰好有一个",
                )

    def test_every_shipped_config_has_a_spec(self):
        for agent_id in all_agent_ids():
            with self.subTest(agent_id=agent_id):
                load_config(agent_id)


class ValidationTests(unittest.TestCase):
    def _load_patched(self, agent_id, mutate, loader=load_agent_config):
        source = config_dir()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            target = root / "configs" / "agents"
            target.mkdir(parents=True)
            for path in source.glob("*.yaml"):
                shutil.copy(path, target / path.name)
            path = target / f"{agent_id}.yaml"
            path.write_text(mutate(path.read_text(encoding="utf-8")), encoding="utf-8")
            with mock.patch.object(agent_config, "REPO_ROOT", root):
                return loader(agent_id)

    def test_unknown_key_is_rejected(self):
        with self.assertRaisesRegex(AgentConfigError, "未知字段"):
            self._load_patched(
                "evocua_32b", lambda text: text.replace("resize_factor:", "resize_factr:")
            )

    def test_missing_live_field_is_rejected(self):
        with self.assertRaisesRegex(AgentConfigError, "缺少 live 字段"):
            self._load_patched(
                "evocua_32b", lambda text: text.replace("prompt_style: S2\n", "")
            )

    def test_value_outside_enum_is_rejected(self):
        with self.assertRaisesRegex(AgentConfigError, "prompt_style"):
            self._load_patched(
                "evocua_32b", lambda text: text.replace("prompt_style: S2", "prompt_style: S3")
            )

    def test_wrong_type_is_rejected(self):
        with self.assertRaisesRegex(AgentConfigError, "必须是整数"):
            self._load_patched(
                "evocua_32b", lambda text: text.replace("resize_factor: 32", "resize_factor: 32.5")
            )

    def test_bool_is_not_accepted_as_number(self):
        with self.assertRaisesRegex(AgentConfigError, "必须是数值"):
            self._load_patched(
                "evocua_32b", lambda text: text.replace("temperature: 0.01", "temperature: true")
            )

    def test_out_of_range_value_is_rejected(self):
        with self.assertRaisesRegex(AgentConfigError, "不能小于"):
            self._load_patched(
                "evocua_32b",
                lambda text: text.replace("max_history_turns: 20", "max_history_turns: 0"),
            )

    def test_agent_id_must_match_filename(self):
        with self.assertRaisesRegex(AgentConfigError, "与文件名不一致"):
            self._load_patched(
                "evocua_32b", lambda text: text.replace("agent_id: evocua_32b", "agent_id: other")
            )

    def test_scaffold_must_match_spec(self):
        with self.assertRaisesRegex(AgentConfigError, "scaffold"):
            self._load_patched(
                "evocua_32b",
                lambda text: text.replace(
                    "scaffold: upstream_official", "scaffold: derail_tool_agent"
                ),
            )

    def test_num_vms_is_type_checked_even_though_bash_reads_it(self):
        with self.assertRaisesRegex(AgentConfigError, "num_vms"):
            self._load_patched(
                "kimi_k3", lambda text: text.replace("num_vms: 4", "num_vms: four")
            )

    _TP_LINE = re.compile(r"^tensor_parallel_size: \d+$", re.MULTILINE)

    def _patch_tp_line(self, replacement):
        def mutate(text):
            patched, count = self._TP_LINE.subn(replacement, text, count=1)
            self.assertEqual(count, 1, "配置里找不到 tensor_parallel_size 声明")
            return patched

        return mutate

    def test_local_agent_may_not_hardcode_num_vms(self):
        with self.assertRaisesRegex(AgentConfigError, "num_vms"):
            self._load_patched(
                "opencua_72b",
                self._patch_tp_line(r"\g<0>" + "\nnum_vms: 1"),
            )

    def test_tensor_parallel_size_is_type_checked(self):
        with self.assertRaisesRegex(AgentConfigError, "tensor_parallel_size"):
            self._load_patched(
                "opencua_72b",
                self._patch_tp_line("tensor_parallel_size: 0"),
            )

    def test_missing_required_metadata_is_rejected(self):
        with self.assertRaisesRegex(AgentConfigError, "缺少必填字段"):
            self._load_patched(
                "evocua_32b",
                lambda text: text.replace(
                    "revision: ce0553438b5f329959b4e740791b580402bbc73e\n", ""
                ),
            )

    def test_missing_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "configs" / "agents").mkdir(parents=True)
            with mock.patch.object(agent_config, "REPO_ROOT", root):
                with self.assertRaisesRegex(AgentConfigError, "找不到 agent 配置"):
                    load_agent_config("evocua_32b")


if __name__ == "__main__":
    unittest.main()
