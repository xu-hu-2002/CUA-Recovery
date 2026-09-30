import types
import unittest
from unittest import mock

from derail.adapters import TARGET_AGENT_IDS, create_native_history_adapter, get_registration
from derail.adapters.qwen35 import Qwen35StateAdapter
from derail.mypcbench.agent_config import AgentConfigError, load_config, resolve_checkpoint
from derail.mypcbench.qwen35_takeover import Qwen35TakeoverTarget, wrap_qwen35_takeover_target


class ReRailAgentTests(unittest.TestCase):
    def test_config_loads_and_reads_checkpoint_from_env(self):
        config = load_config("rerail_35b_a3b")
        qwen = load_config("qwen3_5_35b_a3b")
        for key in ("family", "scaffold", "agent_type", "coordinate_protocol",
                    "context_policy", "tensor_parallel_size"):
            self.assertEqual(config.document[key], qwen.document[key], key)
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(AgentConfigError, "RERAIL_CHECKPOINT"):
                resolve_checkpoint(config)
        with mock.patch.dict("os.environ", {"RERAIL_CHECKPOINT": "/ckpt/rerail"}):
            self.assertEqual(resolve_checkpoint(config), "/ckpt/rerail")

    def test_registry_accepts_rerail_as_target(self):
        self.assertIn("rerail_35b_a3b", TARGET_AGENT_IDS)
        self.assertTrue(get_registration("rerail_35b_a3b").renderer_implemented)

    def test_rerail_uses_qwen35_adapter_and_protocol(self):
        adapter = create_native_history_adapter("rerail_35b_a3b", "Do the task")
        qwen = create_native_history_adapter("qwen3_5_35b_a3b", "Do the task")
        self.assertIsInstance(adapter, Qwen35StateAdapter)
        self.assertEqual(adapter.capabilities.agent_id, "rerail_35b_a3b")
        self.assertEqual(qwen.capabilities.agent_id, "qwen3_5_35b_a3b")
        self.assertEqual(adapter.capabilities.history_format, qwen.capabilities.history_format)
        target = types.SimpleNamespace(_inner=types.SimpleNamespace(restore_state=lambda **_: None))
        self.assertIsInstance(
            wrap_qwen35_takeover_target(target, "rerail_35b_a3b"), Qwen35TakeoverTarget
        )


if __name__ == "__main__":
    unittest.main()
