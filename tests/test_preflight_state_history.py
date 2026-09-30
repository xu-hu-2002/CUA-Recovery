import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import urllib.error
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]


def _load_preflight_module():
    path = REPOSITORY / "scripts" / "10_preflight_takeover_history.py"
    spec = importlib.util.spec_from_file_location("takeover_history_preflight", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREFLIGHT = _load_preflight_module()


class StateHistoryValidationTests(unittest.TestCase):
    def test_evocua_validates_inline_s2_tool_call(self):
        adapter = SimpleNamespace(
            extract_tool_calls=lambda _messages: [{
                "function": {
                    "name": "computer_use",
                    "arguments": json.dumps({"action": "wait", "time": 0.1}),
                }
            }]
        )
        schemas = {
            "computer_use": {
                "type": "object",
                "required": ["action", "time"],
                "properties": {
                    "action": {"const": "wait"},
                    "time": {"type": "number"},
                },
            }
        }

        calls = PREFLIGHT._validate_evocua_step_messages(
            adapter, [{"role": "assistant"}], schemas
        )

        self.assertEqual(calls[0]["function"]["name"], "computer_use")

    def test_shard_selection_matches_worker_job_order_across_cells(self):
        rows = [
            ("t1", "c", "n", "j", "a", "0", "20", "0,10"),
            ("t2", "c", "n", "j", "a", "0", "20", "0"),
            ("t3", "c", "n", "j", "a", "0", "20", "10"),
        ]

        selected = PREFLIGHT._active_shard_conditions(
            rows,
            [0, 10],
            ["unaware", "notified"],
            frozenset(),
            shard_count=4,
            shard_offset=1,
            shard_workers=1,
        )

        self.assertEqual(selected, {(2, 0): ("unaware",), (3, 10): ("unaware",)})

    def test_shard_selection_exclusions_do_not_consume_job_indexes(self):
        rows = [
            ("t1", "c", "n", "j", "a", "0", "20", "0,10"),
            ("t2", "c", "n", "j", "a", "0", "20", "0"),
            ("t3", "c", "n", "j", "a", "0", "20", "10"),
        ]

        selected = PREFLIGHT._active_shard_conditions(
            rows,
            [0, 10],
            ["unaware", "notified"],
            frozenset({"t1"}),
            shard_count=4,
            shard_offset=1,
            shard_workers=1,
        )

        self.assertEqual(selected, {(2, 0): ("notified",)})

    def test_opencua_accepts_system_then_native_state_on_first_step(self):
        step = SimpleNamespace(step_id=0, action_index_within_turn=0)
        messages = [
            {"role": "system", "content": "native prompt"},
            {
                "role": "opencua_state",
                "observation_image_url": "data:image/png;base64,AA==",
                "action": "pyautogui.click(1, 2)",
            },
        ]
        count = PREFLIGHT._validate_state_history_messages(
            "opencua_observations_actions_cots_action_history", messages, step
        )
        self.assertEqual(count, 1)

    def test_opencua_rejects_tool_call_shape(self):
        step = SimpleNamespace(step_id=1, action_index_within_turn=0)
        with self.assertRaisesRegex(ValueError, "sequence"):
            PREFLIGHT._validate_state_history_messages(
                "opencua_observations_actions_cots_action_history",
                [{"role": "assistant", "tool_calls": []}],
                step,
            )

    def test_qwen_allows_no_duplicate_record_within_turn(self):
        step = SimpleNamespace(step_id=2, action_index_within_turn=1)
        count = PREFLIGHT._validate_state_history_messages(
            "qwen35vl_snapshot_state_public_response", [], step
        )
        self.assertEqual(count, 0)

    def test_claude_allows_native_tool_free_termination_only(self):
        messages = [{
            "role": "assistant",
            "content": [{"type": "text", "text": "The task is complete."}],
        }]
        count = PREFLIGHT._validate_claude_step_messages(
            messages, SimpleNamespace(kind="terminate")
        )
        self.assertEqual(count, 0)
        with self.assertRaisesRegex(ValueError, "do not match"):
            PREFLIGHT._validate_claude_step_messages(
                messages, SimpleNamespace(kind="click")
            )


class TokenizeEndpointTests(unittest.TestCase):
    def test_evocua_runtime_model_uses_served_alias(self):
        with mock.patch.dict("os.environ", {"EVOCUA_MODEL": "EvoCUA"}):
            self.assertEqual(
                PREFLIGHT._runtime_model_name("evocua_32b", "meituan/checkpoint"),
                "EvoCUA",
            )

    def test_preflight_environment_rejects_command_execution(self):
        with self.assertRaisesRegex(RuntimeError, "must not execute VM commands"):
            PREFLIGHT.PREFLIGHT_ENVIRONMENT._execute_command("pwd", shell=True)

    def test_chat_usage_reads_prompt_tokens_without_temperature(self):
        response = mock.MagicMock()
        response.__enter__.return_value = io.StringIO(
            json.dumps({"usage": {"prompt_tokens": 157}})
        )
        with mock.patch.object(
            PREFLIGHT.urllib.request, "urlopen", return_value=response
        ) as urlopen:
            count = PREFLIGHT._chat_usage_request(
                "https://gateway/protocol/openai/v1",
                model="kimi-k3",
                messages=[{"role": "user", "content": "ping"}],
                tools=[{"type": "function", "function": {"name": "ping"}}],
                timeout=3,
            )
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(count, 157)
        self.assertNotIn("temperature", payload)
        self.assertEqual(payload["max_completion_tokens"], 1)
        self.assertEqual(
            request.full_url,
            "https://gateway/protocol/openai/v1/chat/completions",
        )

    def test_chat_usage_requires_positive_prompt_count(self):
        response = mock.MagicMock()
        response.__enter__.return_value = io.StringIO(json.dumps({"usage": {}}))
        with mock.patch.object(
            PREFLIGHT.urllib.request, "urlopen", return_value=response
        ):
            with self.assertRaisesRegex(RuntimeError, "usage.prompt_tokens"):
                PREFLIGHT._chat_usage_request(
                    "https://gateway/protocol/openai/v1",
                    model="kimi-k3",
                    messages=[],
                    tools=[],
                    timeout=3,
                )

    def test_anthropic_usage_retries_gateway_all_models_failed(self):
        body = io.BytesIO(b'{"error":"AllModelsFailed: invoke_error"}')
        transient = urllib.error.HTTPError(
            "https://gateway/protocol/anthropic/v1/messages",
            400,
            "Bad Request",
            {},
            body,
        )
        response = mock.MagicMock()
        response.__enter__.return_value = io.StringIO(
            json.dumps({"usage": {"input_tokens": 321}})
        )
        with mock.patch.object(
            PREFLIGHT.urllib.request, "urlopen", side_effect=[transient, response]
        ), mock.patch.object(PREFLIGHT.time, "sleep"):
            count = PREFLIGHT._anthropic_usage_request(
                "https://gateway/protocol/anthropic",
                request_payload={"model": "claude-opus-4-8", "messages": []},
                timeout=3,
            )
        self.assertEqual(count, 321)

    def test_kimi_experiment_cap_is_frozen_in_submitter(self):
        script = (
            REPOSITORY / "scripts/rock/submit_takeover_kimi.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("KIMI_EXPERIMENT_CONTEXT_CAP:-262144", script)
        self.assertIn("takeovewr_annotation/*", script)

    def test_opencua_uses_the_live_served_model_alias(self):
        with mock.patch.dict("os.environ", {"OPENCUA_MODEL": "opencua-72b"}):
            model = PREFLIGHT._runtime_model_name(
                "opencua_72b", "xlangai/OpenCUA-72B"
            )
        self.assertEqual(model, "opencua-72b")

    def test_other_agents_keep_the_configured_model(self):
        with mock.patch.dict("os.environ", {"OPENCUA_MODEL": "opencua-72b"}):
            model = PREFLIGHT._runtime_model_name("qwen3_5_35b_a3b", "qwen")
        self.assertEqual(model, "qwen")

    def test_prefers_versioned_tokenize_route(self):
        self.assertEqual(
            PREFLIGHT._tokenize_endpoints("http://localhost:8000/v1"),
            ("http://localhost:8000/v1/tokenize", "http://localhost:8000/tokenize"),
        )

    def test_falls_back_to_root_route_only_after_404(self):
        response = mock.MagicMock()
        response.__enter__.return_value = io.StringIO(
            json.dumps({"count": 17, "max_model_len": 32768})
        )
        not_found = urllib.error.HTTPError(
            "http://localhost:8000/v1/tokenize", 404, "Not Found", {}, None
        )
        with mock.patch.object(
            PREFLIGHT.urllib.request, "urlopen", side_effect=[not_found, response]
        ) as urlopen:
            result = PREFLIGHT._tokenize_request(
                "http://localhost:8000/v1",
                model="opencua-72b",
                messages=[],
                tools=[],
                timeout=3,
            )
        self.assertEqual(result, (17, 32768))
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(urlopen.call_args_list[0].args[0].full_url, "http://localhost:8000/v1/tokenize")
        self.assertEqual(urlopen.call_args_list[1].args[0].full_url, "http://localhost:8000/tokenize")

    def test_final_http_error_includes_bounded_response_body(self):
        body = io.BytesIO(b'{"error":"The model `wrong` does not exist."}')
        not_found = urllib.error.HTTPError(
            "http://localhost:8000/tokenize", 404, "Not Found", {}, body
        )
        with mock.patch.object(
            PREFLIGHT.urllib.request, "urlopen", side_effect=not_found
        ):
            with self.assertRaisesRegex(RuntimeError, "model `wrong` does not exist"):
                PREFLIGHT._tokenize_request(
                    "http://localhost:8000",
                    model="wrong",
                    messages=[],
                    tools=[],
                    timeout=3,
                )


if __name__ == "__main__":
    unittest.main()
