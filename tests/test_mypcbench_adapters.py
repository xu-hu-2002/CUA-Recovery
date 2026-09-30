import dataclasses
import importlib.util
import json
import io
import os
import pathlib
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

from derail.adapters.kimi_k3 import KimiK3ScaffoldAdapter
from derail.canonical.actions import ScrollAction
from derail.mypcbench import agent_config
from derail.mypcbench.agent_config import load_agent_config
from derail.mypcbench.factory import (
    _OpenCUAMyPCBenchAdapter,
    _upstream_step_budget,
    create_mypcbench_agent,
)
from derail.mypcbench.launch_contract import (
    LaunchContractError,
    fetch_vllm_contract,
    formal_collection_authorized,
    model_context_from_payload,
    positive_int,
    validate_generation_budget,
)
from derail.mypcbench.tool_agent import (
    _BASH_OUTPUT_CAP,
    _FOLDED_SCREENSHOT_PLACEHOLDER,
    _MAX_BASH_ROUNDS_PER_STEP,
    _MAX_CONSECUTIVE_BASH_STEPS,
    NativeToolComputerAgent,
    SafePyAutoGUICompiler,
    ToolCallError,
    build_computer_tools,
    holo31_protocol,
    kimi_k3_cuabash_protocol,
    kimi_k3_protocol,
    qwen36_protocol,
    qwen38_protocol,
    validate_pyautogui_program,
)


def _load_vendored_claude_adapter():
    """Load the ignored vendored adapter without requiring the Anthropic SDK."""

    root = pathlib.Path(__file__).resolve().parents[1]
    path = root / "third_party/MyPCBench/agent-harness/agents/claude_cuabash.py"
    anthropic = types.ModuleType("anthropic")

    class _AnthropicError(Exception):
        pass

    anthropic.APIError = _AnthropicError
    anthropic.APIStatusError = _AnthropicError
    anthropic.APIResponseValidationError = _AnthropicError
    anthropic.Anthropic = object
    anthropic_types = types.ModuleType("anthropic.types")
    anthropic_beta = types.ModuleType("anthropic.types.beta")
    anthropic_beta.BetaMessage = object
    agents = types.ModuleType("agents")
    agents.__path__ = []
    agents_base = types.ModuleType("agents.base")
    agents_base.BaseAgent = object
    agents_base.encode_image = lambda value: value
    agents_prompts = types.ModuleType("agents.prompts")
    agents_prompts.CLAUDE_CUA_SYSTEM_PROMPT = ""
    stubs = {
        "anthropic": anthropic,
        "anthropic.types": anthropic_types,
        "anthropic.types.beta": anthropic_beta,
        "agents": agents,
        "agents.base": agents_base,
        "agents.prompts": agents_prompts,
    }
    spec = importlib.util.spec_from_file_location("_derail_test_claude_cuabash", path)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load vendored Claude adapter")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


def _bare_claude_agent(module, response):
    agent = module.ClaudeCUAAgent.__new__(module.ClaudeCUAAgent)
    agent.enable_computer = False
    agent.messages = []
    agent.actions_log = []
    agent.last_trajectory_tool_messages = []
    agent.agent_metadata = {}
    agent._prompt_caching = False
    agent.system_prompt = "system"
    agent.model = "claude-opus-4-8"
    agent.max_tokens = 100
    agent.tools = []
    agent.betas = []
    agent.enable_thinking = False
    agent._output_config_supported = False
    agent._force_adaptive_thinking = False
    agent.temperature = None
    agent.top_p = None
    agent.api_retry_times = 1
    agent.api_retry_interval = 0
    agent.backup_api_key = None
    agent.last_usage = {}
    agent.total_usage = {}
    agent._resolve_pending_tool_uses = lambda _image: None
    agent._trim_images = lambda: None
    agent._create_message_with_backoff = lambda _kwargs: response
    return agent


class KimiTakeoverPromptTests(unittest.TestCase):
    def test_history_renderer_matches_live_cuabash_prompt(self):
        self.assertEqual(
            KimiK3ScaffoldAdapter.system_prompt(),
            kimi_k3_cuabash_protocol().system_prompt,
        )

    def test_large_scroll_stays_in_kimi_scroll_schema(self):
        action = ScrollAction(kind="scroll", x_px=640, y_px=400, delta_y=-300)
        call = KimiK3ScaffoldAdapter().action_to_calls(action, "scroll-1")[0]
        self.assertEqual(call["function"]["name"], "scroll")
        self.assertEqual(json.loads(call["function"]["arguments"])["delta_y"], -300)


class OpenCUAStateSeedingTests(unittest.TestCase):
    def test_native_state_and_notified_prompt_are_restored(self):
        class Inner:
            system_prompt = "live prompt"

            def reset(self, _logger=None):
                self.observations, self.actions, self.cots = [], [], []

            def predict(self, instruction, obs):
                self.predicted = (instruction, obs)
                return "response", ["WAIT"], {}

        with tempfile.TemporaryDirectory() as directory:
            screenshot = pathlib.Path(directory) / "before.png"
            screenshot.write_bytes(b"\x89PNG\r\n\x1a\ncontent")
            inner = Inner()
            agent = _OpenCUAMyPCBenchAdapter(inner)
            history = [
                {"role": "system", "content": "live prompt"},
                {
                    "role": "opencua_state",
                    "observation_image_url": str(screenshot),
                    "action": "Click Save.",
                    "cot": {
                        "action": "Click Save.",
                        "thought": "Inspect the dialog.",
                    },
                },
            ]
            agent.seed_native_history("Finish", history, condition="notified")
            self.assertEqual(inner.actions, ["Click Save."])
            self.assertEqual(
                inner.cots,
                [{"action": "Click Save.", "thought": "Inspect the dialog."}],
            )
            self.assertEqual(inner.observations[0]["screenshot"], screenshot.read_bytes())
            agent.predict("Finish", {"screenshot": b"current"})
            self.assertIn("Takeover notice:", inner.predicted[0])

    def test_preflight_captures_exact_payload_without_inference_or_state_mutation(self):
        class Inner:
            system_prompt = "live prompt"

            def __init__(self):
                self.actions = ["Click Save."]
                self.observations = [{"screenshot": b"old"}]
                self.cots = [{"action": "Click Save."}]

            def call_llm(self, _payload, _model):
                raise AssertionError("real inference must not run during preflight")

            def predict(self, instruction, obs):
                payload = {
                    "model": "opencua-72b",
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": instruction},
                    ],
                    "max_tokens": 4096,
                }
                self.call_llm(payload, "opencua-72b")
                self.actions.append("mutated")
                return "response", ["WAIT"], {}

        inner = Inner()
        original_call = inner.call_llm
        agent = _OpenCUAMyPCBenchAdapter(inner)
        agent._takeover_prompt = "Takeover notice: prior work may be wrong."

        payload = agent.preflight_next_request("Finish", {"screenshot": b"current"})

        self.assertEqual(payload["model"], "opencua-72b")
        self.assertIn("Takeover notice:", payload["messages"][-1]["content"])
        self.assertEqual(inner.actions, ["Click Save."])
        self.assertEqual(inner.call_llm, original_call)
        self.assertEqual(agent._takeover_prompt, "Takeover notice: prior work may be wrong.")


class ClaudeAdapterTerminationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_vendored_claude_adapter()

    @staticmethod
    def _response(*blocks):
        return types.SimpleNamespace(content=list(blocks), usage=None)

    def test_text_only_without_terminal_marker_is_failure(self):
        samples = (
            "这一步需要调用 computer（action=screenshot），我这边没法直接执行。",
            "Let me take a screenshot to see the current state.",
        )
        for text in samples:
            with self.subTest(text=text):
                block = types.SimpleNamespace(type="text", text=text)
                agent = _bare_claude_agent(self.module, self._response(block))

                response_text, actions = agent.predict("task", {})

                self.assertEqual(response_text, text)
                self.assertEqual(actions, ["FAIL"])
                self.assertEqual(
                    agent.agent_metadata,
                    {
                        "termination_source": "text_only_no_tool",
                        "model_emitted_terminal_marker": False,
                    },
                )

    def test_explicit_done_marker_remains_success(self):
        block = types.SimpleNamespace(type="text", text="Finished. ```DONE```")
        agent = _bare_claude_agent(self.module, self._response(block))

        _, actions = agent.predict("task", {})

        self.assertEqual(actions, ["DONE"])
        self.assertEqual(agent.agent_metadata, {})

    def test_screenshot_tool_call_is_not_text_only_failure(self):
        block = types.SimpleNamespace(
            type="tool_use",
            id="tool-1",
            name="computer",
            input={"action": "screenshot"},
            model_dump=lambda: {
                "type": "tool_use",
                "id": "tool-1",
                "name": "computer",
                "input": {"action": "screenshot"},
            },
        )
        agent = _bare_claude_agent(self.module, self._response(block))

        response_text, actions = agent.predict("task", {})

        self.assertEqual(actions, ["pyautogui.sleep(0.1)"])
        self.assertIn('\"action\": \"screenshot\"', response_text)
        self.assertEqual(agent.agent_metadata, {})


class _FakeCompletions:
    def __init__(self, message):
        self.messages = list(message) if isinstance(message, list) else [message]
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.messages:
            raise AssertionError("fake completion sequence exhausted")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=self.messages.pop(0))])


class _FakeClient:
    def __init__(self, message):
        self.completions = _FakeCompletions(message)
        self.chat = types.SimpleNamespace(completions=self.completions)


class _FakeEnv:
    """替身 VM 控制句柄：记录 _execute_command 调用并回放固定结果。"""

    def __init__(self, output="", error="", returncode=0):
        self.commands = []
        self._result = {"output": output, "error": error, "returncode": returncode}

    def _execute_command(self, command, shell=False):
        self.commands.append((command, shell))
        return dict(self._result)


class SafeCompilerTests(unittest.TestCase):
    def test_holo_normalized_edges_map_inside_screen(self):
        compiler = SafePyAutoGUICompiler((1280, 800), "normalized_0_1000")
        self.assertEqual(
            compiler.compile("click", {"x": 1000, "y": 1000}),
            ["pyautogui.click(1279, 799, button='left')"],
        )

    def test_qwen_absolute_coordinates_are_not_rescaled(self):
        compiler = SafePyAutoGUICompiler((1280, 800), "absolute_pixels")
        self.assertEqual(
            compiler.compile("double_click", {"x": 640, "y": 400, "button": "right"}),
            ["pyautogui.doubleClick(640, 400, button='right')"],
        )
        with self.assertRaises(ToolCallError):
            compiler.compile("click", {"x": 1280, "y": 400})

    def test_qwen_absolute_coordinate_pair_string_is_narrowly_accepted(self):
        compiler = SafePyAutoGUICompiler((1280, 800), "absolute_pixels")
        self.assertEqual(
            compiler.compile("click", {"x": "[878, 160]"}),
            ["pyautogui.click(878, 160, button='left')"],
        )
        for invalid in ("[878]", "[878, 160, 1]", '[878, "160"]', "not-json"):
            with self.subTest(invalid=invalid), self.assertRaises(ToolCallError):
                compiler.compile("click", {"x": invalid})

    def test_holo_bare_coordinate_pair_string_is_accepted(self):
        # qwen3_coder 把两个坐标压进 x 时，Holo-3.1 给出的是裸逗号对而不是 JSON
        # 数组。2026-08-08 smoke10 的真实样本：x="867, 163" 在 repair 轮被模型
        # 自己确认为 x=867, y=163，编译结果必须与那一轮完全一致。
        compiler = SafePyAutoGUICompiler((1280, 800), "normalized_0_1000")
        self.assertEqual(
            compiler.compile("click", {"x": "867, 163"}),
            ["pyautogui.click(1109, 130, button='left')"],
        )

    def test_coordinate_pair_string_keeps_normalized_scaling_and_bounds(self):
        # 补回被 parser 吃掉的 y，不等于放宽归一化协议：串里的数仍按 0--1000
        # 缩放，越界仍然拒绝。
        compiler = SafePyAutoGUICompiler((1280, 800), "normalized_0_1000")
        self.assertEqual(
            compiler.compile("click", {"x": "[500, 500]"}),
            ["pyautogui.click(640, 400, button='left')"],
        )
        for invalid in ("1001, 500", "500, 1001", "-1, 500", "811, 762]", "1, 2, 3"):
            with self.subTest(invalid=invalid), self.assertRaises(ToolCallError):
                compiler.compile("click", {"x": invalid})

    def test_scroll_delta_in_range_is_passed_through_untouched(self):
        # 单位与走上游 scaffold 的对照组（qwen3_5 自己发 -3/-5/-10）保持一致。
        compiler = SafePyAutoGUICompiler((1280, 800), "normalized_0_1000")
        for notches in (-3, -30, 1, 30):
            with self.subTest(delta_y=notches):
                compiler.clamps.clear()
                self.assertEqual(
                    compiler.compile("scroll", {"x": 500, "y": 500, "delta_y": notches}),
                    ["pyautogui.moveTo(640, 400)", f"pyautogui.scroll({notches})"],
                )
                self.assertEqual(compiler.clamps, [])

    def test_pixel_scale_scroll_is_clamped_and_reported_not_rejected(self):
        # 拒绝会消耗 schema repair 预算，耗尽后 predict() 直接返回 FAIL —— 那就
        # 把「单位理解错」变成了「整局作废」，正是要从数据里剔除的那类失败。
        # 所以截断执行，但必须留下可审计的 intervention。
        compiler = SafePyAutoGUICompiler((1280, 800), "normalized_0_1000")
        for requested, executed in ((-300, -30), (500, 30), (-253, -30), (10000, 30)):
            with self.subTest(delta_y=requested):
                compiler.clamps.clear()
                self.assertEqual(
                    compiler.compile("scroll", {"x": 500, "y": 500, "delta_y": requested}),
                    ["pyautogui.moveTo(640, 400)", f"pyautogui.scroll({executed})"],
                )
                self.assertEqual(len(compiler.clamps), 1)
                clamp = compiler.clamps[0]
                self.assertEqual(clamp["type"], "scroll_clamp")
                self.assertEqual(clamp["requested_delta_y"], requested)
                self.assertEqual(clamp["executed_delta_y"], executed)

    def test_scroll_schema_publishes_the_notch_bound(self):
        # schema 里的 minimum/maximum 必须和编译器的拒绝阈值是同一个数，否则
        # 模型看到的契约和实际执行的契约会分叉。
        scroll = next(
            tool for tool in build_computer_tools(1000, 1000)
            if tool["function"]["name"] == "scroll"
        )
        delta = scroll["function"]["parameters"]["properties"]["delta_y"]
        self.assertEqual((delta["minimum"], delta["maximum"]), (-30, 30))
        self.assertIn("notches", scroll["function"]["description"])

    def test_text_is_quoted_as_data_not_executed_as_code(self):
        compiler = SafePyAutoGUICompiler((1280, 800), "absolute_pixels")
        payload = "x'); __import__('os').system('bad') #"
        actions = compiler.compile("write", {"content": payload, "clear_existing": True})
        self.assertEqual(actions[0], "pyautogui.hotkey('ctrl', 'a')")
        self.assertIn(repr(payload), actions[1])

    def test_unknown_tool_hard_fails(self):
        compiler = SafePyAutoGUICompiler((1280, 800), "absolute_pixels")
        with self.assertRaises(ToolCallError):
            compiler.compile("run_python", {"code": "print(1)"})

    def test_unknown_tool_argument_is_not_silently_ignored(self):
        compiler = SafePyAutoGUICompiler((1280, 800), "normalized_0_1000")
        with self.assertRaises(ToolCallError):
            compiler.compile("click", {"x": 500, "y": 500, "unexpected": 1})

    def test_official_text_code_is_ast_sandboxed(self):
        safe = "pyautogui.click(10, 20)\npyautogui.write(\"hello; world\")"
        self.assertEqual(validate_pyautogui_program(safe), safe)
        with self.assertRaises(ToolCallError):
            validate_pyautogui_program("__import__('os').system('id')")
        with self.assertRaises(ToolCallError):
            validate_pyautogui_program("pyautogui.click(get_x(), 20)")


class NativeToolAgentTests(unittest.TestCase):
    def test_visual_fingerprint_ignores_top_bar_clock_only(self):
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            self.skipTest("Pillow is an optional collection dependency")

        def screenshot(top_color, center_color):
            image = Image.new("RGB", (128, 80), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 127, 31), fill=top_color)
            draw.rectangle((40, 45, 88, 70), fill=center_color)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()

        first = screenshot("black", "red")
        clock_changed = screenshot("blue", "red")
        desktop_changed = screenshot("black", "black")
        self.assertEqual(
            NativeToolComputerAgent._screenshot_fingerprint(first),
            NativeToolComputerAgent._screenshot_fingerprint(clock_changed),
        )
        self.assertNotEqual(
            NativeToolComputerAgent._screenshot_fingerprint(first),
            NativeToolComputerAgent._screenshot_fingerprint(desktop_changed),
        )

    def test_native_tool_call_compiles_and_preserves_tool_history(self):
        call = {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "click",
                "arguments": json.dumps({"x": 500, "y": 500}),
            },
        }
        message = types.SimpleNamespace(content=None, tool_calls=[call])
        client = _FakeClient(message)
        agent = NativeToolComputerAgent(
            "Hcompany/Holo-3.1-35B-A3B",
            (1280, 800),
            holo31_protocol(),
            client=client,
        )

        response, actions = agent.predict("Open settings", {"screenshot": b"png"})

        self.assertEqual(actions, ["pyautogui.click(640, 400, button='left')"])
        self.assertEqual(json.loads(response)["compiled_actions"][0]["name"], "click")
        self.assertEqual(client.completions.requests[0]["tool_choice"], "auto")
        self.assertEqual(agent._turns[0][-1]["tool_call_id"], "call_1")

    def test_upstream_context_policy_folds_images_and_keeps_an_action_log(self):
        """kimi_k3 的窗口策略：丢 turn 但不丢动作，留 turn 但可以丢图。

        v1 的「只留最近 3 步」让 hard_app-f010 在第 85 步忘掉自己第 74 步刚确认过
        的结论，整套重做。这里钉住三件事：窗口外的 turn 消失、它们的动作仍出现在
        `Previous actions:` 里、留下来的 turn 超出图片上限时只掉图不掉消息。
        """

        def call(index):
            return {
                "id": f"call_{index}",
                "type": "function",
                "function": {
                    "name": "click",
                    "arguments": json.dumps({"x": 10 * index, "y": 20}),
                },
            }

        steps = 12
        protocol = dataclasses.replace(
            kimi_k3_protocol(),
            history_turns=5,
            max_images_in_context=2,
            image_fold_size=2,
            previous_action_log=True,
        )
        client = _FakeClient(
            [types.SimpleNamespace(content=None, tool_calls=[call(i)]) for i in range(steps)]
        )
        agent = NativeToolComputerAgent(
            "kimi-k3", (1280, 800), protocol, client=client
        )
        for _ in range(steps):
            agent.predict("Do the task", {"screenshot": b"png"})

        sent = client.completions.requests[-1]["messages"]
        # 5 个历史 turn × 3 条消息 + system + 当前 user。
        self.assertEqual(len(sent), 1 + 5 * 3 + 1)

        def images(message):
            content = message.get("content")
            if not isinstance(content, list):
                return 0
            return sum(1 for part in content if part.get("type") == "image_url")

        # 5 个历史 turn，上限 2 张、每次折 2 张：折两轮后只剩 1 张历史图，加上永不
        # 折叠的当前截图共 2 张。成块折叠会打到上限以下，上游同样如此 —— 换来的是
        # 折叠前缀只增不减，prompt cache 不会每步失效。
        self.assertEqual(sum(images(message) for message in sent), 2)
        folded = [
            message
            for message in sent
            if isinstance(message.get("content"), list)
            and any(
                part.get("text") == _FOLDED_SCREENSHOT_PLACEHOLDER
                for part in message["content"]
            )
        ]
        self.assertEqual(len(folded), 4)

        # tool 消息必须仍然紧跟着发出它的 assistant —— 折叠后留下孤儿 tool 消息
        # 就是网关 400 的那一类错误。
        for index, message in enumerate(sent):
            if message["role"] == "tool":
                self.assertEqual(sent[index - 1]["role"], "assistant")
                self.assertIn(
                    message["tool_call_id"],
                    {c["id"] for c in sent[index - 1]["tool_calls"]},
                )

        # 掉出窗口的 7 步仍以动作日志的形式可见。
        text = next(
            part["text"]
            for part in sent[-1]["content"]
            if part.get("type") == "text"
        )
        self.assertIn("Previous actions:", text)
        self.assertIn("Step 1: pyautogui.click(0, 20, button='left')", text)
        self.assertIn("Step 6: pyautogui.click(50, 20, button='left')", text)
        self.assertNotIn("Step 7:", text)

    def test_window_without_action_log_drops_history_outright(self):
        """v1 那套「只留最近 N 步」的策略仍然可配，且行为没变。

        全部 scaffold 现在都对齐到上游的 100/20/10 + 动作日志，这条守的是退路：
        history_turns == max_images_in_context 且关掉动作日志时，窗口外的 turn 连
        同它做过的事一起消失 —— 那正是 hard_app-f010 转圈的成因，改配置时要能一眼
        看出自己退回了哪里。
        """

        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "click", "arguments": json.dumps({"x": 500, "y": 500})},
        }
        protocol = dataclasses.replace(
            holo31_protocol(),
            history_turns=3,
            max_images_in_context=3,
            previous_action_log=False,
        )
        client = _FakeClient(
            [types.SimpleNamespace(content=None, tool_calls=[call]) for _ in range(6)]
        )
        agent = NativeToolComputerAgent(
            "Hcompany/Holo-3.1-35B-A3B", (1280, 800), protocol, client=client
        )
        for _ in range(6):
            agent.predict("Do the task", {"screenshot": b"png"})

        sent = client.completions.requests[-1]["messages"]
        self.assertEqual(len(sent), 1 + 3 * 3 + 1)
        dumped = json.dumps(sent, ensure_ascii=False)
        self.assertNotIn("Previous actions:", dumped)
        self.assertNotIn(_FOLDED_SCREENSHOT_PLACEHOLDER, dumped)

    def test_scroll_clamp_reaches_trajectory_and_is_told_to_the_model(self):
        # 截断只有同时满足两点才算「不静默」：写进 trajectory 供事后审计，且当轮
        # 就通过 tool 消息回告模型，让它下一步能自己改用格数。
        call = {
            "id": "call_scroll",
            "type": "function",
            "function": {
                "name": "scroll",
                "arguments": json.dumps({"x": 500, "y": 500, "delta_y": -300}),
            },
        }
        agent = NativeToolComputerAgent(
            "Hcompany/Holo-3.1-35B-A3B",
            (1280, 800),
            holo31_protocol(),
            client=_FakeClient(types.SimpleNamespace(content=None, tool_calls=[call])),
        )

        response, actions = agent.predict("Scroll down", {"screenshot": b"png"})

        self.assertEqual(actions, ["pyautogui.moveTo(640, 400)", "pyautogui.scroll(-30)"])
        clamps = [
            item
            for item in json.loads(response)["interventions"]
            if item["type"] == "scroll_clamp"
        ]
        self.assertEqual(len(clamps), 1)
        self.assertEqual(clamps[0]["requested_delta_y"], -300)
        self.assertEqual(clamps[0]["executed_delta_y"], -30)
        self.assertEqual(clamps[0]["tool_call_id"], "call_scroll")

        reply = agent._turns[0][-1]
        self.assertEqual(reply["tool_call_id"], "call_scroll")
        self.assertIn("notches", reply["content"])
        self.assertIn("-300", reply["content"])
        self.assertIn("-30", reply["content"])

    def test_in_range_scroll_gets_the_ordinary_tool_reply(self):
        call = {
            "id": "call_scroll",
            "type": "function",
            "function": {
                "name": "scroll",
                "arguments": json.dumps({"x": 500, "y": 500, "delta_y": -3}),
            },
        }
        agent = NativeToolComputerAgent(
            "Hcompany/Holo-3.1-35B-A3B",
            (1280, 800),
            holo31_protocol(),
            client=_FakeClient(types.SimpleNamespace(content=None, tool_calls=[call])),
        )

        _, actions = agent.predict("Scroll down", {"screenshot": b"png"})

        self.assertEqual(actions, ["pyautogui.moveTo(640, 400)", "pyautogui.scroll(-3)"])
        self.assertIn("Accepted for execution", agent._turns[0][-1]["content"])

    def test_invalid_holo_tool_call_gets_one_bounded_schema_repair(self):
        invalid = types.SimpleNamespace(
            content=None,
            tool_calls=[
                {
                    "id": "bad",
                    "type": "function",
                    "function": {
                        "name": "click",
                        "arguments": json.dumps({"546": "", "button": "left"}),
                    },
                }
            ],
        )
        valid = types.SimpleNamespace(
            content=None,
            tool_calls=[
                {
                    "id": "good",
                    "type": "function",
                    "function": {
                        "name": "click",
                        "arguments": json.dumps({"x": 500, "y": 500}),
                    },
                }
            ],
        )
        client = _FakeClient([invalid, valid])
        agent = NativeToolComputerAgent(
            "Hcompany/Holo-3.1-35B-A3B",
            (1280, 800),
            holo31_protocol(),
            client=client,
        )

        response, actions = agent.predict("Click", {"screenshot": b"png"})

        parsed = json.loads(response)
        self.assertEqual(actions, ["pyautogui.click(640, 400, button='left')"])
        self.assertEqual(len(client.completions.requests), 2)
        self.assertEqual(parsed["interventions"][0]["type"], "schema_repair")
        self.assertIn("Rejected by the frozen DERAIL tool schema", agent._turns[0][2]["content"])

    def test_invalid_tool_call_aborts_only_after_the_repair_budget_is_spent(self):
        def invalid(call_id):
            return types.SimpleNamespace(
                content=None,
                tool_calls=[
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "click",
                            "arguments": json.dumps({"546": ""}),
                        },
                    }
                ],
            )

        client = _FakeClient([invalid("bad1"), invalid("bad2"), invalid("bad3")])
        agent = NativeToolComputerAgent(
            "Hcompany/Holo-3.1-35B-A3B",
            (1280, 800),
            holo31_protocol(),
            client=client,
        )

        response, actions = agent.predict("Click", {"screenshot": b"png"})

        parsed = json.loads(response)
        self.assertEqual(actions, ["FAIL"])
        self.assertEqual(parsed["abort"]["type"], "INVALID_TOOL_CALL")
        self.assertEqual(len(parsed["interventions"]), 3)
        self.assertEqual(len(client.completions.requests), 3)

    def test_both_derail_scaffolds_share_schema_repair_budget(self):
        self.assertEqual(
            qwen36_protocol().schema_repair_attempts,
            holo31_protocol().schema_repair_attempts,
            "两个 DERAIL scaffold 的 schema_repair_attempts 必须一致，否则失败分布不可比",
        )
        self.assertEqual(holo31_protocol().schema_repair_attempts, 2)

    def _batch_message(self, call_id, names_and_arguments):
        return types.SimpleNamespace(
            content=None,
            tool_calls=[
                {
                    "id": f"{call_id}_{index}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
                for index, (name, arguments) in enumerate(names_and_arguments)
            ],
        )

    def test_wait_before_an_action_is_executed_in_order(self):
        # 「先等界面稳定，再点」语义上无害：MyPCBench 逐个执行，"WAIT" 就是 sleep。
        message = self._batch_message(
            "lead",
            [("wait", {"seconds": 2}), ("click", {"x": 500, "y": 500})],
        )
        agent = NativeToolComputerAgent(
            "Hcompany/Holo-3.1-35B-A3B",
            (1280, 800),
            holo31_protocol(),
            client=_FakeClient(message),
        )

        response, actions = agent.predict("Wait then click", {"screenshot": b"png"})

        self.assertEqual(actions, ["WAIT", "pyautogui.click(640, 400, button='left')"])
        self.assertEqual(json.loads(response)["interventions"], [])

    def test_qwen_repeated_waits_collapse_into_one_and_stay_auditable(self):
        message = self._batch_message(
            "waits", [("wait", {"seconds": 2}), ("wait", {"seconds": 3})]
        )
        agent = NativeToolComputerAgent(
            "Qwen/Qwen3.6-27B",
            (1280, 800),
            qwen36_protocol(),
            client=_FakeClient(message),
        )

        response, actions = agent.predict("Wait twice", {"screenshot": b"png"})

        parsed = json.loads(response)
        self.assertEqual(actions, ["WAIT"])
        # 折叠是一次 scaffold 干预，必须留痕，而不是静默改写模型输出。
        self.assertEqual(parsed["interventions"][0]["type"], "wait_normalization")
        self.assertEqual(parsed["interventions"][0]["collapsed_tool_call_ids"], ["waits_1"])
        # 模型原样发出的两个 tool call 都保留，且都有对应的 tool 回复。
        self.assertEqual(len(parsed["tool_calls"]), 2)
        self.assertEqual(
            [message["tool_call_id"] for message in agent._turns[0][2:]],
            ["waits_0", "waits_1"],
        )

    def test_holo_repeated_waits_are_all_executed_in_order(self):
        message = self._batch_message(
            "waits", [("wait", {"seconds": 2}), ("wait", {"seconds": 3})]
        )
        agent = NativeToolComputerAgent(
            "Hcompany/Holo-3.1-35B-A3B",
            (1280, 800),
            holo31_protocol(),
            client=_FakeClient(message),
        )

        response, actions = agent.predict("Wait twice", {"screenshot": b"png"})

        self.assertEqual(actions, ["WAIT", "WAIT"])
        self.assertEqual(json.loads(response)["interventions"], [])

    def test_batched_interactions_execute_in_order_without_a_repair(self):
        """多交互动作批次原样执行，不再触发 schema_repair。

        归因不靠"一屏一动作"这条限制：run_mypcbench.py 逐个 env.step、逐个存截图、
        逐个写 traj 行，一批 N 个动作照样得到 N 张图和 N 条记录。v1 里这条限制拒了
        513 次，主力是 click→click 和 click→write（填表单的点击-输入节奏）。
        """

        batched = self._batch_message(
            "batch",
            [
                ("click", {"x": 500, "y": 500}),
                ("write", {"content": "hello"}),
                ("click", {"x": 600, "y": 600}),
            ],
        )
        client = _FakeClient([batched])
        agent = NativeToolComputerAgent(
            "Hcompany/Holo-3.1-35B-A3B", (1280, 800), holo31_protocol(), client=client
        )

        response, actions = agent.predict("Search", {"screenshot": b"png"})

        self.assertEqual(
            actions,
            [
                "pyautogui.click(640, 400, button='left')",
                "pyautogui.write('hello', interval=0.01)",
                "pyautogui.click(767, 479, button='left')",
            ],
        )
        parsed = json.loads(response)
        self.assertEqual(parsed["interventions"], [])
        # 只发了一次请求：没有被打回重说。
        self.assertEqual(len(client.completions.requests), 1)

    def test_answer_may_close_the_turn_it_shares_with_an_action(self):
        # click→answer 在 v1 里出现 27 次："点完这一下就交卷"。以前 answer 被打回，
        # 要等下一轮才发得出去；中途若被别的机制终止，这个答案就永远丢了。
        batched = self._batch_message(
            "finish",
            [("click", {"x": 500, "y": 500}), ("answer", {"status": "success"})],
        )
        agent = NativeToolComputerAgent(
            "Hcompany/Holo-3.1-35B-A3B",
            (1280, 800),
            holo31_protocol(),
            client=_FakeClient([batched]),
        )

        _response, actions = agent.predict("Finish", {"screenshot": b"png"})

        self.assertEqual(actions, ["pyautogui.click(640, 400, button='left')", "DONE"])

    def test_missing_tool_call_gets_one_bounded_schema_repair(self):
        missing = types.SimpleNamespace(content="plain text", tool_calls=None)
        valid = self._action_message("fixed", "wait", {"seconds": 1})
        client = _FakeClient([missing, valid])
        agent = NativeToolComputerAgent(
            "Hcompany/Holo-3.1-35B-A3B",
            (1280, 800),
            holo31_protocol(),
            client=client,
        )

        response, actions = agent.predict("Wait", {"screenshot": b"png"})

        self.assertEqual(actions, ["WAIT"])
        self.assertEqual(json.loads(response)["interventions"][0]["type"], "schema_repair")
        self.assertEqual(agent._turns[0][2]["role"], "user")

    def test_one_interaction_followed_by_wait_is_narrowly_accepted(self):
        message = types.SimpleNamespace(
            content=None,
            tool_calls=[
                {
                    "id": "click",
                    "type": "function",
                    "function": {
                        "name": "click",
                        "arguments": json.dumps({"x": 800, "y": 560}),
                    },
                },
                {
                    "id": "wait",
                    "type": "function",
                    "function": {
                        "name": "wait",
                        "arguments": json.dumps({"seconds": 1}),
                    },
                },
            ],
        )
        agent = NativeToolComputerAgent(
            "Hcompany/Holo-3.1-35B-A3B",
            (1280, 800),
            holo31_protocol(),
            client=_FakeClient(message),
        )

        _response, actions = agent.predict("Click then wait", {"screenshot": b"png"})

        self.assertEqual(actions, ["pyautogui.click(1023, 447, button='left')", "WAIT"])

    def test_consecutive_clicks_run_as_one_batch(self):
        # v1 里被拒最多的形状（299 次）：菜单连选、列表逐条勾。
        batched = types.SimpleNamespace(
            content=None,
            tool_calls=[
                {
                    "id": "first",
                    "type": "function",
                    "function": {
                        "name": "click",
                        "arguments": json.dumps({"x": 500, "y": 500}),
                    },
                },
                {
                    "id": "second",
                    "type": "function",
                    "function": {
                        "name": "click",
                        "arguments": json.dumps({"x": 600, "y": 600}),
                    },
                },
            ],
        )
        agent = NativeToolComputerAgent(
            "Hcompany/Holo-3.1-35B-A3B",
            (1280, 800),
            holo31_protocol(),
            client=_FakeClient([batched]),
        )

        response, actions = agent.predict("Repair", {"screenshot": b"png"})

        self.assertEqual(
            actions,
            [
                "pyautogui.click(640, 400, button='left')",
                "pyautogui.click(767, 479, button='left')",
            ],
        )
        self.assertEqual(json.loads(response)["interventions"], [])

    def test_qwen_consecutive_clicks_still_require_schema_repair(self):
        batched = self._batch_message(
            "batch",
            [("click", {"x": 500, "y": 500}), ("click", {"x": 600, "y": 600})],
        )
        repaired = self._action_message("fixed", "click", {"x": 500, "y": 500})
        client = _FakeClient([batched, repaired])
        agent = NativeToolComputerAgent(
            "Qwen/Qwen3.6-27B", (1280, 800), qwen36_protocol(), client=client
        )

        response, actions = agent.predict("Click once", {"screenshot": b"png"})

        self.assertEqual(actions, ["pyautogui.click(500, 500, button='left')"])
        self.assertEqual(json.loads(response)["interventions"][0]["type"], "schema_repair")
        self.assertEqual(len(client.completions.requests), 2)

    def test_holo_answer_must_be_the_last_call(self):
        invalid = self._batch_message(
            "invalid",
            [("answer", {"status": "success"}), ("click", {"x": 500, "y": 500})],
        )
        repaired = self._action_message("fixed", "answer", {"status": "success"})
        agent = NativeToolComputerAgent(
            "Hcompany/Holo-3.1-35B-A3B",
            (1280, 800),
            holo31_protocol(),
            client=_FakeClient([invalid, repaired]),
        )

        response, actions = agent.predict("Finish", {"screenshot": b"png"})

        self.assertEqual(actions, ["DONE"])
        self.assertEqual(json.loads(response)["interventions"][0]["type"], "schema_repair")

    def test_invalid_argument_inside_a_batch_still_gets_repaired(self):
        # 放开批处理不等于放开校验：批次里任何一个 call 参数不合法，整批仍被打回。
        bad = self._batch_message(
            "bad",
            [("click", {"x": 500, "y": 500}), ("click", {"x": 99999, "y": 1})],
        )
        good = self._action_message("fixed", "click", {"x": 500, "y": 500})
        agent = NativeToolComputerAgent(
            "Hcompany/Holo-3.1-35B-A3B",
            (1280, 800),
            holo31_protocol(),
            client=_FakeClient([bad, good]),
        )

        response, actions = agent.predict("Repair", {"screenshot": b"png"})

        self.assertEqual(actions, ["pyautogui.click(640, 400, button='left')"])
        self.assertEqual(json.loads(response)["interventions"][0]["type"], "schema_repair")

    @staticmethod
    def _action_message(call_id, name, arguments):
        return types.SimpleNamespace(
            content=None,
            tool_calls=[
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        )

    def _alternating_cycle(self, rounds):
        """交替发出 click / esc，共 2*rounds 条消息。"""

        sequence = []
        for index in range(2 * rounds):
            if index % 2 == 0:
                sequence.append(
                    self._action_message(f"a{index}", "click", {"x": 874, "y": 160})
                )
            else:
                sequence.append(self._action_message(f"b{index}", "hotkey", {"keys": ["esc"]}))
        return sequence

    @staticmethod
    def _guarded_protocol():
        """出厂 protocol 已停用守卫；守卫机制本身仍需覆盖，这里显式打开。

        用 v1 曾经的阈值（8 / 15），这样这组测试同时也是"要复现 v1 失败分布该怎么
        配"的可执行文档。
        """

        return dataclasses.replace(
            qwen36_protocol(),
            alternating_action_repeat_limit=8,
            stalled_state_step_limit=15,
        )

    def test_alternating_loop_gets_one_replan_and_accepts_different_action(self):
        protocol = self._guarded_protocol()
        limit = protocol.alternating_action_repeat_limit
        sequence = self._alternating_cycle(limit)
        sequence.append(self._action_message("repair", "move", {"x": 100, "y": 100}))
        client = _FakeClient(sequence)
        agent = NativeToolComputerAgent(
            "Qwen/Qwen3.6-27B", (1280, 800), protocol, client=client
        )
        # 循环要满 limit 轮才算数；在那之前每一步都必须原样执行。
        for index in range(2 * limit - 1):
            _, actions = agent.predict("Task", {"screenshot": f"png-{index}".encode()})
            self.assertNotEqual(actions, ["FAIL"])

        response, actions = agent.predict("Task", {"screenshot": b"png-last"})

        self.assertEqual(actions, ["pyautogui.moveTo(100, 100)"])
        self.assertEqual(json.loads(response)["interventions"][0]["type"], "loop_repair")
        self.assertEqual(len(client.completions.requests), 2 * limit + 1)

    def test_stall_guard_fires_when_screen_frozen_despite_varied_actions(self):
        # 每一步动作都不同，只有屏幕不变。动作重复类的检测器全都不会触发，
        # 命中的必须是空转检测；模型随后用 answer(failure) 逃生。
        sequence = [
            self._action_message(f"m{index}", "move", {"x": 100 + index, "y": 100})
            for index in range(15)
        ]
        sequence.append(self._action_message("bail", "answer", {"status": "failure"}))
        client = _FakeClient(sequence)
        agent = NativeToolComputerAgent(
            "Qwen/Qwen3.6-27B", (1280, 800), self._guarded_protocol(), client=client
        )
        for _ in range(14):
            _, actions = agent.predict("Task", {"screenshot": b"frozen"})
            self.assertNotEqual(actions, ["FAIL"])

        response, actions = agent.predict("Task", {"screenshot": b"frozen"})

        parsed = json.loads(response)
        self.assertEqual(actions, ["FAIL"])
        self.assertEqual(parsed["interventions"][0]["type"], "loop_repair")
        self.assertIn("unchanged for 15", parsed["interventions"][0]["reason"])

    def test_stall_guard_ignores_moving_screen(self):
        # 屏幕每步都在变时，即使远超 15 步也不能触发空转检测。
        sequence = [
            self._action_message(f"m{index}", "move", {"x": 100 + index, "y": 100})
            for index in range(20)
        ]
        agent = NativeToolComputerAgent(
            "Qwen/Qwen3.6-27B", (1280, 800), self._guarded_protocol(), client=_FakeClient(sequence)
        )

        for index in range(20):
            _, actions = agent.predict("Task", {"screenshot": f"png-{index}".encode()})
            self.assertEqual(actions, [f"pyautogui.moveTo({100 + index}, 100)"])

    def test_both_derail_scaffolds_ship_with_loop_guards_disabled(self):
        # 守卫已停用：v1 实测它掐掉的绝大多数是没走完的长任务而不是死循环
        # （Holo 70/184 局被掐，均分 0.066 vs 自然 DONE 的 0.379）。终止只由
        # max_steps 决定。两个 scaffold 必须同口径，否则失败分布不可比。
        qwen, holo = qwen36_protocol(), holo31_protocol()
        for field in ("alternating_action_repeat_limit", "stalled_state_step_limit"):
            self.assertEqual(
                getattr(qwen, field),
                getattr(holo, field),
                f"两个 DERAIL scaffold 的 {field} 必须一致，否则失败分布不可比",
            )
            self.assertEqual(getattr(holo, field), 0, f"{field} 应为 0（守卫停用）")

    def test_disabled_guard_never_aborts_even_on_a_hard_cycle(self):
        # 停用后，把 v1 会被判死循环的形状原样喂进去，必须一步不落地执行。
        sequence = self._alternating_cycle(12)
        agent = NativeToolComputerAgent(
            "Qwen/Qwen3.6-27B", (1280, 800), qwen36_protocol(), client=_FakeClient(sequence)
        )

        for _ in range(24):
            response, actions = agent.predict("Task", {"screenshot": b"frozen"})
            self.assertNotEqual(actions, ["FAIL"])
            self.assertEqual(json.loads(response).get("abort"), None)

    def test_alternating_loop_aborts_if_replan_repeats_cycle(self):
        protocol = self._guarded_protocol()
        limit = protocol.alternating_action_repeat_limit
        sequence = self._alternating_cycle(limit)
        sequence.append(self._action_message("again", "hotkey", {"keys": ["esc"]}))
        agent = NativeToolComputerAgent(
            "Qwen/Qwen3.6-27B",
            (1280, 800),
            protocol,
            client=_FakeClient(sequence),
        )
        for index in range(2 * limit - 1):
            agent.predict("Task", {"screenshot": f"png-{index}".encode()})

        response, actions = agent.predict("Task", {"screenshot": b"png-last"})

        parsed = json.loads(response)
        self.assertEqual(actions, ["FAIL"])
        self.assertEqual(parsed["abort"]["type"], "LOOP_ABORT")
        agent.reset()
        self.assertEqual(agent._action_batches, [])
        self.assertEqual(agent._state_actions, [])

    def test_repeated_same_action_replans_only_after_the_repeat_limit(self):
        protocol = self._guarded_protocol()
        limit = protocol.alternating_action_repeat_limit
        sequence = [
            self._action_message(f"c{index}", "click", {"x": 762, "y": 291})
            for index in range(limit)
        ]
        sequence.append(self._action_message("repair", "hotkey", {"keys": ["ctrl", "l"]}))
        agent = NativeToolComputerAgent(
            "Qwen/Qwen3.6-27B", (1280, 800), protocol, client=_FakeClient(sequence)
        )

        # 前 limit-1 次连点必须原样通过：连点几下同一个位置（翻页、逐条勾选）是
        # 正常操作，不能当死循环处理。
        for _ in range(limit - 1):
            _, actions = agent.predict("Task", {"screenshot": b"state"})
            self.assertEqual(actions, ["pyautogui.click(762, 291, button='left')"])

        response, actions = agent.predict("Task", {"screenshot": b"state"})

        self.assertEqual(actions, ["pyautogui.hotkey('ctrl', 'l')"])
        self.assertEqual(json.loads(response)["interventions"][0]["type"], "loop_repair")

    def test_repeating_a_screen_action_pair_is_not_a_loop_on_its_own(self):
        # 曾经有一条"同画面 + 同动作累计出现 3 次"的判据，已移除：它不要求连续，
        # 会把逐条处理列表这类正常任务误判成死循环。
        actions_by_step = [("click", {"x": 100, "y": 100}), ("click", {"x": 200, "y": 200})]
        sequence = [
            self._action_message(f"s{index}", *actions_by_step[index % 2])
            for index in range(6)
        ]
        agent = NativeToolComputerAgent(
            "Qwen/Qwen3.6-27B", (1280, 800), self._guarded_protocol(), client=_FakeClient(sequence)
        )

        for index in range(6):
            _response, actions = agent.predict(
                "Task", {"screenshot": b"page-a" if index % 2 == 0 else b"page-b"}
            )
            self.assertNotEqual(actions, ["FAIL"])

    def test_content_json_fallback_for_vllm_without_tool_parser(self):
        message = types.SimpleNamespace(
            content='<tool_call>{"name":"wait","arguments":{"seconds":2}}</tool_call>',
            tool_calls=None,
        )
        agent = NativeToolComputerAgent(
            "Qwen/Qwen3.6-27B",
            (1280, 800),
            qwen36_protocol(),
            client=_FakeClient(message),
        )
        _response, actions = agent.predict("Wait", {"screenshot": b"png"})
        self.assertEqual(actions, ["WAIT"])

    def test_factory_exposes_local_tool_agents(self):
        qwen = create_mypcbench_agent("derail_qwen36", "qwen", (1280, 800), "pw")
        qwen38 = create_mypcbench_agent("derail_qwen38", "qwen38", (1280, 800), "pw")
        holo = create_mypcbench_agent("derail_holo31", "holo", (1280, 800), "pw")
        kimi = create_mypcbench_agent("derail_kimi_k3", "kimi-k3", (1280, 800), "pw")
        self.assertEqual(qwen.protocol.coordinate_protocol, "absolute_pixels")
        self.assertEqual(qwen.protocol.tool_choice, "auto")
        self.assertEqual(qwen38.protocol.coordinate_protocol, "absolute_pixels")
        self.assertEqual(qwen38.protocol.tool_choice, "auto")
        self.assertEqual(holo.protocol.coordinate_protocol, "normalized_0_1000")
        # 两个 scaffold 统一 auto：required 逼模型每轮必须调工具，想说"做不到"时
        # 没有出口，v1 里 Holo 因此一次都没主动发过 fail，70 次 FAIL 全是守卫掐的。
        self.assertEqual(holo.protocol.tool_choice, "auto")
        # kimi 同样统一 auto；temperature 必须为 None（网关拒绝显式 temperature）。
        self.assertEqual(kimi.protocol.coordinate_protocol, "absolute_pixels")
        self.assertEqual(kimi.protocol.tool_choice, "auto")
        self.assertIsNone(kimi.protocol.temperature)

    def test_kimi_request_omits_temperature_but_keeps_budget(self):
        """kimi-k3 显式传 temperature 会被 400 拒绝；请求体必须省略该字段。

        对照组 qwen36 继续携带 temperature=0.0，确认改动只影响 None 分支。
        """

        call = {
            "id": "click_0",
            "type": "function",
            "function": {
                "name": "click",
                "arguments": json.dumps({"x": 500, "y": 500}),
            },
        }
        message = types.SimpleNamespace(content=None, tool_calls=[call])

        kimi_client = _FakeClient(message)
        kimi = NativeToolComputerAgent(
            "kimi-k3", (1280, 800), kimi_k3_protocol(), client=kimi_client
        )
        kimi.predict("Open settings", {"screenshot": b"png"})
        kimi_request = kimi_client.completions.requests[0]
        self.assertNotIn(
            "temperature",
            kimi_request,
            "kimi-k3 拒绝 temperature 参数，请求体不得携带",
        )
        self.assertGreaterEqual(
            kimi_request["max_tokens"],
            4096,
            "reasoning tokens 计入预算，kimi 的 max_tokens 不得小于 4096",
        )

        qwen_client = _FakeClient(message)
        qwen = NativeToolComputerAgent(
            "Qwen/Qwen3.6-27B", (1280, 800), qwen36_protocol(), client=qwen_client
        )
        qwen.predict("Open settings", {"screenshot": b"png"})
        self.assertEqual(qwen_client.completions.requests[0]["temperature"], 0.0)

    def test_shipped_protocol_comes_from_yaml(self):
        """改 yaml 必须真的改变 decoder 行为。

        v1 之前这些 yaml 是纯文档，改了不生效：holo 的 yaml 写 temperature 0.0 而
        代码跑的是 0.8，写 coordinate_protocol 却根本没人读。当时的补救是让测试
        比对两边；现在 yaml 是唯一权威，所以这里改的是断言方向 —— 把配置目录指
        向一份改过的副本，agent 必须跟着变。逐字段的取值由
        tests/test_agent_config.py 的金标快照钉住。
        """

        source = pathlib.Path(__file__).resolve().parents[1] / "configs" / "agents"
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "configs" / "agents").mkdir(parents=True)
            for path in source.glob("*.yaml"):
                shutil.copy(path, root / "configs" / "agents" / path.name)
            patched = root / "configs" / "agents" / "holo_3_1_35b_a3b.yaml"
            patched.write_text(
                patched.read_text(encoding="utf-8")
                .replace("tool_choice: auto", "tool_choice: required")
                .replace("max_images_in_context: 20", "max_images_in_context: 7")
                .replace("history_turns: 100", "history_turns: 9")
                .replace("previous_action_log: true", "previous_action_log: false"),
                encoding="utf-8",
            )
            with mock.patch.object(agent_config, "REPO_ROOT", root):
                holo = create_mypcbench_agent("derail_holo31", "holo", (1280, 800), "pw")
                self.assertEqual(holo.protocol.tool_choice, "required")
                self.assertEqual(holo.protocol.max_images_in_context, 7)
                self.assertEqual(holo.protocol.history_turns, 9)
                self.assertFalse(holo.protocol.previous_action_log)
                # 没改的那份不受影响。
                qwen = create_mypcbench_agent("derail_qwen36", "qwen", (1280, 800), "pw")
                self.assertEqual(qwen.protocol.tool_choice, "auto")

    def test_yaml_matches_shipped_protocol(self):
        """四个 derail_tool_agent 的协议必须逐字段来自各自的 yaml。

        kimi_k3 以前在 tool_agent.py 里自建协议，max_tokens 还挂在
        MYPCBENCH_KIMI_MAX_TOKENS 这个环境变量上，既不在 yaml 也不在这条测试的
        字段清单里 —— 两层护栏都漏掉了同两个字段。现在四个走同一条装配路径。
        """

        import yaml

        config_dir = pathlib.Path(__file__).resolve().parents[1] / "configs" / "agents"
        for agent_id, protocol in (
            ("holo_3_1_35b_a3b", holo31_protocol()),
            ("qwen3_6_27b", qwen36_protocol()),
            ("qwen3_8_27b", qwen38_protocol()),
            ("kimi_k3", kimi_k3_protocol()),
            ("kimi_k3_cuabash", kimi_k3_cuabash_protocol()),
        ):
            with self.subTest(agent_id=agent_id):
                config = yaml.safe_load((config_dir / f"{agent_id}.yaml").read_text())
                self.assertEqual(config["scaffold"], "derail_tool_agent")
                self.assertEqual(config["agent_id"], protocol.agent_id)
                for field in (
                    "coordinate_protocol",
                    "temperature",
                    "max_tokens",
                    "max_images_in_context",
                    "history_turns",
                    "image_fold_size",
                    "previous_action_log",
                    "tool_choice",
                    "schema_repair_attempts",
                    "multi_tool_policy",
                    "alternating_action_repeat_limit",
                    "stalled_state_step_limit",
                ):
                    self.assertEqual(
                        config[field],
                        getattr(protocol, field),
                        f"{agent_id}.yaml 的 {field} 与实际装配的协议不一致",
                    )
                # system_prompt 是唯一一个 yaml 存文件名、协议存内容的字段，而且
                # 内容是拼出来的：scaffold 段（yaml 指名的那个文件）+ 共享环境块
                # —— enable_bash 决定拼哪份变体（bash 版带 sudo 密码与 CLI 行）。
                prompt_dir = config_dir.parents[1] / "prompts" / "agents"
                scaffold = (prompt_dir / config["system_prompt_file"]).read_text(
                    encoding="utf-8"
                )
                shared_name = (
                    "mypcbench_shared_block_bash.txt"
                    if config["enable_bash"]
                    else "mypcbench_shared_block.txt"
                )
                shared = (prompt_dir / shared_name).read_text(encoding="utf-8")
                self.assertEqual(protocol.system_prompt, (scaffold + shared).strip())
                # 环境块与模型无关，缺了 agent 就不知道 app 在 localhost:PORT——
                # 2026-08-10 的 smoke 里 EvoCUA 正是因此把 Cheskepdia 当成公网站点。
                # scaffold 文件里则不该再有副本，两份只会不声不响地漂开。
                self.assertIn("3012 | Cheskepdia", protocol.system_prompt)
                self.assertNotIn("## Persona", scaffold)
                # 完成纪律：判官只读最后一条回复，模型必须知道终止前要写下答案。
                self.assertIn("Task completion discipline", protocol.system_prompt)
                # GUI-only agent 不提 bash（对没有 shell 的 agent 提终端，小模型
                # 会去 GUI 终端里敲命令——kimi_k3 的 F3 就是这么丢的 109 次）；
                # cuabash 变体保留从句，并拿到 sudo 密码行。
                if config["enable_bash"]:
                    self.assertIn("different bash command", protocol.system_prompt)
                    self.assertIn("sudo password", protocol.system_prompt)
                else:
                    self.assertNotIn("different bash command", protocol.system_prompt)

    def test_upstream_scaffold_configs_are_marked_as_such(self):
        """不经过 SafePyAutoGUICompiler 的 config 必须自报家门。

        它们的 coordinate_protocol 用的是上游词汇（normalized_0_999 /
        smart_resize_absolute_pixels），不在 SafePyAutoGUICompiler 的枚举里 ——
        因为它们根本不经过那个编译器。没有这个标记就会被误读成配错了。

        两个值的区别：upstream_official 由 DERAIL factory 构造（yaml 是 live
        权威），upstream_runner 由 MyPCBench 自己构造（yaml 纯文档）。以前这两种
        共用 upstream_official 一个值，同一个标签底下配置权威性正好相反。
        """

        import yaml

        config_dir = pathlib.Path(__file__).resolve().parents[1] / "configs" / "agents"
        for agent_id, expected in (
            ("evocua_32b", "upstream_official"),
            ("opencua_72b", "upstream_official"),
            ("qwen3_5_35b_a3b", "upstream_runner"),
        ):
            with self.subTest(agent_id=agent_id):
                config = yaml.safe_load((config_dir / f"{agent_id}.yaml").read_text())
                self.assertEqual(config["scaffold"], expected)
                self.assertNotIn(
                    "tool_choice",
                    config,
                    "上游 scaffold 没有 tool_choice，写上去会让人以为能调",
                )

    def test_upstream_guard_is_one_beyond_runner_budget(self):
        config = load_agent_config("opencua_72b")
        with mock.patch.dict("os.environ", {"DERAIL_AGENT_MAX_STEPS": "8"}):
            self.assertEqual(_upstream_step_budget(config), 9)
        # runner 没导出上限时才回落到 yaml 的 upstream_max_steps_fallback。
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                _upstream_step_budget(config), config["upstream_max_steps_fallback"] + 1
            )
        for invalid in ("0", "-1"):
            with self.subTest(invalid=invalid), mock.patch.dict(
                "os.environ", {"DERAIL_AGENT_MAX_STEPS": invalid}
            ), self.assertRaises(ValueError):
                _upstream_step_budget(config)


class BashToolAgentTests(unittest.TestCase):
    """kimi_k3_cuabash 的 bash 分流：gpt 式 agent-internal 语义的逐条钉死。

    2026-08-20 立项的对照组核心约定（见 configs/agents/kimi_k3_cuabash.yaml）：
    bash 在 predict 内执行、结果以 tool 消息回填后继续对话，不消耗 runner 的
    max_steps 步数；一次一条命令、单独成批；护栏 = 命令长度 + 输出截断 + 轮数
    上限。这里每条都对应一个测试，谁漂了跨组对比就不再成立。
    """

    def _bash_message(self, call_id, command):
        return types.SimpleNamespace(
            content=None,
            tool_calls=[
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": command}),
                    },
                }
            ],
        )

    def _click_message(self, call_id="click"):
        return types.SimpleNamespace(
            content=None,
            tool_calls=[
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "click",
                        "arguments": json.dumps({"x": 500, "y": 500}),
                    },
                }
            ],
        )

    def _agent(self, client, env):
        return NativeToolComputerAgent(
            "kimi-k3",
            (1280, 800),
            kimi_k3_cuabash_protocol(),
            client=client,
            env=env,
        )

    def test_bash_round_executes_via_env_and_continues_the_turn(self):
        client = _FakeClient([self._bash_message("b1", "ls -la"), self._click_message("c1")])
        env = _FakeEnv(output="total 0", returncode=0)
        agent = self._agent(client, env)

        response, actions = agent.predict("Do it", {"screenshot": b"png"})

        # bash 轮不产生 GUI action：最终动作只有 click，runner 只见一步。
        self.assertEqual(actions, ["pyautogui.click(500, 500, button='left')"])
        # 命令走 runner 注入的 env 通道，与 qwen_cuabash / openai_cuabash 同源。
        self.assertEqual(env.commands, [("ls -la", True)])
        # 两次 chat 请求：bash 轮 + 收到结果后的 click 轮 —— 同一 predict 内续话。
        self.assertEqual(len(client.completions.requests), 2)
        sent = client.completions.requests[-1]["messages"]
        assistant, tool_reply = sent[-2], sent[-1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "bash")
        # tool 回复紧跟着它的 assistant，id 配对 —— 网关对孤儿 tool 消息会 400。
        self.assertEqual(tool_reply["role"], "tool")
        self.assertEqual(tool_reply["tool_call_id"], "b1")
        self.assertIn("exit code: 0", tool_reply["content"])
        self.assertIn("total 0", tool_reply["content"])
        parsed = json.loads(response)
        # 每轮 bash 都留痕：命令、退出码、双流长度，供事后审计 bash 依赖度。
        self.assertEqual(parsed["interventions"][0]["type"], "bash_round")
        self.assertEqual(parsed["interventions"][0]["command"], "ls -la")
        self.assertEqual(parsed["interventions"][0]["exit_code"], 0)
        self.assertFalse(parsed["interventions"][0]["truncated"])
        # compiled_actions 只含 GUI 动作，bash 不冒充屏幕交互。
        self.assertEqual(parsed["compiled_actions"][0]["name"], "click")

    def test_bash_must_arrive_alone(self):
        # bash 与 GUI 动作混发：整批打回一次 schema_repair，命令不执行。
        mixed = types.SimpleNamespace(
            content=None,
            tool_calls=[
                {
                    "id": "b1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": "ls"}),
                    },
                },
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "click",
                        "arguments": json.dumps({"x": 500, "y": 500}),
                    },
                },
            ],
        )
        client = _FakeClient([mixed, self._click_message("fixed")])
        env = _FakeEnv()
        agent = self._agent(client, env)

        response, actions = agent.predict("Mixed", {"screenshot": b"png"})

        self.assertEqual(actions, ["pyautogui.click(500, 500, button='left')"])
        parsed = json.loads(response)
        self.assertEqual(parsed["interventions"][0]["type"], "schema_repair")
        self.assertEqual(len(client.completions.requests), 2)
        self.assertEqual(env.commands, [])

    def test_bash_budget_abort_fails_the_step(self):
        # 模型连续 bash 不回头：烧满 8 轮预算后第 9 轮判 FAIL，全部轮次留痕。
        client = _FakeClient(
            [self._bash_message(f"b{i}", f"cmd{i}") for i in range(_MAX_BASH_ROUNDS_PER_STEP + 1)]
        )
        env = _FakeEnv()
        agent = self._agent(client, env)

        response, actions = agent.predict("Loop bash", {"screenshot": b"png"})

        self.assertEqual(actions, ["FAIL"])
        parsed = json.loads(response)
        self.assertEqual(parsed["abort"]["type"], "BASH_BUDGET_ABORT")
        self.assertEqual(len(parsed["interventions"]), _MAX_BASH_ROUNDS_PER_STEP)
        self.assertEqual(len(env.commands), _MAX_BASH_ROUNDS_PER_STEP)
        self.assertEqual(len(client.completions.requests), _MAX_BASH_ROUNDS_PER_STEP + 1)

    def test_steps_accounting_returns_each_bash_round_as_an_empty_step(self):
        # 统一记账（DERAIL_BASH_ACCOUNTING=steps）：bash 轮不再步内续话，而是
        # 作为完整一步返回空 actions，交给 runner 记 TOOL_CALL 行并计入步数。
        # runner 的 has_pending 判定靠 agent.messages（含 system prompt 恒非空）。
        with mock.patch.dict(os.environ, {"DERAIL_BASH_ACCOUNTING": "steps"}):
            client = _FakeClient([self._bash_message("b1", "ls -la"), self._click_message("c1")])
            env = _FakeEnv(output="total 0", returncode=0)
            agent = self._agent(client, env)

            # 第一 predict：bash 轮 -> 空 actions（runner 将记 TOOL_CALL 并计一步）。
            response, actions = agent.predict("Do it", {"screenshot": b"png"})
            self.assertEqual(actions, [])
            parsed = json.loads(response)
            self.assertEqual(parsed["interventions"][0]["type"], "bash_round")
            self.assertEqual(parsed["interventions"][0]["command"], "ls -la")
            self.assertEqual(env.commands, [("ls -la", True)])
            self.assertEqual(agent._consecutive_bash_steps, 1)
            # messages 恒非空 -> runner has_pending 为真 -> 走 TOOL_CALL 分支。
            self.assertTrue(len(agent.messages) > 0)
            self.assertTrue(agent._turns)  # 该 bash 轮已落成一个完整 turn

            # 第二 predict：续话，模型回到 GUI -> click 落地，连发计数清零。
            response2, actions2 = agent.predict("Do it", {"screenshot": b"png"})
            self.assertEqual(actions2, ["pyautogui.click(500, 500, button='left')"])
            self.assertEqual(agent._consecutive_bash_steps, 0)
            # 第二次请求里要能看到上一轮 bash 的 assistant+tool 回填（历史续接）。
            sent = client.completions.requests[-1]["messages"]
            tool_replies = [m for m in sent if m.get("role") == "tool"]
            self.assertTrue(any(m.get("tool_call_id") == "b1" for m in tool_replies))

    def test_steps_accounting_aborts_on_pathological_consecutive_bash(self):
        # steps 模式的安全阀：连发 bash 超过 _MAX_CONSECUTIVE_BASH_STEPS 仍不回
        # GUI，判 FAIL。这里把阈值 patch 小以便快速触发（生产值为 64）。
        with mock.patch.dict(os.environ, {"DERAIL_BASH_ACCOUNTING": "steps"}), \
             mock.patch(
                 "derail.mypcbench.tool_agent._MAX_CONSECUTIVE_BASH_STEPS", 3
             ):
            client = _FakeClient([self._bash_message(f"b{i}", f"cmd{i}") for i in range(4)])
            env = _FakeEnv(output="ok", returncode=0)
            agent = self._agent(client, env)

            for _ in range(3):
                _resp, actions = agent.predict("Loop bash", {"screenshot": b"png"})
                self.assertEqual(actions, [])
            response, actions = agent.predict("Loop bash", {"screenshot": b"png"})
            self.assertEqual(actions, ["FAIL"])
            parsed = json.loads(response)
            self.assertEqual(parsed["abort"]["type"], "BASH_BUDGET_ABORT")
            self.assertIn("consecutive", parsed["abort"]["reason"])
            self.assertEqual(len(env.commands), 3)  # 第 4 轮被拦，未执行

    def test_steps_accounting_rejects_unknown_mode(self):
        with mock.patch.dict(os.environ, {"DERAIL_BASH_ACCOUNTING": "bogus"}):
            with self.assertRaises(ValueError):
                self._agent(_FakeClient([self._click_message()]), _FakeEnv())


    def test_bash_without_env_feeds_error_text_back(self):
        # 漏接 env（装配事故）不崩 predict：错误文本回填，模型转回 GUI 路径。
        client = _FakeClient([self._bash_message("b1", "ls"), self._click_message("c1")])
        agent = self._agent(client, env=None)

        response, actions = agent.predict("No env", {"screenshot": b"png"})

        self.assertEqual(actions, ["pyautogui.click(500, 500, button='left')"])
        sent = client.completions.requests[-1]["messages"]
        self.assertIn("no VM environment", sent[-1]["content"])
        round_info = json.loads(response)["interventions"][0]
        self.assertIsNone(round_info["exit_code"])
        self.assertEqual(round_info["error"], "no_env")

    def test_bash_output_is_truncated_and_audited(self):
        # 单条命令打爆上下文的护栏：回填截断，intervention 记原始长度。
        client = _FakeClient(
            [self._bash_message("b1", "cat big"), self._click_message("c1")]
        )
        huge = "x" * (_BASH_OUTPUT_CAP * 2)
        agent = self._agent(client, env=_FakeEnv(output=huge))

        response, _actions = agent.predict("Big output", {"screenshot": b"png"})

        round_info = json.loads(response)["interventions"][0]
        self.assertTrue(round_info["truncated"])
        self.assertEqual(round_info["stdout_chars"], _BASH_OUTPUT_CAP * 2)
        sent = client.completions.requests[-1]["messages"]
        self.assertIn(f"(output truncated at {_BASH_OUTPUT_CAP} chars per stream)", sent[-1]["content"])
        self.assertNotIn(huge, sent[-1]["content"])

    def test_only_the_cuabash_protocol_ships_the_bash_tool(self):
        gui_names = {tool["function"]["name"] for tool in build_computer_tools(1279, 799)}
        bash_names = {
            tool["function"]["name"]
            for tool in build_computer_tools(1279, 799, include_bash=True)
        }
        self.assertNotIn("bash", gui_names)
        self.assertEqual(bash_names - gui_names, {"bash"})
        self.assertFalse(kimi_k3_protocol().enable_bash)
        self.assertTrue(kimi_k3_cuabash_protocol().enable_bash)
        # schema 公布的长度上限必须与解码校验同源，否则模型看到的契约是假的。
        bash_tool = next(
            tool
            for tool in build_computer_tools(1279, 799, include_bash=True)
            if tool["function"]["name"] == "bash"
        )
        command = bash_tool["function"]["parameters"]["properties"]["command"]
        self.assertEqual(command.get("minLength"), 1)
        self.assertEqual(command.get("maxLength"), 2000)
        self.assertEqual(bash_tool["function"]["parameters"]["required"], ["command"])


    def test_factory_wires_env_into_the_cuabash_agent_only(self):
        # runner 的 get_agent(..., env=env) 经 factory kwargs 落到 agent._env：
        # cuabash 拿到执行通道，GUI-only 的 kimi_k3 保持 None（不触碰）。
        env = _FakeEnv()
        bash_agent = create_mypcbench_agent(
            "derail_kimi_k3_cuabash", "kimi-k3", (1280, 800), "pw", env=env
        )
        self.assertIs(bash_agent._env, env)
        self.assertTrue(bash_agent.protocol.enable_bash)
        self.assertIn(
            "bash", {tool["function"]["name"] for tool in bash_agent.tools}
        )

        gui_agent = create_mypcbench_agent(
            "derail_kimi_k3", "kimi-k3", (1280, 800), "pw", env=env
        )
        self.assertIsNone(gui_agent._env)
        self.assertNotIn(
            "bash", {tool["function"]["name"] for tool in gui_agent.tools}
        )


class LaunchContractTests(unittest.TestCase):
    def test_exact_served_model_and_context_are_required(self):
        payload = {
            "object": "list",
            "data": [
                {
                    "id": "Qwen/Qwen3.5-35B-A3B",
                    "max_model_len": 12288,
                }
            ],
        }
        self.assertEqual(
            model_context_from_payload(payload, "Qwen/Qwen3.5-35B-A3B"), 12288
        )
        with self.assertRaises(LaunchContractError):
            model_context_from_payload(payload, "wrong-model")

    def test_generation_budget_must_leave_context_for_prompt(self):
        self.assertEqual(validate_generation_budget("4096", 12288), (4096, 12288))
        for invalid in ("12288", "32768", "0", "4.5", True):
            with self.subTest(invalid=invalid), self.assertRaises(LaunchContractError):
                validate_generation_budget(invalid, 12288)

    def test_endpoint_contract_records_explicit_history_window(self):
        payload = {
            "data": [
                {
                    "id": "Qwen/Qwen3.5-35B-A3B",
                    "max_model_len": 12288,
                }
            ]
        }
        response = mock.MagicMock()
        response.__enter__.return_value = response
        with mock.patch(
            "derail.mypcbench.launch_contract.urllib.request.urlopen",
            return_value=response,
        ), mock.patch(
            "derail.mypcbench.launch_contract.json.load", return_value=payload
        ):
            contract = fetch_vllm_contract(
                "http://127.0.0.1:8000/v1",
                "Qwen/Qwen3.5-35B-A3B",
                "4096",
                "4",
            )
            self.assertEqual(contract["requested_history_n"], 4)
            with self.assertRaises(LaunchContractError):
                fetch_vllm_contract(
                    "http://127.0.0.1:8000/v1",
                    "Qwen/Qwen3.5-35B-A3B",
                    "4096",
                    "0",
                )

    def test_positive_integer_is_strict(self):
        self.assertEqual(positive_int("N", "8"), 8)
        for invalid in (None, "08", " 8 ", -1, 0, 1.0):
            with self.subTest(invalid=invalid), self.assertRaises(LaunchContractError):
                positive_int("N", invalid)

    def test_formal_authorization_requires_one_top_level_boolean(self):
        self.assertTrue(formal_collection_authorized("formal_collection_authorized: true\n"))
        self.assertFalse(formal_collection_authorized("formal_collection_authorized: false\n"))
        for invalid in (
            "  formal_collection_authorized: true\n",
            "formal_collection_authorized: yes\n",
            "formal_collection_authorized: false\nformal_collection_authorized: true\n",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(LaunchContractError):
                formal_collection_authorized(invalid)


if __name__ == "__main__":
    unittest.main()
