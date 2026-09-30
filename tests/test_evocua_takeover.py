"""EvoCUA S2 self-takeover: the source action is replayed, its reasoning is not."""

from __future__ import annotations

import base64
import json
import unittest
from io import BytesIO

from PIL import Image

from derail.adapters import HistoryStep, create_native_history_adapter
from derail.adapters.evocua import EvoCUAS2Adapter
from derail.canonical.actions import ClickAction
from derail.mypcbench.factory import create_mypcbench_agent
from derail.takeover.history import build_native_history

RESPONSES = [
    "Action: Click the LibreOffice Calc icon.\n"
    '<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", '
    '"coordinate": [26, 811]}}\n</tool_call>',
    "Action: Type the header row.\n"
    '<tool_call>\n{"name": "computer_use", "arguments": {"action": "type", '
    '"text": "Expense"}}\n</tool_call>',
]


def _step(step_id: int) -> HistoryStep:
    buffer = BytesIO()
    Image.new("RGB", (1280, 800), (step_id * 20,) * 3).save(buffer, format="PNG")
    return HistoryStep(
        step_id=step_id,
        observation_image_url="data:image/png;base64,"
        + base64.b64encode(buffer.getvalue()).decode("ascii"),
        action=ClickAction(kind="click", x_px=26, y_px=711),
        trajectory_log=json.dumps({"visible_response": RESPONSES[step_id]}),
    )


class EvoCUATakeoverTests(unittest.TestCase):
    def test_seeded_agent_rebuilds_the_upstream_s2_request(self) -> None:
        adapter = create_native_history_adapter("evocua_32b", "open calc")
        history = build_native_history(adapter, [_step(0), _step(1)])
        # The rendered assistant turn carries the action, not the reasoning.
        self.assertEqual(history[2]["content"][0]["text"], RESPONSES[0])
        self.assertEqual(
            [c["function"]["name"] for c in adapter.extract_tool_calls(history)],
            ["computer_use", "computer_use"],
        )

        agent = create_mypcbench_agent("derail_evocua", "m", (1280, 800), "pw")
        agent.reset()
        agent.seed_native_history("open calc", history, condition="unaware")
        inner = agent.inner
        self.assertEqual(inner.responses, RESPONSES)
        self.assertEqual(inner.actions[0], "Click the LibreOffice Calc icon.")
        with Image.open(BytesIO(base64.b64decode(inner.screenshots[0]))) as image:
            self.assertEqual((image.width % 32, image.height % 32), (0, 0))

        messages = inner._build_s2_messages(
            "open calc", inner.screenshots[-1], 2, inner.max_history_turns,
            EvoCUAS2Adapter.system_prompt(),
        )
        assistants = [m["content"][0]["text"] for m in messages if m["role"] == "assistant"]
        self.assertEqual(assistants, RESPONSES)
        self.assertEqual([p["type"] for p in messages[-1]["content"]], ["image_url"])

        agent.seed_native_history("open calc", history, condition="notified")
        self.assertIn("Takeover notice:", agent._takeover_prompt)

        original_call = agent.inner.call_llm
        payload = agent.preflight_next_request(
            "open calc",
            {
                "screenshot": base64.b64decode(
                    history[1]["content"][0]["image_url"]["url"].split(",", 1)[1]
                )
            },
        )
        self.assertEqual(payload["model"], "m")
        request_text = "\n".join(
            str(part.get("text", ""))
            for message in payload["messages"]
            for part in message.get("content", [])
            if isinstance(part, dict)
        )
        self.assertIn("Takeover notice:", request_text)
        self.assertEqual(agent.inner.call_llm, original_call)
        self.assertIn("Takeover notice:", agent._takeover_prompt)

        observed_prompts = []
        agent.inner.predict = lambda prompt, _obs: (
            observed_prompts.append(prompt) or ("ok", ["pyautogui.click(1, 1)"])
        )
        agent.predict("open calc", {"screenshot": b"unused"})
        agent.predict("open calc", {"screenshot": b"unused"})
        self.assertIn("Takeover notice:", observed_prompts[0])
        self.assertNotIn("Takeover notice:", observed_prompts[1])

        agent.reset()
        self.assertEqual(agent._takeover_prompt, "")
        history[0] = {"role": "system", "content": "drifted"}
        with self.assertRaisesRegex(ValueError, "system prompt"):
            agent.seed_native_history("open calc", history, condition="unaware")


if __name__ == "__main__":
    unittest.main()
