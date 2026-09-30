import asyncio
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "third_party/MyPCBench/agent-harness/utils/osworld_full_traj_judge.py"
)
SPEC = importlib.util.spec_from_file_location("osworld_full_traj_judge", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class _Completions:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        message = type("Message", (), {"content": 'Thoughts: ok\nStatus: "success"'})
        choice = type("Choice", (), {"message": message()})
        return type("Response", (), {"choices": [choice()]})


class _Client:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _Completions()})()


def test_claude_reasoning_effort_uses_output_config(monkeypatch):
    monkeypatch.setenv("MYPCBENCH_OSWORLD_JUDGE_REASONING_EFFORT", "low")
    client = _Client()
    asyncio.run(MODULE._judge_one_rubric_openai(client, "claude-opus-4-8", "x", []))
    assert client.chat.completions.kwargs["extra_body"] == {
        "output_config": {"effort": "low"}
    }


def test_openai_reasoning_effort_uses_standard_parameter(monkeypatch):
    monkeypatch.setenv("MYPCBENCH_OSWORLD_JUDGE_REASONING_EFFORT", "low")
    client = _Client()
    asyncio.run(MODULE._judge_one_rubric_openai(client, "gpt-5.6-terra", "x", []))
    assert client.chat.completions.kwargs["reasoning_effort"] == "low"


def test_openai_request_has_hard_timeout(monkeypatch):
    class SlowCompletions:
        async def create(self, **kwargs):
            await asyncio.sleep(1)

    client = _Client()
    client.chat.completions = SlowCompletions()
    monkeypatch.setenv("MYPCBENCH_OSWORLD_JUDGE_REQUEST_TIMEOUT", "0.01")
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(MODULE._judge_one_rubric_openai(client, "claude-opus-4-8", "x", []))


def test_request_timeout_must_be_positive(monkeypatch):
    monkeypatch.setenv("MYPCBENCH_OSWORLD_JUDGE_REQUEST_TIMEOUT", "0")
    with pytest.raises(ValueError, match="must be positive"):
        MODULE._request_timeout_seconds()


def test_process_timeout_must_be_positive(monkeypatch):
    monkeypatch.setenv("MYPCBENCH_OSWORLD_JUDGE_PROCESS_TIMEOUT", "0")
    with pytest.raises(ValueError, match="must be positive"):
        MODULE._process_timeout_seconds()
