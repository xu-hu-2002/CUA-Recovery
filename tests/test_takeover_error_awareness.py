"""Smallest checks that fail if the EAR thought extraction or slicing breaks."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "takeover_error_awareness",
    Path(__file__).resolve().parents[1] / "scripts" / "judge" / "error_awareness.py",
)
ear = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ear)


def test_thought_of_covers_every_response_shape():
    assert ear.thought_of("I see the data is wrong.\n</think>\n\nAction: undo\n<tool_call>{}") \
        == "I see the data is wrong."
    assert ear.thought_of("<think>plain thought\nAction: click") == "plain thought"
    assert ear.thought_of("None") == ""
    assert ear.thought_of(None) == ""
    assert ear.thought_of("just text") == "just text"


def test_aggregate_counts_multi_label_tasks_in_every_slice():
    failures = {0: {"t1": ["a", "b"], "t2": ["a"]}, 5: {"t3": ["b"]}}
    verdicts = {(0, "t1"): True, (0, "t2"): False, (5, "t3"): True}
    table = {(r["depth"], r["error_type"]): (r["n_failures"], r["n_judged"], r["n_aware"], r["ear"])
             for r in ear.aggregate(failures, verdicts)}
    assert table[(0, "ALL")] == (2, 2, 1, 0.5)
    assert table[(0, "a")] == (2, 2, 1, 0.5)
    assert table[(0, "b")] == (1, 1, 1, 1.0)
    assert table[("ALL", "b")] == (2, 2, 2, 1.0)
    assert table[("ALL", "ALL")] == (3, 3, 2, 0.6667)


def test_missing_episode_policy_is_configurable():
    failures = {0: {"ran": ["a"], "never_ran": ["a"]}}
    verdicts = {(0, "ran"): True}
    row = next(r for r in ear.aggregate(failures, verdicts)
               if r["error_type"] == "a" and r["depth"] == 0)
    assert (row["n_failures"], row["n_judged"], row["n_missing"], row["ear"]) == (2, 1, 1, 1.0)
    row = next(r for r in ear.aggregate(failures, verdicts, "count_as_unaware")
               if r["error_type"] == "a" and r["depth"] == 0)
    assert (row["n_failures"], row["n_missing"], row["n_aware"], row["ear"]) == (2, 1, 1, 0.5)


def test_repeats_count_per_episode_by_default():
    r1, r2 = Path("repeat_1"), Path("repeat_2")
    failures = {r1: {0: {"t": ["a"]}}, r2: {0: {"t": ["a"]}}}
    verdicts = {(r1, 0, "t"): True, (r2, 0, "t"): False}
    row = lambda units: next(r for r in ear.aggregate(*units) if r["depth"] == 0
                             and r["error_type"] == "ALL")
    episode = row(ear.repeat_units(failures, verdicts))
    assert (episode["n_failures"], episode["n_aware"], episode["ear"]) == (2, 1, 0.5)
    state = row(ear.repeat_units(failures, verdicts, "state"))
    assert (state["n_failures"], state["n_aware"], state["ear"]) == (1, 1, 1.0)


def test_prompt_matches_appendix_d_and_reads_every_segment(tmp_path):
    assert "You will read the agent's reasoning segments after the takeover." in ear.SYSTEM_PROMPT
    assert "first three" not in ear.SYSTEM_PROMPT
    traj = tmp_path / "traj.jsonl"
    traj.write_text("".join(f'{{"step_num": {i}, "response": "thought {i}"}}\n' for i in range(1, 6)))
    assert len(ear.post_takeover_thoughts(traj)[0]) == 5
    assert ear.post_takeover_thoughts(traj, 3)[0] == ["thought 1", "thought 2", "thought 3"]
    prompt = ear.user_prompt({"instruction": "i", "error_types": ["x"], "error_description": "d",
                              "thoughts": ["t1", "t2"]})
    assert "Agent's reasoning segments after the takeover:\n\n[Thought 1]\nt1\n\n[Thought 2]\nt2" in prompt


def test_thought_of_keeps_messages_not_tool_payloads():
    assert ear.thought_of('{"content": "The total looks wrong.", "tool_calls": [{"id": "x"}]}') \
        == "The total looks wrong."
    assert ear.thought_of('[tool] {"type": "shell_call"}\nChecking again.') == "Checking again."


def test_parse_verdict_rejects_non_boolean():
    assert ear.parse_verdict('```json\n{"aware": true, "quote": "q", "reason": "r"}\n```')["aware"] is True
    try:
        ear.parse_verdict('{"aware": "yes"}')
    except ValueError:
        pass
    else:
        raise AssertionError("string 'yes' must not pass as a verdict")


def test_reasoning_model_and_completion_budget(monkeypatch):
    assert ear.is_reasoning_model("gpt-5.6-terra")
    assert not ear.is_reasoning_model("claude-opus-4-8")
    monkeypatch.setenv("MYPCBENCH_OSWORLD_JUDGE_MAX_COMPLETION_TOKENS", "4096")
    assert ear.max_completion_tokens() == 4096
