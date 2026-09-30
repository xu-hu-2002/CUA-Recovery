#!/usr/bin/env python3
"""DERAIL full-trajectory rubric Judge: upstream judge + final-state data section.

Runs the upstream MyPCBench judge (third_party, unmodified) with the prompt of
paper Appendix D.  The only addition: when the bundle carries
``artifacts.final_state`` (written by scripts/14_takeover_bundle_prefix.py from
the per-step state probe), the user message gains one data section after the
trajectory, so criteria can be decided from the rollout, its screenshots and the
final state s_T (paper Sec. 2, "Rollouts and Evaluation").  The system prompt is
left verbatim.  Launched through MYPCBENCH_RUBRIC_JUDGE_COMMAND by
scripts/run_judge.sh and scripts/02_judge_rubrics.sh.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from judge_model_registry import require_single_judge  # noqa: E402

UPSTREAM = ROOT / "third_party/MyPCBench/agent-harness/utils/osworld_full_traj_judge.py"
ANCHOR = "\n\nScreenshots attached below:"
FINAL_STATE_HEADER = "Final Environment State (recorded by the harness after the last step):"


def final_state_section(bundle: dict) -> str:
    state = (bundle.get("artifacts") or {}).get("final_state")
    if not isinstance(state, dict) or not state.get("text"):
        return ""
    return f"{FINAL_STATE_HEADER}\n{state['text']}"


def with_section(text: str, section: str) -> str:
    if not section or ANCHOR not in text:
        return text
    return text.replace(ANCHOR, f"\n\n{section}{ANCHOR}", 1)


def load_upstream(section: str):
    spec = importlib.util.spec_from_file_location("derail_osworld", UPSTREAM)
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load upstream full-trajectory judge")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shared, per_rubric = module._shared_prefix_text, module._user_text_for_rubric
    module._shared_prefix_text = lambda *a, **k: with_section(shared(*a, **k), section)
    module._user_text_for_rubric = lambda *a, **k: with_section(per_rubric(*a, **k), section)
    return module


def main() -> None:
    os.environ["MYPCBENCH_RUBRIC_JUDGE_MODEL"] = require_single_judge(
        os.environ.get("MYPCBENCH_RUBRIC_JUDGE_MODEL", "").strip() or None)
    bundle_path = os.environ.get("MYPCBENCH_RUBRIC_BUNDLE_PATH", "").strip()
    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8")) if bundle_path else {}
    load_upstream(final_state_section(bundle)).main()


if __name__ == "__main__":
    main()
