import importlib.util
import shlex
import sys
import time
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "third_party/MyPCBench/agent-harness/utils/rubric_judge.py"
SPEC = importlib.util.spec_from_file_location("rubric_judge_timeout_test", MODULE_PATH)
RUBRIC_JUDGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUBRIC_JUDGE
SPEC.loader.exec_module(RUBRIC_JUDGE)


def test_timeout_kills_descendant_holding_output_pipe(tmp_path):
    marker = tmp_path / "orphan_wrote.txt"
    child = f"import time; time.sleep(1); open({str(marker)!r}, 'w').write('bad')"
    parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',sys.argv[1]]); time.sleep(10)"
    command = shlex.join([sys.executable, "-c", parent, child])

    result = RUBRIC_JUDGE.run_rubric_judge_command(
        command, bundle_path=tmp_path / "bundle.json", save_dir=str(tmp_path), timeout=0.1
    )
    time.sleep(1.2)

    assert result[1] == "timeout"
    assert not marker.exists()
