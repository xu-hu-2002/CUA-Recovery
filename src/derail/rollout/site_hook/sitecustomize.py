"""DERAIL 环境钩子的进程入口（不改 third_party 的包装方式）。

01_collect_trajectories.sh 把本目录放进 PYTHONPATH 并导出 DERAIL_ENVIRONMENT_CONFIG；
官方 run_parallel_tasks.py 以子进程起 run_mypcbench.py 时继承这两个变量，Python 启动时
自动 import 本模块，于是只在 run_mypcbench.py 进程里给 MyPCBenchEnv 装上
derail.rollout.state_probe 的 reset/step 包装。其余 Python 进程不受影响。
"""

import os
import sys

if os.environ.get("DERAIL_ENVIRONMENT_CONFIG") and os.path.basename(
    sys.argv[0] if sys.argv else ""
) == "run_mypcbench.py":
    try:
        from derail.rollout.state_probe import install_for_runner

        install_for_runner()
    except Exception as exc:  # site 会吞掉异常，这里改成硬失败，免得静默采到没有钩子的数据
        print("[DERAIL env] failed to install environment hooks: %r" % (exc,), file=sys.stderr)
        raise SystemExit(2)
