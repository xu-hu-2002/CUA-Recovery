#!/usr/bin/env python3
"""DERAIL ROCK 采集 Nebula job 的 Python entry shim。

Nebula mdl launcher 用 `python <entry>` 执行 --entry（不是 bash），所以裸 .sh
会被当 Python 解析直接 SyntaxError。照 MCUA/ProtAgent 模式：.py shim 转调
同目录的 entry_derail_rock_nebula.sh（preflight + driver）。

所有运行参数经代码包里的 .rock_run.env 传递（bash wrapper 自己 source），
这里无需转发任何环境变量。
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WRAPPER = os.path.join(HERE, "entry_derail_rock_nebula.sh")


def main():
    print(f"[entry-shim] launching: bash {WRAPPER}", flush=True)
    if not os.path.isfile(WRAPPER):
        print(f"[entry-shim] ERROR: {WRAPPER} not found", file=sys.stderr, flush=True)
        return 2
    # 前台运行并继承 stdout/stderr，日志直接流进 Nebula。
    return subprocess.call(["bash", WRAPPER], cwd=os.path.dirname(os.path.dirname(HERE)))


if __name__ == "__main__":
    sys.exit(main())
