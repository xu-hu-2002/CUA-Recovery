#!/usr/bin/env python3
"""开发期快速检查；等价于 `python -m derail.cli check-repository`。"""

from derail.cli import check_repository


if __name__ == "__main__":
    raise SystemExit(check_repository())
