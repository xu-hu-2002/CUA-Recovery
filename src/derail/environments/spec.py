"""冻结环境协议所需的最小 metadata。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentSpec:
    environment_id: str
    screen_width: int
    screen_height: int
    snapshot_backend: str
    version: str

    def __post_init__(self) -> None:
        if not self.environment_id or not self.version:
            raise ValueError("environment_id 和 version 必须冻结")
        if self.screen_width <= 0 or self.screen_height <= 0:
            raise ValueError("屏幕尺寸必须为正整数")
