"""通过受控 backend 重放 canonical actions。"""

from .executor import CanonicalExecutor, PyAutoGUIBackend, RecordingBackend, compile_pyautogui
from .mypcbench import MyPCBenchVMReplayBackend
from .verification import (
    DeterministicSyntheticBackend,
    ReplayPlan,
    ReplayVerification,
    ReplayVerificationError,
    StateFingerprint,
    execute_replay_plan,
)

__all__ = [
    "CanonicalExecutor",
    "DeterministicSyntheticBackend",
    "MyPCBenchVMReplayBackend",
    "PyAutoGUIBackend",
    "RecordingBackend",
    "ReplayPlan",
    "ReplayVerification",
    "ReplayVerificationError",
    "StateFingerprint",
    "compile_pyautogui",
    "execute_replay_plan",
]
