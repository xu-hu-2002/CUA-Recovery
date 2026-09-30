import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _load_runner_env_module():
    harness = Path(__file__).resolve().parents[1] / "third_party/MyPCBench/agent-harness"
    sys.path.insert(0, str(harness))
    spec = importlib.util.spec_from_file_location("mypcbench_runner_env", harness / "env.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner_env = _load_runner_env_module()


class HostDiscoveryTests(unittest.TestCase):
    def test_fedora_qemu_fallback_is_supported(self):
        def resolve(candidate):
            return "/usr/libexec/qemu-kvm" if candidate == "/usr/libexec/qemu-kvm" else None

        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            runner_env.shutil, "which", side_effect=resolve
        ):
            self.assertEqual(runner_env._discover_qemu_binary(), "/usr/libexec/qemu-kvm")

    def test_explicit_missing_qemu_fails(self):
        with mock.patch.dict(
            os.environ, {"MYPCBENCH_QEMU_BINARY": "/missing/qemu"}, clear=True
        ), mock.patch.object(runner_env.shutil, "which", return_value=None), self.assertRaises(
            RuntimeError
        ):
            runner_env._discover_qemu_binary()

    def test_fedora_code_cc_uses_plain_vars(self):
        with tempfile.TemporaryDirectory() as directory:
            code = Path(directory) / "OVMF_CODE.cc.fd"
            variables = Path(directory) / "OVMF_VARS.fd"
            code.touch()
            variables.touch()
            with mock.patch.dict(
                os.environ, {"MYPCBENCH_OVMF_CODE": str(code)}, clear=True
            ):
                self.assertEqual(
                    runner_env._discover_ovmf(), (str(code), str(variables))
                )


if __name__ == "__main__":
    unittest.main()
