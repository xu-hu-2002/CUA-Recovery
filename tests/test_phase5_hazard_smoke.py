from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import shlex
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

if not (Path(__file__).resolve().parents[1]
        / "artifacts/phase5/preflight/hazards-option-a-20260912.jsonl").is_file():
    raise unittest.SkipTest("Phase 5 hazard catalog is run data, not shipped with the code")

from scripts.phase5.run_hazard_smoke import (
    EVIDENCE_TEMPLATE,
    download_evidence_file,
    guest_exec,
    guest_fetch_file,
    guest_output,
    guest_reset,
    sandbox_run_background,
    snapshot_report,
    verify_snapshot,
)


class HazardSmokeTests(unittest.TestCase):
    def test_guest_output_returns_only_command_stdout(self):
        payload = {"status": "success", "returncode": 0, "output": '{"errors": []}\n'}
        self.assertEqual(guest_output(json.dumps(payload)), payload["output"])

    def test_guest_output_rejects_false_success(self):
        for payload in (
            {"status": "error", "returncode": 0, "output": ""},
            {"status": "success", "returncode": 1, "output": '"returncode":0'},
            {"status": "success", "returncode": False, "output": ""},
            {"status": "success", "returncode": 0},
            {"message": 'HTTP_CODE=200 "returncode":0'},
            [],
        ):
            with self.subTest(payload=payload), self.assertRaises(RuntimeError):
                guest_output(json.dumps(payload))
        with self.assertRaises(json.JSONDecodeError):
            guest_output('Failed to execute: {"returncode":0}')

    def test_transport_error_never_replays_post(self):
        sandbox = SimpleNamespace(arun=AsyncMock(side_effect=RuntimeError('"returncode":0')))
        with self.assertRaises(RuntimeError):
            asyncio.run(guest_exec(sandbox, "mutate", timeout=600))
        self.assertEqual(sandbox.arun.await_count, 1)
        args, kwargs = sandbox.arun.call_args
        self.assertEqual(kwargs["mode"], "nohup")
        self.assertGreater(kwargs["wait_timeout"], 600)
        self.assertIn("--max-time 600", args[0])
        self.assertTrue(args[0].startswith("bash -lc "))

    def test_nonzero_transport_cannot_report_success(self):
        response = SimpleNamespace(exit_code=1, output=json.dumps(
            {"status": "success", "returncode": 0, "output": "ok"}
        ))
        sandbox = SimpleNamespace(arun=AsyncMock(return_value=response))
        with self.assertRaises(RuntimeError):
            asyncio.run(guest_exec(sandbox, "mutate"))
        self.assertEqual(sandbox.arun.await_count, 1)

    def test_reset_is_direct_single_request(self):
        response = SimpleNamespace(exit_code=0, output='{"errors": []}')
        sandbox = SimpleNamespace(arun=AsyncMock(return_value=response))
        self.assertEqual(asyncio.run(guest_reset(sandbox)), {"errors": []})
        self.assertEqual(sandbox.arun.await_count, 1)
        args, kwargs = sandbox.arun.call_args
        self.assertIn("/pcapi/reset", args[0])
        self.assertNotIn("/execute", args[0])
        self.assertEqual(kwargs["mode"], "nohup")
        self.assertGreater(kwargs["wait_timeout"], 1200)

    def test_reset_requires_explicit_empty_error_list(self):
        for payload in ({}, {"errors": ["seeder failed"]}, {"errors": None}, []):
            response = SimpleNamespace(exit_code=0, output=json.dumps(payload))
            sandbox = SimpleNamespace(arun=AsyncMock(return_value=response))
            with self.subTest(payload=payload), self.assertRaises(RuntimeError):
                asyncio.run(guest_reset(sandbox))
            self.assertEqual(sandbox.arun.await_count, 1)

    def test_download_rejects_sdk_failure_even_with_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            local_path = Path(directory) / "base.tar.gz"
            local_path.write_bytes(b"partial transfer")
            download = AsyncMock(return_value=SimpleNamespace(
                success=False, message="Failed to ensure ossutil is installed and working",
            ))
            sandbox = SimpleNamespace(fs=SimpleNamespace(download_file=download))
            with self.assertRaisesRegex(RuntimeError, "evidence download failed.*ossutil"):
                asyncio.run(download_evidence_file(sandbox, "/tmp/base.tar.gz", local_path))
            download.assert_awaited_once_with(
                remote_path="/tmp/base.tar.gz", local_path=str(local_path),
            )

    def test_download_rejects_success_without_file(self):
        with tempfile.TemporaryDirectory() as directory:
            local_path = Path(directory) / "base.tar.gz"
            download = AsyncMock(return_value=SimpleNamespace(success=True))
            sandbox = SimpleNamespace(fs=SimpleNamespace(download_file=download))
            with self.assertRaisesRegex(RuntimeError, "evidence download produced no file"):
                asyncio.run(download_evidence_file(sandbox, "/tmp/base.tar.gz", local_path))
            self.assertEqual(download.await_count, 1)

    def test_download_rejects_success_with_empty_file(self):
        with tempfile.TemporaryDirectory() as directory:
            local_path = Path(directory) / "base.tar.gz"
            local_path.touch()
            download = AsyncMock(return_value=SimpleNamespace(success=True))
            sandbox = SimpleNamespace(fs=SimpleNamespace(download_file=download))
            with self.assertRaisesRegex(RuntimeError, "evidence download produced no file"):
                asyncio.run(download_evidence_file(sandbox, "/tmp/base.tar.gz", local_path))
            self.assertEqual(download.await_count, 1)

    def test_download_accepts_success_with_file(self):
        async def transfer(*, remote_path, local_path):
            Path(local_path).write_bytes(b"snapshot evidence")
            return SimpleNamespace(success=True)

        with tempfile.TemporaryDirectory() as directory:
            local_path = Path(directory) / "base.tar.gz"
            download = AsyncMock(side_effect=transfer)
            sandbox = SimpleNamespace(fs=SimpleNamespace(download_file=download))
            asyncio.run(download_evidence_file(sandbox, "/tmp/base.tar.gz", local_path))
            self.assertEqual(local_path.read_bytes(), b"snapshot evidence")
            download.assert_awaited_once_with(
                remote_path="/tmp/base.tar.gz", local_path=str(local_path),
            )

    def test_guest_fetch_uses_nohup_and_one_round_trip(self):
        response = SimpleNamespace(exit_code=0, output="size=174963277\n")
        sandbox = SimpleNamespace(arun=AsyncMock(return_value=response))
        with contextlib.redirect_stdout(io.StringIO()):
            size = asyncio.run(guest_fetch_file(
                sandbox, "/tmp/BASE.tar.gz", "/tmp/phase5-base.tar.gz", timeout=600,
            ))
        self.assertEqual(size, "size=174963277")
        self.assertEqual(sandbox.arun.await_count, 1)
        args, kwargs = sandbox.arun.call_args
        self.assertEqual(kwargs["mode"], "nohup")
        self.assertEqual(kwargs["wait_timeout"], 600)
        self.assertNotIn("session", kwargs)
        script = shlex.split(kwargs["cmd"])[2]
        self.assertIn("http://127.0.0.1:8080/pcapi/file", script)
        self.assertIn("stat -c 'size=%s'", script)
        self.assertIn("/tmp/phase5-base.tar.gz.part1", script)

    def test_guest_fetch_retries_with_its_own_staging_path(self):
        responses = [
            SimpleNamespace(exit_code=1, output="curl: (22) The requested URL returned error: 502"),
            SimpleNamespace(exit_code=0, output="size=4096\n"),
        ]
        scripts = []

        async def arun(cmd, **kwargs):
            scripts.append(shlex.split(cmd)[2])
            return responses[len(scripts) - 1]

        sandbox = SimpleNamespace(arun=arun)
        with contextlib.redirect_stdout(io.StringIO()):
            size = asyncio.run(guest_fetch_file(
                sandbox, "/tmp/BASE.tar.gz", "/tmp/phase5-base.tar.gz",
                timeout=60, attempts=3, retry_delay=0,
            ))
        self.assertEqual(size, "size=4096")
        self.assertEqual(len(scripts), 2)
        self.assertIn("tar.gz.part1", scripts[0])
        self.assertNotIn("part2", scripts[0])
        self.assertIn("tar.gz.part2", scripts[1])

    def test_guest_fetch_gives_up_after_bounded_attempts(self):
        response = SimpleNamespace(exit_code=0, output="")
        sandbox = SimpleNamespace(arun=AsyncMock(return_value=response))
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "guest file fetch failed for /tmp/BASE"):
                asyncio.run(guest_fetch_file(
                    sandbox, "/tmp/BASE.tar.gz", "/tmp/phase5-base.tar.gz",
                    timeout=30, attempts=2, retry_delay=0,
                ))
        self.assertEqual(sandbox.arun.await_count, 2)

    def test_snapshot_report_reads_guest_digest_and_counts(self):
        digest = "77fa22f9cb9599826b183ef5369baaeef03f4e12dfb38aa38d75346fd3f86a28"
        line = (
            "writing snapshot\nSNAPSHOT_BASE_READY sha256=" + digest
            + " size=174963277 manifest_components=67\n"
        )
        self.assertEqual(snapshot_report(line, "BASE"), {
            "sha256": digest, "size": 174963277, "manifest_components": 67,
        })
        for report in ("", "SNAPSHOT_BASE_READY sha256=77fa size=174963277",
                       "SNAPSHOT_VARIANT_READY sha256=" + digest + " size=1 manifest_components=1"):
            with self.subTest(report=report), self.assertRaises(RuntimeError):
                snapshot_report(report, "BASE")

    def test_upload_marker_must_be_complete_final_line(self):
        for output in ("OK", "UPLOAD_DONE failed", "UPLOAD_DONE\nfailed", ""):
            response = SimpleNamespace(exit_code=0, output=output)
            sandbox = SimpleNamespace(arun=AsyncMock(return_value=response))
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                asyncio.run(sandbox_run_background(sandbox, "false", "upload", 60))
        response = SimpleNamespace(exit_code=0, output="copied\nUPLOAD_DONE\n")
        sandbox = SimpleNamespace(arun=AsyncMock(return_value=response))
        asyncio.run(sandbox_run_background(sandbox, "true", "upload", 60))

    def test_background_shell_propagates_and_chain_failure(self):
        async def arun(cmd, **kwargs):
            self.assertEqual(kwargs["mode"], "nohup")
            process = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
            return SimpleNamespace(exit_code=0, output=process.stdout)

        sandbox = SimpleNamespace(arun=arun)
        with self.assertRaisesRegex(RuntimeError, "upload failed"):
            asyncio.run(sandbox_run_background(sandbox, "false && true", "upload", 60))
        output = asyncio.run(sandbox_run_background(sandbox, "true && true", "upload", 60))
        self.assertEqual(output.strip(), "UPLOAD_DONE")

    def test_snapshot_rejects_nonempty_wal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "data" / "mail.sqlite-wal").write_bytes(b"pending data")
            source = (EVIDENCE_TEMPLATE.replace("{stage}", "BASE").replace("{dirs}", "[]")
                      .replace('"/tmp/phase5-evidence"', repr(str(root / "evidence")))
                      .replace('"/tmp/BASE.tar.gz"', repr(str(root / "BASE.tar.gz")))
                      .replace('"/data"', repr(str(root / "data"))))
            with self.assertRaisesRegex(RuntimeError, "uncheckpointed SQLite WAL"):
                exec(compile(source, "snapshot.py", "exec"), {})
            self.assertFalse((root / "BASE.tar.gz").exists())

    def test_snapshot_writes_external_and_embedded_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home = root / "home"
            (home / "Documents").mkdir(parents=True)
            (home / "Documents" / "example.txt").write_text("snapshot evidence")
            for stage in ("BASE", "VARIANT"):
                source = (EVIDENCE_TEMPLATE.replace("{stage}", stage)
                          .replace("{dirs}", '["Documents", "Desktop", "Downloads"]')
                          .replace('"/tmp/phase5-evidence"', repr(str(root / "evidence")))
                          .replace(f'"/tmp/{stage}.tar.gz"', repr(str(root / f"{stage}.tar.gz")))
                          .replace('"/data"', repr(str(root / "data")))
                          .replace('"/home/user"', repr(str(home))))
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    exec(compile(source, "snapshot.py", "exec"), {})
                self.assertIn(f"SNAPSHOT_{stage}_READY", output.getvalue())
                archive = root / f"{stage}.tar.gz"
                manifest = root / "evidence" / f"{stage}_manifest.json"
                self.assertEqual(len(verify_snapshot(archive, manifest, stage)), 64)
                with tarfile.open(archive) as tar:
                    self.assertEqual(tar.extractfile(f"{stage}_manifest.json").read(), manifest.read_bytes())
                manifest.write_text(manifest.read_text() + " ")
                with self.assertRaisesRegex(RuntimeError, "embedded manifest mismatch"):
                    verify_snapshot(archive, manifest, stage)


    def test_snapshot_stores_symlinked_sources_as_regular_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            home = root / "home"
            data.mkdir()
            (home / "Documents").mkdir(parents=True)
            (data / "real.sqlite").write_bytes(b"sqlite world content")
            os.symlink("real.sqlite", data / "link.sqlite")
            (home / "Documents" / "note.txt").write_bytes(b"note bytes")
            os.symlink("note.txt", home / "Documents" / "link.txt")
            source = (EVIDENCE_TEMPLATE.replace("{stage}", "BASE")
                      .replace("{dirs}", '["Documents"]')
                      .replace('"/tmp/phase5-evidence"', repr(str(root / "evidence")))
                      .replace('"/tmp/BASE.tar.gz"', repr(str(root / "BASE.tar.gz")))
                      .replace('"/data"', repr(str(data)))
                      .replace('"/home/user"', repr(str(home))))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                exec(compile(source, "snapshot.py", "exec"), {})
            self.assertIn("SNAPSHOT_BASE_READY", output.getvalue())
            archive = root / "BASE.tar.gz"
            manifest = root / "evidence" / "BASE_manifest.json"
            sha = verify_snapshot(archive, manifest, "BASE")
            self.assertEqual(len(sha), 64)
            with tarfile.open(archive) as tar:
                for name in ("data/link.sqlite", "home/Documents/link.txt"):
                    member = tar.getmember(name)
                    self.assertTrue(member.isfile(), f"{name} stored as {member.type!r}")
                    self.assertEqual(tar.extractfile(member).read(),
                                     b"sqlite world content" if name.startswith("data") else b"note bytes")


if __name__ == "__main__":
    unittest.main()
