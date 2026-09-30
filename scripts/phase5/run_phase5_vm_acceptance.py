#!/usr/bin/env python3
"""Run Phase 5 VM acceptance in ROCK and ship the terminal evidence."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shlex
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
DRIVER_PATH = REPOSITORY / "scripts" / "rock" / "derail_rock_driver.py"


def load_driver():
    spec = importlib.util.spec_from_file_location("phase5_acceptance_driver", DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


async def run() -> int:
    driver = load_driver()
    from rock.actions import CreateBashSessionRequest
    from rock.sdk.sandbox.client import Sandbox

    expected_sha = os.environ["PHASE5_QCOW2_SHA256"]
    output_uri = os.environ["PHASE5_ACCEPTANCE_OSS_URI"].rstrip("/")
    sandbox = Sandbox(driver._sandbox_config())
    local_artifacts = REPOSITORY / "artifacts" / "phase5" / "stage1" / "acceptance"
    local_artifacts.mkdir(parents=True, exist_ok=True)
    try:
        sandbox = await driver._setup_sandbox_proxy(sandbox, CreateBashSessionRequest)
        sandbox_id = driver._sandbox_id(sandbox)
        identity = await sandbox.arun(
            cmd=f"sha256sum {driver.SANDBOX_VM_DIR}/mypcbench.qcow2",
            session="default", wait_timeout=600,
        )
        if expected_sha not in str(identity.output):
            raise RuntimeError(f"Phase 5 qcow2 SHA mismatch: {identity.output}")
        print(f"[acceptance] sandbox={sandbox_id} qcow2_sha256={expected_sha}", flush=True)
        for source, target in (
            (REPOSITORY / "scripts" / "phase5" / "guest_admin.py", "/opt/phase5/guest_admin.py"),
            (
                REPOSITORY / "scripts" / "phase5" / "run_vm_acceptance_guest.py",
                "/opt/phase5/run_vm_acceptance_guest.py",
            ),
        ):
            await driver._stage_sandbox_file(sandbox, source, target)
        await sandbox.arun(
            cmd=("$(command -v python3 || command -v python3.12) "
                 "/opt/phase5/guest_admin.py put "
                 "--source /opt/phase5/run_vm_acceptance_guest.py "
                 "--target /tmp/run_vm_acceptance_guest.py"),
            session="default", wait_timeout=180,
        )
        await sandbox.arun(
            cmd=("$(command -v python3 || command -v python3.12) "
                 "/opt/phase5/guest_admin.py start --target /tmp/run_vm_acceptance_guest.py"),
            session="default", wait_timeout=180,
        )
        terminal = ""
        for attempt in range(1, 91):
            await asyncio.sleep(30)
            status = await sandbox.arun(
                cmd=("$(command -v python3 || command -v python3.12) "
                     "/opt/phase5/guest_admin.py acceptance-status"),
                session="default", wait_timeout=180,
            )
            terminal = str(status.output or "")
            print(f"[acceptance] poll {attempt}/90\n{terminal[-3000:]}", flush=True)
            if "ACCEPTANCE_DONE" in terminal:
                break
            if "ACCEPTANCE_FAILED" in terminal:
                await preserve_failure_log(sandbox, local_artifacts)
                raise RuntimeError("guest acceptance process failed")
        else:
            raise RuntimeError("guest acceptance did not finish within 45 minutes")
        package = await sandbox.arun(
            cmd=("curl -fsS -X POST http://127.0.0.1:8080/pcapi/execute "
                 "-H 'Content-Type: application/json' "
                 "-d '{\"command\":\"cp /tmp/derail-phase5-acceptance/"
                 "vm_acceptance_report.json /tmp/vm_acceptance_report.json && "
                 "tar -czf /tmp/phase5-acceptance.tar.gz "
                 "-C /tmp derail-phase5-acceptance\",\"shell\":true}'"),
            session="default", wait_timeout=180,
        )
        if '"returncode":0' not in str(package.output).replace(" ", ""):
            raise RuntimeError(f"guest acceptance packaging failed: {package.output}")
        fetches = (
            ("/tmp/vm_acceptance_report.json", "/tmp/vm_acceptance_report.json"),
            ("/tmp/phase5-acceptance.tar.gz", "/tmp/phase5-acceptance.tar.gz"),
        )
        for guest_path, sandbox_path in fetches:
            request_body = shlex.quote(json.dumps({"file_path": guest_path}))
            fetched = await sandbox.arun(
                cmd=("curl -fsS -X POST -H 'Content-Type: application/json' "
                     f"-d {request_body} "
                     f"http://127.0.0.1:8080/pcapi/file -o {shlex.quote(sandbox_path)}"),
                session="default", wait_timeout=300,
            )
            print(f"[acceptance] fetched {guest_path}: {fetched.output}", flush=True)
        await driver._stage_sandbox_file(
            sandbox, REPOSITORY / "scripts" / "phase5" / "validate_vm_acceptance.py",
            "/opt/phase5/validate_vm_acceptance.py",
        )
        verdict = await sandbox.arun(
            cmd=("set +e; $(command -v python3 || command -v python3.12) "
                 "/opt/phase5/validate_vm_acceptance.py /tmp/vm_acceptance_report.json "
                 "--output /tmp/vm_acceptance_verdict.json; "
                 "validator_rc=$?; cat /tmp/vm_acceptance_verdict.json; "
                 "echo VALIDATOR_RC=$validator_rc; true"),
            session="default", wait_timeout=180,
        )
        print(f"[acceptance] verdict:\n{verdict.output}", flush=True)
        accepted = '"accepted": true' in str(verdict.output).lower()
        upload_script = (
            f"set -e; {driver._ossutil()} cp -f /tmp/vm_acceptance_report.json "
            f"{shlex.quote(output_uri)}/vm_acceptance_report.json; "
            f"{driver._ossutil()} cp -f /tmp/vm_acceptance_verdict.json "
            f"{shlex.quote(output_uri)}/vm_acceptance_verdict.json; "
            f"{driver._ossutil()} cp -f /tmp/phase5-acceptance.tar.gz "
            f"{shlex.quote(output_uri)}/phase5-acceptance.tar.gz"
        )
        upload_output = await driver_run_background(
            sandbox, upload_script, "phase5-acceptance-upload", 900
        )
        print(f"[acceptance] upload:\n{upload_output}", flush=True)
        for name in ("vm_acceptance_report.json", "vm_acceptance_verdict.json"):
            await sandbox.fs.download_file(f"/tmp/{name}", local_artifacts / name)
        (local_artifacts / "sandbox.json").write_text(
            json.dumps({"sandbox_id": sandbox_id, "qcow2_sha256": expected_sha}, indent=2) + "\n"
        )
        if not accepted:
            raise RuntimeError("Phase 5 VM acceptance validator rejected the report")
        return 0
    finally:
        print("[acceptance] stopping sandbox", flush=True)
        await sandbox.stop()


async def preserve_failure_log(sandbox, local_artifacts: Path) -> None:
    """Return the complete guest failure log before the sandbox is stopped."""
    sandbox_path = "/tmp/phase5-acceptance-failure.log"
    copied = await sandbox.arun(
        cmd=("curl -fsS -X POST http://127.0.0.1:8080/pcapi/execute "
             "-H 'Content-Type: application/json' "
             "-d '{\"command\":\"cp /tmp/phase5-acceptance.log "
             "/tmp/phase5-acceptance-failure.log\",\"shell\":true}'"),
        session="default", wait_timeout=180,
    )
    print(f"[acceptance] staged failure log: {copied.output}", flush=True)
    request_body = shlex.quote(
        json.dumps({"file_path": "/tmp/phase5-acceptance-failure.log"})
    )
    fetched = await sandbox.arun(
        cmd=("curl -fsS -X POST -H 'Content-Type: application/json' "
             f"-d {request_body} "
             f"http://127.0.0.1:8080/pcapi/file -o {sandbox_path}"),
        session="default", wait_timeout=300,
    )
    print(f"[acceptance] preserved failure log: {fetched.output}", flush=True)
    await sandbox.fs.download_file(
        sandbox_path, local_artifacts / "phase5-acceptance-failure.log"
    )


async def driver_run_background(sandbox, script: str, name: str, timeout: int) -> str:
    marker = name.upper().replace("-", "_") + "_DONE"
    await sandbox.arun(
        cmd=f"cat > /tmp/{name}.sh <<'EOF'\n{script}\necho {marker}\nEOF",
        session="default", wait_timeout=120,
    )
    await sandbox.arun(
        cmd=f"nohup bash /tmp/{name}.sh >/tmp/{name}.log 2>&1 & echo $!",
        session="default", wait_timeout=120,
    )
    for _ in range(max(1, timeout // 20)):
        await asyncio.sleep(20)
        result = await sandbox.arun(
            cmd=f"tail -20 /tmp/{name}.log; grep -q {marker} /tmp/{name}.log && echo OK || true",
            session="default", wait_timeout=120,
        )
        output = str(result.output or "")
        if "OK" in output:
            return output
    raise RuntimeError(f"{name} did not finish")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
