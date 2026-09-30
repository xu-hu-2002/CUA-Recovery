#!/usr/bin/env python3
"""Build and upload a versioned Phase 5 MyPCBench qcow2 inside ROCK."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shlex
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
DRIVER_PATH = REPOSITORY / "scripts" / "rock" / "derail_rock_driver.py"


def load_driver():
    spec = importlib.util.spec_from_file_location("phase5_rock_driver", DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


async def run_background(sandbox, script: str, name: str, timeout: int) -> str:
    script_path = f"/tmp/{name}.sh"
    log_path = f"/tmp/{name}.log"
    marker = f"{name.upper().replace('-', '_')}_DONE"
    await sandbox.arun(
        cmd=f"cat > {script_path} <<'PHASE5EOF'\n{script}\necho {marker}\nPHASE5EOF",
        session="default",
        wait_timeout=120,
    )
    await sandbox.arun(
        cmd=f"nohup bash {script_path} > {log_path} 2>&1 & echo $!",
        session="default",
        wait_timeout=120,
    )
    for _ in range(max(1, timeout // 30)):
        await asyncio.sleep(30)
        check = await sandbox.arun(
            cmd=f"tail -30 {log_path}; grep -q {marker} {log_path} && echo MARKER_OK || true",
            session="default",
            wait_timeout=120,
        )
        output = str(check.output or "")
        if "MARKER_OK" in output:
            return output
    raise RuntimeError(f"{name} did not reach {marker} within {timeout}s")


async def run() -> int:
    driver = load_driver()
    from rock.actions import CreateBashSessionRequest
    from rock.sdk.sandbox.client import Sandbox

    infra_commit = os.environ.get("PHASE5_INFRA_COMMIT")
    if not infra_commit:
        raise RuntimeError("PHASE5_INFRA_COMMIT is required")
    boot_date = datetime.now(timezone.utc).date().isoformat()
    tag = f"phase5-{boot_date}-{infra_commit[:8]}"
    remote_dir = f"{driver.SANDBOX_VM_DIR}/phase5"
    remote_qcow2 = f"{remote_dir}/mypcbench-{tag}.qcow2"
    output_uri = os.environ["PHASE5_VM_OSS_URI"].rstrip("/")
    sandbox = Sandbox(driver._sandbox_config())
    try:
        sandbox = await driver._setup_sandbox_proxy(sandbox, CreateBashSessionRequest)
        sandbox_id = driver._sandbox_id(sandbox)
        print(f"[phase5-build] sandbox={sandbox_id}", flush=True)
        await sandbox.fs.upload_dir(
            source_dir=REPOSITORY / "infra", target_dir="/opt/phase5-infra", extract_timeout=120
        )
        await driver._stage_sandbox_file(
            sandbox, REPOSITORY / "scripts" / "phase5" / "guest_admin.py",
            "/opt/phase5/guest_admin.py",
        )
        deploy = await sandbox.arun(
            cmd=("$(command -v python3 || command -v python3.12) "
                 "/opt/phase5/guest_admin.py deploy --infra /opt/phase5-infra"),
            session="default", wait_timeout=600,
        )
        print(f"[phase5-build] deploy:\n{deploy.output}", flush=True)
        reset = await sandbox.arun(
            cmd=("$(command -v python3 || command -v python3.12) "
                 "/opt/phase5/guest_admin.py reset --run-patchers"),
            session="default", mode="nohup", wait_timeout=1200,
            output_file="/tmp/phase5-reset.log",
        )
        print(f"[phase5-build] persona reset:\n{reset.output}", flush=True)
        reset_output = str(reset.output or "")
        if "Traceback" in reset_output or '"errors": []' not in reset_output:
            raise RuntimeError("persona reset did not return a clean terminal record")
        if reset_output.count(".sqlite") < 17:
            raise RuntimeError("persona reset did not report all 17 databases")
        cursor = await sandbox.arun(
            cmd=("curl -fsS -X POST http://127.0.0.1:8080/pcapi/derail/cursor "
                 "-H 'Content-Type: application/json' -d '{\"action_index\":-1}'"),
            session="default", wait_timeout=120,
        )
        if '"action_index":-1' not in str(cursor.output).replace(" ", ""):
            raise RuntimeError(f"reset cursor gate failed: {cursor.output}")
        try:
            await sandbox.arun(
                cmd=("$(command -v python3 || command -v python3.12) "
                     "/opt/phase5/guest_admin.py shutdown"),
                session="default", wait_timeout=120,
            )
        except Exception as exc:  # noqa: BLE001 - shutdown may close its own HTTP request
            print(f"[phase5-build] shutdown response closed: {exc}", flush=True)
        await asyncio.sleep(20)
        build_command = f"""
set -e
for _ in $(seq 1 30); do
  pid=$(cat /tmp/mypcbench-guest.pid 2>/dev/null || true)
  {{ [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; }} && break
  sleep 5
done
pid=$(cat /tmp/mypcbench-guest.pid 2>/dev/null || true)
{{ [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; }}
mkdir -p {shlex.quote(remote_dir)}
qemu-img convert -p -O qcow2 -o cluster_size=64K \
  /tmp/mypcbench-guest-overlay.qcow2 {shlex.quote(remote_qcow2)}
sha256sum {shlex.quote(remote_qcow2)} > {shlex.quote(remote_qcow2)}.sha256
""".strip()
        build_output = await run_background(sandbox, build_command, "phase5-convert", 3600)
        print(f"[phase5-build] convert:\n{build_output}", flush=True)
        identity = await sandbox.arun(
            cmd=f"sha256sum {shlex.quote(remote_qcow2)}; stat -c %s {shlex.quote(remote_qcow2)}",
            session="default", wait_timeout=120,
        )
        values = str(identity.output).strip().splitlines()
        qcow2_sha = values[0].split()[0]
        qcow2_size = int(values[-1])
        manifest = {
            "schema": "phase5-vm-image/1.0",
            "tag": tag,
            "boot_date": boot_date,
            "infra_commit": infra_commit,
            "mypcbench_commit": driver.MYPCBENCH_COMMIT,
            "base_qcow2_sha256": "29e6ef0230655501920ad7b08c38e16c0d39e94c5be8220de2cf723885759e90",
            "qcow2_sha256": qcow2_sha,
            "qcow2_size": qcow2_size,
            "rock_image": driver.ROCK_SANDBOX_IMAGE,
            "rock_cluster": driver.ROCK_CLUSTER,
            "sandbox_id": sandbox_id,
        }
        with tempfile.TemporaryDirectory(prefix="phase5-manifest-") as directory:
            manifest_path = Path(directory) / "phase5_vm_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            await sandbox.fs.upload_dir(
                source_dir=directory,
                target_dir="/tmp/phase5-manifest-stage",
                extract_timeout=120,
            )
        upload_command = (
                f"set -e; set -a; . {shlex.quote(driver.SECRETS_REMOTE_PATH)}; set +a; "
                f"{driver._ossutil()} cp -f {shlex.quote(remote_qcow2)} "
                f"{shlex.quote(output_uri)}/{Path(remote_qcow2).name} && "
                f"{driver._ossutil()} cp -f {shlex.quote(remote_qcow2)}.sha256 "
                f"{shlex.quote(output_uri)}/{Path(remote_qcow2).name}.sha256 && "
                f"{driver._ossutil()} cp -f /tmp/phase5-manifest-stage/phase5_vm_manifest.json "
                f"{shlex.quote(output_uri)}/vm_acceptance_manifest.json"
            )
        upload_output = await run_background(sandbox, upload_command, "phase5-upload", 3600)
        print(f"[phase5-build] upload:\n{upload_output}", flush=True)
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
        return 0
    finally:
        print("[phase5-build] stopping sandbox", flush=True)
        await sandbox.stop()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
