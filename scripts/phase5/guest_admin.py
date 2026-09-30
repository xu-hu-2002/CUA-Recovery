#!/usr/bin/env python3
"""Administer a MyPCBench guest through its control API."""

from __future__ import annotations

import argparse
import base64
import io
import json
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


def request_json(base_url: str, path: str, payload: Optional[dict] = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=1200) as response:
        return json.loads(response.read())


def execute(base_url: str, command, shell: bool = False) -> dict:
    result = request_json(base_url, "/execute", {"command": command, "shell": shell})
    if result.get("returncode") != 0:
        raise RuntimeError(f"guest command failed: {result}")
    return result


def archive_directory(root: Path) -> str:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        archive.add(root, arcname="infra")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def put_file(base_url: str, source: Path, target: str) -> dict:
    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    writer = (
        "import base64,pathlib; p=pathlib.Path(" + repr(target) + "); "
        "p.parent.mkdir(parents=True,exist_ok=True); "
        "p.write_bytes(base64.b64decode(" + repr(encoded) + ")); p.chmod(0o755)"
    )
    return execute(base_url, ["python3", "-c", writer])


def start_acceptance(base_url: str, script: str) -> dict:
    command = (
        f"nohup python3 {script} >/tmp/phase5-acceptance.log 2>&1 & "
        "echo $! >/tmp/phase5-acceptance.pid"
    )
    return execute(base_url, command, shell=True)


def acceptance_status(base_url: str) -> dict:
    command = (
        "if [ -s /tmp/recovery-phase5-acceptance/vm_acceptance_report.json ]; then "
        "echo ACCEPTANCE_DONE; "
        "elif [ -s /tmp/phase5-acceptance.pid ] && "
        "kill -0 $(cat /tmp/phase5-acceptance.pid) 2>/dev/null; then echo ACCEPTANCE_RUNNING; "
        "else echo ACCEPTANCE_FAILED; fi; tail -30 /tmp/phase5-acceptance.log 2>/dev/null || true"
    )
    return execute(base_url, command, shell=True)


def deploy(base_url: str, infra: Path) -> dict:
    encoded = archive_directory(infra)
    writer = (
        "import base64,pathlib; "
        "pathlib.Path('/tmp/recovery-infra.tar.gz').write_bytes(base64.b64decode(" +
        repr(encoded) + "))"
    )
    execute(base_url, ["python3", "-c", writer])
    command = r"""
set -e
rm -rf /tmp/recovery-upload
mkdir -p /tmp/recovery-upload
tar -xzf /tmp/recovery-infra.tar.gz -C /tmp/recovery-upload
sudo mkdir -p /opt/recovery /data/_trace /data/_recovery
sudo install -m 755 /tmp/recovery-upload/infra/triggers/install_triggers.py /opt/recovery/install_triggers.py
sudo install -m 755 /tmp/recovery-upload/infra/snapshot/changelog_replay.py /opt/recovery/changelog_replay.py
sudo install -m 644 /tmp/recovery-upload/infra/tracer/trace.js /opt/recovery/trace.js
sudo install -m 644 /tmp/recovery-upload/infra/volatile_columns.json /opt/recovery/volatile_columns.json
sudo install -m 644 /tmp/recovery-upload/infra/control_api_recovery.py /opt/recovery/control_api_recovery.py
    sudo install -m 755 /tmp/recovery-upload/infra/canon_timeout_patch.py /opt/recovery/canon_timeout_patch.py
sudo install -m 755 /tmp/recovery-upload/infra/guest_seeder_guard.sh /opt/recovery/guest_seeder_guard.sh
sudo chown -R user:user /opt/recovery /data/_trace /data/_recovery
sudo python3 /opt/recovery/install_triggers.py --glob '/data/*.sqlite' --glob '/data/vms/*/*.sqlite' --glob '/data/worlds/*/*.sqlite'
for unit in /etc/systemd/system/mypcbench-*.service; do
  name=$(basename "$unit" .service)
  case "$name" in mypcbench-control-api|mypcbench-firstboot|mypcbench-runtime-fixup|mypcbench-*-on-boot) continue;; esac
  grep -q 'server.js' "$unit" || continue
  sudo mkdir -p "$unit.d"
  sudo install -m 644 /tmp/recovery-upload/infra/tracer/systemd-dropin.conf "$unit.d/recovery-trace.conf"
done
if [ ! -e /usr/bin/sqlite3.real ]; then sudo mv /usr/bin/sqlite3 /usr/bin/sqlite3.real; fi
sudo install -m 755 /tmp/recovery-upload/infra/sqlite3_wrapper.sh /usr/bin/sqlite3
sudo python3 /opt/recovery/canon_timeout_patch.py
sudo python3 /tmp/recovery-upload/infra/control_api_patch.py --main /opt/desktop-seed/server/main.py --infra-dir /opt/recovery
sudo systemctl daemon-reload
sudo systemctl enable mypcbench-control-api.service
nohup sh -c 'sleep 1; sudo systemctl restart mypcbench-control-api.service mypcbench-apps.target' >/tmp/recovery-restart.log 2>&1 &
"""
    execute(base_url, command, shell=True)
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        try:
            status = request_json(base_url, "/recovery/status")
            if status.get("databases"):
                return status
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(5)
    raise RuntimeError("RECOVERY control API did not recover after deployment")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("deploy", "status", "reset", "shutdown", "put", "start", "acceptance-status"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/pcapi")
    parser.add_argument("--infra", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--run-patchers", action="store_true")
    args = parser.parse_args()
    if args.command == "deploy":
        if args.infra is None:
            parser.error("deploy requires --infra")
        result = deploy(args.base_url, args.infra.resolve())
    elif args.command == "status":
        result = request_json(args.base_url, "/recovery/status")
    elif args.command == "reset":
        result = request_json(
            args.base_url, "/reset", {"skip_patchers": not args.run_patchers}
        )
    elif args.command == "shutdown":
        result = execute(args.base_url, "sudo sync; sudo shutdown -h now", shell=True)
    elif args.command == "put":
        if args.source is None or not args.target:
            parser.error("put requires --source and --target")
        result = put_file(args.base_url, args.source, args.target)
    elif args.command == "start":
        if not args.target:
            parser.error("start requires --target")
        result = start_acceptance(args.base_url, args.target)
    else:
        result = acceptance_status(args.base_url)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
