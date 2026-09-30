#!/usr/bin/env python3
"""Run same-day ROCK network and KVM probes for DERAIL Phase 5."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone

from rock.actions import CreateBashSessionRequest
from rock.sdk.sandbox.client import Sandbox
from rock.sdk.sandbox.config import SandboxConfig

IMAGE = os.environ.get("ROCK_SANDBOX_IMAGE", "<rock-sandbox-image>")
CLUSTER = os.environ.get("ROCK_CLUSTER", "")
PROBE_HOST = os.environ.get("PHASE5_PROBE_HOST", "")
EXPERIMENT = os.environ.get("ROCK_EXPERIMENT_ID", "derail-phase5-probe")


def config(cpus: int, memory: str) -> SandboxConfig:
    kwargs = {
        "base_url": os.environ["ROCK_BASE_URL"],
        "image": IMAGE,
        "auto_clear_seconds": 900,
        "startup_timeout": 600,
        "memory": memory,
        "cpus": cpus,
        "disk": "20g",
        "cluster": CLUSTER,
        "extra_headers": {
            "XRL-Authorization": f"Bearer {os.environ['ROCK_API_KEY']}"
        },
        "user_id": os.environ.get("ROCK_USER_ID", "522240"),
        "experiment_id": EXPERIMENT,
    }
    return SandboxConfig(**kwargs)


async def run_probe(name: str, cpus: int, memory: str, command: str) -> dict:
    sandbox = Sandbox(config(cpus, memory))
    record = {"name": name, "sandbox_id": None, "passed": False, "stopped": False}
    try:
        await sandbox.start()
        record["sandbox_id"] = next(
            (getattr(sandbox, field, None) for field in ("sandbox_id", "id", "instance_id")
             if getattr(sandbox, field, None)),
            None,
        )
        await sandbox.create_session(CreateBashSessionRequest(session="probe"))
        try:
            observation = await sandbox.arun(command, session="probe", wait_timeout=60)
            output = str(getattr(observation, "output", observation) or "")
            record["passed"] = f"PHASE5_{name.upper()}_OK" in output
            record["output"] = output[-2000:]
        except Exception as exc:  # noqa: BLE001 - retain a terminal probe record
            record["error"] = str(exc)
    finally:
        try:
            await sandbox.stop()
            record["stopped"] = True
        except Exception as exc:  # noqa: BLE001 - preserve stop failure in the gate artifact
            record["stop_error"] = str(exc)
    return record


async def main() -> int:
    if not CLUSTER or not PROBE_HOST:
        raise SystemExit("Phase 5 probes require ROCK_CLUSTER and PHASE5_PROBE_HOST")
    network = await run_probe(
        "network",
        1,
        "2g",
        f"getent hosts {PROBE_HOST} >/dev/null && "
        "code=$(curl -sS --connect-timeout 10 -o /dev/null -w '%{http_code}' "
        "https://www.aliyun.com) && test \"$code\" != 000 && echo PHASE5_NETWORK_OK",
    )
    kvm = await run_probe(
        "kvm",
        8,
        "16g",
        "ls /dev/kvm >/dev/null 2>&1 && echo PHASE5_KVM_OK",
    )
    report = {
        "schema": "rock-probe/1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cluster": CLUSTER,
        "image": IMAGE,
        "probes": [network, kvm],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(item["passed"] and item["stopped"] for item in report["probes"]) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
