#!/usr/bin/env python3
"""Run one real Phase 5 hazard seed smoke against the frozen guest image,
extract base and variant world snapshots, and upload evidence to OSS."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import json
import os
import re
import shlex
import tarfile
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DRIVER_PATH = REPO / "scripts" / "rock" / "derail_rock_driver.py"
OPTION_A_CATALOG = (
    REPO / "artifacts/phase5/preflight/hazards-option-a-20260912.jsonl"
)


def load_option_a_smoke_record() -> dict:
    for line in OPTION_A_CATALOG.read_text().splitlines():
        record = json.loads(line)
        if record.get("hazard_type") == "distractor_candidates" and record.get("status") == "pending":
            return record
    raise RuntimeError("option A catalog has no pending distractor hazard")


SMOKE_RECORD = load_option_a_smoke_record()
HAZARD_ID = SMOKE_RECORD["injection_id"]
PATCH = SMOKE_RECORD["persona_diff"]
EVIDENCE_APPS = ("etaxi", "hoolicalendar", "lockedin", "mail", "speedtax", "sprintboard", "workbuzz")
EVIDENCE_DIRS = ("Documents", "Desktop", "Downloads")
EVIDENCE_TAG = f"{HAZARD_ID}-20260912"


def load_driver():
    spec = importlib.util.spec_from_file_location("phase5_hazard_driver", DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _evidence_oss_uri() -> str:
    bucket = os.environ["OSS_BUCKET"]
    prefix = os.environ["OSS_PREFIX"]
    override = os.environ.get("PHASE5_EVIDENCE_OSS_URI")
    if override:
        return override.rstrip("/")
    tag = f"{EVIDENCE_TAG}-{uuid.uuid4().hex[:10]}"
    return f"oss://{bucket}/{prefix}/DERAIL/phase5/hazard-evidence/{tag}"


def guest_output(output: str) -> str:
    result = json.loads(output)
    if (
        not isinstance(result, dict)
        or type(result.get("returncode")) is not int
        or result["returncode"] != 0
        or result.get("status") != "success"
        or not isinstance(result.get("output"), str)
    ):
        raise RuntimeError(f"guest command failed: {result}")
    return result["output"]


async def guest_exec(sandbox, command: str, timeout: int = 300) -> str:
    payload = json.dumps({"command": ["bash", "-lc", command], "shell": False})
    encoded = base64.b64encode(payload.encode()).decode()
    remote = (
        f"printf %s {shlex.quote(encoded)} | base64 -d | "
        f"curl -fsS --max-time {timeout} -X POST "
        "http://127.0.0.1:8080/pcapi/execute "
        "-H 'Content-Type: application/json' --data-binary @-"
    )
    # A timed-out POST may already have mutated the guest; never replay it.
    result = await sandbox.arun(
        "bash -lc " + shlex.quote(remote), mode="nohup",
        wait_timeout=timeout + 30, wait_interval=5,
    )
    if result.exit_code != 0:
        raise RuntimeError(f"guest transport failed: {result.output}")
    return guest_output(str(result.output or ""))


async def guest_reset(sandbox) -> dict:
    result = await sandbox.arun(
        "curl -fsS --max-time 1200 -X POST http://127.0.0.1:8080/pcapi/reset "
        "-H 'Content-Type: application/json' -d '{\"skip_patchers\":false}'",
        mode="nohup", wait_timeout=1230, wait_interval=5,
    )
    if result.exit_code != 0:
        raise RuntimeError(f"guest reset transport failed: {result.output}")
    response = json.loads(str(result.output or ""))
    if not isinstance(response, dict) or response.get("errors") != []:
        raise RuntimeError(f"guest reset failed: {response}")
    return response


async def guest_fetch_file(sandbox, guest_path: str, sandbox_path: str,
                           timeout: int = 600, attempts: int = 3,
                           retry_delay: int = 15) -> str:
    """Proven by vm_acceptance: /pcapi/file streams a guest file via gateway.

    Runs in nohup mode because normal mode ignores wait_timeout and is capped by the
    SDK's fixed 300s HTTP read timeout. /pcapi/file only reads guest state, so a
    stalled attempt may be retried; each attempt stages to its own path and renames
    atomically, so a still-running earlier attempt cannot race the next one.
    """
    body = shlex.quote(json.dumps({"file_path": guest_path}))
    target = shlex.quote(sandbox_path)
    failure = "no attempt made"
    for attempt in range(1, attempts + 1):
        staging = shlex.quote(f"{sandbox_path}.part{attempt}")
        script = (
            "curl -fsS -X POST -H 'Content-Type: application/json' "
            f"-d {body} http://127.0.0.1:8080/pcapi/file -o {staging} && "
            f"mv -f {staging} {target} && stat -c 'size=%s' {target}"
        )
        result = await sandbox.arun(
            cmd="bash -lc " + shlex.quote(script),
            mode="nohup", wait_timeout=timeout, wait_interval=5,
        )
        output = str(result.output or "")
        size_line = next((line for line in output.splitlines()
                          if line.startswith("size=")), "")
        if result.exit_code == 0 and size_line:
            print(f"[evidence] fetched {guest_path} -> {sandbox_path}: {size_line} "
                  f"(attempt {attempt})", flush=True)
            return size_line
        failure = f"exit_code={result.exit_code} output={output[-300:]}"
        print(f"[evidence] fetch attempt {attempt}/{attempts} failed for "
              f"{guest_path}: {failure}", flush=True)
        if attempt < attempts:
            await asyncio.sleep(retry_delay)
    raise RuntimeError(f"guest file fetch failed for {guest_path}: {failure}")


async def download_evidence_file(sandbox, remote_path: str, local_path: Path) -> None:
    result = await sandbox.fs.download_file(remote_path=remote_path, local_path=str(local_path))
    if not result.success:
        raise RuntimeError(f"evidence download failed for {remote_path}: {result.message}")
    if not local_path.is_file() or local_path.stat().st_size == 0:
        raise RuntimeError(f"evidence download produced no file: {local_path}")


def _oss_download(oss_uri: str, local_path: Path) -> None:
    """Download from OSS to local file via locally-configured ossutil (v1.7.19)."""
    import subprocess
    result = subprocess.run(
        ["ossutil", "cp", "-f", oss_uri, str(local_path)],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"oss download failed: {result.stderr.strip() or result.stdout[-500:]}"
        )
    if not local_path.is_file() or local_path.stat().st_size == 0:
        raise RuntimeError(f"oss download produced no file: {local_path}")


def verify_snapshot(archive_path: Path, manifest_path: Path, stage: str) -> str:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["stage"] != stage or not manifest["files"]:
        raise RuntimeError(f"empty or incorrect snapshot manifest: {stage}")
    if manifest["tar_components"] != len(manifest["files"]):
        raise RuntimeError(f"snapshot component count mismatch: {stage}")
    with tarfile.open(archive_path) as archive:
        if archive.extractfile(f"{stage}_manifest.json").read() != manifest_bytes:
            raise RuntimeError(f"embedded manifest mismatch: {stage}")
        expected = {f"{stage}_manifest.json"}
        for entry in manifest["files"]:
            name = entry["arcname"]
            member = archive.getmember(name)
            if not member.isfile():
                raise RuntimeError(f"snapshot member is not a regular file: {name}")
            data = archive.extractfile(member).read()
            if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"snapshot file mismatch: {name}")
            if name in expected:
                raise RuntimeError(f"duplicate snapshot entry: {name}")
            expected.add(name)
        actual = [m.name for m in archive.getmembers() if not m.isdir()]
        if len(actual) != len(expected) or set(actual) != expected:
            raise RuntimeError(f"snapshot inventory mismatch: {stage}")
    return hashlib.sha256(archive_path.read_bytes()).hexdigest()


async def sandbox_run_background(sandbox, script: str, name: str, timeout: int) -> str:
    marker = name.upper().replace("-", "_") + "_DONE"
    command = f"set -e\nbash -lc {shlex.quote(script)}\nprintf '%s\\n' {shlex.quote(marker)}"
    result = await sandbox.arun(
        cmd="bash -lc " + shlex.quote(command), mode="nohup",
        wait_timeout=timeout, wait_interval=5,
    )
    output = str(result.output or "")
    if result.exit_code != 0 or output.rstrip().splitlines()[-1:] != [marker]:
        raise RuntimeError(f"{name} failed: {output}")
    return output


async def sandbox_file_sha(sandbox, sandbox_path: str) -> str:
    output = await sandbox_run_background(
        sandbox, f"sha256sum {shlex.quote(sandbox_path)}",
        f"sha-{Path(sandbox_path).name}", 600,
    )
    return output.split()[0]


def snapshot_report(result: str, stage: str) -> dict:
    """Parse the SNAPSHOT_<stage>_READY line the guest snapshot script prints."""
    match = re.search(
        rf"SNAPSHOT_{stage}_READY sha256=([0-9a-f]{{64}}) size=(\d+) "
        rf"manifest_components=(\d+)",
        result,
    )
    if not match:
        raise RuntimeError(f"snapshot report missing for {stage}: {result[-300:]}")
    return {
        "sha256": match.group(1),
        "size": int(match.group(2)),
        "manifest_components": int(match.group(3)),
    }


EVIDENCE_TEMPLATE = """\
import hashlib, json, pathlib, tarfile, io

mk = pathlib.Path("/tmp/phase5-evidence")
mk.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = mk / "{stage}_manifest.json"
TAR_PATH = pathlib.Path("/tmp/{stage}.tar.gz")

sidecars = [{"path": str(p), "size": p.stat().st_size}
            for p in sorted(pathlib.Path("/data").glob("*.sqlite-wal"))]
if any(p["size"] for p in sidecars):
    raise RuntimeError("uncheckpointed SQLite WAL files: " + json.dumps(sidecars))

gens = []
members = []

def add_source(src, arcname):
    data = src.read_bytes()
    gens.append({"src": str(src), "arcname": arcname, "size": len(data),
                 "sha256": hashlib.sha256(data).hexdigest()})
    members.append((arcname, data))

for p in sorted(pathlib.Path("/data").glob("*.sqlite")):
    if not p.is_file() or p.stat().st_size == 0: continue
    add_source(p, "data/" + p.name)
for d in {dirs}:
    base = pathlib.Path("/home/user") / d
    if not base.is_dir(): continue
    for f in sorted(base.rglob("*")):
        if not f.is_file(): continue
        add_source(f, "home/" + str(f.relative_to("/home/user")))

manifest = {"stage": "{stage}", "files": gens, "tar_components": len(gens), "sqlite_wal_files": sidecars}

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    manifest_bytes = json.dumps(manifest, indent=1, sort_keys=True).encode()
    MANIFEST_PATH.write_bytes(manifest_bytes)
    info = tarfile.TarInfo(name="{stage}_manifest.json")
    info.size = len(manifest_bytes)
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(manifest_bytes))
    for arcname, data in members:
        ti = tarfile.TarInfo(name=arcname)
        ti.size = len(data)
        ti.mode = 0o644
        tar.addfile(ti, io.BytesIO(data))
tar_path = TAR_PATH
tar_path.write_bytes(buf.getvalue())
tar_sha = hashlib.sha256(tar_path.read_bytes()).hexdigest()
tar_size = tar_path.stat().st_size
print(f"SNAPSHOT_{stage}_READY sha256={tar_sha} size={tar_size} manifest_components={len(gens)}")"""


async def run() -> int:
    driver = load_driver()
    os.environ["PHASE5_KEEP_SUPERVISOR"] = "1"
    from rock.actions import CreateBashSessionRequest
    from rock.sdk.sandbox.client import Sandbox

    evidence_oss_uri = _evidence_oss_uri()
    local_dir = REPO / "artifacts/phase5/hazard-evidence" / evidence_oss_uri.rsplit("/", 1)[-1]
    local_dir.mkdir(parents=True, exist_ok=False)
    print(f"[evidence] local directory: {local_dir}", flush=True)
    print(f"[evidence] OSS target: {evidence_oss_uri}", flush=True)

    sandbox = Sandbox(driver._sandbox_config())
    sandbox_id = None
    try:
        sandbox = await driver._setup_sandbox_proxy(sandbox, CreateBashSessionRequest)
        sandbox_id = driver._sandbox_id(sandbox)
        print(f"[smoke] sandbox={sandbox_id}", flush=True)
        await asyncio.sleep(60)

        asset_check = await sandbox.arun(
            "test -s /storage/mypcbench-vm/mypcbench.qcow2 && "
            "stat -c 'QCOW2_PRESENT size=%s' /storage/mypcbench-vm/mypcbench.qcow2",
            session="default", wait_timeout=60,
        )
        if "QCOW2_PRESENT" not in str(asset_check.output or ""):
            raise RuntimeError(f"sandbox asset check failed: {asset_check}")
        expected_qcow2_sha = os.environ.get("PHASE5_QCOW2_SHA256", "")
        if expected_qcow2_sha:
            sha_output = await sandbox_run_background(
                sandbox, "sha256sum /storage/mypcbench-vm/mypcbench.qcow2",
                "qcow2-checksum", 600,
            )
            actual_qcow2_sha = sha_output.split()[0]
            if actual_qcow2_sha != expected_qcow2_sha:
                raise RuntimeError(
                    f"qcow2 SHA mismatch: expected {expected_qcow2_sha}, got {actual_qcow2_sha}"
                )
            print(f"[smoke] qcow2 present sha256={actual_qcow2_sha} (matches frozen Phase 5 image)",
                  flush=True)
        else:
            print("[smoke] qcow2 present (no PHASE5_QCOW2_SHA256 set; skipped SHA check)",
                  flush=True)

        await driver._stage_sandbox_file(
            sandbox, REPO / "scripts/phase5/guest_admin.py", "/opt/phase5/guest_admin.py"
        )

        # =============================== PHASE 1: reset + guard + base snapshot
        print("[phase1] reset + guard + base evidence snapshot", flush=True)
        base_snap_py = EVIDENCE_TEMPLATE.replace("{stage}", "BASE").replace(
            "{dirs}", json.dumps(list(EVIDENCE_DIRS))
        )
        base_b64 = base64.b64encode(base_snap_py.encode()).decode()
        await guest_exec(
            sandbox,
            "python3 -c 'import base64,pathlib; "
            f"pathlib.Path(\"/tmp/phase5-snapshot-base.py\").write_bytes(base64.b64decode(\"{base_b64}\"))'",
            timeout=60,
        )
        # Step 1: reset the guest (synchronous, waits for completion)
        print("[phase1] step 1/3: reset", flush=True)
        reset_result = await guest_reset(sandbox)
        (local_dir / "reset.json").write_text(json.dumps(reset_result, indent=2) + "\n")
        print(f"[phase1] reset: {json.dumps(reset_result)}", flush=True)

        # Step 2: run guard to stop apps and clear sidecars
        print("[phase1] step 2/3: guard", flush=True)
        guard_result = await guest_exec(
            sandbox,
            "sudo -n /opt/derail/guest_seeder_guard.sh",
            timeout=60,
        )
        if guard_result.rstrip().splitlines()[-1:] != ["GUEST_SEEDER_QUIESCENT"]:
            raise RuntimeError(f"guest guard did not confirm quiescence: {guard_result}")
        (local_dir / "guard.txt").write_text(guard_result)
        print(f"[phase1] guard: {guard_result[:500]}", flush=True)

        # Step 3: build base evidence snapshot
        print("[phase1] step 3/3: base snapshot", flush=True)
        snap_result = await guest_exec(
            sandbox,
            "mkdir -p /tmp/phase5-evidence && python3 /tmp/phase5-snapshot-base.py",
            timeout=600,
        )
        print(f"[phase1] snapshot: {snap_result[-300:]}", flush=True)
        base_guest = snapshot_report(snap_result, "BASE")

        print("[phase1] fetching base snapshot", flush=True)
        await guest_fetch_file(sandbox, "/tmp/BASE.tar.gz", "/tmp/phase5-base.tar.gz")
        await guest_fetch_file(
            sandbox, "/tmp/phase5-evidence/BASE_manifest.json",
            "/tmp/phase5-base-manifest.json",
        )
        base_tar_sha = await sandbox_file_sha(sandbox, "/tmp/phase5-base.tar.gz")
        if base_tar_sha != base_guest["sha256"]:
            raise RuntimeError(
                f"base snapshot guest-to-sandbox mismatch: guest {base_guest['sha256']} "
                f"sandbox {base_tar_sha}"
            )
        for name in ("base.tar.gz", "base-manifest.json"):
            sandbox_path = f"/tmp/phase5-{name}"
            oss_target = f"{evidence_oss_uri}/{name}"
            upload_cmd = (
                f"set -a; . {shlex.quote(driver.SECRETS_REMOTE_PATH)} 2>/dev/null; set +a; "
                f"{driver._ossutil()} cp -f {shlex.quote(sandbox_path)} {shlex.quote(oss_target)}"
            )
            await sandbox_run_background(sandbox, upload_cmd, f"ev-upload-{name}", 300)
            await asyncio.to_thread(_oss_download, oss_target, local_dir / name)
        if verify_snapshot(local_dir / "base.tar.gz", local_dir / "base-manifest.json", "BASE") != base_tar_sha:
            raise RuntimeError("base snapshot transfer checksum mismatch")
        print(f"[phase1] base snapshot verified sha256={base_tar_sha}", flush=True)

        # =============================== PHASE 2: patch + seed + variant snapshot
        print("[phase2] launching persona patch + seed", flush=True)
        patch_json = json.dumps(PATCH, separators=(",", ":"))
        patch_b64 = base64.b64encode(patch_json.encode()).decode()
        phase2 = f"""
set -e
mkdir /tmp/phase5-personas
cp /opt/personas/michael_scott.json /tmp/phase5-personas/michael_scott.json
printf %s {shlex.quote(patch_b64)} | base64 -d >/tmp/phase5-persona-patch.json
python3 - <<'PY'
import json
p='/tmp/phase5-personas/michael_scott.json'
d=json.load(open(p)); patch=json.load(open('/tmp/phase5-persona-patch.json'))
d['app_overrides']['calendar']['events'].append(patch[0]['value'])
json.dump(d, open(p, 'w'), indent=2)
PY
"""
        await guest_exec(sandbox, phase2, timeout=60)
        seed = (
            "python3 generate.py --persona michael_scott --personas-dir /tmp/phase5-personas "
            "--data-dir /data --home-dir /home/user --strict "
            ">/tmp/phase5_hazard_seed.log 2>&1; "
            "rc=$?; printf '%s\\n' \"$rc\" >/tmp/phase5_hazard_seed.rc; exit \"$rc\""
        )
        await guest_exec(
            sandbox,
            "sudo -n systemd-run --unit=phase5-hazard-seed --uid=user "
            "--working-directory=/opt/generator /bin/bash -lc " + shlex.quote(seed),
            timeout=60,
        )
        print("[phase2] single seed unit launched, polling for terminal", flush=True)
        for _ in range(60):
            await asyncio.sleep(30)
            status = json.loads(await guest_exec(sandbox, """python3 - <<'PY'
import json, pathlib, subprocess
rc = pathlib.Path('/tmp/phase5_hazard_seed.rc')
log = pathlib.Path('/tmp/phase5_hazard_seed.log')
unit = subprocess.run(['systemctl', 'show', 'phase5-hazard-seed', '--property=ActiveState', '--value'], capture_output=True, text=True)
print(json.dumps({'returncode': int(rc.read_text()) if rc.exists() else None,
                  'unit_state': unit.stdout.strip(),
                  'log_tail': log.read_text()[-6000:] if log.exists() else ''}))
PY""", timeout=60))
            print(f"[phase2] seed status: {json.dumps(status)}", flush=True)
            if status["returncode"] is not None:
                if status["returncode"] != 0:
                    raise RuntimeError(f"hazard seeder failed; hazard remains PENDING: {status}")
                break
            if status["unit_state"] != "active":
                raise RuntimeError(f"seed unit stopped without completion evidence: {status}")
        else:
            raise RuntimeError("hazard seeder exceeded 30-minute timeout")

        # Integrity check (in-guest)
        integrity = await guest_exec(sandbox, """python3 - <<'PY'
import glob, json, sqlite3
errors=[]
for p in glob.glob('/data/*.sqlite'):
  try:
    c=sqlite3.connect('file:'+p+'?mode=ro', uri=True); ok=c.execute('PRAGMA integrity_check').fetchone()[0]; c.close()
    if ok != 'ok': errors.append((p, ok))
  except Exception as exc: errors.append((p, str(exc)))
stats=json.load(open('/data/_persona_stats.json'))
print(json.dumps({'hazard_id': 'hz-2a6d5c9f09', 'sqlite_integrity_ok': bool(glob.glob('/data/*.sqlite')) and not errors, 'persona_stats': stats, 'errors': errors}, sort_keys=True))
PY""")
        integrity_result = json.loads(integrity)
        (local_dir / "integrity.json").write_text(json.dumps(integrity_result, indent=2) + "\n")
        (local_dir / "seed.json").write_text(json.dumps(status, indent=2) + "\n")
        print(f"[phase2] integrity check:\n{integrity}", flush=True)

        # Variant snapshot
        print("[phase2] building variant snapshot", flush=True)
        variant_snap_py = EVIDENCE_TEMPLATE.replace("{stage}", "VARIANT").replace(
            "{dirs}", json.dumps(list(EVIDENCE_DIRS))
        )
        variant_b64 = base64.b64encode(variant_snap_py.encode()).decode()
        await guest_exec(
            sandbox,
            "python3 -c 'import base64,pathlib; "
            f"pathlib.Path(\"/tmp/phase5-snapshot-variant.py\").write_bytes(base64.b64decode(\"{variant_b64}\"))'",
            timeout=60,
        )
        snap_result = await guest_exec(
            sandbox, "mkdir -p /tmp/phase5-evidence && python3 /tmp/phase5-snapshot-variant.py",
            timeout=600,
        )
        print(f"[phase2] variant snapshot: {snap_result[-300:]}", flush=True)
        variant_guest = snapshot_report(snap_result, "VARIANT")

        await guest_fetch_file(sandbox, "/tmp/VARIANT.tar.gz", "/tmp/phase5-variant.tar.gz")
        await guest_fetch_file(
            sandbox, "/tmp/phase5-evidence/VARIANT_manifest.json",
            "/tmp/phase5-variant-manifest.json",
        )
        variant_sandbox_sha = await sandbox_file_sha(sandbox, "/tmp/phase5-variant.tar.gz")
        if variant_sandbox_sha != variant_guest["sha256"]:
            raise RuntimeError(
                f"variant snapshot guest-to-sandbox mismatch: guest {variant_guest['sha256']} "
                f"sandbox {variant_sandbox_sha}"
            )

        for name in ("variant.tar.gz", "variant-manifest.json"):
            sandbox_path = f"/tmp/phase5-{name}"
            oss_target = f"{evidence_oss_uri}/{name}"
            upload_cmd = (
                f"set -a; . {shlex.quote(driver.SECRETS_REMOTE_PATH)} 2>/dev/null; set +a; "
                f"{driver._ossutil()} cp -f {shlex.quote(sandbox_path)} {shlex.quote(oss_target)}"
            )
            await sandbox_run_background(sandbox, upload_cmd, f"ev-upload-{name}", 300)
            await asyncio.to_thread(_oss_download, oss_target, local_dir / name)
        variant_tar_sha = verify_snapshot(
            local_dir / "variant.tar.gz", local_dir / "variant-manifest.json", "VARIANT",
        )
        if variant_tar_sha != variant_sandbox_sha:
            raise RuntimeError(
                f"variant snapshot transfer mismatch: sandbox {variant_sandbox_sha} "
                f"local {variant_tar_sha}"
            )
        summary = {
            "schema": "phase5-hazard-evidence/1.0",
            "sandbox_id": sandbox_id,
            "hazard_id": HAZARD_ID,
            "patch": PATCH,
            "qcow2_sha256": expected_qcow2_sha,
            "evidence_oss_uri": evidence_oss_uri,
            "base_tar_sha256": base_tar_sha,
            "variant_tar_sha256": variant_tar_sha,
            "integrity": integrity_result,
            "status": "snapshots_collected",
            "hazard_status": "pending",
        }
        summary_path = local_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        for name in ("summary.json", "integrity.json", "seed.json", "reset.json", "guard.txt"):
            await driver._stage_sandbox_file(sandbox, local_dir / name, f"/tmp/phase5-{name}")

        # =============================== UPLOAD to OSS
        # Secrets already staged by _setup_sandbox_proxy; ossutil config at OSS_CONFIG_REMOTE_PATH.
        print("[evidence] uploading to OSS", flush=True)
        upload_script = (
            f"set -e; set -a; . {shlex.quote(driver.SECRETS_REMOTE_PATH)}; set +a; "
            f"{driver._ossutil()} cp -f /tmp/phase5-base.tar.gz "
            f"{shlex.quote(evidence_oss_uri)}/base.tar.gz && "
            f"{driver._ossutil()} cp -f /tmp/phase5-base-manifest.json "
            f"{shlex.quote(evidence_oss_uri)}/base_manifest.json && "
            f"{driver._ossutil()} cp -f /tmp/phase5-variant.tar.gz "
            f"{shlex.quote(evidence_oss_uri)}/variant.tar.gz && "
            f"{driver._ossutil()} cp -f /tmp/phase5-variant-manifest.json "
            f"{shlex.quote(evidence_oss_uri)}/variant_manifest.json"
        )
        for name in ("summary.json", "integrity.json", "seed.json", "reset.json", "guard.txt"):
            upload_script += (
                f" && {driver._ossutil()} cp /tmp/phase5-{name} "
                f"{shlex.quote(evidence_oss_uri + '/' + name)}"
            )
        upload_result = await sandbox_run_background(
            sandbox, upload_script, "hazard-evidence-upload", 600
        )
        print(f"[evidence] upload:\n{upload_result}", flush=True)
        print("\n=== EVIDENCE SUMMARY ===", flush=True)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0

    finally:
        if sandbox_id:
            print(f"[smoke] stopping sandbox {sandbox_id}", flush=True)
        else:
            print("[smoke] stopping sandbox", flush=True)
        supervisor = getattr(driver, "_PHASE5_SUPERVISOR_TASK", None)
        if supervisor is not None:
            supervisor.cancel()
        await sandbox.stop()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
