#!/usr/bin/env python3
"""Decide how a Phase 5 variant world can be realized, without mutating the golden world.

The frozen guest already ships a seeded world, and /reset restores that same seeded state, so
seeding a patched persona on top of it collides on deterministic primary keys. This probe asks
the one question that gates Stage 2: can /opt/generator build a complete world into an empty
--data-dir, and does such a regeneration reproduce the baked BASE world?
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import json
import os
import shlex
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SMOKE_PATH = REPO / "scripts" / "phase5" / "run_hazard_smoke.py"
PROBE_ROOT = "/tmp/phase5-probe"
PERSONA = "michael_scott"
PERSONA_SOURCE = "/opt/personas/michael_scott.json"
GENERATOR_DIR = "/opt/generator"
GENERATOR_ARCHIVE = "/tmp/phase5-generator.tar.gz"
CALENDAR_DB = "hoolicalendar.sqlite"
TAGS = ("base", "variant")
# configs/synthesis/hazards_v1.yaml: distractor_candidates / same_title_prefix_events
RATE_QUERY = (
    "SELECT COUNT(*) FROM (SELECT substr(title, 1, 12) p FROM events "
    "GROUP BY p HAVING COUNT(*) > 1)"
)
# generate.py derives a fresh seed and reference time when these flags are absent, which made the
# two arms differ by more than the patch. Both arms must pass the same values verbatim.
PROBE_SEED = "4242424242"
PROBE_REFERENCE_TIME = "2026-09-01T07:24:24.351485+00:00"
SCAN_TOKENS = ("CREATE TABLE", "executescript", "sqlite_master", "INSERT OR IGNORE",
               "INSERT OR REPLACE", "ON CONFLICT", "required=True", "reference_time",
               "_seed_meta", "deterministic")
CANDIDATE_FLAGS = ("--persona", "--personas-dir", "--data-dir", "--home-dir", "--strict",
                   "--seed", "--reference-time", "--apps", "--skip", "--world", "--force",
                   "--dry-run", "--no-email", "--quiet")


def load_smoke():
    spec = importlib.util.spec_from_file_location("phase5_hazard_smoke_for_probe", SMOKE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def assert_isolated(*paths: str) -> None:
    """Generation must never point at the golden world; /tmp/phase5-probe is the only target."""
    for path in paths:
        if path != PROBE_ROOT and not path.startswith(f"{PROBE_ROOT}/"):
            raise RuntimeError(f"probe path escapes the isolated root: {path}")


def probe_paths(tag: str) -> dict:
    root = f"{PROBE_ROOT}/{tag}"
    paths = {
        "personas": f"{root}/personas",
        "data": f"{root}/data",
        "home": f"{root}/home",
        "log": f"{root}/generate.log",
        "rc": f"{root}/generate.rc",
    }
    assert_isolated(*paths.values())
    return paths


def generation_script(paths: dict) -> str:
    assert_isolated(paths["data"], paths["home"], paths["personas"])
    return (
        f"python3 generate.py --persona {PERSONA} "
        f"--personas-dir {paths['personas']} --data-dir {paths['data']} "
        f"--home-dir {paths['home']} --seed {PROBE_SEED} "
        f"--reference-time {PROBE_REFERENCE_TIME} --strict "
        f">{paths['log']} 2>&1; "
        f"rc=$?; printf '%s\\n' \"$rc\" >{paths['rc']}; exit \"$rc\""
    )


def variant_patch_command(persona: str, patch: list) -> str:
    assert_isolated(persona)
    encoded = base64.b64encode(json.dumps(patch, separators=(",", ":")).encode()).decode()
    body = (
        "import base64, json\n"
        f"path = {json.dumps(persona)}\n"
        f"patch = json.loads(base64.b64decode({json.dumps(encoded)}).decode())\n"
        "doc = json.load(open(path))\n"
        "events = doc.setdefault('app_overrides', {}).setdefault('calendar', {})\n"
        "events.setdefault('events', []).append(patch[0]['value'])\n"
        "json.dump(doc, open(path, 'w'), indent=2)\n"
    )
    return f"python3 -c {shlex.quote(body)}"


def stage_script(patch: list) -> str:
    """Copy the baked persona aside once per arm and add the hazard event to the variant arm."""
    commands = []
    for tag in TAGS:
        paths = probe_paths(tag)
        persona = f"{paths['personas']}/{PERSONA}.json"
        commands.append(
            # The control API may stage as root while generate.py runs as --uid=user.
            f"mkdir -p {shlex.quote(paths['personas'])} {shlex.quote(paths['data'])} "
            f"{shlex.quote(paths['home'])}\n"
            f"chmod 777 {shlex.quote(paths['personas'])} {shlex.quote(paths['data'])} "
            f"{shlex.quote(paths['home'])}\n"
            f"cp {shlex.quote(PERSONA_SOURCE)} {shlex.quote(persona)}"
        )
        if tag == "variant":
            commands.append(variant_patch_command(persona, patch))
    return "\n".join(commands) + "\n"


INVENTORY_SCRIPT = f"""python3 - <<'PY'
import json, os, subprocess

TOKENS = {SCAN_TOKENS!r}
FLAGS = {CANDIDATE_FLAGS!r}
GENERATOR_DIR = {GENERATOR_DIR!r}

def run(argv, cwd=None, shell=False):
    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                              timeout=90, shell=shell)
        return {{"rc": proc.returncode, "out": proc.stdout[-4000:], "err": proc.stderr[-1500:]}}
    except Exception as exc:
        return {{"rc": None, "out": "", "err": str(exc)[:500]}}

files, sources = [], {{}}
for root, _dirs, names in os.walk(GENERATOR_DIR):
    for name in sorted(names):
        path = os.path.join(root, name)
        try:
            size = os.path.getsize(path)
        except OSError:
            size = -1
        files.append({{"path": path, "size": size}})
        if path.endswith(".py"):
            try:
                sources[path] = open(path, encoding="utf-8", errors="replace").read()
            except OSError:
                sources[path] = ""

scan = {{}}
for path, text in sources.items():
    scan[path] = {{"lines": text.count("\\n")}}
    for token in TOKENS + FLAGS:
        hits = text.count(token)
        if hits:
            scan[path][token] = hits

report = {{
    "identity": run("id; echo ---; sudo -n -l 2>&1 | head -20", shell=True),
    "generator_files": files,
    "python_files": len(sources),
    "scan": scan,
    "help": run(["python3", "generate.py", "--help"], cwd=GENERATOR_DIR),
    "personas": run(["ls", "-l", "/opt/personas"]),
    "world_layout": run("ls -l /data | head -25; echo ---VMS; ls /data/vms 2>&1 | head -25; "
                        "echo ---WORLDS; ls /data/worlds 2>&1 | head -25; "
                        "echo ---META; head -c 1200 /data/_seed_meta.json 2>&1", shell=True),
    "apps": run(["systemctl", "is-active", "mypcbench-apps.target"]),
}}
print(json.dumps(report, sort_keys=True))
PY"""


def measure_script() -> str:
    data_dirs = {tag: probe_paths(tag)["data"] for tag in TAGS}
    title_prefix = load_smoke().PATCH[0]["value"]["summary"][:12]
    return f"""python3 - <<'PY'
import glob, json, os, sqlite3

DATA_DIRS = {json.dumps(data_dirs)}
CALENDAR_DB = {json.dumps(CALENDAR_DB)}
RATE_QUERY = {json.dumps(RATE_QUERY)}
TITLE_PREFIX = {json.dumps(title_prefix)}

def summarize(path):
    entry = {{"name": os.path.basename(path), "size": os.path.getsize(path)}}
    conn = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    try:
        tables = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        entry["tables"] = len(tables)
        rows = 0
        for table in tables:
            try:
                rows += conn.execute('SELECT COUNT(*) FROM "%s"' % table).fetchone()[0]
            except sqlite3.Error:
                pass
        entry["rows"] = rows
        if entry["name"] == CALENDAR_DB and "events" in tables:
            entry["events"] = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            entry["same_title_prefix_events"] = conn.execute(RATE_QUERY).fetchone()[0]
            entry["target_title_events"] = conn.execute(
                "SELECT COUNT(*) FROM events WHERE title LIKE ?",
                (TITLE_PREFIX + "%",)).fetchone()[0]
    finally:
        conn.close()
    return entry

out = {{}}
for tag, data_dir in DATA_DIRS.items():
    arm = {{"databases": [], "error": None, "other_files": []}}
    if not os.path.isdir(data_dir):
        arm["error"] = "data dir absent"
        out[tag] = arm
        continue
    for path in sorted(glob.glob(os.path.join(data_dir, "*.sqlite"))):
        try:
            arm["databases"].append(summarize(path))
        except sqlite3.Error as exc:
            arm["databases"].append({{"name": os.path.basename(path),
                                      "error": str(exc)[:200]}})
    meta = os.path.join(data_dir, "_seed_meta.json")
    if os.path.exists(meta):
        arm["seed_meta"] = open(meta, errors="replace").read()[:1200]
    arm["other_files"] = sorted(os.path.basename(p) for p in
                                glob.glob(os.path.join(data_dir, "*"))
                                if not p.endswith(".sqlite"))[:40]
    out[tag] = arm
print(json.dumps(out, sort_keys=True))
PY"""


def gold_uniqueness_evidence(smoke) -> dict:
    record = smoke.SMOKE_RECORD
    task_id = record["task_id"]
    candidate_path = REPO / "data/synthesis/generation/final_v1/records" / f"{task_id}.json"
    candidate = json.loads(candidate_path.read_text())
    verifier_path = REPO / "data/synthesis/generation/final_v1" / candidate["verifier_bundle_ref"]
    injected_title = record["persona_diff"][0]["value"]["summary"]
    collides = any(
        injected_title.casefold() in item["instruction"].casefold()
        for item in candidate["realization"]["instructions"]
    )
    return {
        "gold_unique": candidate["gold_execution"]["final_verifier_passed"] is True and not collides,
        "gold_execution_passed": candidate["gold_execution"]["passed"],
        "injected_title_collides_with_instruction": collides,
        "verifier_bundle": candidate["verifier_bundle_ref"],
        "verifier_sha256": hashlib.sha256(verifier_path.read_bytes()).hexdigest(),
    }


def poll_script(paths: dict, unit: str) -> str:
    return f"""python3 - <<'PY'
import json, pathlib, subprocess
rc = pathlib.Path({paths['rc']!r})
log = pathlib.Path({paths['log']!r})
state = subprocess.run(['systemctl', 'show', {unit!r}, '--property=ActiveState', '--value'],
                       capture_output=True, text=True).stdout.strip()
print(json.dumps({{'returncode': int(rc.read_text().strip()) if rc.exists() else None,
                   'unit_state': state,
                   'log_tail': log.read_text(errors='replace')[-6000:] if log.exists() else ''}}))
PY"""


async def generate_probe(smoke, sandbox, paths: dict, tag: str, minutes: int) -> dict:
    unit = f"phase5-probe-{tag}"
    launch = (
        f"sudo -n systemd-run --unit={unit} --uid=user --working-directory={GENERATOR_DIR} "
        f"/bin/bash -lc {shlex.quote(generation_script(paths))}"
    )
    await smoke.guest_exec(sandbox, launch, timeout=60)
    deadline = time.monotonic() + minutes * 60
    status = {"returncode": None, "unit_state": "unknown", "log_tail": ""}
    while time.monotonic() < deadline:
        await asyncio.sleep(20)
        status = json.loads(await smoke.guest_exec(sandbox, poll_script(paths, unit), timeout=60))
        if status["returncode"] is not None:
            return status
        if status["unit_state"] not in ("active", "activating"):
            status["note"] = "unit left active state without writing the rc file"
            return status
    status["note"] = f"generation exceeded the {minutes}-minute budget"
    return status


def verdict(measurement: dict, generations: dict) -> dict:
    """State what the probe observed; hazard status stays PENDING either way."""

    summary = {}
    for tag in TAGS:
        arm = measurement.get(tag, {})
        calendar = next((db for db in arm.get("databases", [])
                         if db.get("name") == CALENDAR_DB), {})
        summary[tag] = {
            "returncode": generations.get(tag, {}).get("returncode"),
            "databases_built": len(arm.get("databases", [])),
            "events": calendar.get("events"),
            "same_title_prefix_events": calendar.get("same_title_prefix_events"),
            "target_title_events": calendar.get("target_title_events"),
            "error": arm.get("error"),
        }
    summary["generator_builds_empty_data_dir"] = any(
        summary[tag]["databases_built"] for tag in TAGS)
    reference_times = {}
    for tag in TAGS:
        try:
            meta = json.loads(measurement.get(tag, {}).get("seed_meta") or "{}")
        except json.JSONDecodeError:
            meta = {}
        reference_times[tag] = meta.get("bake_reference_time")
    summary["arm_reference_times"] = reference_times
    summary["arms_controlled"] = (
        reference_times["base"] is not None
        and reference_times["base"] == reference_times["variant"])
    base_rate = summary["base"]["same_title_prefix_events"]
    variant_rate = summary["variant"]["same_title_prefix_events"]
    # An uncontrolled pair differs by more than the patch, so no rate verdict is reportable.
    summary["variant_rate_is_base_plus_one"] = None if not summary["arms_controlled"] else (
        isinstance(base_rate, int) and variant_rate == base_rate + 1)
    summary["seeder_consistent"] = (
        summary["arms_controlled"]
        and all(summary[tag]["returncode"] == 0 for tag in TAGS)
        and all(summary[tag]["databases_built"] == 17 for tag in TAGS)
    )
    return summary


async def upload_evidence(smoke, driver, sandbox, sandbox_path: str, oss_target: str,
                          local_dir: Path) -> str:
    upload = (
        f"set -a; . {shlex.quote(driver.SECRETS_REMOTE_PATH)} 2>/dev/null; set +a; "
        f"{driver._ossutil()} cp -f {shlex.quote(sandbox_path)} {shlex.quote(oss_target)}"
    )
    await smoke.sandbox_run_background(sandbox, upload, f"probe-upload-{Path(sandbox_path).name}", 300)
    local_path = local_dir / Path(sandbox_path).name
    await asyncio.to_thread(smoke._oss_download, oss_target, local_path)
    local_sha = hashlib.sha256(local_path.read_bytes()).hexdigest()
    if local_sha != await smoke.sandbox_file_sha(sandbox, sandbox_path):
        raise RuntimeError(f"probe evidence transfer mismatch: {sandbox_path}")
    return local_sha


async def run() -> int:
    smoke = load_smoke()
    driver = smoke.load_driver()
    os.environ["PHASE5_KEEP_SUPERVISOR"] = "1"
    from rock.actions import CreateBashSessionRequest
    from rock.sdk.sandbox.client import Sandbox

    stamp = datetime.now(timezone.utc)
    probe_id = f"probe-generator-{stamp:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    local_dir = REPO / "artifacts" / "phase5" / "probes" / probe_id
    local_dir.mkdir(parents=True, exist_ok=False)
    oss_uri = (f"oss://{os.environ['OSS_BUCKET']}/"
               f"{os.environ['OSS_PREFIX']}/DERAIL/phase5/probes/{probe_id}")
    print(f"[probe] local={local_dir}\n[probe] oss={oss_uri}", flush=True)

    sandbox = Sandbox(driver._sandbox_config())
    sandbox_id = None
    report: dict = {"schema": "phase5-generator-probe/1.0", "probe_id": probe_id,
                    "started": stamp.isoformat(), "oss_uri": oss_uri,
                    "hazard_id": smoke.HAZARD_ID, "hazard_status": "pending"}
    try:
        sandbox = await driver._setup_sandbox_proxy(sandbox, CreateBashSessionRequest)
        sandbox_id = driver._sandbox_id(sandbox)
        report["sandbox_id"] = sandbox_id
        print(f"[probe] sandbox={sandbox_id}", flush=True)
        await asyncio.sleep(60)

        inventory = json.loads(await smoke.guest_exec(sandbox, INVENTORY_SCRIPT, timeout=180))
        report["inventory"] = inventory
        (local_dir / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
        print(f"[probe] generator python files: {inventory['python_files']}; "
              f"flags seen: {sorted({t for scan in inventory['scan'].values() for t in scan})}",
              flush=True)

        guest_digest = await smoke.guest_exec(
            sandbox,
            f"tar czf {GENERATOR_ARCHIVE} -C /opt generator && "
            f"sha256sum {GENERATOR_ARCHIVE} && stat -c 'size=%s' {GENERATOR_ARCHIVE}",
            timeout=180,
        )
        (local_dir / "generator-guest-digest.txt").write_text(guest_digest)
        await smoke.guest_fetch_file(sandbox, GENERATOR_ARCHIVE, GENERATOR_ARCHIVE)
        sandbox_sha = await smoke.sandbox_file_sha(sandbox, GENERATOR_ARCHIVE)
        if sandbox_sha not in guest_digest:
            raise RuntimeError(f"generator archive digest mismatch: sandbox {sandbox_sha}")
        report["generator_sha256"] = await upload_evidence(
            smoke, driver, sandbox, GENERATOR_ARCHIVE, f"{oss_uri}/generator.tar.gz", local_dir)

        await smoke.guest_exec(sandbox, stage_script(smoke.PATCH), timeout=120)
        generations, report["generation"] = {}, {}
        for tag in TAGS:
            print(f"[probe] generating {tag} world into {probe_paths(tag)['data']}", flush=True)
            status = await generate_probe(smoke, sandbox, probe_paths(tag), tag, minutes=25)
            (local_dir / f"generate-{tag}.log").write_text(status.pop("log_tail", ""))
            generations[tag] = status
            report["generation"][tag] = status
            print(f"[probe] {tag}: rc={status['returncode']} state={status['unit_state']}",
                  flush=True)

        measurement = json.loads(await smoke.guest_exec(sandbox, measure_script(), timeout=900))
        report["measurement"] = measurement
        (local_dir / "measurement.json").write_text(json.dumps(measurement, indent=2) + "\n")
        report["verdict"] = verdict(measurement, generations)
        report["gold_uniqueness"] = gold_uniqueness_evidence(smoke)
        report["profile_after"] = report["verdict"]["variant"]
        report["attempt_lineage"] = [{
            "attempt_id": probe_id,
            "hazard_id": smoke.HAZARD_ID,
            "seed": PROBE_SEED,
            "reference_time": PROBE_REFERENCE_TIME,
            "generator_sha256": report["generator_sha256"],
        }]
        report["status"] = "completed"
    except Exception as exc:  # noqa: BLE001 - the probe must always land a record
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[probe] failed: {report['error']}", flush=True)
    finally:
        report["finished"] = datetime.now(timezone.utc).isoformat()
        (local_dir / "probe_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("\n=== PROBE VERDICT ===", flush=True)
        print(json.dumps({"status": report.get("status"), "verdict": report.get("verdict"),
                          "error": report.get("error")}, indent=2, sort_keys=True), flush=True)
        supervisor = getattr(driver, "_PHASE5_SUPERVISOR_TASK", None)
        if supervisor is not None:
            supervisor.cancel()
        await sandbox.stop()
        print(f"[probe] sandbox {sandbox_id} stop requested; record at {local_dir}", flush=True)
    return 0 if report.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
