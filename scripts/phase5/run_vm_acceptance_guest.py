#!/usr/bin/env python3
"""Run the ten Phase 5 VM acceptance trajectories inside MyPCBench."""

from __future__ import annotations

import base64
import hashlib
import hmac
import importlib.util
import json
import shlex
import shutil
import sqlite3
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1:5000"
ROOT = Path("/tmp/derail-phase5-acceptance")
RULES_PATH = "/opt/derail/volatile_columns.json"
DATABASES = (
    "batbucks", "buzzchat", "cheskepdia", "dinoco-airlines", "etaxi",
    "hangrydash", "hoolicalendar", "hoolishop", "kwik-e-mart", "lockedin",
    "mail", "oddsmarket", "speedtax", "sprintboard", "tablefind", "vaultbank",
    "workbuzz",
)
APP_UNITS = tuple(f"mypcbench-{name}.service" for name in DATABASES)
TRAJECTORIES = (
    ("acceptance-gui-00", "gui", ("hoolicalendar",)),
    ("acceptance-bash-01", "bash", ("workbuzz",)),
    ("acceptance-cross-app-02", "cross_app", ("mail", "hoolicalendar")),
    ("acceptance-bash-03", "bash", ("batbucks",)),
    ("acceptance-bash-04", "bash", ("buzzchat",)),
    ("acceptance-cross-app-05", "cross_app", ("sprintboard", "workbuzz")),
    ("acceptance-bash-06", "bash", ("hoolishop",)),
    ("acceptance-bash-07", "bash", ("lockedin",)),
    ("acceptance-cross-app-08", "cross_app", ("hangrydash", "vaultbank")),
    ("acceptance-bash-09", "bash", ("tablefind",)),
)


def request(path: str, payload=None, raw: bool = False):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=1200) as response:
        body = response.read()
    return body if raw else json.loads(body)


def load_replayer():
    spec = importlib.util.spec_from_file_location(
        "phase5_changelog_replay", "/opt/derail/changelog_replay.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def copy_database(source: str | Path, target: Path) -> None:
    """Create a consistent snapshot that includes committed WAL pages."""
    source_connection = sqlite3.connect(str(source))
    target_connection = sqlite3.connect(target)
    try:
        source_connection.backup(target_connection)
    finally:
        target_connection.close()
        source_connection.close()


def mutation(db: str, suffix: str) -> tuple[str, int]:
    path = f"/data/{db}.sqlite"
    conn = sqlite3.connect(path)
    try:
        table_types = {str(row[1]): str(row[2]) for row in conn.execute("PRAGMA table_list")}
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        for (table,) in rows:
            if table.startswith(("sqlite_", "_")) or table_types.get(table) != "table":
                continue
            columns = conn.execute(f"PRAGMA table_info({quote(table)})").fetchall()
            for column in columns:
                name, kind = str(column[1]), str(column[2]).upper()
                if "CHAR" not in kind and "TEXT" not in kind and kind:
                    continue
                row = conn.execute(f"SELECT rowid FROM {quote(table)} ORDER BY rowid LIMIT 1").fetchone()
                if row:
                    literal = suffix.replace("'", "''")
                    sql = (
                        f"UPDATE {quote(table)} SET {quote(name)} = "
                        f"COALESCE({quote(name)}, '') || '{literal}' "
                        f"WHERE rowid = {int(row[0])}"
                    )
                    return sql, int(row[0])
    finally:
        conn.close()
    raise RuntimeError(f"no mutable text row in {db}")


def run_sql(db: str, sql: str, gui: bool = False) -> float:
    command = f"sqlite3 {shlex.quote(f'/data/{db}.sqlite')} {shlex.quote(sql)}"
    started = time.monotonic()
    if gui:
        script = (
            "import os,time; os.environ['DISPLAY']=':0'; "
            "os.environ['XAUTHORITY']='/run/user/1000/gdm/Xauthority'; "
            "import pyautogui; pyautogui.FAILSAFE=False; pyautogui.PAUSE=0.1; "
            "pyautogui.hotkey('ctrl','alt','t'); time.sleep(2); "
            f"pyautogui.write({command!r}, interval=0.001); pyautogui.press('enter'); "
            "time.sleep(2); pyautogui.hotkey('alt','f4')"
        )
        subprocess.run(["python3", "-c", script], check=True, timeout=30)
    else:
        subprocess.run(["sqlite3", f"/data/{db}.sqlite", sql], check=True, timeout=30)
    return time.monotonic() - started


def untraced_duration(baseline: Path, sql: str) -> float:
    scratch = baseline.with_name(baseline.stem + "-untraced.sqlite")
    shutil.copyfile(baseline, scratch)
    conn = sqlite3.connect(scratch)
    try:
        for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'trg_derail_%'"
        ).fetchall():
            conn.execute(f"DROP TRIGGER IF EXISTS {quote(str(name))}")
        conn.commit()
        started = time.monotonic()
        conn.execute(sql)
        conn.commit()
        return time.monotonic() - started
    finally:
        conn.close()


def traced_duration(baseline: Path, sql: str) -> float:
    scratch = baseline.with_name(baseline.stem + "-traced.sqlite")
    shutil.copyfile(baseline, scratch)
    conn = sqlite3.connect(scratch)
    try:
        conn.execute("UPDATE _cursor SET action_index = 0 WHERE id = 1")
        conn.commit()
        started = time.monotonic()
        conn.execute(sql)
        conn.commit()
        return time.monotonic() - started
    finally:
        conn.close()


def probe_pages() -> None:
    payload = json.dumps({
        "email": "michael.scott@dundermifflin.com",
        "app": "batbucks",
    })
    signature = hmac.new(
        b"mypcbench-session-2026", payload.encode(), hashlib.sha256
    ).hexdigest()
    token = base64.b64encode(f"{payload}.{signature}".encode()).decode()
    page = urllib.request.Request(
        "http://127.0.0.1:3002/api/holdings",
        headers={"Cookie": f"session_batbucks={token}"},
    )
    deadline = time.monotonic() + 60
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(page, timeout=10) as response:
                response.read(64)
            return
        except urllib.error.URLError as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"BatBucks page probe did not become ready: {last_error}")


def probe_all_apps() -> None:
    """Wait until every app has bound its HTTP port and completed startup migrations."""
    pending = set(range(3001, 3018))
    deadline = time.monotonic() + 90
    while pending and time.monotonic() < deadline:
        for port in tuple(pending):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2).close()
                pending.remove(port)
            except urllib.error.HTTPError:
                pending.remove(port)
            except urllib.error.URLError:
                pass
        if pending:
            time.sleep(1)
    if pending:
        raise RuntimeError(f"app ports did not become ready: {sorted(pending)}")
    # Listeners can bind immediately before their final migration transaction commits.
    time.sleep(2)


def set_apps(command: str) -> None:
    subprocess.run(
        ["sudo", "-n", "systemctl", command, *APP_UNITS],
        check=True,
        timeout=180,
    )


def wait_for_changelog_quiescence() -> dict[str, int]:
    """Return only after all database changelog heads are stable for five polls."""
    previous = None
    stable_polls = 0
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        current = request("/derail/seq")["seq"]
        if current == previous:
            stable_polls += 1
            if stable_polls >= 5:
                return current
        else:
            previous = current
            stable_polls = 0
        time.sleep(1)
    raise RuntimeError("changelog heads did not quiesce after app shutdown")


def replay_digests(directory: Path, rows: list[dict]) -> dict[str, str]:
    replayer = load_replayer()
    rules = replayer.load_volatile_rules(RULES_PATH)
    by_db = {name: [] for name in DATABASES}
    for row in rows:
        by_db.setdefault(row["db"], []).append(row)
    digests = {}
    for db in DATABASES:
        baseline = directory / "baseline" / f"{db}.sqlite"
        replayed = directory / "replayed" / f"{db}.sqlite"
        replayed.parent.mkdir(exist_ok=True)
        replayer.replay(str(baseline), by_db.get(db, []), str(replayed))
        digests[db] = replayer.digest(str(replayed), rules=rules, app=db)
    return digests


def reset_failure(reset: dict) -> dict:
    """Keep reset failures diagnostic without dumping multi-megabyte patch logs."""
    copied = reset.get("copied_dbs", [])
    return {
        "copied_db_count": len(copied),
        "copied_dbs": copied,
        "errors": reset.get("errors", []),
        "status": reset.get("status"),
    }


def run_trajectory(index: int, spec: tuple) -> dict:
    trajectory_id, kind, databases = spec
    directory = ROOT / trajectory_id
    (directory / "baseline").mkdir(parents=True, exist_ok=True)
    set_apps("stop")
    reset = request("/reset", {})
    if reset.get("errors") or len(reset.get("copied_dbs", [])) != 17:
        raise RuntimeError(
            f"{trajectory_id}: reset failed: "
            f"{json.dumps(reset_failure(reset), sort_keys=True)}"
        )
    set_apps("stop")
    request("/derail/cursor", {"action_index": -1})
    # The shipped persona databases intentionally precede app-owned migrations.
    # Warm every service once so the replay baseline matches the runtime schema;
    # changelog replay is a DML ledger and does not attempt to reproduce DDL.
    set_apps("start")
    probe_all_apps()
    probe_pages()
    set_apps("stop")
    for db in DATABASES:
        copy_database(f"/data/{db}.sqlite", directory / "baseline" / f"{db}.sqlite")
    start_seq = request("/derail/seq")["seq"]
    observation_start = len(request("/derail/observations?action_index=0")["records"])
    set_apps("start")
    mutations = [(db, *mutation(db, f" [phase5-{index:02d}]")) for db in databases]
    request("/derail/cursor", {"action_index": 0})
    durations = []
    untraced = []
    for position, (db, sql, _rowid) in enumerate(mutations):
        baseline = directory / "baseline" / f"{db}.sqlite"
        durations.append(traced_duration(baseline, sql))
        untraced.append(untraced_duration(baseline, sql))
        run_sql(db, sql, gui=kind == "gui" and position == 0)
    probe_pages()
    screenshot = request("/screenshot", raw=True)
    screenshot_path = directory / "action_000.png"
    screenshot_path.write_bytes(screenshot)
    set_apps("stop")
    end_seq = wait_for_changelog_quiescence()
    since = urllib.parse.quote(json.dumps(start_seq, separators=(",", ":")))
    rows = request(f"/derail/changelog?since={since}")["rows"]
    observations = request("/derail/observations?action_index=0")["records"][observation_start:]
    direct = request("/derail/digest")
    direct_directory = directory / "direct"
    direct_directory.mkdir(exist_ok=True)
    for db in DATABASES:
        copy_database(f"/data/{db}.sqlite", direct_directory / f"{db}.sqlite")
    if request("/derail/seq")["seq"] != end_seq:
        raise RuntimeError(f"{trajectory_id}: database changed during terminal snapshot")
    replayed = replay_digests(directory, rows)
    attributed = [row for row in rows if row["action_index"] == 0]
    action = {
        "action_index": 0,
        "kind": kind,
        "cursor_written_before_dispatch": True,
        "changed": True,
        "changelog_attributed": bool(attributed),
        "changelog_rows": len(attributed),
        "observation_lag_actions": 0 if observations else None,
        "api_observation_count": sum(record.get("source") == "api" for record in observations),
        "cli_observation_count": sum(record.get("source") == "cli" for record in observations),
        "screenshot_sha256": hashlib.sha256(screenshot).hexdigest(),
        "tracing_overhead_seconds": max(0.0, max(durations) - max(untraced)),
    }
    return {
        "trajectory_id": trajectory_id,
        "kind": kind,
        "reset_cursor": -1,
        "reset_elapsed_seconds": reset.get("elapsed_s"),
        "task_start_seq": start_seq,
        "direct_digest": direct,
        "replay_digest": replayed,
        "actions": [action],
    }


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for index, spec in enumerate(TRAJECTORIES):
        print(f"[acceptance] {index + 1}/10 {spec[0]}", flush=True)
        records.append(run_trajectory(index, spec))
    report = {
        "schema": "vm-acceptance-report/1.0",
        "coverage": ["gui", "bash", "cross_app", "reset"],
        "trajectories": records,
    }
    output = ROOT / "vm_acceptance_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[acceptance] report={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
