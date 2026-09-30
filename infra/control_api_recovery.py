from __future__ import annotations

import glob
import importlib.util
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Mapping, Optional

DB_DIR = os.environ.get("RECOVERY_DB_DIR", "/data")
TRACE_DIR = os.environ.get("RECOVERY_TRACE_DIR", "/data/_trace")
CURSOR_FILE = os.environ.get("RECOVERY_CURSOR_FILE", "/data/_recovery/cursor.json")
INFRA_DIR = os.environ.get("RECOVERY_INFRA_DIR", "/opt/recovery")


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _triggers():
    return _load("recovery_install_triggers", os.path.join(INFRA_DIR, "install_triggers.py"))


def _replayer():
    return _load("recovery_changelog_replay", os.path.join(INFRA_DIR, "changelog_replay.py"))


def live_databases(db_dir: str = DB_DIR) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(db_dir, "*.sqlite"))):
        if os.path.islink(path) or os.path.getsize(path) == 0:
            continue
        out[os.path.splitext(os.path.basename(path))[0]] = path
    return out


def write_cursor(
    action_index: int, db_dir: str = DB_DIR, cursor_file: str = CURSOR_FILE, triggers=None
) -> Dict[str, Any]:
    triggers = triggers or _triggers()
    updated = []
    for app, path in live_databases(db_dir).items():
        conn = sqlite3.connect(path, timeout=5)
        try:
            tables = {
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if triggers.CURSOR_TABLE not in tables:
                continue
            triggers.set_cursor(conn, action_index)
            updated.append(app)
        finally:
            conn.close()
    os.makedirs(os.path.dirname(cursor_file), exist_ok=True)
    tmp = cursor_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump({"action_index": int(action_index), "ts": time.time()}, handle)
    os.replace(tmp, cursor_file)
    return {"action_index": int(action_index), "databases": updated}


def read_changelog(
    since: Mapping[str, int], db_dir: str = DB_DIR, triggers=None
) -> List[Dict[str, Any]]:
    triggers = triggers or _triggers()
    rows: List[Dict[str, Any]] = []
    for app, path in live_databases(db_dir).items():
        conn = sqlite3.connect("file:%s?mode=ro" % path, uri=True, timeout=5)
        try:
            tables = {
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if triggers.CHANGELOG_TABLE not in tables:
                continue
            rows.extend(triggers.read_changelog(conn, int(since.get(app, 0)), db=app))
        finally:
            conn.close()
    rows.sort(key=lambda r: (float(r["ts"]), r["db"], int(r["seq"])))
    return rows


def latest_seq(db_dir: str = DB_DIR, triggers=None) -> Dict[str, int]:
    triggers = triggers or _triggers()
    out: Dict[str, int] = {}
    for app, path in live_databases(db_dir).items():
        conn = sqlite3.connect("file:%s?mode=ro" % path, uri=True, timeout=5)
        try:
            tables = {
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if triggers.CHANGELOG_TABLE not in tables:
                continue
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM %s" % triggers.CHANGELOG_TABLE
            ).fetchone()
            out[app] = int(row[0]) if row else 0
        finally:
            conn.close()
    return out


def read_observations(action_index: int, trace_dir: str = TRACE_DIR) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(trace_dir, "*.jsonl"))):
        source = "cli" if os.path.basename(path) == "cli.jsonl" else "api"
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if int(record.get("action_index", -1)) != int(action_index):
                    continue
                record.setdefault("source", source)
                record.setdefault("app", os.path.splitext(os.path.basename(path))[0])
                out.append(record)
    return out


def digests(db_dir: str = DB_DIR, replayer=None, infra_dir: str = INFRA_DIR) -> Dict[str, str]:
    replayer = replayer or _replayer()
    rules_path = os.path.join(infra_dir, "volatile_columns.json")
    rules = replayer.load_volatile_rules(rules_path if os.path.isfile(rules_path) else None)
    return {
        app: replayer.digest(path, rules=rules, app=app)
        for app, path in live_databases(db_dir).items()
    }


def install_all(db_dir: str = DB_DIR, triggers=None) -> Dict[str, Any]:
    triggers = triggers or _triggers()
    report = {}
    for app, path in live_databases(db_dir).items():
        conn = sqlite3.connect(path, timeout=30)
        try:
            report[app] = triggers.install(conn)
        finally:
            conn.close()
    return report


def status(
    db_dir: str = DB_DIR, trace_dir: str = TRACE_DIR, cursor_file: str = CURSOR_FILE, triggers=None
) -> Dict[str, Any]:
    triggers = triggers or _triggers()
    per_db = {}
    for app, path in live_databases(db_dir).items():
        conn = sqlite3.connect("file:%s?mode=ro" % path, uri=True, timeout=5)
        try:
            per_db[app] = {"triggers": len(triggers.installed_triggers(conn))}
        finally:
            conn.close()
    cursor: Optional[Dict[str, Any]] = None
    if os.path.isfile(cursor_file):
        with open(cursor_file, "r", encoding="utf-8") as handle:
            cursor = json.load(handle)
    return {
        "databases": per_db,
        "cursor": cursor,
        "trace_files": sorted(
            os.path.basename(p) for p in glob.glob(os.path.join(trace_dir, "*.jsonl"))
        ),
    }


def register(app) -> None:
    from flask import jsonify, request

    @app.route("/recovery/cursor", methods=["POST"])
    def recovery_cursor():
        data = request.get_json(force=True, silent=True) or {}
        if "action_index" not in data:
            return jsonify({"error": "action_index required"}), 400
        return jsonify(write_cursor(int(data["action_index"])))

    @app.route("/recovery/changelog", methods=["GET"])
    def recovery_changelog():
        since: Dict[str, int] = {}
        if request.args.get("since"):
            since = {k: int(v) for k, v in json.loads(request.args["since"]).items()}
        elif request.args.get("since_all"):
            since = {app_name: int(request.args["since_all"]) for app_name in live_databases()}
        return jsonify({"rows": read_changelog(since)})

    @app.route("/recovery/seq", methods=["GET"])
    def recovery_seq():
        return jsonify({"seq": latest_seq()})

    @app.route("/recovery/observations", methods=["GET"])
    def recovery_observations():
        if "action_index" not in request.args:
            return jsonify({"error": "action_index required"}), 400
        return jsonify({"records": read_observations(int(request.args["action_index"]))})

    @app.route("/recovery/digest", methods=["GET"])
    def recovery_digest():
        return jsonify(digests())

    @app.route("/recovery/install", methods=["POST"])
    def recovery_install():
        return jsonify(install_all())

    @app.route("/recovery/status", methods=["GET"])
    def recovery_status():
        return jsonify(status())
