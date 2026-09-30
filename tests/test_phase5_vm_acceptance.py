from __future__ import annotations

import base64
import importlib.util
import io
import sqlite3
import tarfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts" / "phase5" / "validate_vm_acceptance.py"
SPEC = importlib.util.spec_from_file_location("validate_vm_acceptance", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

ADMIN_SPEC = importlib.util.spec_from_file_location(
    "guest_admin", REPOSITORY / "scripts" / "phase5" / "guest_admin.py"
)
ADMIN = importlib.util.module_from_spec(ADMIN_SPEC)
assert ADMIN_SPEC.loader is not None
ADMIN_SPEC.loader.exec_module(ADMIN)

GUEST_SPEC = importlib.util.spec_from_file_location(
    "run_vm_acceptance_guest",
    REPOSITORY / "scripts" / "phase5" / "run_vm_acceptance_guest.py",
)
GUEST = importlib.util.module_from_spec(GUEST_SPEC)
assert GUEST_SPEC.loader is not None
GUEST_SPEC.loader.exec_module(GUEST)


def test_reset_failure_is_compact():
    result = GUEST.reset_failure(
        {"copied_dbs": ["a"], "errors": ["boom"], "status": "partial", "logs": "x" * 1000}
    )
    assert result == {
        "copied_db_count": 1,
        "copied_dbs": ["a"],
        "errors": ["boom"],
        "status": "partial",
    }


def test_guest_database_snapshot_includes_committed_wal_pages(tmp_path):
    source = tmp_path / "source.sqlite"
    snapshot = tmp_path / "snapshot.sqlite"
    connection = sqlite3.connect(source)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA wal_autocheckpoint=0")
    connection.execute("CREATE TABLE records (value TEXT)")
    connection.commit()
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.execute("INSERT INTO records VALUES ('from-wal')")
    connection.commit()
    GUEST.copy_database(source, snapshot)
    with sqlite3.connect(snapshot) as copied:
        assert copied.execute("SELECT value FROM records").fetchall() == [("from-wal",)]
    connection.close()


def test_acceptance_warms_runtime_schema_before_baseline_snapshot():
    source = (REPOSITORY / "scripts" / "phase5" / "run_vm_acceptance_guest.py").read_text()
    warm = source.index(
        'set_apps("start")\n    probe_all_apps()\n    probe_pages()\n    set_apps("stop")'
    )
    baseline = source.index('copy_database(f"/data/{db}.sqlite"')
    start_seq = source.index('start_seq = request("/derail/seq")')
    assert warm < baseline < start_seq


def test_acceptance_quiesces_changelog_before_terminal_snapshot():
    source = (REPOSITORY / "scripts" / "phase5" / "run_vm_acceptance_guest.py").read_text()
    stopped = source.index('set_apps("stop")\n    end_seq = wait_for_changelog_quiescence()')
    changelog = source.index('rows = request(f"/derail/changelog?since={since}")')
    direct = source.index('direct = request("/derail/digest")')
    assert stopped < changelog < direct


def valid_report() -> dict:
    trajectories = []
    for index in range(10):
        trajectories.append(
            {
                "trajectory_id": f"acceptance-{index:02d}",
                "reset_cursor": -1,
                "direct_digest": {"calendar": "abc"},
                "replay_digest": {"calendar": "abc"},
                "actions": [
                    {
                        "action_index": 0,
                        "cursor_written_before_dispatch": True,
                        "changed": True,
                        "changelog_attributed": True,
                        "observation_lag_actions": index % 2,
                        "api_observation_count": 1,
                        "screenshot_sha256": "a" * 64,
                        "tracing_overhead_seconds": 0.25,
                    }
                ],
            }
        )
    return {"coverage": ["gui", "bash", "cross_app", "reset"], "trajectories": trajectories}


def test_accepts_complete_report():
    assert MODULE.validate(valid_report()) == []


def test_rejects_digest_cursor_and_overhead_failures():
    report = valid_report()
    report["trajectories"][0]["reset_cursor"] = 0
    report["trajectories"][1]["replay_digest"] = {"calendar": "different"}
    report["trajectories"][2]["actions"][0]["tracing_overhead_seconds"] = 1.01
    errors = MODULE.validate(report)
    assert any("reset cursor" in error for error in errors)
    assert any("digest mismatch" in error for error in errors)
    assert any("overhead" in error for error in errors)


def test_rejects_missing_coverage_and_non_unique_ids():
    report = valid_report()
    report["coverage"].remove("bash")
    report["trajectories"][1]["trajectory_id"] = report["trajectories"][0]["trajectory_id"]
    errors = MODULE.validate(report)
    assert "missing coverage: bash" in errors
    assert "trajectory ids are not unique" in errors


def test_guest_admin_archive_preserves_infra_tree(tmp_path):
    infra = tmp_path / "infra"
    (infra / "triggers").mkdir(parents=True)
    (infra / "triggers" / "install.py").write_text("pass\n")
    encoded = ADMIN.archive_directory(infra)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(encoded)), mode="r:gz") as archive:
        assert "infra/triggers/install.py" in archive.getnames()


def test_guest_boot_accepts_versioned_qcow2_override():
    source = (REPOSITORY / "scripts" / "rock" / "derail_guest_boot.sh").read_text()
    assert 'BASE_QCOW2="${GUEST_QCOW2:-${VM_DIR}/mypcbench.qcow2}"' in source
    assert 'qemu-img create -f qcow2 -b "$BASE_QCOW2"' in source


def test_control_api_patch_suppresses_only_deprecation_warnings():
    source = (REPOSITORY / "infra" / "control_api_patch.py").read_text()
    assert "ignore::DeprecationWarning" in source


def test_control_api_patch_defaults_phase5_reset_to_frozen_baseline():
    path = REPOSITORY / "infra" / "control_api_patch.py"
    spec = importlib.util.spec_from_file_location("control_api_patch_phase5", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    source = 'skip_patchers = bool(data.get("skip_patchers", False))\napp = Flask(__name__)\n'
    patched = module.patched_text(source, "/opt/derail")
    assert 'data.get("skip_patchers", True)' in patched


def test_guest_deploy_enables_control_api_for_fresh_boot():
    source = (REPOSITORY / "scripts" / "phase5" / "guest_admin.py").read_text()
    assert "sudo systemctl enable mypcbench-control-api.service" in source


def test_acceptance_file_fetch_uses_post_json_contract():
    source = (REPOSITORY / "scripts" / "phase5" / "run_phase5_vm_acceptance.py").read_text()
    assert "curl -fsS -X POST -H 'Content-Type: application/json'" in source
    assert 'json.dumps({"file_path": guest_path})' in source
    assert "--data-urlencode" not in source
    assert "/tmp/phase5-acceptance.tar.gz" in source
    assert "/home/oai/share" not in source
    assert "range(1, 91)" in source


def test_acceptance_page_probe_is_read_only_and_keeps_trace_streams_linked():
    source = (REPOSITORY / "scripts" / "phase5" / "run_vm_acceptance_guest.py").read_text()
    assert 'http://127.0.0.1:3002/api/holdings' in source
    assert 'b"mypcbench-session-2026"' in source
    assert 'session_batbucks=' in source
    assert 'deadline = time.monotonic() + 60' in source
    assert 'except urllib.error.URLError' in source
    assert 'request("/reset", {})' in source
    assert '["sudo", "-n", "systemctl", command, *APP_UNITS]' in source
    assert 'set_apps("start")' in source
    assert 'set_apps("stop")' in source
    assert '[observation_start:]' in source
    assert 'trace.unlink()' not in source
    assert 'range(3001, 3019)' not in source


def test_canon_timeout_patch_is_exact_and_idempotent():
    path = REPOSITORY / "infra" / "canon_timeout_patch.py"
    spec = importlib.util.spec_from_file_location("canon_timeout_patch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    source = "import os\n\ndef run():\n" + module.OLD
    patched = module.patched_text(source)
    assert "MYPCBENCH_CANON_PATCH_TIMEOUT" in patched
    assert "'300'" in patched
    assert module.patched_text(patched) == patched


def test_image_build_runs_canonical_patchers_once():
    source = (REPOSITORY / "scripts" / "phase5" / "build_phase5_vm.py").read_text()
    assert "guest_admin.py reset --run-patchers" in source
