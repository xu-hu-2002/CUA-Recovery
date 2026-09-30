"""Annotation-bound takeover orchestration tests (no live VM or model)."""

import hashlib
import json
import os
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from derail.canonical.actions import ClickAction, SequenceAction, ShellAction
from derail.canonical.trajectory import CanonicalStep
from derail.mypcbench.takeover_agent import PrefixTakeoverAgent
from derail.replay.mypcbench import MyPCBenchVMReplayBackend
from derail.replay.verification import ReplayVerificationError
from derail.takeover.diagnosis import (
    HumanDiagnosisEvidence,
    load_human_diagnosis_evidence,
)
from derail.takeover.selection import select_takeover_failures
from derail.takeover.protocol import load_takeover_config
from derail.takeover.source_logs import load_public_trajectory_log


def _load_stage_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts/takeover/stage_inputs.py"
    spec = importlib.util.spec_from_file_location("stage_takeover_inputs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ProductionBundleDefaultsTests(unittest.TestCase):
    def test_portable_bundle_requires_claude_native_messages_sidecar(self):
        stage = _load_stage_module()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source_dir = repo / "raw/claude/task"
            source_dir.mkdir(parents=True)
            trajectory_source = source_dir / "traj.jsonl"
            trajectory_source.write_text("{}\n", encoding="utf-8")
            messages = source_dir / "messages.json"
            messages.write_text("[]", encoding="utf-8")
            canonical = repo / "build/canonical/task/trajectory.jsonl"
            canonical.parent.mkdir(parents=True)
            canonical.write_text(
                json.dumps({
                    "source_agent": "claude_opus_4_8",
                    "source_record_uri": f"{trajectory_source}#line=1",
                }) + "\n",
                encoding="utf-8",
            )
            supporting = []
            for name in ("normalization_report.json", "task_config.json", "annotation.json", "rubric.json"):
                path = repo / name
                path.write_text("{}", encoding="utf-8")
                supporting.append(path)
            selection = repo / "selection.json"
            selection.write_text(json.dumps({"included": [{
                "canonical_trajectory_uri": str(canonical),
                "normalization_report_uri": str(supporting[0]),
                "task_config_uri": str(supporting[1]),
                "annotation_uri": str(supporting[2]),
                "rubric_review_uri": str(supporting[3]),
            }]}), encoding="utf-8")
            config = {"experiments": [{
                "selection_manifest": "selection.json",
                "build_dir": "build",
            }]}
            repaired = repo / "build/canonical_repaired/task/trajectory.jsonl"
            repaired.parent.mkdir(parents=True)
            repaired.write_text(canonical.read_text(encoding="utf-8"), encoding="utf-8")
            (source_dir / "state_probes.jsonl").write_text("{}\n", encoding="utf-8")

            required, missing = stage.selected_files(config, repo, load_takeover_config())

            self.assertIn(messages.resolve(), required)
            self.assertIn(repaired.resolve(), required)
            self.assertIn((source_dir / "state_probes.jsonl").resolve(), required)
            self.assertFalse(missing)

    def test_proxy_port_allocator_is_stable_and_collection_scoped(self):
        repo = Path(__file__).resolve().parents[1]
        allocator = repo / "scripts/rock/allocate_proxy_ports.sh"

        def allocate(collection_id: str) -> str:
            command = (
                "unset PROXY_PORT_BASE PROXY_LIFECYCLE_PORT; "
                f"COLLECTION_ID={collection_id} source {allocator}; "
                'printf "%s:%s" "$PROXY_PORT_BASE" "$PROXY_LIFECYCLE_PORT"'
            )
            return subprocess.run(
                ["bash", "-c", command], check=True, capture_output=True, text=True
            ).stdout.splitlines()[-1]

        first = allocate("collection-a")
        self.assertEqual(first, allocate("collection-a"))
        self.assertNotEqual(first, allocate("collection-b"))
        base, lifecycle = (int(value) for value in first.split(":"))
        self.assertEqual(lifecycle, base + 28)

    def test_rock_entrypoints_default_to_the_frozen_v2_bundle(self):
        repo = Path(__file__).resolve().parents[1]
        for relative_path in (
            "scripts/rock/derail_rock_driver.py",
            "scripts/rock/submit_derail_opencua_smoke.sh",
            "scripts/rock/submit_derail_rock_nebula.sh",
        ):
            source = (repo / relative_path).read_text(encoding="utf-8")
            self.assertIn("takeover_inputs/failure_prefix_v1_full_v3", source)

    def test_submit_entrypoints_reject_tracked_dirty_worktrees(self):
        repo = Path(__file__).resolve().parents[1]
        for relative_path in (
            "scripts/rock/submit_derail_opencua_smoke.sh",
            "scripts/rock/submit_derail_rock_nebula.sh",
        ):
            source = (repo / relative_path).read_text(encoding="utf-8")
            self.assertIn('ALLOW_DIRTY_SUBMIT="${ALLOW_DIRTY_SUBMIT:-0}"', source)
            self.assertIn('git -C "$REPO" diff --quiet --ignore-submodules', source)
            self.assertIn('git -C "$REPO" diff --cached --quiet --ignore-submodules', source)

    def test_submit_entrypoints_serialize_fixed_run_config_packaging(self):
        repo = Path(__file__).resolve().parents[1]
        for relative_path in (
            "scripts/rock/submit_derail_opencua_smoke.sh",
            "scripts/rock/submit_derail_rock_nebula.sh",
        ):
            source = (repo / relative_path).read_text(encoding="utf-8")
            self.assertIn('SUBMIT_LOCK_DIR="$REPO/.submit_derail_nebula.lock"', source)
            self.assertIn('mkdir "$SUBMIT_LOCK_DIR"', source)
            self.assertIn("trap _release_submit_lock EXIT", source)
            self.assertIn("fi; _release_submit_lock' EXIT", source)

    def test_rock_driver_keeps_oss_credentials_out_of_action_argv(self):
        repo = Path(__file__).resolve().parents[1]
        source = (repo / "scripts/rock/derail_rock_driver.py").read_text(encoding="utf-8")
        self.assertIn("await sandbox.fs.upload_dir(", source)
        self.assertIn("ossutil -c {shlex.quote(OSS_CONFIG_REMOTE_PATH)}", source)
        self.assertNotIn('ossutil -e \"$OSS_ENDPOINT\" -i', source)
        self.assertNotIn("f\"{secrets_body}DERAILEOF", source)

    def test_opencua_fleet_preserves_running_sibling_sandboxes(self):
        repo = Path(__file__).resolve().parents[1]
        fleet = repo / "scripts/rock/submit_takeover_opencua_fleet.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = root / "calls.txt"
            submit = root / "submit.sh"
            submit.write_text(
                'printf "%s:%s:%s\\n" "$SHARD_OFFSET" "$SHARD_COUNT" '
                '"$SWEEP_STALE_SANDBOXES" >> "$CALLS_FILE"\n',
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.update(
                {
                    "SHARD_COUNT": "3",
                    "MAX_FLEET_WORKERS": "3",
                    "SUBMIT_SCRIPT": str(submit),
                    "CALLS_FILE": str(calls),
                }
            )
            subprocess.run(["bash", str(fleet)], env=env, check=True)
            self.assertEqual(calls.read_text().splitlines(), ["0:3:0", "1:3:0", "2:3:0"])

    def test_kimi_fleet_preserves_running_sibling_sandboxes(self):
        repo = Path(__file__).resolve().parents[1]
        fleet = repo / "scripts/rock/submit_takeover_kimi_fleet.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = root / "calls.txt"
            submit = root / "submit.sh"
            submit.write_text(
                'printf "%s:%s:%s:%s\\n" "$SHARD_OFFSET" "$SHARD_COUNT" '
                '"$SWEEP_STALE_SANDBOXES" "$COLLECTION_ID" >> "$CALLS_FILE"\n',
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.update(
                {
                    "SHARD_COUNT": "3",
                    "MAX_FLEET_WORKERS": "3",
                    "SUBMIT_SCRIPT": str(submit),
                    "CALLS_FILE": str(calls),
                    "COLLECTION_ID_PREFIX": "retry",
                }
            )
            subprocess.run(["bash", str(fleet)], env=env, check=True)
            self.assertEqual(
                calls.read_text().splitlines(),
                ["0:3:0:retry-s0of3", "1:3:0:retry-s1of3", "2:3:0:retry-s2of3"],
            )

    def test_opencua_takeover_uses_the_frozen_result_layout(self):
        repo = Path(__file__).resolve().parents[1]
        source = (repo / "scripts/rock/submit_derail_opencua_smoke.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("results/takeover/failure_prefix_v1/", source)
        self.assertIn("${TAKEOVER_SOURCE_AGENT}/${TAKEOVER_TARGET_AGENT}", source)
        self.assertIn("${TAKEOVER_CONDITION}/d${TAKEOVER_DEPTH}", source)

    def test_takeover_preflight_uses_the_same_external_shard_range_as_workers(self):
        repo = Path(__file__).resolve().parents[1]
        source = (repo / "scripts/rock/run_takeover.sh").read_text()
        self.assertIn('--shard-count "$SHARD_COUNT"', source)
        self.assertIn('--shard-offset "$SHARD_OFFSET"', source)
        self.assertIn('--shard-workers "$NUM_WORKERS"', source)
        self.assertIn("--overflow-policy exclude", source)
        self.assertIn("PROTOCOL_EXCLUSION:", source)
        self.assertIn('reason=%s report=%s', source)
        self.assertIn('"token_overflow"', source)
        self.assertIn('--takeover-config "$TAKEOVER_CONFIG"', source)

    def test_qwen_runtime_budget_contract_is_packaged_for_nebula(self):
        repo = Path(__file__).resolve().parents[1]
        source = (repo / "scripts/rock/submit_derail_opencua_smoke.sh").read_text()
        for variable in (
            "MYPCBENCH_QWEN_MAX_TOKENS",
            "MYPCBENCH_QWEN_HISTORY_N",
            "MYPCBENCH_QWEN_CONTEXT_POLICY",
        ):
            self.assertIn(f"export {variable}=", source)

    def test_open_model_takeover_wrappers_freeze_protocol_and_layout(self):
        repo = Path(__file__).resolve().parents[1]
        qwen35 = (repo / "scripts/rock/submit_takeover_qwen35.sh").read_text()
        self.assertIn("TAKEOVER_TOKENIZE_MODE=vllm", qwen35)
        self.assertIn("TAKEOVER_CONTEXT_CAP=49152", qwen35)
        self.assertIn("Qwen/Qwen3.5-35B-A3B", qwen35)
        self.assertIn("--disable-custom-all-reduce --enforce-eager", qwen35)
        self.assertIn("--enable-auto-tool-choice", qwen35)
        self.assertIn("--tool-call-parser hermes", qwen35)
        self.assertNotIn("--reasoning-parser", qwen35)
        self.assertIn("MYPCBENCH_QWEN_MAX_TOKENS=4096", qwen35)
        self.assertIn("MYPCBENCH_QWEN_CONTEXT_POLICY=tokenize_oldest_first_v1", qwen35)
        self.assertIn("notified is frozen to depth 0/10/20", qwen35)

        evocua = (repo / "scripts/rock/submit_takeover_evocua.sh").read_text()
        self.assertIn('CONDITION="${CONDITION:-notified}"', evocua)
        self.assertIn('case "$DEPTH" in 0|10|20)', evocua)
        self.assertIn("${OPENCUA_WEIGHTS_OSS_DIR:?", evocua)
        self.assertIn("OPENCUA_MODEL=EvoCUA", evocua)
        self.assertIn("TAKEOVER_CONTEXT_CAP=49152", evocua)
        self.assertIn("--max-model-len 49152", evocua)
        self.assertIn("TAKEOVER_TOKENIZE_MODE=vllm", evocua)
        self.assertIn("failure_prefix_v1/evocua_32b/evocua_32b", evocua)

        driver = (repo / "scripts/rock/derail_rock_driver.py").read_text()
        self.assertIn('"EVOCUA_BASE_URLS": OPENCUA_BASE_URLS', driver)
        self.assertIn('"EVOCUA_MODEL": EVOCUA_MODEL', driver)

        entry = (repo / "scripts/rock/entry_derail_opencua_nebula.sh").read_text()
        self.assertIn('export EVOCUA_BASE_URLS="$OPENCUA_BASE_URLS"', entry)
        self.assertIn('export EVOCUA_MODEL="$OPENCUA_MODEL"', entry)
        self.assertIn("OSS_EVOCUA_SNAPSHOT_URI", driver)
        self.assertIn("setup_evocua", driver)
        self.assertIn("evocua-4a0ad5f.tar.gz", driver)

        preflight = (repo / "scripts/takeover/preflight_history.py").read_text()
        self.assertIn('args.target_agent == "evocua_32b"', preflight)
        self.assertIn('"EVOCUA_MODEL"', preflight)
        self.assertIn("EvoCUA target has no native request preflight API", preflight)

    def test_qwen_probe_checks_exact_tokenizer_and_tool_call_contracts(self):
        repo = Path(__file__).resolve().parents[1]
        entry = (repo / "scripts/rock/entry_derail_opencua_nebula.sh").read_text()
        self.assertIn('"name": "tokenize"', entry)
        self.assertIn('base.removesuffix("/v1") + "/tokenize"', entry)
        self.assertIn('"name": "structured_tool_call"', entry)
        self.assertIn('"tool_choice": "required"', entry)
        self.assertIn('if agent in {"qwen3_8_27b", "qwen3_5_35b_a3b"}:', entry)
        self.assertIn("chat completions probe failed: HTTP", entry)
        self.assertIn('text_message.get("content") or ""', entry)
        self.assertIn('text_max_tokens = 4096 if agent == "qwen3_5_35b_a3b" else 256', entry)
        self.assertIn('"finish_reason": body["choices"][0].get("finish_reason")', entry)
        self.assertIn('"usage": body.get("usage")', entry)
        self.assertIn('gui_message.get("content") or ""', entry)

    def test_fleets_do_not_sweep_other_models_by_default(self):
        repo = Path(__file__).resolve().parents[1]
        for name in (
            "submit_takeover_kimi_fleet.sh",
            "submit_takeover_opencua_fleet.sh",
        ):
            source = (repo / "scripts/rock" / name).read_text()
            self.assertIn('shard_sweep="${SWEEP_STALE_SANDBOXES:-0}"', source)

        for name in (
            "submit_takeover_qwen35_fleet.sh",
            "submit_takeover_claude_fleet.sh",
        ):
            self.assertTrue((repo / "scripts/rock" / name).is_file())

    def test_proxy_driver_forwards_frozen_trajectory_selection(self):
        repo = Path(__file__).resolve().parents[1]
        driver = (repo / "scripts/rock/derail_rock_driver.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"TRAJECTORY_ID_FILTER": TAKEOVER_TRAJECTORY_ID_FILTER', driver)
        self.assertIn('"TRAJECTORY_ID_FILE": TAKEOVER_TRAJECTORY_ID_FILE', driver)
        for relative_path in (
            "scripts/rock/submit_derail_rock_nebula.sh",
            "scripts/rock/submit_derail_opencua_smoke.sh",
        ):
            source = (repo / relative_path).read_text(encoding="utf-8")
            self.assertIn("TAKEOVER_TRAJECTORY_ID_FILTER", source)
            self.assertIn("TAKEOVER_TRAJECTORY_ID_FILE", source)

    def test_proxy_driver_requires_auditable_ship_parity(self):
        repo = Path(__file__).resolve().parents[1]
        driver = (repo / "scripts/rock/derail_rock_driver.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('SHIP_LEDGER_NAME = "ship_ledger.json"', driver)
        self.assertIn('"object_parity": not (missing or unexpected or mismatched)', driver)
        self.assertIn('ledger["object_parity"] and ledger["completed_result_count"]', driver)

    def test_claude_takeover_submit_uses_native_protocol_token_counting(self):
        repo = Path(__file__).resolve().parents[1]
        submit = (repo / "scripts/rock/submit_takeover_claude.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("TAKEOVER_TOKENIZE_MODE=anthropic_usage", submit)
        self.assertIn("CLAUDE_OPUS_4_8_MODEL", submit)
        self.assertIn("TAKEOVER_ANNOTATOR=xuhu", submit)


class TakeoverDiagnosisTests(unittest.TestCase):
    def test_diagnosis_is_only_the_human_evidence_block_and_is_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.json"
            raw = {
                "annotation_id": "traj-1__human",
                "trajectory_id": "traj-1",
                "annotator_id": "human",
                "annotator_role": "human",
                "root_cause_action_index": 3,
                "source_trajectory_sha256": "a" * 64,
                "rationale": (
                    "Summary that must not enter the diagnosed prompt.\n\n"
                    "Error type evidence:\n- wrong_target [Action #3]: clicked Alice"
                ),
            }
            path.write_text(json.dumps(raw), encoding="utf-8")
            result = load_human_diagnosis_evidence(
                path,
                expected_trajectory_id="traj-1",
                expected_source_trajectory_sha256="a" * 64,
                maximum_action_index=4,
            )
            self.assertEqual(
                result.evidence, "- wrong_target [Action #3]: clicked Alice"
            )
            self.assertNotIn("Summary", result.evidence)
            self.assertEqual(result.annotation_sha256, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_annotation_binding_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.json"
            path.write_text(
                json.dumps(
                    {
                        "annotation_id": "a",
                        "trajectory_id": "other",
                        "annotator_id": "h",
                        "annotator_role": "human",
                        "root_cause_action_index": 0,
                        "source_trajectory_sha256": "b" * 64,
                        "rationale": "Error type evidence:\n- evidence",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_human_diagnosis_evidence(path, expected_trajectory_id="traj")


class PublicTrajectoryLogTests(unittest.TestCase):
    def test_strips_source_reasoning_and_keeps_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traj.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "step_num": 1,
                        "action": "WAIT",
                        "response": (
                            "private visual reasoning and plan\n</think>\n\n"
                            "Action: wait for the application"
                        ),
                        "reward": 0.0,
                        "done": False,
                        "info": {"status": "running", "reasoning": "private metadata"},
                        "agent_metadata": {
                            "tool_messages": [{"name": "bash", "content": "public output"}],
                            "analysis": "private adapter analysis",
                        },
                        "screenshot_file": "",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            step = CanonicalStep(
                step_id=0,
                action=ClickAction(kind="click", x_px=10, y_px=20),
                observation_before_sha256="a" * 64,
                source_agent="evocua_32b",
                source_step_id=0,
                source_record_uri=f"{path.resolve()}#line=1",
            )
            rendered = load_public_trajectory_log(step)
            self.assertNotIn("private visual reasoning", rendered)
            self.assertNotIn("private metadata", rendered)
            self.assertNotIn("private adapter analysis", rendered)
            self.assertNotIn("</think>", rendered)
            self.assertIn("Action: wait for the application", rendered)
            self.assertIn("public output", rendered)
            self.assertIn('"source_screenshot_available":false', rendered)

    def test_explicit_path_mapping_makes_a_staged_source_record_portable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traj.jsonl"
            path.write_text('{"step_num":1,"response":"Action: wait"}\n', encoding="utf-8")
            missing = "/original/host/collection/traj.jsonl"
            step = CanonicalStep(
                step_id=0,
                action=ClickAction(kind="click", x_px=10, y_px=20),
                observation_before_sha256="a" * 64,
                source_agent="kimi_k3",
                source_step_id=0,
                source_record_uri=f"{missing}#line=1",
            )
            with patch.dict(
                "os.environ", {"DERAIL_PATH_REMAP_JSON": json.dumps({missing: str(path)})}
            ):
                self.assertIn("Action: wait", load_public_trajectory_log(step))

    def test_path_mapping_file_avoids_large_environment_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "traj.jsonl"
            path.write_text('{"step_num":1,"response":"Action: wait"}\n', encoding="utf-8")
            mappings_file = root / "mappings.json"
            missing = "/original/host/collection/traj.jsonl"
            mappings_file.write_text(json.dumps({missing: str(path)}), encoding="utf-8")
            step = CanonicalStep(
                step_id=0,
                action=ClickAction(kind="click", x_px=10, y_px=20),
                observation_before_sha256="a" * 64,
                source_agent="kimi_k3",
                source_step_id=0,
                source_record_uri=f"{missing}#line=1",
            )
            with patch.dict(
                "os.environ",
                {"DERAIL_PATH_REMAP_FILE": str(mappings_file)},
                clear=True,
            ):
                self.assertIn("Action: wait", load_public_trajectory_log(step))


class TakeoverSelectionTests(unittest.TestCase):
    def test_cross_annotator_wrong_rollout_does_not_veto_a_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "build"
            labels = root / "labels"
            (labels / "rubric_scores").mkdir(parents=True)
            (labels / "rollout_flags").mkdir()
            (build / "build_manifest.json").parent.mkdir(parents=True, exist_ok=True)
            (build / "build_manifest.json").write_text(
                json.dumps({"normalization_reports": []}), encoding="utf-8"
            )

            for trajectory_id in (
                "valid-failure",
                "cross-annotator-conflict",
                "wrong-rollout-only",
                "success",
            ):
                canonical = build / "canonical" / trajectory_id
                canonical.mkdir(parents=True)
                step = CanonicalStep(
                    step_id=0,
                    action=ClickAction(kind="click", x_px=10, y_px=20),
                    observation_before_sha256="a" * 64,
                    source_agent="evocua_32b",
                )
                trajectory_path = canonical / "trajectory.jsonl"
                trajectory_path.write_text(json.dumps(step.to_dict()) + "\n")
                source_sha = hashlib.sha256((trajectory_id + "-source").encode()).hexdigest()
                (canonical / "normalization_report.json").write_text(
                    json.dumps(
                        {
                            "trajectory_id": trajectory_id,
                            "source_trajectory_sha256": source_sha,
                            "canonical_trajectory_sha256": hashlib.sha256(
                                trajectory_path.read_bytes()
                            ).hexdigest(),
                        }
                    )
                )
                (canonical / "task_config.json").write_text(
                    json.dumps({"id": trajectory_id, "instruction": "Finish"})
                )
                review_name = f"{trajectory_id}__alice.json"
                (labels / "rubric_scores" / review_name).write_text(
                    json.dumps(
                        {
                            "trajectory_id": trajectory_id,
                            "source_trajectory_sha256": source_sha,
                            "reviewer_id": "alice",
                            "reviewer_role": "human",
                            "task_success": trajectory_id == "success",
                        }
                    )
                )
                if trajectory_id in {"valid-failure", "cross-annotator-conflict"}:
                    (labels / review_name).write_text(
                        json.dumps(
                            {
                                "annotation_id": review_name[:-5],
                                "trajectory_id": trajectory_id,
                                "source_trajectory_sha256": source_sha,
                                "annotator_id": "alice",
                                "annotator_role": "human",
                                "root_cause_action_index": 0,
                                "error_horizon_actions": 0,
                                "identifiable_at_action_index": 0,
                                "rationale": "Error type evidence:\n- [Action #0]: wrong value",
                            }
                        )
                    )
                if trajectory_id in {"cross-annotator-conflict", "wrong-rollout-only"}:
                    (labels / "rollout_flags" / f"{trajectory_id}__bob.json").write_text(
                        json.dumps(
                            {
                                "trajectory_id": trajectory_id,
                                "source_trajectory_sha256": source_sha,
                                "annotator_id": "bob",
                                "annotator_role": "human",
                                "rollout_status": "needs_rerun",
                                "reason": "wrong_rollout",
                            }
                        )
                    )

            rejected_id = "hash-mismatched-source"
            (labels / f"{rejected_id}__alice.json").write_text(
                json.dumps(
                    {
                        "trajectory_id": rejected_id,
                        "annotator_id": "alice",
                        "annotator_role": "human",
                    }
                ),
                encoding="utf-8",
            )
            (build / "build_manifest.json").write_text(
                json.dumps(
                    {
                        "normalization_reports": [
                            {
                                "trajectory_id": rejected_id,
                                "normalization_complete": False,
                                "uri": "",
                                "rejection_reason": "source trajectory sha256 mismatch",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = select_takeover_failures(
                build_dir=build,
                human_labels_dir=labels,
                source_agent="evocua_32b",
                annotator_id="alice",
                depths=(0, 5),
            )
            self.assertEqual(
                [item["trajectory_id"] for item in result["included"]],
                ["cross-annotator-conflict", "valid-failure"],
            )
            self.assertEqual(result["excluded_reason_counts"]["wrong_rollout"], 1)
            self.assertEqual(
                result["excluded_reason_counts"]["no_human_failure_annotation"], 1
            )
            self.assertEqual(
                result["excluded_reason_counts"]["canonicalization_rejected"], 1
            )
            conflict = result["included"][0]
            self.assertEqual(conflict["available_depths"], [0])
            self.assertEqual(conflict["unavailable_depths"], [5])
            self.assertEqual(
                conflict["ignored_cross_annotator_wrong_rollout_flags"][0][
                    "annotator_id"
                ],
                "bob",
            )
            frozen = select_takeover_failures(
                build_dir=build,
                human_labels_dir=labels,
                source_agent="evocua_32b",
                annotator_id="alice",
                trajectory_id_allowlist=("valid-failure",),
                depths=(0, 5),
            )
            self.assertEqual(
                [item["trajectory_id"] for item in frozen["included"]],
                ["valid-failure"],
            )
            self.assertEqual(
                frozen["selection_policy"]["trajectory_id_allowlist"],
                ["valid-failure"],
            )


class TakeoverAgentTests(unittest.TestCase):
    def test_first_predict_replays_through_root_and_seeds_diagnosed_history(self):
        class Target:
            agent_metadata = {}
            last_trajectory_tool_messages = []
            total_usage = {}
            messages = []

            def reset(self, _logger=None):
                pass

            def seed_native_history(self, instruction, history, **kwargs):
                self.seeded = (instruction, history, kwargs)
                self.messages = history

            def predict(self, instruction, observation):
                self.predicted = (instruction, observation)
                return "next", ["WAIT"]

        class Replay:
            def __init__(self, **_kwargs):
                self.executed = []
                self.initial_state_alignment = {"kind": "test_alignment"}

            def adopt_current_snapshot(self, *_args, **_kwargs):
                pass

            def execute(self, action):
                index = len(self.executed)
                self.executed.append(action)
                return {
                    "result": "ok",
                    "observation_before_uri": f"/tmp/before-{index}.png",
                    "observation_before_sha256": str(index) * 64,
                    "observation_after_uri": f"/tmp/after-{index}.png",
                    "observation_after_sha256": str(index + 1) * 64,
                }

            def observe(self):
                return {"screenshot": b"current"}

        steps = tuple(
            CanonicalStep(
                step_id=index,
                action=ClickAction(kind="click", x_px=10 + index, y_px=20),
                observation_before_sha256="a" * 64,
                source_agent="evocua_32b",
            )
            for index in range(3)
        )
        diagnosis = HumanDiagnosisEvidence(
            annotation_id="ann",
            trajectory_id="traj",
            annotator_id="human",
            root_cause_action_index=1,
            evidence="[Action #1]: selected the wrong item",
            annotation_uri="/tmp/ann.json",
            annotation_sha256="a" * 64,
            source_trajectory_sha256="b" * 64,
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "derail.mypcbench.takeover_agent.MyPCBenchVMReplayBackend", Replay
        ):
            target = Target()
            agent = PrefixTakeoverAgent(
                target_agent=target,
                environment=object(),
                task_config={"instruction": "Finish"},
                canonical_steps=steps,
                diagnosis=diagnosis,
                condition="diagnosed",
                source_agent="evocua_32b",
                target_agent_id="qwen3_8_27b",
                qcow2_path=Path("/tmp/base.qcow2"),
                qcow2_sha256="c" * 64,
                artifact_dir=Path(directory),
                depth=1,
            )
            response, actions = agent.predict("Finish", {"screenshot": b"stale"})
            self.assertEqual((response, actions), ("next", ["WAIT"]))
            _instruction, history, kwargs = target.seeded
            self.assertEqual(kwargs["condition"], "diagnosed")
            self.assertEqual(kwargs["root_cause_action_index"], 1)
            self.assertEqual(kwargs["diagnosis"], diagnosis.evidence)
            assistant_turns = [m for m in history if m["role"] == "assistant"]
            self.assertEqual(len(assistant_turns), 3)
            manifest = json.loads((Path(directory) / "takeover_manifest.json").read_text())
            self.assertEqual(manifest["depth"], 1)
            self.assertEqual(manifest["prefix_end_action_index"], 2)
            self.assertEqual(manifest["first_target_action_index"], 3)


class HybridReplayTests(unittest.TestCase):
    def test_seeded_firefox_focus_uses_window_class_with_retries(self):
        class Environment:
            def __init__(self):
                self.commands = []

            def _execute_shell(self, command):
                self.commands.append(command)
                return {"returncode": 0, "output": "", "error": ""}

            def _execute_pyautogui(self, command):
                return {"returncode": 0, "output": "", "error": ""}

        with tempfile.TemporaryDirectory() as directory:
            backend = MyPCBenchVMReplayBackend(
                env_factory=lambda: None,
                task_config={},
                state_probe_commands={"x": "true"},
                evidence_dir=Path(directory),
            )
            env = Environment()
            backend._refresh_seeded_firefox_tabs(env)
            self.assertIn("wmctrl -lx", env.commands[0])
            self.assertIn("/(firefox|navigator)/", env.commands[0])
            self.assertIn("find /run/user", env.commands[0])
            self.assertIn("/home/oai/.Xauthority", env.commands[0])
            self.assertIn("for attempt in $(seq 1 60)", env.commands[0])
            self.assertIn("xdotool windowactivate --sync", env.commands[0])
            self.assertIn("nohup firefox", env.commands[0])
            self.assertEqual(
                backend.initial_state_alignment["focus_strategy"],
                "wmctrl_readiness_retry_v2",
            )

    def test_seeded_firefox_focus_failure_keeps_diagnostics(self):
        class Environment:
            def _execute_shell(self, _command):
                return {"returncode": 1, "output": "", "error": "no X11 windows"}

            def _execute_pyautogui(self, _command):
                raise AssertionError("refresh must not run without a focused window")

        with tempfile.TemporaryDirectory() as directory:
            backend = MyPCBenchVMReplayBackend(
                env_factory=lambda: None,
                task_config={},
                state_probe_commands={"x": "true"},
                evidence_dir=Path(directory),
            )
            with self.assertRaisesRegex(ReplayVerificationError, "no X11 windows"):
                backend._refresh_seeded_firefox_tabs(Environment())

    def test_mixed_sequence_executes_shell_and_gui_in_order(self):
        class Environment:
            def __init__(self):
                self.events = []

            def _execute_shell(self, command):
                self.events.append(("shell", command))
                return {"returncode": 0, "output": "ok", "error": ""}

            def step(self, action, pause):
                self.events.append(("gui", action, pause))
                return {"screenshot": b"\x89PNG\r\n\x1a\npost"}, 0.0, False, {}

            def _get_obs(self):
                return {"screenshot": b"\x89PNG\r\n\x1a\npost"}

        with tempfile.TemporaryDirectory() as directory:
            backend = MyPCBenchVMReplayBackend(
                env_factory=lambda: None,
                task_config={},
                state_probe_commands={"x": "true"},
                evidence_dir=Path(directory),
            )
            env = Environment()
            backend._env = env
            backend._last_screenshot_sha256 = "a" * 64
            backend._last_screenshot_uri = "/tmp/before.png"
            record = backend.execute(
                SequenceAction(
                    kind="sequence",
                    actions=(
                        ShellAction(kind="shell", commands=("pwd",)),
                        ClickAction(kind="click", x_px=10, y_px=20),
                    ),
                )
            )
            self.assertEqual(env.events[0], ("shell", "pwd"))
            self.assertEqual(env.events[1][0], "gui")
            self.assertIn("shell_results", json.loads(record["result"]))


if __name__ == "__main__":
    unittest.main()
