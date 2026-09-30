from __future__ import annotations

import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.phase5.probe_guest_generator import (
    PROBE_REFERENCE_TIME,
    PROBE_SEED,
    assert_isolated,
    generation_script,
    probe_paths,
    stage_script,
    variant_patch_command,
    verdict,
)

PATCH = [{"op": "add", "path": "/app_overrides/calendar/events/-",
          "value": {"summary": "THE DUNDIES ARE BACK BABY (prep)", "start": "", "end": ""}}]
REF = "2026-09-01T07:24:24.351485+00:00"


class GeneratorProbeTests(unittest.TestCase):
    def test_probe_paths_stay_inside_the_isolated_root(self):
        paths = probe_paths("variant")
        self.assertTrue(paths)
        for key, value in paths.items():
            with self.subTest(key=key):
                self.assertTrue(value.startswith("/tmp/phase5-probe/variant/"), value)
        with self.assertRaisesRegex(RuntimeError, "escapes the isolated root"):
            assert_isolated("/data/vms/michael_scott/hoolicalendar.sqlite")
        with self.assertRaisesRegex(RuntimeError, "escapes the isolated root"):
            assert_isolated("/tmp/phase5-probe-evil/data")

    def test_generation_script_uses_probe_dirs_and_records_returncode(self):
        script = generation_script(probe_paths("base"))
        self.assertIn("--data-dir /tmp/phase5-probe/base/data", script)
        self.assertIn("--home-dir /tmp/phase5-probe/base/home", script)
        self.assertIn("--strict", script)
        self.assertIn(f"--seed {PROBE_SEED}", script)
        self.assertIn(f"--reference-time {PROBE_REFERENCE_TIME}", script)
        self.assertIn('>/tmp/phase5-probe/base/generate.rc', script)
        absolute = [token for token in shlex.split(script) if token.startswith("/")]
        self.assertTrue(all(token.startswith("/tmp/phase5-probe/") for token in absolute),
                        absolute)

    def test_both_arms_pin_identical_deterministic_inputs(self):
        tokens = {tag: shlex.split(generation_script(probe_paths(tag))) for tag in ("base", "variant")}
        for flag in ("--seed", "--reference-time", "--persona"):
            pinned = {tag: tokens[tag][tokens[tag].index(flag) + 1] for tag in tokens}
            with self.subTest(flag=flag):
                self.assertEqual(pinned["base"], pinned["variant"], pinned)

    def test_stage_script_copies_both_arms_and_patches_only_variant(self):
        script = stage_script(PATCH)
        self.assertEqual(script.count("/opt/personas/michael_scott.json"), 2)
        command = variant_patch_command(
            "/tmp/phase5-probe/variant/personas/michael_scott.json", PATCH)
        self.assertIn(command.rstrip(), script)
        with tempfile.TemporaryDirectory() as directory:
            persona = Path(directory) / "michael_scott.json"
            persona.write_text(json.dumps(
                {"app_overrides": {"calendar": {"events": [{"summary": "existing"}]}}}))
            body = shlex.split(command)[2].replace(
                "/tmp/phase5-probe/variant/personas/michael_scott.json", str(persona))
            subprocess.run([sys.executable, "-c", body], check=True)
            events = json.loads(persona.read_text())["app_overrides"]["calendar"]["events"]
            self.assertEqual([event["summary"] for event in events],
                             ["existing", "THE DUNDIES ARE BACK BABY (prep)"])

    def test_generation_script_rejects_a_golden_world_target(self):
        paths = dict(probe_paths("base"), data="/data")
        with self.assertRaisesRegex(RuntimeError, "escapes the isolated root"):
            generation_script(paths)

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1]
         / "artifacts/phase5/preflight/hazards-option-a-20260912.jsonl").is_file(),
        "Phase 5 hazard catalog is run data, not shipped with the code",
    )
    def test_embedded_guest_scripts_are_valid_python(self):
        from scripts.phase5.probe_guest_generator import (
            INVENTORY_SCRIPT,
            measure_script,
            poll_script,
        )

        scripts = [
            INVENTORY_SCRIPT,
            measure_script(),
            poll_script(probe_paths("base"), "phase5-probe-base"),
        ]
        for script in scripts:
            body = script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
            with self.subTest(first_line=body.splitlines()[0]):
                compile(body, "<guest>", "exec")

    def test_verdict_reads_rates_and_capability_from_measurement(self):
        meta = json.dumps({"bake_reference_time": REF})

        def arm(rate, events=625):
            return {"databases": [{"name": "hoolicalendar.sqlite", "events": events,
                                   "same_title_prefix_events": rate,
                                   "target_title_events": 1}],
                    "error": None, "seed_meta": meta}

        measurement = {"base": arm(18), "variant": arm(19)}
        generations = {"base": {"returncode": 0}, "variant": {"returncode": 0}}
        result = verdict(measurement, generations)
        self.assertTrue(result["generator_builds_empty_data_dir"])
        self.assertTrue(result["arms_controlled"])
        self.assertTrue(result["variant_rate_is_base_plus_one"])
        self.assertEqual(result["base"]["same_title_prefix_events"], 18)
        self.assertEqual(result["variant"]["target_title_events"], 1)

        stalled = {"base": {"databases": [], "error": "data dir absent"},
                   "variant": {"databases": [], "error": "data dir absent"}}
        result = verdict(stalled, {"base": {"returncode": 1}, "variant": {"returncode": 1}})
        self.assertFalse(result["generator_builds_empty_data_dir"])
        self.assertIsNone(result["variant_rate_is_base_plus_one"])
        self.assertEqual(result["base"]["error"], "data dir absent")

    def test_verdict_refuses_a_rate_conclusion_for_uncontrolled_arms(self):
        def arm(rate, when):
            return {"databases": [{"name": "hoolicalendar.sqlite", "events": 625,
                                   "same_title_prefix_events": rate,
                                   "target_title_events": 0}],
                    "error": None, "seed_meta": json.dumps({"bake_reference_time": when})}

        result = verdict({"base": arm(13, REF), "variant": arm(14, REF + "Z")},
                         {"base": {"returncode": 0}, "variant": {"returncode": 0}})
        self.assertFalse(result["arms_controlled"])
        self.assertIsNone(result["variant_rate_is_base_plus_one"])
        self.assertEqual(result["arm_reference_times"]["variant"], REF + "Z")

        corrupt = verdict({"base": arm(13, REF), "variant": arm(14, "not json")},
                          {"base": {"returncode": 0}, "variant": {"returncode": 0}})
        self.assertIsNone(corrupt["variant_rate_is_base_plus_one"])


if __name__ == "__main__":
    unittest.main()
