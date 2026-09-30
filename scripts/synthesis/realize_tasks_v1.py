#!/usr/bin/env python3
"""Realize instructions for the selected tasks of a generation bundle."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(os.environ.get("RECOVERY_REPO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO_ROOT / "src"))

from recovery.gen.graft import lineage  # noqa: E402
from recovery.gen.pipeline import Candidate, realization_step  # noqa: E402
from recovery.gen.realize import RealizationConfig, style_stats_from_instructions  # noqa: E402
from recovery.ir.extract import V1ExtractorConfig  # noqa: E402

PURPOSE = "realization"


def rescore(bundle, records, config, persona, instruction_of, done_path) -> int:
    """Offline re-verdict of saved replies (leak / literal rules changed)."""

    from recovery.gen.realize import round_trip_compare

    rows, counts = [], {}
    for record in records:
        task_id = record["candidate_id"]
        reply_path = bundle / "realization" / "replies" / ("%s.json" % task_id)
        if not reply_path.is_file():
            continue
        saved = json.loads(reply_path.read_text(encoding="utf-8"))
        task_ir = json.loads(
            (bundle / "task_ir" / ("%s.json" % task_id)).read_text(encoding="utf-8")
        )
        gold_path = bundle / "gold_lineage" / ("%s.gold_lineage.json" % task_id)
        gold = json.loads(gold_path.read_text(encoding="utf-8")) if gold_path.is_file() else None
        public = [instruction_of.get(p, "") for p in lineage(task_ir)[0]]
        for variant, item in enumerate(saved.get("instructions", ())):
            if "instruction" not in item:
                continue
            verdict = round_trip_compare(
                task_ir, None, gold, item["instruction"], config, persona, public
            )
            item["status"] = verdict["verdict"]
            item["round_trip"] = {k: v for k, v in verdict.items() if k != "instruction"}
            rows.append(
                {
                    "task_id": task_id,
                    "variant": variant,
                    "instruction": item["instruction"],
                    "status": item["status"],
                }
            )
            counts[item["status"]] = counts.get(item["status"], 0) + 1
        accepted = [i for i in saved.get("instructions", ()) if i.get("status") == "accepted"]
        saved["status"] = "accepted" if accepted else saved.get("status", "re_realize")
        saved["instruction"] = accepted[0]["instruction"] if accepted else saved.get("instruction")
        reply_path.write_text(
            json.dumps(saved, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        record_path = bundle / "records" / ("%s.json" % task_id)
        if record_path.is_file():
            stored = json.loads(record_path.read_text(encoding="utf-8"))
            stored["realization"] = {k: v for k, v in saved.items() if k not in ("prompt", "reply")}
            record_path.write_text(
                json.dumps(stored, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    done_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    _update_manifest(bundle, rows)
    print("rescored %d instructions: %s" % (len(rows), counts), file=sys.stderr)
    return 0


def _update_manifest(bundle: Path, rows) -> None:
    """Record the accepted instruction count in generation_bundle.json."""

    manifest_path = bundle / "generation_bundle.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    accepted = [r for r in rows if r["status"] == "accepted"]
    manifest["realized"] = len(accepted)
    manifest["realized_tasks"] = len({r["task_id"] for r in accepted})
    manifest_path.write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True, help="seed tasks.json (style stats)")
    parser.add_argument(
        "--all", action="store_true", help="every accepted candidate, not only selected"
    )
    parser.add_argument("--limit", type=int, help="stop after this many tasks (smoke run)")
    parser.add_argument("--call-model", action="store_true")
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="re-judge saved replies offline (no model call) and rewrite instructions.jsonl",
    )
    cfg = REPO_ROOT / "configs/synthesis"
    parser.add_argument("--realization", type=Path, default=cfg / "realization_v1.yaml")
    parser.add_argument("--extractor-config", type=Path, default=cfg / "task_ir_v1_extractor.yaml")
    args = parser.parse_args()
    config = RealizationConfig.from_yaml(args.realization, REPO_ROOT)
    persona = V1ExtractorConfig.from_yaml(args.extractor_config, REPO_ROOT).persona_literals
    client = None
    if args.call_model:
        from recovery.longhorizon.extraction import OpenAICompatibleClient

        client = OpenAICompatibleClient(config.base, PURPOSE)
    seed_tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    instruction_of = {str(t.get("id")): str(t.get("instruction") or "") for t in seed_tasks}
    style = style_stats_from_instructions([i for i in instruction_of.values() if i])
    records = [
        json.loads(line)
        for line in (args.bundle / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    todo = [r for r in records if r["status"] == "accepted" and (args.all or r["selected"])]
    out_dir = args.bundle / "realization"
    (out_dir / "prompts").mkdir(parents=True, exist_ok=True)
    (out_dir / "replies").mkdir(parents=True, exist_ok=True)
    done_path = out_dir / "instructions.jsonl"
    if args.rescore:
        return rescore(args.bundle, todo, config, persona, instruction_of, done_path)
    done = set()
    if done_path.is_file():
        done = {
            json.loads(line)["task_id"]
            for line in done_path.read_text(encoding="utf-8").splitlines()
        }
    todo = [r for r in todo if r["candidate_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(
        "realizing %d tasks (%d already done), model calls %s\nwriting %s"
        % (len(todo), len(done), "on" if client else "OFF (dry run)", out_dir),
        file=sys.stderr,
    )
    started = time.time()
    for index, record in enumerate(todo, 1):
        tick = time.time()
        task_id = record["candidate_id"]
        task_ir = json.loads(
            (args.bundle / "task_ir" / ("%s.json" % task_id)).read_text(encoding="utf-8")
        )
        gold_path = args.bundle / "gold_lineage" / ("%s.gold_lineage.json" % task_id)
        gold = json.loads(gold_path.read_text(encoding="utf-8")) if gold_path.is_file() else None
        candidate = Candidate(task_ir, {}, 0.0, [], 0, record["origin"], gold=gold)
        parents = lineage(task_ir)[0]
        result = realization_step(
            candidate,
            config,
            style,
            persona,
            client,
            public_texts=[instruction_of.get(p, "") for p in parents],
        )
        (out_dir / "prompts" / ("%s.txt" % task_id)).write_text(result["prompt"], encoding="utf-8")
        if client is not None:
            (out_dir / "replies" / ("%s.json" % task_id)).write_text(
                json.dumps(
                    {k: v for k, v in result.items() if k != "prompt"}, indent=1, ensure_ascii=False
                )
                + "\n",
                encoding="utf-8",
            )
            with done_path.open("a", encoding="utf-8") as handle:
                for variant, item in enumerate(result.get("instructions", ())):
                    if "instruction" in item:
                        handle.write(
                            json.dumps(
                                {
                                    "task_id": task_id,
                                    "variant": variant,
                                    "instruction": item["instruction"],
                                    "status": item["status"],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
            record_path = args.bundle / "records" / ("%s.json" % task_id)
            if record_path.is_file():
                stored = json.loads(record_path.read_text(encoding="utf-8"))
                stored["realization"] = {
                    k: v for k, v in result.items() if k not in ("prompt", "reply")
                }
                stored["provenance"]["llm_calls"] = list(result.get("llm_calls", ()))
                record_path.write_text(
                    json.dumps(stored, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
                )
        elapsed = time.time() - started
        print(
            "[%d/%d] %s %s %.1fs | elapsed %dm eta %dm"
            % (
                index,
                len(todo),
                task_id,
                result["status"],
                time.time() - tick,
                elapsed // 60,
                (elapsed / index * (len(todo) - index)) // 60,
            ),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
