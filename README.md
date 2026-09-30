# CUA-Recovery

Code for **CUA-Recovery**, a benchmark for error detection and recovery in long-horizon computer use, and **ReRail**, the pipeline that builds it.

ReRail composes MyPCBench tasks into long multi-application workflows with executable ground truth, collects verifier-confirmed failures from several agents, and turns them into two suites:

- **Test suite**: human-verified erroneous states. An agent takes over each state at depth *d* ∈ {0, 5, 10, 15, 20, 25} steps after the root cause, without being told that an error occurred.
- **Training suite**: verified recovery trajectories `(l, H_c, r*)` for supervised fine-tuning.

This repository contains code only. Rollouts, annotations, and judge outputs are not included.

## Setup

```bash
pip install -e ".[collection,dev]"
make setup        # clone MyPCBench into third_party/ at the pinned commit and apply patches/
make setup-all    # also fetch EvoCUA and OpenCUA-OSWorld
```

`third_party/` is not version-controlled. `scripts/setup_third_party.sh` clones each upstream at a pinned commit and applies the patches in `patches/`. The MyPCBench VM image is about 16.5 GB.

Credentials are read from environment variables and are never stored in the repository. Deployment-specific locations must also be set as environment variables before running the corresponding scripts:

| Variable | Used by |
|---|---|
| `ROCK_BASE_URL`, `ROCK_CLUSTER`, `ROCK_SANDBOX_IMAGE` | ROCK sandbox runs under `scripts/rock/` |
| `OSS_ENDPOINT`, `OSS_BUCKET`, `OSS_PREFIX` | Object-storage staging |
| `JUDGE_OSS_ROOT` | Shipping judge archives (`--ship`) |

## Pipeline

Each stage reads its parameters from `configs/`. The defaults follow the paper.

| Stage | Entry point | Code | Config |
|---|---|---|---|
| Task IR extraction and gold lineage | `scripts/extract_task_ir_v1.py`, `scripts/gold_interpret.py` | `src/derail/ir`, `src/derail/world` | `configs/synthesis/task_ir_v1_extractor.yaml` |
| Workflow composition, frozen verifiers, mutation test, composed rubrics | `scripts/generate_tasks_v1.py` | `src/derail/gen` | `configs/synthesis/sampling_v1.yaml` |
| Workflow-level train/test split | `scripts/freeze_source_splits.py` | `src/derail/gen/splits.py` | `configs/synthesis/workflow_splits_v1.yaml` |
| Rollout collection (clean start; per-step state probes) | `scripts/01_collect_all.sh`, `scripts/run_rollout_v1.py` | `src/derail/rollout` | `configs/collection/`, `configs/environments/` |
| Rubric judging | `scripts/02_judge_rubrics.sh`, `scripts/run_judge.sh` | `scripts/30_full_traj_judge.py` | `configs/judges/default.yaml` |
| Root-cause analysis against gold lineage | `scripts/analyze_failures_v1.py` | `src/derail/failure_analysis` | `configs/synthesis/typing_rules_v1.yaml` |
| Human annotation, double annotation, adjudication | `scripts/serve_annotation_ui.py` | `src/derail/annotation`, `analysis/derail_annotation_stats` | `analysis/derail_annotation_stats/config/field_mapping.yaml` |
| Prefix repair (one root cause per prefix) | `scripts/prepare_clean_prefix.py` | `src/derail/construction` | `configs/benchmark/derail_v1.yaml` |
| Takeover-state selection | `scripts/09_select_takeover_failures.py`, `scripts/export_error_depth_slices.py` | `src/derail/construction/cases.py` | `configs/benchmark/derail_v1.yaml` |
| Takeover with replay-state verification | `scripts/rock/run_takeover.sh` | `src/derail/takeover`, `src/derail/mypcbench`, `src/derail/replay` | `configs/takeover/takeover.yaml` |
| Error Awareness Rate, Rubric Score, Pass@3 | `artifacts/takeover/run_takeover_judge.sh`, `scripts/13_takeover_error_awareness.py`, `scripts/11_summarize_takeover_judges.py` | `src/derail/evaluation/metrics.py` | `configs/judges/default.yaml` |
| Recovery trajectory generation (hint schedule, teacher fallback) | `scripts/recovery_gen_v1.py` | `src/derail/train/recovery_gen.py` | `configs/train/sft_v1.yaml` |
| SFT samples and success-only control | `scripts/build_training_v1.py` | `src/derail/train/build_samples.py` | `configs/train/sft_v1.yaml` |
| Error-type and horizon statistics | `analysis/derail_error_taxonomy/analyze.py` | | `configs/synthesis/failure_taxonomy_v0.1.yaml` |

## Repository layout

```text
src/derail/   Python package
scripts/      entry points; scripts/rock/ runs on ROCK sandboxes
configs/      agents, environments, collection, synthesis, takeover, judges, training
prompts/      verbatim agent and judge prompts
schemas/      JSON Schemas for released data formats
analysis/     annotation and error-taxonomy statistics
infra/        in-VM state digest, change-log triggers, control-API patch
patches/      patches applied to MyPCBench by scripts/setup_third_party.sh
tests/        unit tests
```

## Tests

```bash
PYTHONPATH=src python3 -m pytest -q tests
```

Tests that need rollout data or local VM database copies are skipped when that data is absent.

## License

MIT
