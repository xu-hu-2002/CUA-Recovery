#!/usr/bin/env python3
"""Local annotation UI for human failure labelling.

Serves ``src/derail/annotation/ui/index.html`` plus a tiny JSON API over one or more compatible
derived builds.
The browser never sees provenance fields: ``annotation_id``, ``trajectory_id``,
``source_trajectory_sha256``, ``taxonomy_version``, ``schema_version`` and ``annotator_role``
are filled in here, and ``error_horizon_actions`` is derived from the two action indices the
annotator clicked.  Every accepted submission is validated against
``schemas/annotation.schema.json`` and re-parsed through ``HumanAnnotation`` before it is
written, so a file on disk is always a legal annotation.

Annotation protocol (paper App. E), all under the out-dir:

* ``auto_analysis/<trajectory_id>.json``  automatic root cause / type / horizon proposal from
  ``scripts/analyze_failures_v1.py --proposals-dir``; shown with the task for verification.
* ``protocol/double_annotation_plan.json``  the stratified double-annotation sample written by
  ``derail_annot_stats.agreement --write-plan``; those trajectories are flagged in the config.
* ``adjudications/<trajectory_id>__<adjudicator>.json``  a third annotator's resolution of two
  independent reviews (``POST /api/adjudication``).

This is a development tool, not a pipeline stage, so it carries no stage number.
"""

from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import math
import mimetypes
import os
import re
import threading
from datetime import datetime, timezone
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from derail.annotation.records import Adjudication, AnnotationError, HumanAnnotation
from derail.construction.cases import eligible_depths
from derail.derived.layout import DEPTH_GRID, atomic_write_json, sha256_file
from derail.derived.schema import validate_schema

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MAX_BODY_BYTES = 1 << 20
OPEN_LABEL_CATEGORIES = ("planning", "perception", "execution", "termination", "others")
OPEN_LABEL_REGISTRY_SCHEMA = "0.3.0"
OPEN_LABEL_DESCRIPTION_MAX_CHARS = 300
ERROR_TYPE_GROUPS = {
    "planning": (
        "fabricate_data",
        "misunderstand_task_objective",
        "lack_of_knowledge",
    ),
    "perception": (
        "progress_misperception",
        "detail_misperception",
        "state_misinterpretation",
        "ineffective_action",
    ),
    "execution": (
        "grounding_failure",
        "incorrect_ui_element",
        "typing_or_parameter_error",
    ),
    "termination": ("fail_to_terminate", "premature_completion"),
}
ERROR_TYPE_DESCRIPTIONS = {
    "detail_misperception": "漏看或误读了影响结果的局部信息、数值、文本或具体要求",
    "fabricate_data": "编造了未从界面、文件、工具结果或其他可靠证据中获得的数据",
    "fail_to_terminate": "任务已明确成功或无法继续时仍未结束，并继续执行无效或重复操作",
    "grounding_failure": "知道要操作什么，但未能准确定位目标，导致点击、拖拽或操作落点失败",
    "incorrect_ui_element": "选择了与目标功能不同的按钮、字段、菜单或其他界面元素",
    "ineffective_action": "操作已执行但没有推进任务，也没有产生预期的界面或系统状态变化",
    "lack_of_knowledge": "缺少完成任务所必需的领域知识、规则知识或操作方法",
    "misunderstand_task_objective": "错误理解了任务的最终目标、交付物、对象范围或关键约束",
    "premature_completion": "尚未满足全部任务要求，就报告成功、提交结果或结束任务",
    "progress_misperception": "错误判断已完成的工作、剩余步骤或当前任务进度",
    "state_misinterpretation": (
        "错误判断当前页面、应用、数据、登录/加载状态或工具能力，"
        "例如把已生效当作未生效、把可用工具当作不可用，并据此采取错误行动"
    ),
    "typing_or_parameter_error": "输入的文本、数值、日期、路径、坐标或工具参数不正确",
}
EDITABLE_SEED_LABELS = {
    label for labels in ERROR_TYPE_GROUPS.values() for label in labels
}
RETIRED_SEED_LABELS = {"wrong_subgoal"}
OPEN_LABEL_DEFAULT_DESCRIPTIONS = {
    "scope_error": "处理的数据范围、时间范围或对象范围与任务要求不一致。",
    "section_content_misplacement": "把内容写入了错误的文档章节、表格区域或其他不合适的位置。",
}
DERIVED_BUDGET_LABEL = "hit_budget_limit"
# Out-dir layout of the annotation-protocol records (see the module docstring).
AUTO_ANALYSIS_DIR = "auto_analysis"
DOUBLE_ANNOTATION_PLAN = ("protocol", "double_annotation_plan.json")
ADJUDICATION_DIR = "adjudications"
ADJUDICATION_SCHEMA_VERSION = "adjudication-record/0.1"


class UIError(RuntimeError):
    """A request cannot be served; the message is safe to show the annotator."""


class ConflictError(UIError):
    """The write would overwrite an existing immutable annotation."""


def _safe_id(value: str, field: str) -> str:
    value = str(value).strip()
    if not SAFE_ID.match(value):
        raise UIError("%s may contain only letters, digits, dots, underscores, and hyphens: %r" % (field, value))
    return value


def _build_export_stem(build_id: str) -> str:
    stem = re.sub(r"_human_label_traj$", "", build_id)
    return re.sub(r"_vm\d+$", "", stem)


def _normalize_open_label(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    value = re.sub(r"_+", "_", value)
    if not value:
        raise UIError("label must contain at least one letter or digit")
    if len(value) > 96:
        raise UIError("label must be at most 96 characters after normalization")
    return value


def _normalize_open_label_description(value: Any, *, required: bool) -> str:
    description = " ".join(str(value or "").split()).strip()
    if not description:
        if required:
            raise UIError("新标签必须填写中文解释")
        return ""
    if len(description) > OPEN_LABEL_DESCRIPTION_MAX_CHARS:
        raise UIError(
            "中文解释不能超过 %d 个字符" % OPEN_LABEL_DESCRIPTION_MAX_CHARS
        )
    if not re.search(r"[\u3400-\u9fff]", description):
        raise UIError("新标签的解释必须包含中文")
    return description


def _taxonomy_groups(taxonomy: Dict[str, Any]) -> Dict[str, List[str]]:
    """Recover the YAML grouping for display, but only if the snapshot still matches.

    ``03_prepare_annotations.py`` flattens the taxonomy into ``seed_labels`` and pins its
    sha256 into the build manifest.  Showing the grouped view is nicer for annotators, yet it
    must never contradict the pinned snapshot, so a changed file falls back to the flat list.
    """

    seed = list(taxonomy.get("seed_labels", ()))
    uri = taxonomy.get("uri", "")
    pinned = taxonomy.get("sha256", "")
    path = Path(uri) if uri else None
    if not (path and path.is_file() and pinned and sha256_file(path) == pinned):
        return {"seed_labels": seed}
    groups: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        stripped = raw.strip()
        if stripped.startswith("- "):
            if current is not None:
                groups[current].append(stripped[2:].strip())
        elif raw.startswith("    ") or raw.startswith("  ") and stripped.endswith(":"):
            current = stripped.rstrip(":")
            groups[current] = []
    groups = {name: labels for name, labels in groups.items() if labels}
    flat = sorted(label for labels in groups.values() for label in labels)
    return groups if flat == sorted(seed) else {"seed_labels": seed}


class AnnotationService:
    """Read-only view of one derived build plus a write path for annotations."""

    def __init__(self, build_dir: Path, out_dir: Path, repository: Path) -> None:
        self.build_dir = build_dir
        self.out_dir = out_dir
        self.repository = repository
        self.ui_dir = repository / "src" / "derail" / "annotation" / "ui"
        if not (self.ui_dir / "index.html").is_file():
            raise UIError("Annotation UI not found: %s" % (self.ui_dir / "index.html"))
        manifest_path = build_dir / "build_manifest.json"
        if not manifest_path.is_file():
            raise UIError("Build manifest not found: %s" % manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.taxonomy = dict(self.manifest["taxonomy"])
        collection_root = Path(self.manifest["source_collection"]["uri"]).resolve()
        self.collection_root = collection_root
        # Screenshots live in the immutable raw collection; canonical artifacts live in the
        # build.  Nothing outside those two trees may ever be read through the image route.
        self.image_roots: Tuple[Path, ...] = (collection_root, build_dir.resolve())
        self.allowed_images = self._referenced_images()
        self._submit_lock = threading.Lock()
        self._label_lock = threading.Lock()

    def _referenced_images(self) -> set[Path]:
        """Return the exact image allowlist referenced by this build's annotation tasks."""

        allowed: set[Path] = set()
        for task_path in self._task_paths():
            task = json.loads(task_path.read_text(encoding="utf-8"))
            for action in task.get("actions", ()):
                for side in ("before", "after"):
                    raw = str(action.get("observation_%s_uri" % side, ""))
                    if not raw:
                        continue
                    resolved = Path(unquote(raw.partition("#")[0])).resolve()
                    if any(
                        resolved == root or root in resolved.parents
                        for root in self.image_roots
                    ):
                        allowed.add(resolved)
        return allowed

    # ---- reads ----------------------------------------------------------------

    def _task_paths(self) -> List[Path]:
        canonical = self.build_dir / "canonical"
        if not canonical.is_dir():
            return []
        return sorted(canonical.glob("*/annotation_task.json"))

    def _submitted(self) -> Dict[str, List[str]]:
        by_trajectory: Dict[str, List[str]] = {}
        paths = (
            list(self.out_dir.glob("*.json"))
            + list((self.out_dir / "rubric_scores").glob("*.json"))
            + list((self.out_dir / "rollout_flags").glob("*.json"))
        )
        for path in sorted(paths):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            trajectory = str(record.get("trajectory_id", ""))
            annotator = str(record.get("annotator_id", record.get("reviewer_id", "")))
            if trajectory and annotator:
                values = by_trajectory.setdefault(trajectory, [])
                if annotator not in values:
                    values.append(annotator)
        return by_trajectory

    def _rerun_requested(self) -> Dict[str, List[str]]:
        by_trajectory: Dict[str, List[str]] = {}
        for path in sorted((self.out_dir / "rollout_flags").glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if record.get("rollout_status") != "needs_rerun":
                continue
            trajectory = str(record.get("trajectory_id", ""))
            annotator = str(record.get("annotator_id", ""))
            if trajectory and annotator:
                by_trajectory.setdefault(trajectory, []).append(annotator)
        return by_trajectory

    def _open_label_registry_path(self) -> Path:
        return self.out_dir / "taxonomy" / "open_coded_labels.json"

    def _open_label_registry(
        self,
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
        path = self._open_label_registry_path()
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}, {}
        except (OSError, json.JSONDecodeError) as exc:
            raise UIError("Open-coded label registry cannot be read: %s" % exc)
        raw_labels = record.get("labels", {}) if isinstance(record, dict) else {}
        if not isinstance(raw_labels, dict):
            raise UIError("Open-coded label registry has an invalid labels object")
        labels: Dict[str, Dict[str, str]] = {}
        for raw_label, raw_metadata in raw_labels.items():
            try:
                label = _normalize_open_label(raw_label)
            except UIError:
                continue
            metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
            category = str(metadata.get("category", "others"))
            if category not in OPEN_LABEL_CATEGORIES:
                category = "others"
            labels[label] = {
                "category": category,
                "created_by": str(metadata.get("created_by", "")),
                "created_at": str(metadata.get("created_at", "")),
                "updated_by": str(metadata.get("updated_by", "")),
                "updated_at": str(metadata.get("updated_at", "")),
                "description_zh": _normalize_open_label_description(
                    metadata.get(
                        "description_zh",
                        OPEN_LABEL_DEFAULT_DESCRIPTIONS.get(label, ""),
                    ),
                    required=False,
                ),
            }
        raw_deleted = record.get("deleted_labels", {}) if isinstance(record, dict) else {}
        if not isinstance(raw_deleted, dict):
            raise UIError("Open-coded label registry has an invalid deleted_labels object")
        deleted: Dict[str, Dict[str, str]] = {}
        for raw_label, raw_metadata in raw_deleted.items():
            try:
                label = _normalize_open_label(raw_label)
            except UIError:
                continue
            metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
            deleted[label] = {
                "deleted_by": str(metadata.get("deleted_by", "")),
                "deleted_at": str(metadata.get("deleted_at", "")),
                "last_category": str(metadata.get("last_category", "")),
                "description_zh": str(metadata.get("description_zh", "")),
            }
        return labels, deleted

    def _registered_open_labels(self) -> Dict[str, Dict[str, str]]:
        return self._open_label_registry()[0]

    def _write_open_label_registry(
        self,
        labels: Dict[str, Dict[str, str]],
        deleted: Dict[str, Dict[str, str]],
    ) -> None:
        atomic_write_json(
            self._open_label_registry_path(),
            {
                "schema_version": OPEN_LABEL_REGISTRY_SCHEMA,
                "labels": labels,
                "deleted_labels": deleted,
            },
        )

    def open_coded_label_groups(self) -> Dict[str, List[str]]:
        """Return shared open-coded labels grouped for every annotator and trajectory."""

        seed = (
            set(self.taxonomy.get("seed_labels", ()))
            | EDITABLE_SEED_LABELS
            | RETIRED_SEED_LABELS
        )
        groups: Dict[str, set[str]] = {
            category: set() for category in OPEN_LABEL_CATEGORIES
        }
        registered, deleted = self._open_label_registry()
        for label, metadata in registered.items():
            if label not in seed:
                groups[metadata["category"]].add(label)

        # Preserve labels from annotations created before the shared registry existed.
        for path in sorted(self.out_dir.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for item in record.get("error_types", ()):
                try:
                    label = _normalize_open_label(item)
                except UIError:
                    continue
                if label not in seed and label not in registered and label not in deleted:
                    groups["others"].add(label)
        return {category: sorted(groups[category]) for category in OPEN_LABEL_CATEGORIES}

    def open_coded_label_descriptions(self) -> Dict[str, str]:
        """Return the shared Chinese definition for each registered open-coded label."""

        return {
            label: metadata["description_zh"]
            for label, metadata in sorted(self._registered_open_labels().items())
            if label not in EDITABLE_SEED_LABELS and metadata["description_zh"]
        }

    def error_type_overrides(self) -> Dict[str, Dict[str, str]]:
        """Return shared category and Chinese-description overrides for seed labels."""

        return {
            label: {
                "category": metadata["category"],
                "description_zh": metadata["description_zh"],
            }
            for label, metadata in sorted(self._registered_open_labels().items())
            if label in EDITABLE_SEED_LABELS
        }

    def add_open_coded_label(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Register one categorized open-coded label for all annotators and tasks."""

        if self.taxonomy.get("mode") != "open_coding":
            raise UIError("This taxonomy does not allow open-coded labels")
        label = _normalize_open_label(payload.get("label", ""))
        category = str(payload.get("category", "")).strip().lower()
        if category not in OPEN_LABEL_CATEGORIES:
            raise UIError(
                "category must be one of: %s" % ", ".join(OPEN_LABEL_CATEGORIES)
            )
        annotator_id = _safe_id(payload.get("annotator_id", ""), "annotator_id")
        description_zh = _normalize_open_label_description(
            payload.get("description_zh", ""), required=True
        )
        if label in EDITABLE_SEED_LABELS or label in RETIRED_SEED_LABELS:
            raise UIError("Label already exists in the seed taxonomy: %s" % label)

        with self._label_lock:
            labels, deleted = self._open_label_registry()
            existing = labels.get(label)
            if existing is not None:
                if existing["category"] != category:
                    raise ConflictError(
                        "Label %s already exists in category %s"
                        % (label, existing["category"])
                    )
                if existing["description_zh"] != description_zh:
                    raise ConflictError(
                        "Label %s already exists with a different Chinese explanation" % label
                    )
                return {
                    "label": label,
                    "category": category,
                    "description_zh": existing["description_zh"],
                }
            labels[label] = {
                "category": category,
                "created_by": annotator_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "description_zh": description_zh,
            }
            deleted.pop(label, None)
            self._write_open_label_registry(labels, deleted)
        return {
            "label": label,
            "category": category,
            "description_zh": description_zh,
        }

    def update_open_coded_label(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Update the category or Chinese explanation of any active error type."""

        label = _normalize_open_label(payload.get("label", ""))
        category = str(payload.get("category", "")).strip().lower()
        if category not in OPEN_LABEL_CATEGORIES:
            raise UIError(
                "category must be one of: %s" % ", ".join(OPEN_LABEL_CATEGORIES)
            )
        annotator_id = _safe_id(payload.get("annotator_id", ""), "annotator_id")
        with self._label_lock:
            labels, deleted = self._open_label_registry()
            metadata = labels.get(label)
            if metadata is None and label in EDITABLE_SEED_LABELS:
                default_category = next(
                    group for group, members in ERROR_TYPE_GROUPS.items() if label in members
                )
                metadata = {
                    "category": default_category,
                    "created_by": annotator_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_by": "",
                    "updated_at": "",
                    "description_zh": ERROR_TYPE_DESCRIPTIONS[label],
                }
                labels[label] = metadata
            if metadata is None or label in RETIRED_SEED_LABELS:
                raise UIError("Open-coded label is not registered: %s" % label)
            if "description_zh" in payload:
                metadata["description_zh"] = _normalize_open_label_description(
                    payload.get("description_zh"), required=True
                )
            metadata["category"] = category
            metadata["updated_by"] = annotator_id
            metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write_open_label_registry(labels, deleted)
        return {
            "label": label,
            "category": category,
            "description_zh": metadata["description_zh"],
        }

    def delete_open_coded_label(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Hide one shared label without rewriting historical annotations that used it."""

        label = _normalize_open_label(payload.get("label", ""))
        annotator_id = _safe_id(payload.get("annotator_id", ""), "annotator_id")
        if label in EDITABLE_SEED_LABELS or label in RETIRED_SEED_LABELS:
            raise UIError("Seed taxonomy labels cannot be deleted: %s" % label)
        with self._label_lock:
            labels, deleted = self._open_label_registry()
            metadata = labels.pop(label, None)
            if metadata is None:
                if label in deleted:
                    return {"label": label, "deleted": True}
                raise UIError("Open-coded label is not registered: %s" % label)
            deleted[label] = {
                "deleted_by": annotator_id,
                "deleted_at": datetime.now(timezone.utc).isoformat(),
                "last_category": metadata["category"],
                "description_zh": metadata["description_zh"],
            }
            self._write_open_label_registry(labels, deleted)
        return {"label": label, "deleted": True}

    def _known_extra_labels(self) -> List[str]:
        groups = self.open_coded_label_groups()
        return sorted(label for labels in groups.values() for label in labels)

    def _trajectory_context(
        self, task: Dict[str, Any], *, include_details: bool = False
    ) -> Dict[str, Any]:
        """Recover selector metadata from the immutable rollout task bundle."""

        trajectory_id = str(task["trajectory_id"])
        actions = list(task.get("actions", ()))
        source_agents = sorted(
            {str(action.get("source_agent", "")).strip() for action in actions}
            - {""}
        )
        agent_id = str(task.get("agent_id", "")).strip()
        if not agent_id:
            agent_id = source_agents[0] if len(source_agents) == 1 else "unknown"

        context = {
            "agent_id": agent_id,
            "task_id": str(task.get("task_id", "")).strip() or trajectory_id,
            "task_category": str(task.get("task_category", "")).strip() or "uncategorized",
            "task_instruction": str(task.get("task_instruction", "")).strip(),
        }
        # Resumed MyPCBench collection runs may overwrite ``_tasks/batch.json``.  The
        # canonicalizer freezes the matched task config beside every trajectory, so prefer that
        # immutable per-trajectory copy whenever the annotation task does not embed metadata.
        canonical_config_path = (
            self.build_dir / "canonical" / trajectory_id / "task_config.json"
        )
        try:
            canonical_config = json.loads(canonical_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            canonical_config = {}
        if isinstance(canonical_config, dict):
            context["task_id"] = (
                str(canonical_config.get("id", "")).strip() or context["task_id"]
            )
            context["task_category"] = (
                str(canonical_config.get("category", "")).strip()
                or context["task_category"]
            )
            context["task_instruction"] = (
                str(canonical_config.get("instruction", "")).strip()
                or context["task_instruction"]
            )
            if include_details:
                grading = canonical_config.get("grading", {})
                raw_rubrics = grading.get("rubrics", ()) if isinstance(grading, dict) else ()
                context["task_rubrics"] = [
                    {
                        "rubric_id": str(rubric.get("rubric_id", "")).strip()
                        or "R%d" % (index + 1),
                        "requirement": str(
                            rubric.get("requirement", rubric.get("criterion", ""))
                        ).strip(),
                        "weight": rubric.get("weight"),
                    }
                    for index, rubric in enumerate(raw_rubrics)
                    if isinstance(rubric, dict)
                    and rubric.get("requirement", rubric.get("criterion"))
                ]
        if include_details:
            embedded_rubrics = list(task.get("task_rubrics", ()))
            if embedded_rubrics:
                context["task_rubrics"] = embedded_rubrics
            else:
                context.setdefault("task_rubrics", [])
        source_uri = next(
            (
                str(action.get("source_record_uri", "")).partition("#")[0]
                for action in actions
                if action.get("source_record_uri")
            ),
            "",
        )
        if not source_uri:
            return context
        source_path = Path(unquote(source_uri)).resolve()
        if not (source_path == self.collection_root or self.collection_root in source_path.parents):
            return context

        inferred_task_id = source_path.parent.name
        batch_path = source_path.parent.parent / "_tasks" / "batch.json"
        if not batch_path.is_file():
            context["task_id"] = inferred_task_id or context["task_id"]
            return context
        try:
            batch = json.loads(batch_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return context
        metadata = next(
            (item for item in batch if str(item.get("id", "")) == inferred_task_id),
            None,
        )
        if metadata is None:
            return context
        context["task_id"] = str(metadata.get("id", "")).strip() or context["task_id"]
        context["task_category"] = (
            str(metadata.get("category", "")).strip() or context["task_category"]
        )
        context["task_instruction"] = (
            str(metadata.get("instruction", "")).strip() or context["task_instruction"]
        )
        if include_details:
            grading = metadata.get("grading", {})
            raw_rubrics = grading.get("rubrics", ()) if isinstance(grading, dict) else ()
            context["task_rubrics"] = [
                {
                    # MyPCBench's current task batches use ``criterion`` and do not assign an
                    # ID.  Older batches use ``requirement``/``rubric_id``.  Stable positional
                    # IDs keep both formats independently scoreable in the annotation UI.
                    "rubric_id": str(rubric.get("rubric_id", "")).strip()
                    or "R%d" % (index + 1),
                    "requirement": str(
                        rubric.get("requirement", rubric.get("criterion", ""))
                    ).strip(),
                    "weight": rubric.get("weight"),
                }
                for index, rubric in enumerate(raw_rubrics)
                if isinstance(rubric, dict)
                and rubric.get("requirement", rubric.get("criterion"))
            ]
        return context

    # ---- annotation protocol (paper App. E) -------------------------------------

    def _optional_record(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise UIError("%s cannot be read: %s" % (path.name, exc))

    def _double_annotation_ids(self) -> set:
        plan = self._optional_record(self.out_dir.joinpath(*DOUBLE_ANNOTATION_PLAN)) or {}
        return {str(item.get("trajectory_id")) for item in plan.get("trajectories", ())}

    def _adjudicators(self) -> Dict[str, List[str]]:
        by_trajectory: Dict[str, List[str]] = {}
        for path in sorted((self.out_dir / ADJUDICATION_DIR).glob("*.json")):
            record = self._optional_record(path) or {}
            if record.get("trajectory_id") and record.get("adjudicator_id"):
                by_trajectory.setdefault(str(record["trajectory_id"]), []).append(
                    str(record["adjudicator_id"])
                )
        return by_trajectory

    def auto_proposal(self, trajectory_id: str) -> Optional[Dict[str, Any]]:
        record = self._optional_record(self.out_dir / AUTO_ANALYSIS_DIR / (trajectory_id + ".json"))
        if record is not None and record.get("trajectory_id") != trajectory_id:
            raise UIError("Automatic proposal identity does not match %s" % trajectory_id)
        return record

    def submit_adjudication(self, payload: Dict[str, Any]) -> Path:
        """Record a third annotator's resolution of two independent reviews.

        The adjudicator must not be one of the reviewers.  ``task_success`` settles the
        full-completion judgment; for a failure, ``failure_adjudication`` optionally settles the
        root cause, horizon and error types as a ``derail.annotation.records.Adjudication`` over
        the reviewers' failure annotations.
        """

        trajectory_id = _safe_id(payload.get("trajectory_id", ""), "trajectory_id")
        adjudicator_id = _safe_id(payload.get("annotator_id", ""), "annotator_id")
        task = self.task(trajectory_id)
        source_sha = task["source_trajectory_sha256"]
        reviews: Dict[str, Dict[str, Any]] = {}
        for path in sorted((self.out_dir / "rubric_scores").glob(trajectory_id + "__*.json")):
            record = self._optional_record(path) or {}
            if record.get("trajectory_id") == trajectory_id and (
                record.get("source_trajectory_sha256") == source_sha
            ):
                reviews[str(record.get("reviewer_id"))] = record
        if len(reviews) < 2:
            raise UIError("Adjudication needs two independent submitted reviews")
        if adjudicator_id in reviews:
            raise UIError("The adjudicator must not be one of the reviewers")
        task_success = payload.get("task_success")
        if not isinstance(task_success, bool):
            raise UIError("task_success must be true or false")
        rationale = str(payload.get("resolution_rationale", "")).strip()
        if not rationale:
            raise UIError("resolution_rationale is required")
        disagreement = [
            field
            for field in ("task_success", "scores")
            if len({json.dumps(r.get(field), sort_keys=True) for r in reviews.values()}) > 1
        ]
        failure = None
        raw_failure = payload.get("failure_adjudication")
        if raw_failure is not None:
            if task_success or not isinstance(raw_failure, dict):
                raise UIError("failure_adjudication applies only to an adjudicated failure")
            annotations = [
                HumanAnnotation.from_dict(json.loads(path.read_text(encoding="utf-8")))
                for path in sorted(self.out_dir.glob(trajectory_id + "__*.json"))
            ]
            adjudication = Adjudication.from_dict(
                {
                    **raw_failure,
                    "adjudication_id": "%s__%s" % (trajectory_id, adjudicator_id),
                    "trajectory_id": trajectory_id,
                    "adjudicator_id": adjudicator_id,
                    "resolution_rationale": rationale,
                }
            )
            adjudication.validate_inputs(annotations)
            failure = adjudication.to_dict()
            validate_schema(failure, "adjudication.schema.json", self.repository)
        record = {
            "adjudication_id": "%s__%s" % (trajectory_id, adjudicator_id),
            "trajectory_id": trajectory_id,
            "source_trajectory_sha256": source_sha,
            "adjudicator_id": adjudicator_id,
            "adjudicator_role": "human",
            "input_review_ids": sorted(
                "%s__%s" % (trajectory_id, reviewer) for reviewer in reviews
            ),
            "disagreement_fields": disagreement,
            "task_success": task_success,
            "resolution_rationale": rationale,
            "failure_adjudication": failure,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": ADJUDICATION_SCHEMA_VERSION,
        }
        target = self.out_dir / ADJUDICATION_DIR / (
            self._review_id(trajectory_id, adjudicator_id) + ".json"
        )
        with self._submit_lock:
            atomic_write_json(target, record)
        return target

    def config(self) -> Dict[str, Any]:
        submitted = self._submitted()
        rerun_requested = self._rerun_requested()
        double_annotation = self._double_annotation_ids()
        adjudicators = self._adjudicators()
        trajectories = []
        for path in self._task_paths():
            task = json.loads(path.read_text(encoding="utf-8"))
            trajectory_id = str(task["trajectory_id"])
            context = self._trajectory_context(task)
            trajectories.append(
                {
                    "trajectory_id": trajectory_id,
                    "action_count": len(task.get("actions", ())),
                    "normalization_gate_passed": bool(task.get("normalization_gate_passed")),
                    "annotation_gate_passed": bool(
                        task.get(
                            "annotation_gate_passed",
                            task.get("normalization_gate_passed"),
                        )
                    ),
                    "annotated_by": sorted(submitted.get(trajectory_id, ())),
                    "rerun_requested_by": sorted(rerun_requested.get(trajectory_id, ())),
                    "double_annotation": trajectory_id in double_annotation,
                    "adjudicated_by": sorted(adjudicators.get(trajectory_id, ())),
                    **context,
                }
            )
        return {
            "build_id": self.manifest["build_id"],
            "taxonomy": {
                "version": self.taxonomy.get("version", ""),
                "mode": self.taxonomy.get("mode", ""),
                "seed_labels": list(self.taxonomy.get("seed_labels", ())),
                "groups": _taxonomy_groups(self.taxonomy),
            },
            "known_extra_labels": self._known_extra_labels(),
            "open_coded_label_groups": self.open_coded_label_groups(),
            "open_coded_label_descriptions": self.open_coded_label_descriptions(),
            "error_type_overrides": self.error_type_overrides(),
            "trajectories": trajectories,
        }

    def task(self, trajectory_id: str) -> Dict[str, Any]:
        trajectory_id = _safe_id(trajectory_id, "trajectory_id")
        path = self.build_dir / "canonical" / trajectory_id / "annotation_task.json"
        if not path.is_file():
            raise UIError("annotation_task.json not found: %s" % trajectory_id)
        task = json.loads(path.read_text(encoding="utf-8"))
        task.update(self._trajectory_context(task, include_details=True))
        report_path = path.parent / "normalization_report.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = {}
        provenance = report.get("task_provenance", {})
        task["rubric_bundle_sha256"] = (
            str(provenance.get("rubric_bundle_sha256", ""))
            if isinstance(provenance, dict)
            else ""
        )
        task["cleaning_depths"] = list(DEPTH_GRID)
        source_files: Dict[Path, List[str]] = {}
        for action in task.get("actions", ()):
            for side in ("before", "after"):
                uri = str(action.get("observation_%s_uri" % side, ""))
                action["observation_%s_url" % side] = (
                    "api/image?p=" + _quote(uri) if uri else ""
                )
            source_uri = str(action.get("source_record_uri", ""))
            source_path_raw, separator, fragment = source_uri.partition("#line=")
            if not separator:
                action["source_traj_error"] = "source_record_uri has no line number"
                continue
            try:
                source_path = Path(unquote(source_path_raw)).resolve()
                source_line = int(fragment)
                if source_line < 1:
                    raise ValueError("line number must be positive")
            except (TypeError, ValueError):
                action["source_traj_error"] = "source_record_uri has an invalid line number"
                continue
            if not (
                source_path == self.collection_root
                or self.collection_root in source_path.parents
            ):
                action["source_traj_error"] = "source traj path is outside the collection"
                continue
            try:
                lines = source_files.get(source_path)
                if lines is None:
                    lines = source_path.read_text(encoding="utf-8").splitlines()
                    source_files[source_path] = lines
                raw_record = json.loads(lines[source_line - 1])
            except FileNotFoundError:
                action["source_traj_error"] = "source traj file is missing"
                continue
            except (IndexError, OSError, json.JSONDecodeError) as exc:
                action["source_traj_error"] = "source traj row cannot be read: %s" % exc
                continue
            if not isinstance(raw_record, dict):
                action["source_traj_error"] = "source traj row is not a JSON object"
                continue
            action["source_traj_record"] = {
                key: raw_record.get(key)
                for key in (
                    "step_num",
                    "action_timestamp",
                    "action",
                    "response",
                    "reward",
                    "done",
                    "info",
                    "agent_metadata",
                    "screenshot_file",
                )
                if key in raw_record
            }
        task["derived_error_types"] = (
            [DERIVED_BUDGET_LABEL] if self._hit_budget_limit(task) else []
        )
        task["auto_proposal"] = self.auto_proposal(trajectory_id)
        task["double_annotation"] = trajectory_id in self._double_annotation_ids()
        return task

    @staticmethod
    def _hit_budget_limit(task: Dict[str, Any]) -> bool:
        """Recognize collection-budget termination from the authoritative raw final row.

        Ordinary wall-clock/step-budget truncation leaves the final raw row with ``done=false``.
        Kimi's consecutive-shell guard instead emits a terminal FAIL row whose structured payload
        contains ``BASH_BUDGET_ABORT``.  PREDICT_CRASH and other terminal failures are deliberately
        excluded: they ended the rollout, but did not exhaust an agent budget.
        """

        actions = list(task.get("actions", ()))
        if not actions:
            return False
        source = actions[-1].get("source_traj_record")
        if not isinstance(source, dict):
            return False
        if source.get("done") is False:
            return True
        return "BASH_BUDGET_ABORT" in json.dumps(source, ensure_ascii=False)

    def image(self, raw_path: str) -> Tuple[bytes, str]:
        path = Path(unquote(raw_path))
        if not path.is_absolute():
            raise UIError("Image path must be absolute")
        resolved = path.resolve()
        if resolved not in self.allowed_images:
            raise UIError("Image is not referenced by this annotation build")
        if not resolved.is_file():
            raise UIError("Image not found")
        mime = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        if not mime.startswith("image/"):
            raise UIError("Only image files may be read")
        return resolved.read_bytes(), mime

    # ---- write ----------------------------------------------------------------

    def _review_id(self, trajectory_id: str, annotator_id: str) -> str:
        return "%s__%s" % (
            _safe_id(trajectory_id, "trajectory_id"),
            _safe_id(annotator_id, "annotator_id"),
        )

    def _draft_target(self, trajectory_id: str, annotator_id: str) -> Path:
        return self.out_dir / "drafts" / (
            self._review_id(trajectory_id, annotator_id) + ".json"
        )

    def _rollout_flag_target(self, trajectory_id: str, annotator_id: str) -> Path:
        return self.out_dir / "rollout_flags" / (
            self._review_id(trajectory_id, annotator_id) + ".json"
        )

    def save_draft(self, payload: Dict[str, Any]) -> Path:
        """Atomically save a recoverable, mutable annotation draft.

        Drafts intentionally have a small private schema rather than using the immutable final
        annotation schema: incomplete rubric scores and partially filled failure fields are valid
        while an annotator is still working.
        """

        trajectory_id = _safe_id(payload.get("trajectory_id", ""), "trajectory_id")
        annotator_id = _safe_id(payload.get("annotator_id", ""), "annotator_id")
        task = self.task(trajectory_id)
        state = payload.get("draft")
        raw_scores = payload.get("rubric_scores", {})
        if not isinstance(state, dict):
            raise UIError("draft must be a JSON object")
        if not isinstance(raw_scores, dict):
            raise UIError("rubric_scores must be a JSON object")
        expected_ids = {
            str(item.get("rubric_id", "")) for item in task.get("task_rubrics", ())
        }
        scores: Dict[str, int] = {}
        for rubric_id, value in raw_scores.items():
            if rubric_id not in expected_ids:
                raise UIError("Unknown rubric in draft: %s" % rubric_id)
            if isinstance(value, bool) or value not in {0, 1}:
                raise UIError("Draft rubric %s score must be 0 or 1" % rubric_id)
            scores[rubric_id] = int(value)
        # The HTTP body cap already bounds this, but service-level callers receive the same
        # protection and a clear error rather than creating an unexpectedly large draft file.
        if len(json.dumps(state, ensure_ascii=False).encode("utf-8")) > MAX_BODY_BYTES:
            raise UIError("draft is too large")
        record = {
            "draft_id": self._review_id(trajectory_id, annotator_id),
            "trajectory_id": trajectory_id,
            "source_trajectory_sha256": task["source_trajectory_sha256"],
            "annotator_id": annotator_id,
            "draft": state,
            "rubric_scores": scores,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": "draft-0.1.0",
        }
        target = self._draft_target(trajectory_id, annotator_id)
        atomic_write_json(target, record)
        return target

    def draft(self, trajectory_id: str, annotator_id: str) -> Optional[Dict[str, Any]]:
        """Return one server-side draft, or ``None`` when no recoverable draft exists."""

        target = self._draft_target(trajectory_id, annotator_id)
        try:
            record = json.loads(target.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise UIError("Saved draft cannot be read: %s" % exc)
        if (
            record.get("trajectory_id") != trajectory_id
            or record.get("annotator_id") != annotator_id
        ):
            raise UIError("Saved draft identity does not match the request")
        return record

    def submission(self, trajectory_id: str, annotator_id: str) -> Dict[str, Any]:
        """Read back the records one annotator already submitted for one trajectory.

        Submitting supersedes the mutable draft, so without this an annotator loses sight of
        their own decision the moment they leave the page even though the files are on disk.
        This is strictly a read: it opens the four possible submission files and returns them
        untouched, and never writes, moves, or deletes anything under ``out_dir``.
        """

        trajectory_id = _safe_id(trajectory_id, "trajectory_id")
        annotator_id = _safe_id(annotator_id, "annotator_id")
        # Verify the trajectory belongs to this build before resolving any label path.
        self.task(trajectory_id)
        review_id = self._review_id(trajectory_id, annotator_id)
        sources = {
            "rubric_scores": self.out_dir / "rubric_scores" / (review_id + ".json"),
            "annotation": self.out_dir / (review_id + ".json"),
            "cleaning_proposal": self.out_dir / "cleaning_proposals" / (review_id + ".json"),
            "rollout_flag": self._rollout_flag_target(trajectory_id, annotator_id),
        }
        submission: Dict[str, Any] = {}
        for name, path in sources.items():
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                submission[name] = None
                continue
            except (OSError, json.JSONDecodeError) as exc:
                raise UIError("Submitted %s cannot be read: %s" % (name, exc))
            owner = str(record.get("annotator_id", record.get("reviewer_id", "")))
            if record.get("trajectory_id") != trajectory_id or owner != annotator_id:
                raise UIError("Submitted %s identity does not match the request" % name)
            submission[name] = record
        return submission

    def clear_review(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Archive and clear one annotator's state for exactly one trajectory."""

        trajectory_id = _safe_id(payload.get("trajectory_id", ""), "trajectory_id")
        annotator_id = _safe_id(payload.get("annotator_id", ""), "annotator_id")
        # Verify that the requested trajectory belongs to this build before resolving targets.
        self.task(trajectory_id)
        review_id = self._review_id(trajectory_id, annotator_id)
        targets = {
            "failure_annotation": self.out_dir / (review_id + ".json"),
            "server_draft": self._draft_target(trajectory_id, annotator_id),
            "rubric_scores": self.out_dir / "rubric_scores" / (review_id + ".json"),
            "cleaning_proposal": self.out_dir / "cleaning_proposals" / (review_id + ".json"),
            "wrong_rollout": self._rollout_flag_target(trajectory_id, annotator_id),
        }

        with self._submit_lock:
            existing = [(name, path) for name, path in targets.items() if path.is_file()]
            if not existing:
                return {"cleared": [], "archive_path": None}
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            archive = self.out_dir / "cleared_reviews" / review_id / stamp
            moved: List[str] = []
            try:
                for name, source in existing:
                    relative = source.relative_to(self.out_dir)
                    destination = archive / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    source.replace(destination)
                    moved.append(name)
            except OSError as exc:
                raise UIError("Could not archive cleared review state: %s" % exc)
            return {"cleared": moved, "archive_path": str(archive)}

    def export_results(
        self,
        annotator_id: str,
        task_loader: Optional[Callable[[str], Dict[str, Any]]] = None,
        active_trajectory_ids: Optional[set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Export compact, analysis-ready records for one annotator."""

        annotator_id = _safe_id(annotator_id, "annotator_id")
        task_loader = task_loader or self.task
        records: List[Dict[str, Any]] = []
        rerun_trajectory_ids: set[str] = set()

        def optional_json(path: Path) -> Optional[Dict[str, Any]]:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return None
            except (OSError, json.JSONDecodeError) as exc:
                raise UIError("Submitted result cannot be read: %s" % exc)
            if not isinstance(value, dict):
                raise UIError("Submitted result is not a JSON object: %s" % path.name)
            return value

        for flag_path in sorted((self.out_dir / "rollout_flags").glob("*.json")):
            flag = optional_json(flag_path)
            if (
                flag is not None
                and str(flag.get("annotator_id", "")) == annotator_id
                and flag.get("rollout_status") == "needs_rerun"
            ):
                rerun_trajectory_ids.add(str(flag.get("trajectory_id", "")))

        for review_path in sorted((self.out_dir / "rubric_scores").glob("*.json")):
            try:
                review = json.loads(review_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(review.get("reviewer_id", "")) != annotator_id:
                continue
            trajectory_id = str(review.get("trajectory_id", ""))
            if (
                active_trajectory_ids is not None
                and trajectory_id not in active_trajectory_ids
            ):
                continue
            if trajectory_id in rerun_trajectory_ids:
                continue
            review_id = self._review_id(trajectory_id, annotator_id)
            annotation_path = self.out_dir / (review_id + ".json")

            scores = review.get("scores")
            if not isinstance(scores, dict) or not scores:
                raise UIError("Submitted rubric scores are missing: %s" % review_path.name)
            task = task_loader(trajectory_id)
            rubrics = list(task.get("task_rubrics", ()))
            rubric_ids = [str(rubric.get("rubric_id", "")) for rubric in rubrics]
            if not rubric_ids or set(scores) != set(rubric_ids):
                raise UIError(
                    "Submitted rubric scores do not match the task bundle: %s"
                    % review_path.name
                )
            weights: List[float] = []
            for rubric in rubrics:
                try:
                    weight = float(rubric.get("weight"))
                except (TypeError, ValueError):
                    weight = 1.0
                weights.append(weight if math.isfinite(weight) and weight > 0 else 1.0)
            total_weight = sum(weights) or 1.0
            passed_weight = sum(
                weight * int(scores[rubric_id])
                for rubric_id, weight in zip(rubric_ids, weights)
            )
            weighted_score = max(0.0, min(1.0, passed_weight / total_weight))

            records.append(
                {
                    "trajectory_id": trajectory_id,
                    "annotator_id": annotator_id,
                    "scores": scores,
                    "weighted_rubric_score": weighted_score,
                    "task_score": int(all(scores[rubric_id] == 1 for rubric_id in rubric_ids)),
                    "source_trajectory_sha256": str(
                        review.get("source_trajectory_sha256", "")
                    ),
                    "rubric_bundle_sha256": str(review.get("rubric_bundle_sha256", "")),
                    "failure_annotation": optional_json(annotation_path),
                    "rollout_status": "valid",
                }
            )
        for flag_path in sorted((self.out_dir / "rollout_flags").glob("*.json")):
            flag = optional_json(flag_path)
            if flag is None or str(flag.get("annotator_id", "")) != annotator_id:
                continue
            if flag.get("rollout_status") != "needs_rerun":
                continue
            trajectory_id = str(flag.get("trajectory_id", ""))
            if (
                active_trajectory_ids is not None
                and trajectory_id not in active_trajectory_ids
            ):
                continue
            records.append(
                {
                    "trajectory_id": trajectory_id,
                    "annotator_id": annotator_id,
                    "scores": None,
                    "weighted_rubric_score": None,
                    "task_score": None,
                    "source_trajectory_sha256": str(
                        flag.get("source_trajectory_sha256", "")
                    ),
                    "rubric_bundle_sha256": str(flag.get("rubric_bundle_sha256", "")),
                    "failure_annotation": None,
                    "rollout_status": "needs_rerun",
                }
            )
        records.sort(key=lambda item: item["trajectory_id"])
        return records

    def export_results_jsonl(
        self,
        annotator_id: str,
        task_loader: Optional[Callable[[str], Dict[str, Any]]] = None,
        active_trajectory_ids: Optional[set[str]] = None,
        filename_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Write compact, analysis-ready records for one annotator to local JSONL."""

        annotator_id = _safe_id(annotator_id, "annotator_id")
        records = self.export_results(
            annotator_id,
            task_loader=task_loader,
            active_trajectory_ids=active_trajectory_ids,
        )
        if not records:
            raise UIError(
                "No submitted results to export; drafts are not included. "
                "Run Validate and Submit first."
            )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        prefix = _safe_id(
            filename_prefix
            or "%s_human_label_results_%s"
            % (_build_export_stem(str(self.manifest["build_id"])), annotator_id),
            "filename_prefix",
        )
        target = self.out_dir / "exports" / ("%s_%s.jsonl" % (prefix, stamp))
        target.parent.mkdir(parents=True, exist_ok=True)
        body = "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in records
        )
        tmp = target.with_name("." + target.name + ".tmp")
        try:
            tmp.write_text(body, encoding="utf-8")
            tmp.replace(target)
        except OSError as exc:
            raise UIError("Could not write export JSONL: %s" % exc)
        return {"path": target, "record_count": len(records)}

    def submit_review(
        self, payload: Dict[str, Any], allow_resubmit: bool
    ) -> Dict[str, Optional[Path]]:
        """Keep the latest valid submission for one trajectory/annotator identity."""

        # ThreadingHTTPServer may receive two tabs at once. Serializing the replacement keeps
        # "latest submission wins" deterministic and prevents mixed artifacts across states.
        with self._submit_lock:
            return self._submit_review_latest(payload)

    def _submit_review_latest(
        self, payload: Dict[str, Any]
    ) -> Dict[str, Optional[Path]]:
        """Validate and atomically replace one annotator's previous task decision."""

        trajectory_id = _safe_id(payload.get("trajectory_id", ""), "trajectory_id")
        annotator_id = _safe_id(payload.get("annotator_id", ""), "annotator_id")
        task = self.task(trajectory_id)
        review_id = self._review_id(trajectory_id, annotator_id)
        review_target = self.out_dir / "rubric_scores" / (review_id + ".json")
        annotation_target = self.out_dir / (review_id + ".json")
        cleaning_target = self.out_dir / "cleaning_proposals" / (review_id + ".json")
        rollout_flag_target = self._rollout_flag_target(trajectory_id, annotator_id)

        def remove_stale(*paths: Path) -> None:
            for path in paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

        if payload.get("wrong_rollout") is True:
            flag_record = {
                "flag_id": review_id,
                "trajectory_id": trajectory_id,
                "source_trajectory_sha256": task["source_trajectory_sha256"],
                "rubric_bundle_sha256": task.get("rubric_bundle_sha256", ""),
                "annotator_id": annotator_id,
                "annotator_role": "human",
                "rollout_status": "needs_rerun",
                "reason": "wrong_rollout",
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "schema_version": "0.1.0",
            }
            atomic_write_json(rollout_flag_target, flag_record)
            remove_stale(
                review_target,
                annotation_target,
                cleaning_target,
                self._draft_target(trajectory_id, annotator_id),
            )
            return {
                "annotation_path": None,
                "cleaning_proposal_path": None,
                "rubric_score_path": None,
                "rollout_flag_path": rollout_flag_target,
            }

        rubrics = list(task.get("task_rubrics", ()))
        raw_scores = payload.get("rubric_scores")
        if not isinstance(raw_scores, dict):
            raise UIError("Complete every rubric score before submitting")
        expected_ids = [str(item.get("rubric_id", "")) for item in rubrics]
        if not expected_ids or set(raw_scores) != set(expected_ids):
            raise UIError("Rubric scores must cover every rubric exactly once")
        scores: Dict[str, int] = {}
        for rubric_id in expected_ids:
            score = raw_scores[rubric_id]
            if isinstance(score, bool) or score not in {0, 1}:
                raise UIError("Rubric %s score must be 0 or 1" % rubric_id)
            scores[rubric_id] = int(score)
        task_success = all(score == 1 for score in scores.values())
        if payload.get("task_success") is not task_success:
            raise UIError("task_success does not match the submitted rubric scores")

        review_record = {
            "review_id": review_id,
            "trajectory_id": trajectory_id,
            "source_trajectory_sha256": task["source_trajectory_sha256"],
            "rubric_bundle_sha256": task.get("rubric_bundle_sha256", ""),
            "reviewer_id": annotator_id,
            "reviewer_role": "human",
            "scores": scores,
            "task_success": task_success,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": "raw-0.1.0",
        }

        annotation_target = None
        if not task_success:
            annotation_target = self.submit(payload, allow_resubmit=True)
        atomic_write_json(review_target, review_record)
        if task_success:
            remove_stale(self.out_dir / (review_id + ".json"), cleaning_target)
        remove_stale(rollout_flag_target)
        # A final, validated review supersedes its mutable draft.  Failure to remove the draft
        # must never turn a successful submission into an HTTP error; the UI also ignores drafts
        # for trajectories already listed as submitted by this annotator.
        remove_stale(self._draft_target(trajectory_id, annotator_id))
        return {
            "annotation_path": annotation_target,
            "cleaning_proposal_path": (
                annotation_target.parent / "cleaning_proposals" / annotation_target.name
                if annotation_target is not None
                else None
            ),
            "rubric_score_path": review_target,
            "rollout_flag_path": None,
        }

    def _cleaning_proposal(
        self,
        payload: Dict[str, Any],
        task: Dict[str, Any],
        annotation_id: str,
        root: int,
    ) -> Dict[str, Any]:
        if payload.get("cleaning_review_complete") is not True:
            raise UIError("Complete the clean-prefix review before submitting")
        # The audit covers every depth the suffix allows (the shared App. C rule without the
        # error-horizon cut, which narrows cases later, not the reviewed prefix).
        available_depths, _ = eligible_depths(
            root, len(task["actions"]) - 1, None, require_error_explicit=False
        )
        if not available_depths:
            raise UIError("No clean-prefix depth is available for this root cause")
        audit_end = root + max(available_depths)
        actions_by_id = {
            int(action["action_index_global"]): action for action in task["actions"]
        }
        raw_candidates = payload.get("cleaning_drop_candidates", ())
        if not isinstance(raw_candidates, list):
            raise UIError("cleaning_drop_candidates must be an array")
        candidates = []
        seen = set()
        for raw in raw_candidates:
            if not isinstance(raw, dict):
                raise UIError("Each clean-prefix candidate must be an object")
            index = raw.get("action_index_global")
            recovery_index = raw.get("recovery_action_index")
            if not isinstance(index, int) or isinstance(index, bool):
                raise UIError("Cleaning candidate action_index_global must be an integer")
            if not isinstance(recovery_index, int) or isinstance(recovery_index, bool):
                raise UIError("Cleaning candidate recovery_action_index must be an integer")
            if index in seen:
                raise UIError("Duplicate cleaning candidate action: %d" % index)
            if index == root:
                raise UIError("The root-cause action cannot be removed")
            if index < 0 or index > audit_end or index not in actions_by_id:
                raise UIError("Cleaning candidate is outside the audited prefix: %d" % index)
            if (
                recovery_index <= index
                or recovery_index > audit_end
                or recovery_index not in actions_by_id
            ):
                raise UIError("Cleaning recovery step is outside the audited prefix")
            reason = str(raw.get("reason", "")).strip()
            if not reason:
                raise UIError("Cleaning candidate %d requires a reason" % index)
            seen.add(index)
            candidates.append(
                {
                    "action_index_global": index,
                    "recovery_action_index": recovery_index,
                    "old_action": actions_by_id[index]["action"],
                    "self_recovered": True,
                    "persistent_state_effect": False,
                    "causal_to_root_or_task": False,
                    "reason": reason,
                }
            )
        candidates.sort(key=lambda item: item["action_index_global"])
        candidate_indices = {item["action_index_global"] for item in candidates}
        if any(item["recovery_action_index"] in candidate_indices for item in candidates):
            raise UIError("Every recovery step must be retained")
        by_recovery: Dict[int, List[int]] = {}
        for item in candidates:
            by_recovery.setdefault(item["recovery_action_index"], []).append(
                item["action_index_global"]
            )
        for recovery_index, indices in by_recovery.items():
            if indices != list(range(min(indices), recovery_index)):
                raise UIError(
                    "Each removed detour must be contiguous through the step before recovery"
                )
        proposal = {
            "proposal_id": annotation_id,
            "trajectory_id": task["trajectory_id"],
            "source_trajectory_sha256": task["source_trajectory_sha256"],
            "reviewer_id": payload["annotator_id"],
            "root_cause_action_index": root,
            "audit_end_action_index": audit_end,
            "audited_action_indices": list(range(audit_end + 1)),
            "drop_candidates": candidates,
            "review_complete": True,
            "schema_version": "0.2.0",
        }
        validate_schema(proposal, "cleaning_proposal.schema.json", self.repository)
        return proposal

    def submit(self, payload: Dict[str, Any], allow_resubmit: bool) -> Path:
        trajectory_id = _safe_id(payload.get("trajectory_id", ""), "trajectory_id")
        annotator_id = _safe_id(payload.get("annotator_id", ""), "annotator_id")
        task = self.task(trajectory_id)
        if not task.get("annotation_gate_passed", task.get("normalization_gate_passed")):
            raise UIError("This trajectory did not pass the annotation gate")

        root = payload.get("root_cause_action_index")
        identifiable = payload.get("identifiable_at_action_index")
        if not isinstance(root, int) or isinstance(root, bool):
            raise UIError("root_cause_action_index must be an integer")
        if identifiable is not None and (
            not isinstance(identifiable, int) or isinstance(identifiable, bool)
        ):
            raise UIError("identifiable_at_action_index must be an integer or null")
        # The schema stores the horizon, the UI collects the two indices; deriving it here is
        # the only way root + horizon == identifiable can never drift.
        horizon = None if identifiable is None else identifiable - root

        annotation_id = "%s__%s" % (trajectory_id, annotator_id)
        # Persist the annotator's selected labels exactly.  The raw-terminal
        # detector is exposed as a UI hint, not promoted to a human label.
        error_types = list(payload.get("error_types", ()))
        record = {
            "annotation_id": annotation_id,
            "trajectory_id": trajectory_id,
            "source_trajectory_sha256": task["source_trajectory_sha256"],
            "annotator_id": annotator_id,
            "annotator_role": "human",
            "root_cause_action_index": root,
            "error_horizon_actions": horizon,
            "identifiable_at_action_index": identifiable,
            "error_types": error_types,
            "reversibility": payload.get("reversibility", ""),
            "rationale": str(payload.get("rationale", "")),
            "taxonomy_version": self.taxonomy["version"],
            "schema_version": "0.2.0",
        }
        validate_schema(record, "annotation.schema.json", self.repository)
        annotation = HumanAnnotation.from_dict(record)
        annotation.validate_against_action_count(len(task["actions"]))
        cleaning_proposal = self._cleaning_proposal(
            payload, task, annotation_id, root
        )

        target = self.out_dir / (annotation_id + ".json")
        cleaning_target = self.out_dir / "cleaning_proposals" / (annotation_id + ".json")
        if (target.exists() or cleaning_target.exists()) and not allow_resubmit:
            raise ConflictError(
                "Already exists and cannot be overwritten: %s (restart with --allow-resubmit to replace it)"
                % target.name
            )
        atomic_write_json(target, annotation.to_dict())
        atomic_write_json(cleaning_target, cleaning_proposal)
        return target


class MultiBuildAnnotationService:
    """Present compatible derived builds through one annotation UI."""

    def __init__(self, services: List[AnnotationService]) -> None:
        if not services:
            raise UIError("At least one annotation build is required")
        self.services = tuple(services)
        first = services[0]
        self.ui_dir = first.ui_dir
        self.out_dir = first.out_dir
        self.repository = first.repository
        self.taxonomy = first.taxonomy
        self.build_ids = [str(service.manifest["build_id"]) for service in services]
        taxonomy_keys = ("version", "mode", "seed_labels", "sha256")
        expected_taxonomy = {key: self.taxonomy.get(key) for key in taxonomy_keys}
        self._services_by_trajectory: Dict[str, AnnotationService] = {}
        self._services_by_image: Dict[Path, AnnotationService] = {}

        for service in services:
            if service.out_dir != self.out_dir:
                raise UIError("Multi-build annotation services must share one out-dir")
            actual_taxonomy = {key: service.taxonomy.get(key) for key in taxonomy_keys}
            if actual_taxonomy != expected_taxonomy:
                raise UIError(
                    "All annotation builds must use the same taxonomy: %s differs"
                    % service.manifest["build_id"]
                )
            for path in service._task_paths():
                task = json.loads(path.read_text(encoding="utf-8"))
                trajectory_id = _safe_id(task.get("trajectory_id", ""), "trajectory_id")
                if trajectory_id in self._services_by_trajectory:
                    raise UIError("Duplicate trajectory ID across builds: %s" % trajectory_id)
                self._services_by_trajectory[trajectory_id] = service
            for image_path in service.allowed_images:
                self._services_by_image.setdefault(image_path, service)

    def _service_for(self, trajectory_id: str) -> AnnotationService:
        trajectory_id = _safe_id(trajectory_id, "trajectory_id")
        service = self._services_by_trajectory.get(trajectory_id)
        if service is None:
            raise UIError("annotation_task.json not found: %s" % trajectory_id)
        return service

    def config(self) -> Dict[str, Any]:
        configs = [service.config() for service in self.services]
        trajectories = [
            trajectory
            for config in configs
            for trajectory in config["trajectories"]
        ]
        trajectories.sort(
            key=lambda item: (
                item["agent_id"],
                item["task_category"],
                item["task_id"],
                item["trajectory_id"],
            )
        )
        return {
            "build_id": "+".join(self.build_ids),
            "build_ids": list(self.build_ids),
            "taxonomy": configs[0]["taxonomy"],
            "known_extra_labels": sorted(
                {
                    label
                    for config in configs
                    for label in config["known_extra_labels"]
                }
            ),
            "open_coded_label_groups": configs[0]["open_coded_label_groups"],
            "open_coded_label_descriptions": configs[0][
                "open_coded_label_descriptions"
            ],
            "error_type_overrides": configs[0]["error_type_overrides"],
            "trajectories": trajectories,
        }

    def task(self, trajectory_id: str) -> Dict[str, Any]:
        return self._service_for(trajectory_id).task(trajectory_id)

    def image(self, raw_path: str) -> Tuple[bytes, str]:
        path = Path(unquote(raw_path))
        if not path.is_absolute():
            raise UIError("Image path must be absolute")
        resolved = path.resolve()
        service = self._services_by_image.get(resolved)
        if service is None:
            raise UIError("Image is not referenced by any annotation build")
        return service.image(raw_path)

    def save_draft(self, payload: Dict[str, Any]) -> Path:
        trajectory_id = str(payload.get("trajectory_id", ""))
        return self._service_for(trajectory_id).save_draft(payload)

    def draft(self, trajectory_id: str, annotator_id: str) -> Optional[Dict[str, Any]]:
        return self._service_for(trajectory_id).draft(trajectory_id, annotator_id)

    def submission(self, trajectory_id: str, annotator_id: str) -> Dict[str, Any]:
        return self._service_for(trajectory_id).submission(trajectory_id, annotator_id)

    def clear_review(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        trajectory_id = str(payload.get("trajectory_id", ""))
        return self._service_for(trajectory_id).clear_review(payload)

    def export_results(self, annotator_id: str) -> List[Dict[str, Any]]:
        return self.services[0].export_results(
            annotator_id,
            task_loader=self.task,
            active_trajectory_ids=set(self._services_by_trajectory),
        )

    def export_results_jsonl(self, annotator_id: str) -> Dict[str, Any]:
        records = self.export_results(annotator_id)
        build_stems = {
            _build_export_stem(
                str(self._service_for(str(record["trajectory_id"])).manifest["build_id"])
            )
            for record in records
        }
        export_stem = build_stems.pop() if len(build_stems) == 1 else "derail"
        return self.services[0].export_results_jsonl(
            annotator_id,
            task_loader=self.task,
            active_trajectory_ids=set(self._services_by_trajectory),
            filename_prefix="%s_human_label_results_%s"
            % (export_stem, annotator_id),
        )

    def open_coded_label_groups(self) -> Dict[str, List[str]]:
        return self.services[0].open_coded_label_groups()

    def open_coded_label_descriptions(self) -> Dict[str, str]:
        return self.services[0].open_coded_label_descriptions()

    def error_type_overrides(self) -> Dict[str, Dict[str, str]]:
        return self.services[0].error_type_overrides()

    def add_open_coded_label(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.services[0].add_open_coded_label(payload)

    def update_open_coded_label(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.services[0].update_open_coded_label(payload)

    def delete_open_coded_label(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.services[0].delete_open_coded_label(payload)

    def submit_review(
        self, payload: Dict[str, Any], allow_resubmit: bool
    ) -> Dict[str, Optional[Path]]:
        trajectory_id = str(payload.get("trajectory_id", ""))
        return self._service_for(trajectory_id).submit_review(payload, allow_resubmit)

    def submit_adjudication(self, payload: Dict[str, Any]) -> Path:
        trajectory_id = str(payload.get("trajectory_id", ""))
        return self._service_for(trajectory_id).submit_adjudication(payload)


def _quote(value: str) -> str:
    from urllib.parse import quote

    return quote(value, safe="")


class Handler(BaseHTTPRequestHandler):
    server_version = "DERAILAnnotationUI/0.1"

    def __init__(
        self,
        *args: Any,
        service: AnnotationService,
        allow_resubmit: bool,
        token: str,
        annotator_tokens: Dict[str, str],
        **kw: Any,
    ):
        self.service = service
        self.allow_resubmit = allow_resubmit
        self.token = token
        self.annotator_tokens = annotator_tokens
        super().__init__(*args, **kw)

    def _presented_token(self, query: str) -> str:
        header = self.headers.get("Authorization", "")
        return (
            header[len("Bearer "):] if header.startswith("Bearer ")
            else parse_qs(query).get("token", [""])[0]
        )

    def _authorized(self, query: str) -> bool:
        """Gate every data route behind a shared token when one is configured.

        The page itself is served unauthenticated because it carries no data; a stray link then
        renders a readable prompt instead of raw JSON.  Screenshots load through ``<img>``, which
        cannot carry a header, so the query parameter is accepted alongside the Bearer header.
        """

        if not self.token and not self.annotator_tokens:
            return True
        presented = self._presented_token(query)
        return (
            bool(self.token and hmac.compare_digest(presented, self.token))
            or any(
                hmac.compare_digest(presented, expected)
                for expected in self.annotator_tokens.values()
            )
        )

    def _bound_annotator(self, query: str) -> Optional[str]:
        presented = self._presented_token(query)
        return next(
            (
                annotator
                for annotator, expected in self.annotator_tokens.items()
                if hmac.compare_digest(presented, expected)
            ),
            None,
        )

    # ---- plumbing -------------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - stdlib hook
        print("[ui] " + fmt % args)

    def _send(self, code: int, body: bytes, mime: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, value: Any) -> None:
        self._send(code, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json")

    def _download(self, body: bytes, mime: str, filename: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", 'attachment; filename="%s"' % filename)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _requested_annotator(self, query: str) -> str:
        annotator = _safe_id(
            parse_qs(query).get("annotator", [""])[0], "annotator_id"
        )
        bound = self._bound_annotator(query)
        if bound is not None and annotator != bound:
            raise UIError("This access token is bound to annotator %s" % bound)
        return annotator

    # ---- routes ---------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - stdlib hook
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"
        try:
            if route in {"/", "/index.html"}:
                body = (self.service.ui_dir / "index.html").read_bytes()
                self._send(200, body, "text/html; charset=utf-8")
            elif not self._authorized(parsed.query):
                self._json(401, {"error": "Missing or wrong access token"})
            elif route == "/api/config":
                self._json(200, self.service.config())
            elif route == "/api/labels":
                self._json(
                    200,
                    {
                        "groups": self.service.open_coded_label_groups(),
                        "descriptions": self.service.open_coded_label_descriptions(),
                        "overrides": self.service.error_type_overrides(),
                    },
                )
            elif route.startswith("/api/task/"):
                self._json(200, self.service.task(unquote(route[len("/api/task/"):])))
            elif route.startswith("/api/draft/"):
                annotator = self._requested_annotator(parsed.query)
                trajectory = unquote(route[len("/api/draft/"):])
                self._json(200, {"draft": self.service.draft(trajectory, annotator)})
            elif route.startswith("/api/submission/"):
                annotator = self._requested_annotator(parsed.query)
                trajectory = unquote(route[len("/api/submission/"):])
                self._json(
                    200,
                    {"submission": self.service.submission(trajectory, annotator)},
                )
            elif route == "/api/export":
                annotator = self._requested_annotator(parsed.query)
                records = self.service.export_results(annotator)
                export_format = parse_qs(parsed.query).get("format", ["jsonl"])[0]
                if export_format == "jsonl":
                    body = b"".join(
                        (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
                        for record in records
                    )
                    self._download(
                        body,
                        "application/x-ndjson; charset=utf-8",
                        "derail_%s_compact_results.jsonl" % annotator,
                    )
                elif export_format == "json":
                    body = json.dumps(records, ensure_ascii=False, indent=2).encode("utf-8")
                    self._download(
                        body,
                        "application/json; charset=utf-8",
                        "derail_%s_compact_results.json" % annotator,
                    )
                else:
                    raise UIError("format must be jsonl or json")
            elif route == "/api/image":
                raw = parse_qs(parsed.query).get("p", [""])[0]
                body, mime = self.service.image(raw)
                self._send(200, body, mime)
            else:
                self._json(404, {"error": "unknown route"})
        except UIError as exc:
            self._json(400, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802 - stdlib hook
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/")
        if route not in {
            "/api/annotation",
            "/api/draft",
            "/api/labels",
            "/api/clear",
            "/api/export-local",
            "/api/adjudication",
        }:
            self._json(404, {"error": "unknown route"})
            return
        if not self._authorized(parsed.query):
            self._json(401, {"error": "Missing or wrong access token"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self._json(400, {"error": "Request body is empty or too large"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._json(400, {"error": "Request body is not valid JSON: %s" % exc})
            return
        if not isinstance(payload, dict):
            self._json(400, {"error": "Request body must be a JSON object"})
            return
        bound_annotator = self._bound_annotator(parsed.query)
        if bound_annotator is not None and payload.get("annotator_id") != bound_annotator:
            self._json(403, {"error": "This access token is bound to annotator %s" % bound_annotator})
            return
        try:
            if route == "/api/clear":
                result = self.service.clear_review(payload)
                self._json(
                    200,
                    {
                        "cleared": result["cleared"],
                        "cleared_count": len(result["cleared"]),
                        "archive_path": result["archive_path"],
                    },
                )
                return
            if route == "/api/labels":
                action = str(payload.get("action", "add")).strip().lower()
                if action == "add":
                    label = self.service.add_open_coded_label(payload)
                elif action == "update":
                    label = self.service.update_open_coded_label(payload)
                elif action == "delete":
                    label = self.service.delete_open_coded_label(payload)
                else:
                    raise UIError("label action must be add, update, or delete")
                self._json(
                    200,
                    {
                        **label,
                        "groups": self.service.open_coded_label_groups(),
                        "descriptions": self.service.open_coded_label_descriptions(),
                        "overrides": self.service.error_type_overrides(),
                    },
                )
                return
            if route == "/api/adjudication":
                target = self.service.submit_adjudication(payload)
                self._json(200, {"path": str(target), "adjudicated": True})
                return
            if route == "/api/draft":
                target = self.service.save_draft(payload)
                self._json(200, {"path": str(target), "saved": True})
                return
            if route == "/api/export-local":
                annotator = _safe_id(payload.get("annotator_id", ""), "annotator_id")
                result = self.service.export_results_jsonl(annotator)
                self._json(
                    200,
                    {
                        "path": str(result["path"]),
                        "record_count": result["record_count"],
                    },
                )
                return
            targets = self.service.submit_review(payload, self.allow_resubmit)
        except ConflictError as exc:
            self._json(409, {"error": str(exc)})
            return
        except UIError as exc:
            self._json(400, {"error": str(exc)})
            return
        except AnnotationError as exc:
            self._json(400, {"error": "Annotation semantic validation failed: %s" % exc})
            return
        except Exception as exc:  # schema violations surface as the annotator's problem
            # jsonschema renders a multi-line report; the annotator only needs the first line.
            self._json(400, {"error": str(exc).splitlines()[0] or type(exc).__name__})
            return
        self._json(
            200,
            {
                "path": (
                    str(targets["annotation_path"])
                    if targets["annotation_path"] is not None
                    else None
                ),
                "cleaning_proposal_path": (
                    str(targets["cleaning_proposal_path"])
                    if targets["cleaning_proposal_path"] is not None
                    else None
                ),
                "rubric_score_path": (
                    str(targets["rubric_score_path"])
                    if targets["rubric_score_path"] is not None
                    else None
                ),
                "rollout_flag_path": (
                    str(targets["rollout_flag_path"])
                    if targets["rollout_flag_path"] is not None
                    else None
                ),
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument(
        "--build-id",
        action="append",
        required=True,
        help="Derived build ID to serve; repeat to combine compatible builds.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Where raw per-annotator files are written; feeds 04_build_benchmark --annotations.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--allow-resubmit",
        action="store_true",
        help="Legacy direct-annotation overwrite switch; UI reviews always keep the latest submission.",
    )
    parser.add_argument(
        "--require-path-prefix",
        type=Path,
        default=(Path(os.environ["DERAIL_DATA_ROOT"]) if os.environ.get("DERAIL_DATA_ROOT") else None),
        help="Refuse to write outside this prefix; defaults to $DERAIL_DATA_ROOT when set.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("DERAIL_ANNOTATION_TOKEN", ""),
        help="Shared access token for every /api route; defaults to $DERAIL_ANNOTATION_TOKEN. "
        "Required whenever the server is reachable beyond loopback.",
    )
    parser.add_argument(
        "--annotator-token",
        action="append",
        default=[],
        metavar="ID=TOKEN",
        help="Bind a distinct access token to one annotator ID; may be repeated.",
    )
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    annotator_tokens: Dict[str, str] = {}
    for raw in args.annotator_token:
        annotator, separator, value = raw.partition("=")
        annotator = _safe_id(annotator, "annotator_id")
        if not separator or not value:
            raise SystemExit("--annotator-token must use ID=TOKEN")
        if annotator in annotator_tokens:
            raise SystemExit("duplicate --annotator-token ID: %s" % annotator)
        if value in annotator_tokens.values():
            raise SystemExit("each --annotator-token value must be unique")
        annotator_tokens[annotator] = value

    build_dirs = [(args.build_root / build_id).resolve() for build_id in args.build_id]
    out_dir = args.out_dir.resolve()
    if args.require_path_prefix is not None:
        prefix = args.require_path_prefix.resolve()
        if prefix != out_dir and prefix not in out_dir.parents:
            raise SystemExit("out-dir must be inside %s; got %s" % (prefix, out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    # A loopback bind is only private until something tunnels it out, so the token is demanded
    # for any other bind address and merely urged for 127.0.0.1.
    try:
        loopback = ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        loopback = args.host == "localhost"
    if not loopback and not args.token and not annotator_tokens:
        raise SystemExit(
            "--host %s reaches beyond this machine; set --token or $DERAIL_ANNOTATION_TOKEN"
            % args.host
        )

    services = [
        AnnotationService(build_dir, out_dir, args.repository.resolve())
        for build_dir in build_dirs
    ]
    service = (
        services[0]
        if len(services) == 1
        else MultiBuildAnnotationService(services)
    )
    handler = partial(
        Handler,
        service=service,
        allow_resubmit=args.allow_resubmit,
        token=args.token,
        annotator_tokens=annotator_tokens,
    )
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    config = service.config()
    for build_dir in build_dirs:
        print("build        %s" % build_dir)
    print("out-dir      %s" % out_dir)
    print("taxonomy     %s (%s)" % (config["taxonomy"]["version"], config["taxonomy"]["mode"]))
    print("trajectories %d" % len(config["trajectories"]))
    # The token is a credential: report only that one is set and how long it is, never its value.
    if args.token:
        print("token        set, %d chars" % len(args.token))
        print("open         http://%s:%d/?token=<your token>" % (args.host, args.port))
    elif annotator_tokens:
        print("tokens       %d annotator-bound tokens set" % len(annotator_tokens))
        print("open         http://%s:%d/?annotator=<assigned ID>&token=<assigned token>" % (args.host, args.port))
    else:
        print("token        NOT SET — every /api route is open to anyone who reaches this port")
        print("open         http://%s:%d/?annotator=<ID>" % (args.host, args.port))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
