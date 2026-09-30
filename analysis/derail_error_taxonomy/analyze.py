#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from derail.evaluation.metrics import PassAtK, pass_at_k, rubric_verdict  # noqa: E402


@dataclass(frozen=True)
class Build:
    name: str
    model: str
    source_group: str
    annotator: str


BUILDS = (
    Build("claudeopus48_xuhu_traj", "claude_opus_4_8", "closed", "xuhu"),
    Build("evocua32b_jinxin_traj", "evocua_32b", "open", "Jinxin"),
    Build("gpt5_5_haoming_traj", "gpt_5_5", "closed", "Haoming"),
    Build("gpt5_5_recollected_haoming_traj", "gpt_5_5", "closed", "Haoming"),
    Build("gpt5_5_recollected_retry38_haoming_traj", "gpt_5_5", "closed", "Haoming"),
    Build("gpt5_5_recollected_retry38_unfinished13_haoming_traj", "gpt_5_5", "closed", "Haoming"),
    Build("kimi_k3_26_kangshuo_traj", "kimi_k3_26", "open", "kangshuo"),
    Build("opencua_72b_dingyi_traj", "opencua_72b", "open", "dingyi"),
    Build("qwen35_licheng_traj", "qwen3_5_35b_a3b", "open", "licheng"),
)
INCLUDED_MODELS = (
    "claude_opus_4_8",
    "evocua_32b",
    "gpt_5_5",
    "kimi_k3_26",
    "opencua_72b",
    "qwen3_5_35b_a3b",
)
EXCLUDED_INCOMPLETE_MODELS: tuple[str, ...] = ()
PILOT_MODELS = tuple(model for model in INCLUDED_MODELS if model != "claude_opus_4_8")
ROLLOUTS_PER_MODEL = 184
PILOT_SIZE = 100
AGENT_DISPLAY_NAMES = {
    "evocua_32b": "EvoCUA-32B",
    "claude_opus_4_8": "Claude-Opus-4.8",
    "gpt_5_5": "GPT-5.5",
    "kimi_k3_26": "Kimi-K3",
    "opencua_72b": "OpenCUA-72B",
    "qwen3_5_35b_a3b": "Qwen3.5-35B-A3B",
}

RENAMES = {
    "wrong_subgoal": "misunderstand_task_objective",
    "hallucinated_data": "fabricate_data",
    "constraint_loss": "scope_error",
    "section_content_misplacement": "wrong_target",
}
DROP_AS_FAILURE = {"gui_workflow_bypass", "exceed_bash_budget"}
CODEBOOK = {
    "planning": {
        "fabricate_data": "输出轨迹证据不支持的具体事实、数值或记录。",
        "misunderstand_task_objective": "追求了错误的最终目标、交付物、对象或关键约束。",
        "lack_of_knowledge": "缺少选择可行方法所必需的领域、规则或应用知识。",
        "scope_error": "使用了错误的时间、对象、记录、总体或覆盖范围。",
    },
    "perception": {
        "progress_misperception": "误判已完成工作、剩余步骤或任务进度。",
        "detail_misperception": "漏看或误读当前可见的局部数值、字段、文本或要求。",
        "state_misinterpretation": "误判当前页面、应用、加载、登录、保存、数据或工具状态。",
        "ineffective_action": "动作没有推进任务或产生预期状态变化；仅作残余/结果标签，不能替代有证据的原因。动作无效说明 agent 没有回读状态，因此归入 perception 而非 execution。",
    },
    "execution": {
        "grounding_failure": "语义目标正确，但坐标、焦点、拖拽或动作落点没有命中。",
        "incorrect_ui_element": "选择了语义功能错误的控件或 UI 元素。",
        "typing_or_parameter_error": "输入错误文本、值、路径，或提供错误动作/工具参数；这是等待重标拆分的临时合并类。",
        "wrong_target": "在错误的文件、章节、单元格/范围、人物或记录上操作或写入内容。",
    },
    "termination": {
        "fail_to_terminate": "成功或不可完成已经明确后，仍继续无效或重复操作。",
        "premature_completion": "仍有实质要求未满足时停止、提交或报告成功。",
        "hit_budget_limit": "任务完成前耗尽可用动作预算。",
    },
}
FINAL_LABELS = tuple(label for labels in CODEBOOK.values() for label in labels)
CATEGORY_OF = {label: category for category, labels in CODEBOOK.items() for label in labels}


def load_category_priority(taxonomy_path: Path) -> tuple[str, ...]:
    import yaml

    raw = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))
    configured = {c: set(labels) for c, labels in raw["paper_categories"].items()}
    if configured != {c: set(labels) for c, labels in CODEBOOK.items()}:
        raise AssertionError(f"codebook and {taxonomy_path} disagree on the paper categories")
    return tuple(raw["primary_selection"]["category_priority"])


def primary_label(raw: list[str], final: list[str], priority: tuple[str, ...]) -> str:
    ordered = [RENAMES.get(str(x).strip(), str(x).strip()) for x in raw]
    ordered = [x for x in dict.fromkeys(ordered) if x in final]
    for category in priority:
        for label in ordered:
            if CATEGORY_OF[label] == category:
                return label
    return ""


def distribution_rows(rows: list[dict], group: str, basis: str) -> list[dict]:
    failures = [row for row in rows if row["state"] == "failure"]
    if basis == "root_cause":
        units = [row["primary_error_type"] for row in failures]
    else:
        units = [label for row in failures for label in row["error_types"].split("|") if label]
    types = Counter(units)
    categories = Counter(CATEGORY_OF[label] for label in units)
    out = []
    for category, labels in CODEBOOK.items():
        out.append({"group": group, "basis": basis, "level": "category", "value": category,
                    "count": categories[category], "denominator": len(units),
                    "typed_failures": len(failures),
                    "share": f"{categories[category] / len(units):.6f}" if units else ""})
        for label in labels:
            out.append({"group": group, "basis": basis, "level": "error_type", "value": label,
                        "count": types[label], "denominator": len(units),
                        "typed_failures": len(failures),
                        "share": f"{types[label] / len(units):.6f}" if units else ""})
    return out

MODEL_INSIGHTS = {
    "claude_opus_4_8": (
        "当前已标失败更像“接近完成后漏字段/漏推导并提前结束”：17 条失败中 7 条已通过至少 80% 的二元 rubric。",
        "`detail_misperception` 与 `premature_completion` 占主导；当前主标注里没有 grounding/UI-element 类，不能据此推断完整 184 条都不存在该问题，因为 104 条仍是 N/A。",
    ),
    "evocua_32b": (
        "最突出的链条是错点或动作无效后不读取状态，继而长期重复；`ineffective_action` 与 grounding/UI-element 大量共现。",
        "第二类是未充分取证便填入具体数据并宣布完成；平均每条 failure 约 3 个标签，已暴露 cause、belief 与 outcome 被平铺的问题。",
    ),
    "gpt_5_5": (
        "原 failure-90 已重采并重标，184 条的有效分母是 180：153 条 perfect pass、27 条带冻结类型的 failure，另有 4 条被标注者判为 wrong rollout。",
        "`detail_misperception` 与 `ineffective_action` 并列最高频（各 10 条）：前者多是中间量都算对、最终报告漏掉 rubric 点名要求的字段；后者多是长段无状态变化的重复动作或空 tool call。",
        "`fail_to_terminate` 在当前主标注中为 0 条。旧 build 里仅因“没有 DONE/用户答复”而误标的那批已撤回，重采后的 90 条也没有产生新的该类型。",
    ),
    "kimi_k3_26": (
        "重审 wrong-rollout 后，当前统一 build 的有效分母为 178：119 条 perfect pass、59 条带冻结类型的 failure。",
        "已标 failure 仍以 scope、完成度和状态判断为主；grounding 与 incorrect-UI 类型相对少见。",
        "`hit_budget_limit` 是 termination 类的 error type；它由原始 terminal row 确定性派生，避免依赖 annotator 对预算耗尽的自由表述。",
    ),
    "opencua_72b": (
        "典型失败是“范围失控 → 进度/状态误判 → 无效动作 → 提前结束”的长链，`scope_error` 与 progress/ineffective 高频共现。",
        "平均 5.42 labels/failure 远高于其他模型，主要反映当前标注把根因和下游症状同时多标，不能直接解释成五个独立根因。",
    ),
    "qwen3_5_35b_a3b": (
        "最高频模式是“局部观察 → 无依据补全 → 错误交付”；`fabricate_data` 是第一高频标签。",
        "长任务还容易进入无状态推进循环；完成度误判明显，但真正符合严格定义的 `fail_to_terminate` 很少。",
    ),
}

EXAMPLES = {
    "claude_opus_4_8": (
        ("aggregation-f020", "claudeopus48-r1-vm1-aggregation-f020", 1,
         "画面已有 balance=$1,351、available credit=$1,649，但最终漏算 credit limit/runway 后结束。",
         "artifacts/raw_rollouts/mypcbench/v1/claude_opus_4_8/vm1/aggregation-f020/step_2_20260818@212706506142.png"),
        ("aggregation-f036", "claudeopus48-r1-vm2-aggregation-f036", 5,
         "计算流程可行，但导出 2026 数据；任务要求 last full calendar year（2025），后续结果都建立在错误时间范围上。",
         "artifacts/raw_rollouts/mypcbench/v1/claude_opus_4_8/vm2/aggregation-f036/step_6_20260809@092118900453.png"),
        ("contradiction-f014", "claudeopus48-r1-vm2-contradiction-f014", 15,
         "实际已有当年收入记录，却把 8–12 月判断为缺失并生成估算值。",
         "artifacts/raw_rollouts/mypcbench/v1/claude_opus_4_8/vm2/contradiction-f014/step_16_20260808@013359551183.png"),
    ),
    "evocua_32b": (
        ("aggregation-f005", "evocua32b-r1-vm0-aggregation-f005", 13,
         "未打开三项数据源便在 Calc 填入 hotel=1200、flight=850，随后报告完成。",
         "artifacts/raw_rollouts/mypcbench/v1/evocua_32b/repeat_1/vm0/aggregation-f005/step_29_20260811@223004912748.png"),
        ("cua_only-f002", "evocua32b-r1-vm3-cua_only-f002", 10,
         "点击非交互的 completed-status pill 后，在同一坐标重复约 110 次，流程停在 Step 4/7。",
         "artifacts/raw_rollouts/mypcbench/v1/evocua_32b/repeat_1/vm3/cua_only-f002/step_11_20260811@225841368646.png"),
        ("long_horizon-f020", "evocua32b-r1-vm1-long_horizon-f020", 23,
         "命令未落入终端、prompt 已空闲，模型仍连续等待约 96 次并声称任务仍在处理。",
         "artifacts/raw_rollouts/mypcbench/v1/evocua_32b/repeat_1/vm1/long_horizon-f020/step_25_20260811@080113896087.png"),
    ),
    "gpt_5_5": (
        ("aggregation-f020", "gpt55-recollected-v1-aggregation-f020", 52,
         "中间量都已算出，最终报告却漏掉 rubric 明确要求的 months-to-limit / N/A 结论。",
         "artifacts/raw_rollouts/mypcbench/v1/gpt5_5_rerun/gpt55_recollected_done52/vm1/aggregation-f020/step_29_20260909@125610025048.png"),
        ("hard_app-f014", "gpt55-recollected-v2-hard_app-f014", 1,
         "开局即转去翻后端文件和服务器，而不是在 HangryDash 里复购 Cooper's；这条绕路始终未纠正，耗尽了整条 rollout。",
         "artifacts/raw_rollouts/mypcbench/v1/gpt5_5_rerun/gpt55_recollected_retry38_unfinished13/vm0/hard_app-f014/step_2_20260914@135908292247.png"),
        ("situated_action-f018", "gpt55-recollected-v1-situated_action-f018", 34,
         "遇到缺货时自行下调订购数量，没有向用户确认，交付了错误的订单量。",
         "artifacts/raw_rollouts/mypcbench/v1/gpt5_5_rerun/gpt55_recollected_done52/vm2/situated_action-f018/step_35_20260909@122451299035.png"),
    ),
    "kimi_k3_26": (
        ("long_horizon-f026", "kimik326-long_horizon-f026", 19,
         "把任务要求的 HooliChat committee chat 当成可见的 HooliWork channel，并在未满足目标时 success terminate。",
         "artifacts/raw_rollouts/mypcbench/v1/kimi_k3_26/repeat_1/vm1/long_horizon-f026/step_20_20260825@022741068791.png"),
        ("long_horizon-f045", "kimik326-long_horizon-f045", 4,
         "只看到 Documents 根目录和五个未打开子目录，就声称已 visual walk 整棵目录树。",
         "artifacts/raw_rollouts/mypcbench/v1/kimi_k3_26/repeat_1/vm3/long_horizon-f045/step_5_20260825@201919274131.png"),
        ("aggregation-f040", "kimik326-aggregation-f040", 12,
         "声称数据已齐，却硬编码 improv=$120/月并漏掉已确认的 rent=$2,800/月，随后提前结束。",
         "artifacts/raw_rollouts/mypcbench/v1/kimi_k3_26/repeat_1/vm1/aggregation-f040/step_13_20260824@224529763359.png"),
    ),
    "opencua_72b": (
        ("aggregation-f001", "opencua72b-r1-vm0-aggregation-f001", 4,
         "把 upcoming flight 混入、遗漏 completed trips，又输入遗漏行的 SUM range，最终只汇总到 4,227。",
         "artifacts/raw_rollouts/mypcbench/v1/opencua_72b/vm0/aggregation-f001/step_19_20260817@061316884524.png"),
        ("long_horizon-f038", "opencua72b-r1-vm0-long_horizon-f038", 4,
         "日历时间被设为 09:00 AM–10:10 PM，又凭空写增长率和产品线；SprintBoard 未核对便结束。",
         "artifacts/raw_rollouts/mypcbench/v1/opencua_72b/vm0/long_horizon-f038/step_20_20260817@081337260717.png"),
        ("situated_action-f039", "opencua72b-r1-vm0-situated_action-f039", 15,
         "Review modal 已显示 42 件商品，模型仍反复 checkout，没有使用可见的 Deselect all，订单未提交。",
         "artifacts/raw_rollouts/mypcbench/v1/opencua_72b/vm0/situated_action-f039/step_16_20260817@175051616410.png"),
    ),
    "qwen3_5_35b_a3b": (
        ("aggregation-f001", "qwen35-r1-vm0-aggregation-f001", 22,
         "已观察到 8 个航班和 12,140 miles，随后写入未出现且越过时间窗的记录，并提交错误 SUM 范围。",
         "artifacts/raw_rollouts/mypcbench/v1/qwen3_5_35b_a3b/repeat_1/vm0/aggregation-f001/step_40_20260805@034800172493.png"),
        ("hard_app-f034", "qwen35-r1-vm0-hard_app-f034", 5,
         "任务要求 HooliCalendar，却语义上选择本地 Calendar；随后反复操作同一编辑器直到上限。",
         "artifacts/raw_rollouts/mypcbench/v1/qwen3_5_35b_a3b/repeat_1/vm0/hard_app-f034/step_6_20260805@070637348167.png"),
        ("cua_only-f023", "qwen35-r1-vm1-cua_only-f023", 26,
         "LockedIn composer 已打开，但正文落入 Firefox quick-find；观察到失败后仍未重新聚焦 textarea。",
         "artifacts/raw_rollouts/mypcbench/v1/qwen3_5_35b_a3b/repeat_1/vm1/cua_only-f023/step_27_20260805@144101697722.png"),
    ),
}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def row_for_json(path: Path) -> dict:
    row = read_json(path)
    row["_path"] = str(path.resolve())
    return row


def screenshot_at(trajectory_path: Path, action_index: int | None) -> str:
    if action_index is None or not trajectory_path.exists():
        return ""
    with trajectory_path.open(encoding="utf-8") as handle:
        for line in handle:
            action = json.loads(line)
            if action.get("action_index_global") == action_index:
                return str(action.get("observation_after_uri") or "")
    return ""


def catalogue(build_root: Path) -> tuple[dict[str, dict], list[str]]:
    items: dict[str, dict] = {}
    warnings: list[str] = []
    for spec in BUILDS:
        root = build_root / spec.name
        tasks = sorted(root.glob("canonical/*/annotation_task.json"))
        tasks += sorted(root.glob("*/canonical/*/annotation_task.json"))
        for task_path in tasks:
            task = read_json(task_path)
            tid = str(task["trajectory_id"])
            if tid in items:
                raise AssertionError(f"duplicate canonical trajectory_id: {tid}")
            trajectory_path = task_path.with_name("trajectory.jsonl")
            task_config_path = task_path.with_name("task_config.json")
            if not task_config_path.is_file():
                raise AssertionError(f"missing task_config.json: {tid}")
            items[tid] = {
                "trajectory_id": tid,
                "task_id": str(read_json(task_config_path).get("id") or tid),
                "model": spec.model,
                "source_group": spec.source_group,
                "build": spec.name,
                "annotator": spec.annotator,
                "sha256": str(task["source_trajectory_sha256"]),
                "annotation_task_path": str(task_path.resolve()),
                "trajectory_path": str(trajectory_path.resolve()),
                "task_config_path": str(task_config_path.resolve()),
                "task_config_sha256": hashlib.sha256(task_config_path.read_bytes()).hexdigest(),
            }
        if not tasks:
            warnings.append(f"no canonical annotation tasks found: {spec.name}")
    return items, warnings


def load_owned_records(directory: Path, owner_field: str, catalog: dict[str, dict]) -> tuple[dict[str, dict], Counter, list[dict]]:
    kept: dict[str, dict] = {}
    audit: Counter = Counter()
    excluded: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        record = row_for_json(path)
        tid = str(record.get("trajectory_id") or "")
        if tid not in catalog:
            audit["noncanonical_or_orphan"] += 1
            excluded.append({"path": str(path.resolve()), "trajectory_id": tid, "reason": "noncanonical_or_orphan"})
            continue
        expected = catalog[tid]
        if str(record.get("source_trajectory_sha256") or "") != expected["sha256"]:
            audit["sha_mismatch"] += 1
            excluded.append({"path": str(path.resolve()), "trajectory_id": tid, "reason": "sha_mismatch"})
            continue
        if str(record.get(owner_field) or "") != expected["annotator"]:
            audit["nonprimary_annotator"] += 1
            excluded.append({"path": str(path.resolve()), "trajectory_id": tid, "reason": "nonprimary_annotator"})
            continue
        if tid in kept:
            raise AssertionError(f"duplicate primary record in {directory}: {tid}")
        kept[tid] = record
        audit["kept"] += 1
    return kept, audit, excluded


def load_flags(directory: Path, catalog: dict[str, dict]) -> tuple[dict[str, dict], Counter]:
    flags: dict[str, dict] = {}
    audit: Counter = Counter()
    for path in sorted(directory.glob("*.json")):
        record = row_for_json(path)
        tid = str(record.get("trajectory_id") or "")
        if tid not in catalog:
            audit["noncanonical_or_orphan"] += 1
            continue
        if str(record.get("source_trajectory_sha256") or "") != catalog[tid]["sha256"]:
            audit["sha_mismatch"] += 1
            continue
        if str(record.get("annotator_id") or "") != catalog[tid]["annotator"]:
            audit["nonprimary_annotator"] += 1
            continue
        flags[tid] = record
        audit["kept"] += 1
    return flags, audit


def normalize_labels(raw: list[str]) -> tuple[list[str], list[str], list[str]]:
    final, dropped, unknown = set(), [], []
    for value in sorted(set(str(item).strip() for item in raw if str(item).strip())):
        if value in DROP_AS_FAILURE:
            dropped.append(value)
            continue
        value = RENAMES.get(value, value)
        if value not in FINAL_LABELS:
            unknown.append(value)
            continue
        final.add(value)
    if {"premature_completion", "hit_budget_limit"} <= final:
        final.discard("premature_completion")
        dropped.append("premature_completion_under_budget_limit")
    return sorted(final), dropped, unknown


def human_rubric_verdict(rubric: dict | None, task_config_path: Path) -> tuple[float, bool] | None:
    if not rubric:
        return None
    scores = rubric.get("scores")
    if not isinstance(scores, dict) or not scores:
        return None
    config = read_json(task_config_path)
    specs = list(config.get("grading", {}).get("rubrics") or [])
    rubric_ids = [f"R{index}" for index in range(1, len(specs) + 1)]
    if not rubric_ids or set(scores) != set(rubric_ids):
        raise AssertionError(f"rubric/config mismatch: {rubric.get('trajectory_id')}")
    weights: list[float] = []
    for spec in specs:
        try:
            weight = float(spec.get("weight"))
        except (TypeError, ValueError):
            weight = 1.0
        weights.append(weight if math.isfinite(weight) and weight > 0 else 1.0)
    return rubric_verdict([
        {"weight": weight, "success": bool(int(scores[rubric_id]))}
        for rubric_id, weight in zip(rubric_ids, weights)
    ])


def clean_start_pass(rows: list[dict], aggregation: dict, exclude_na: bool = False) -> PassAtK:
    runs: dict[tuple[str, str], list] = {}
    for row in rows:
        verdict = None if row["state"] == "n/a" else (
            float(row["weighted_rubric_score"]), row["rubric_all_pass"] == "true")
        runs.setdefault((row["model"], row["task_id"]), []).append(verdict)
    return pass_at_k(runs, runs, int(aggregation["repeats"]),
                     bool(aggregation["missing_counts_as_failure"]) and not exclude_na)


def classify(catalog: dict[str, dict], annotations: dict[str, dict], rubrics: dict[str, dict], flags: dict[str, dict]) -> tuple[list[dict], Counter]:
    rows: list[dict] = []
    audit: Counter = Counter()
    for tid, base in sorted(catalog.items()):
        annotation, rubric, flag = annotations.get(tid), rubrics.get(tid), flags.get(tid)
        raw = list(annotation.get("error_types") or []) if annotation else []
        labels, dropped, unknown = normalize_labels(raw)
        task_success = rubric.get("task_success") if rubric else None
        reason = ""
        if flag:
            state, reason = "n/a", "wrong_rollout"
            labels = []
        elif rubric and task_success is True and annotation:
            state, reason = "n/a", "success_failure_annotation_conflict"
            labels = []
        elif annotation and labels:
            state = "failure"
        elif annotation:
            state, reason = "n/a", "only_excluded_or_unknown_labels"
        elif rubric and task_success is True:
            state = "perfect_pass"
        elif rubric and task_success is False:
            state, reason = "n/a", "failed_but_untyped"
        else:
            state, reason = "n/a", "unreviewed"
        root = annotation.get("root_cause_action_index") if annotation else None
        trajectory_path = Path(base["trajectory_path"])
        verdict = human_rubric_verdict(rubric, Path(base["task_config_path"]))
        weighted_score = verdict[0] if verdict else None
        row = {
            **base,
            "state": state,
            "n_a_reason": reason,
            "task_success": "" if task_success is None else str(bool(task_success)).lower(),
            "weighted_rubric_score": "" if weighted_score is None else f"{weighted_score:.6f}",
            "rubric_all_pass": "" if verdict is None else str(verdict[1]).lower(),
            "error_types": "|".join(labels),
            "raw_error_types": "|".join(sorted(set(raw))),
            "dropped_labels": "|".join(dropped),
            "unknown_labels": "|".join(unknown),
            "root_cause_action_index": "" if root is None else root,
            "screenshot_at_root": screenshot_at(trajectory_path, root),
            "annotation_path": annotation.get("_path", "") if annotation else "",
            "rubric_path": rubric.get("_path", "") if rubric else "",
            "flag_path": flag.get("_path", "") if flag else "",
        }
        rows.append(row)
        audit[f"state_{state}"] += 1
        if reason:
            audit[f"n_a_{reason}"] += 1
        audit.update(f"dropped_{label}" for label in dropped)
        audit.update(f"unknown_{label}" for label in unknown)
    return rows, audit


def count_rows(rows: list[dict], selector) -> tuple[Counter, int, int]:
    chosen = [row for row in rows if selector(row)]
    counts: Counter = Counter()
    failures = 0
    for row in chosen:
        if row["state"] == "failure":
            failures += 1
            counts.update(row["error_types"].split("|"))
        elif row["state"] == "perfect_pass":
            counts["perfect_pass"] += 1
        elif row["state"] == "n/a":
            counts["n/a"] += 1
    return counts, len(chosen), failures


def mean_weighted_rubric_score(rows: list[dict], selector=lambda row: True) -> float:
    eligible = [row for row in rows if selector(row) and row["state"] != "n/a"]
    values = [float(row["weighted_rubric_score"]) for row in eligible if row["weighted_rubric_score"] != ""]
    if len(values) != len(eligible):
        missing = [row["trajectory_id"] for row in eligible if row["weighted_rubric_score"] == ""]
        raise AssertionError(f"eligible trajectories missing weighted rubric score: {missing}")
    return sum(values) / len(values) if values else float("nan")


def balanced_pilot(rows: list[dict], size: int = 100) -> list[dict]:
    by_model: dict[str, deque] = {}
    for model in sorted({row["model"] for row in rows}):
        candidates = [row for row in rows if row["model"] == model and row["annotation_path"]]
        candidates.sort(key=lambda row: row["trajectory_id"])
        by_model[model] = deque(candidates)
    sample: list[dict] = []
    while len(sample) < size and any(by_model.values()):
        for model in sorted(by_model):
            if by_model[model] and len(sample) < size:
                sample.append(by_model[model].popleft())
    return sample


def percent(count: int, denominator: int) -> str:
    return f"{100 * count / denominator:.1f}%" if denominator else "n/a"


def md_table(headers: list[str], body: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in body]
    return "\n".join(lines)


def success_cell(summary: PassAtK) -> str:
    return (f"{summary.solved_count} / {summary.unit_count} "
            f"({percent(summary.solved_count, summary.unit_count)})")


def write_report(path: Path, repo: Path, rows: list[dict], counts: dict[str, tuple[Counter, int, int]], pilot: list[dict], audit: dict, passes: dict[str, PassAtK]) -> None:
    overall, open_models, closed_models = counts["overall"], counts["open"], counts["closed"]
    success_rule = audit["success_policy"]["rule"]
    included_names = "、".join(AGENT_DISPLAY_NAMES.get(m, m) for m in INCLUDED_MODELS)
    excluded_names = "、".join(
        AGENT_DISPLAY_NAMES.get(m, m) for m in EXCLUDED_INCOMPLETE_MODELS
    ) or "（无）"
    n_included = len(INCLUDED_MODELS)
    n_pilot_models = len(PILOT_MODELS)
    pilot_names = "、".join(AGENT_DISPLAY_NAMES.get(m, m) for m in PILOT_MODELS)
    n_models = len({spec.model for spec in BUILDS})
    n_catalog = n_models * ROLLOUTS_PER_MODEL
    exclusion_clause = (
        f"主统计明确排除尚未完成标注的 {excluded_names}。"
        if EXCLUDED_INCOMPLETE_MODELS
        else "每个 build 的标注均已完成，主统计不排除任何模型。"
    )
    inventory_clause = (
        f"磁盘完整 inventory 的 `{n_models} × {ROLLOUTS_PER_MODEL} = {n_catalog}`，"
        f"其中被排除的模型各有 {ROLLOUTS_PER_MODEL} 条；{n_catalog} 不再用于主统计。"
        if EXCLUDED_INCOMPLETE_MODELS
        else f"这与磁盘完整 inventory 的 `{n_models} × {ROLLOUTS_PER_MODEL} = {n_catalog}` 一致。"
    )
    order = list(FINAL_LABELS) + ["perfect_pass"]
    count_body = []
    for label in order:
        overall_denominator = overall[1] - overall[0]["n/a"]
        open_denominator = open_models[1] - open_models[0]["n/a"]
        closed_denominator = closed_models[1] - closed_models[0]["n/a"]
        count_body.append([
            f"`{label}`", f"{overall[0][label]} / {overall_denominator} ({percent(overall[0][label], overall_denominator)})",
            f"{open_models[0][label]} / {open_denominator} ({percent(open_models[0][label], open_denominator)})",
            f"{closed_models[0][label]} / {closed_denominator} ({percent(closed_models[0][label], closed_denominator)})",
        ])
    denominator_body = []
    for group_name, summary, group_rows, group in (
        ("总体", overall, rows, "overall"),
        ("开源模型", open_models, [row for row in rows if row["source_group"] == "open"], "open"),
        ("闭源模型", closed_models, [row for row in rows if row["source_group"] == "closed"], "closed"),
    ):
        counter, raw_n, _ = summary
        n1 = counter["n/a"]
        denominator = raw_n - n1
        n2 = counter["perfect_pass"]
        denominator_body.append([
            group_name, raw_n, n1, denominator, n2,
            success_cell(passes[group]),
            f"{mean_weighted_rubric_score(group_rows):.3f}",
        ])
    model_body = []
    model_detail = []
    row_by_id = {row["trajectory_id"]: row for row in rows}
    for model in sorted({row["model"] for row in rows}):
        subset = [row for row in rows if row["model"] == model]
        states = Counter(row["state"] for row in subset)
        denominator = len(subset) - states["n/a"]
        model_body.append([
            model, len(subset), states["n/a"], denominator, states["failure"], states["perfect_pass"],
            success_cell(passes[model]), f"{mean_weighted_rubric_score(subset):.3f}",
        ])
        label_counts = Counter(
            label
            for row in subset if row["state"] == "failure"
            for label in row["error_types"].split("|") if label
        )
        assignments = sum(label_counts.values())
        top = "; ".join(f"`{label}`={count}" for label, count in label_counts.most_common(6)) or "无"
        detail = (
            f"### {model}\n\n"
            f"- 状态：raw n={len(subset)}，N1 (filtered N/A)={states['n/a']}，有效分母={denominator}；typed failure={states['failure']}，N2 (perfect pass)={states['perfect_pass']}，success rate={success_cell(passes[model])}，mean weighted rubric score={mean_weighted_rubric_score(subset):.3f}。\n"
            f"- 高频标签：{top}。\n"
            f"- 标签密度：{assignments / states['failure']:.2f} labels/typed failure。\n"
        )
        detail += "".join(f"- {insight}\n" for insight in MODEL_INSIGHTS.get(model, ()))
        examples = EXAMPLES.get(model, ())
        if examples:
            detail += "\n代表例：\n"
        for task_id, tid, action_index, evidence, screenshot_rel in examples:
            evidence_row = row_by_id[tid]
            screenshot = (repo / screenshot_rel).resolve()
            trajectory = Path(evidence_row["trajectory_path"])
            annotation = Path(evidence_row["annotation_path"])
            assert screenshot.is_file(), screenshot
            assert trajectory.is_file(), trajectory
            assert annotation.is_file(), annotation
            labels = evidence_row["error_types"] or evidence_row["raw_error_types"] or "n/a"
            detail += (
                f"\n#### {task_id} / `{tid}`\n\n"
                f"- Types：`{labels.replace('|', '`, `')}`；关键 `action_index_global={action_index}`。\n"
                f"- 证据：{evidence}\n"
                f"- [screenshot]({screenshot}) · [trajectory.jsonl]({trajectory}) · [annotation]({annotation})\n"
            )
        model_detail.append(detail + "\n")
    agent_body = []
    for model in INCLUDED_MODELS:
        subset = [row for row in rows if row["model"] == model]
        states = Counter(row["state"] for row in subset)
        denominator = len(subset) - states["n/a"]
        agent_body.append([
            AGENT_DISPLAY_NAMES[model], len(subset), states["n/a"], denominator, states["failure"], states["perfect_pass"],
            success_cell(passes[model]), f"{mean_weighted_rubric_score(subset):.3f}",
        ])
    inventory_body = [
        [item["build"], item["model"], item["status"], item["canonical_count"]]
        for item in audit["inventory_by_build"]
    ]
    taxonomy_body = []
    for category, labels in CODEBOOK.items():
        for label, description in labels.items():
            taxonomy_body.append([category, f"`{label}`", description])
    pilot_raw = Counter()
    pilot_final = Counter()
    for row in pilot:
        pilot_raw.update(row["raw_error_types"].split("|") if row["raw_error_types"] else [])
        pilot_final.update(row["error_types"].split("|") if row["error_types"] else ["n/a"])
    pilot_final_labels = sorted(label for label in pilot_final if label in set(FINAL_LABELS))
    pilot_unobserved = sorted(set(FINAL_LABELS) - set(pilot_final_labels))
    pilot_unobserved_note = (
        "" if not pilot_unobserved
        else "；未在 pilot 窗口内出现的是 "
        + "、".join(f"`{label}`" for label in pilot_unobserved)
        + "（该类型仅存在于 pilot 抽样窗口之外的轨迹中）"
    )
    na_reasons = Counter(row["n_a_reason"] for row in rows if row["state"] == "n/a")
    report = f"""# DERAIL error taxonomy 与 multi-label counts

由 `analysis/derail_error_taxonomy/analyze.py` 从当前非 backup artifacts 生成。快照时间：{audit['generated_at_utc']}；输入摘要：`{audit['input_digest_sha256']}`。

## 统计口径

- {exclusion_clause}raw canonical pool 是 {overall[1]} 条（开源 {open_models[1]}、闭源 {closed_models[1]}）；过滤 N/A 后的有效统计集是 {overall[1] - overall[0]['n/a']} 条（开源 {open_models[1] - open_models[0]['n/a']}、闭源 {closed_models[1] - closed_models[0]['n/a']}）。
- 分母来源：{included_names} 各 {ROLLOUTS_PER_MODEL} 条，所以 `n = {n_included} × {ROLLOUTS_PER_MODEL} = {overall[1]}`。{inventory_clause}
- 一个标签的 count 是“带该标签的不同轨迹数”；同一轨迹的同一标签最多计一次。标签可多选，因此各行不会加总为分母。
- 每条 trajectory 先且只能归入一个结果状态：`failure`、`perfect_pass` 或 `n/a`。记 raw canonical 数为 `n`，过滤掉的 `n/a` 数为 `N1`，有效统计分母为 `n−N1`；记 `perfect_pass` 数为 `N2`。Success rate：{success_rule}。只有 `failure` 才继续选择一个或多个 error types。
- 先应用仓库已记录的主标注者归属，再统计。SHA 不匹配、orphan 和非主标注者提交只进入审计，不静默合并。

`{n_catalog}` 的逐 build 来源（直接数每个非 backup build 下的 canonical `annotation_task.json`）。GPT-5.5 的 184 条分布在 4 个 build 上：重跑前保留的 94 条，加上分三批重采的原 failure-90。

{md_table(["Build", "模型", "主统计状态", "Canonical n"], inventory_body)}

### 分母、N1、N2 与 success rate

{md_table(["分组", "Raw n", "N1: filtered N/A", "有效分母 n−N1", "N2: perfect pass", "Success rate", "Mean weighted rubric score [0,1]"], denominator_body)}

{md_table(["Error type / 状态", "总体", "开源模型", "闭源模型"], count_body)}

上表所有 rate 均已过滤 N/A，分母分别是总体 {overall[1] - overall[0]['n/a']}、开源 {open_models[1] - open_models[0]['n/a']}、闭源 {closed_models[1] - closed_models[0]['n/a']}。若分析“已完成实质 failure typing 的组成”，条件分母分别是总体 {overall[2]}、开源 {open_models[2]}、闭源 {closed_models[2]}。

**终止标签审计：**表中的 `fail_to_terminate` 使用专项审计后的当前人工标签。此前因“没有 DONE/用户答复”而误标的 12 条已经撤回；单纯触及动作预算也不会自动得到该类型。预算耗尽单列为 termination 类的 `hit_budget_limit`，与“成功或不可能已经明确后仍继续”的 `fail_to_terminate` 分开统计。

### 各模型覆盖

{md_table(["模型", "Raw n", "N1: filtered N/A", "有效分母", "Typed failure", "N2: perfect pass", "Success rate", "Mean weighted rubric score [0,1]"], model_body)}

N/A 原因：{', '.join(f'`{key}`={value}' for key, value in sorted(na_reasons.items())) or 'none'}。

进入主统计的 {n_included} 个 agent：

{md_table(["Agent", "Raw n", "N1: filtered N/A", "有效分母", "Typed failure", "N2: perfect pass", "Success rate", "Mean weighted rubric score [0,1]"], agent_body)}

Weighted rubric score 是逐 trajectory 的派生量：`Σ(rubric pass × weight) / Σ(weight)`，范围为 `[0,1]`。表中报告过滤 N/A 后的 trajectory-level macro mean；逐条分数保存在 `trajectory_labels.csv`。

## Taxonomy 建议

参考基线包括用户提供的 GUI-RobustEval Table 6 截图、当前 annotation UI，以及本地 `_taxonomy_open_coding_db548656.yaml` 和 `human_labels/taxonomy/open_coded_labels.json`。GUI-RobustEval 原论文为 [Recovering Policy-Induced Errors](https://arxiv.org/abs/2605.29447)。

这里要避免符号混淆：`n={overall[1]}` 是 raw canonical 数；`N1={overall[0]['n/a']}` 和 `N2={overall[0]['perfect_pass']}` 是上面定义的状态 count；`N_taxonomy={len(FINAL_LABELS)}` 才是 taxonomy 类别数。可辩护的流程应明确写为：**我们先在 100 条轨迹上开放编码，归并观测标签得到 N_taxonomy 类，冻结带版本号的 codebook，再用固定 codebook 标注剩余样本。** 对 codebook 冻结时已有 open-coding 提交的 {n_pilot_models} 个模型（{pilot_names}）做确定性的模型平衡回溯 pilot：每个模型 {PILOT_SIZE // n_pilot_models} 条，共 {len(pilot)} 条轨迹。后并入的 Claude-Opus-4.8 是对着已冻结的 codebook 标注的（raw label 无需 rename、未触发 drop），不参与 pilot，以免回溯改写 codebook 的形成过程。pilot 观测到 {len(pilot_raw)} 个 raw labels；删除 `gui_workflow_bypass`，将 `wrong_subgoal` 归入已存在的 `misunderstand_task_objective`，并将 `section_content_misplacement` 重命名为 `wrong_target`，得到冻结 codebook 的 **N_taxonomy={len(FINAL_LABELS)} 个 provisional error types**，其中 pilot 实际观测到 {len(pilot_final_labels)} 个{pilot_unobserved_note}。完整 raw/final label 列表保存在 `audit.json`。这是来自仓库的可复现证据，但不能替代前瞻冻结的 pilot 和独立双人标注。

{md_table(["阶段", "最终类型", "操作性定义"], taxonomy_body)}

`n/a` 只作为 N1 保留在 coverage/audit 中，并从所有 error-type rate 的分母过滤（success rate 的口径见上）；它不计入 taxonomy 的 N_taxonomy。它用于 invalid/wrong rollout、尚未完成 review、关键观察缺失、证据无法裁决，或 adjudication 后只剩被删除标签的 failure；成功轨迹使用 `perfect_pass`。

### 与参考截图 taxonomy 的逐项处理

| 参考类型 | 建议 | 理由/边界 |
| --- | --- | --- |
| Incorrect UI Element | 保留 | 仅指语义上选错控件；不包含坐标落点错误。 |
| Grounding Failure | 保留 | 语义目标正确但坐标、焦点或拖拽失败；与上一项在同一根因 action 上原则上互斥。 |
| Ineffective Action | 降为 secondary/residual | 它描述“没有状态变化”的结果，不是与原因同层的机制；有更具体原因时不能用它替代根因。 |
| Typing Error / Incorrect Parameter | 当前暂合并，下一轮重标拆分 | 二者机制不同，但当前 `typing_or_parameter_error` 不能可靠地事后自动拆开。 |
| Miss Necessary Step | 作为 pilot 候选，不直接并入 v1 | 它是 omission mechanism；不能用 `premature_completion` 或 `progress_misperception` 代替。只有 100 条 adjudication 显示可稳定区分时才加入。 |
| Incorrect Tool Usage | 收紧后再考虑加入 | 非 GUI、API、shell 本身不是 failure。只有无效/不支持/上下文不适用的工具选择造成实质错误时才可标。 |
| Wrong Target | 保留为 `wrong_target` | 错文件、对象、记录、章节或单元格；与错误控件和错误范围分开。 |
| Misunderstand Task Objective | 保留 | 仅用于最终目标/交付物/关键约束被根本误解，不能泛化为任意局部错误。 |
| Fail to Terminate | 保留但收紧 | 必须是成功或不可完成已经明确后仍继续；预算截断、没保存、没有 DONE 都不自动属于此类。 |
| Lack of Knowledge | 保留但要求行为证据 | 不能仅从失败结果推断 latent knowledge；证据不足时标可观察机制或 `n/a`。 |

当前数据还实证支持参考截图没有单列的 `fabricate_data`、三类 perception、`scope_error` 与 `premature_completion`；它们在长轨迹中频繁出现，不建议删掉。

## 各模型 failure profile

{''.join(model_detail)}

### 冻结前必须修正

1. 从 failure taxonomy 删除 `gui_workflow_bypass`。API、shell、browser scripting 或其他非 GUI 路径本身不是失败；应标最早独立可观察的实质错误。
2. 将 `ineffective_action` 作为无进展结果/残余类；已有 grounding、控件选择、状态或参数根因时，不再把它当同层根因。
3. 固定边界：错语义控件 → `incorrect_ui_element`；正确语义目标但坐标/焦点错 → `grounding_failure`；错文件/记录/章节 → `wrong_target`；错总体/时间/对象覆盖 → `scope_error`。
4. 分开进度信念与终止：`progress_misperception` 是中间信念，`premature_completion` 是要求未满足便停止，`fail_to_terminate` 是成功/不可完成已清楚却仍继续。
5. 当前数据不能支持把 `typing_or_parameter_error` 回溯拆成两个可靠 count；本次保留临时合并类，只能通过重标拆分。
6. 把 `miss_necessary_step` 作为下一轮 pilot 候选；它是遗漏机制，不是 premature completion 的同义词。
7. 删除类做显式迁移：`hallucinated_data` → `fabricate_data`，`constraint_loss` → `scope_error`，稀有 `wrong_subgoal` → `misunderstand_task_objective`；记录 `section_content_misplacement` → `wrong_target` 的重命名。
8. 冻结单一 taxonomy snapshot。当前服务仍是 `draft/open_coding`，manifest/API seed 与浏览器实际分组对 `wrong_subgoal`/`fabricate_data` 还不一致，registry 甚至仍把 `gui_workflow_bypass` 作为 active label；不冻结就无法复现 count。

## 质量限制

- 当前 annotations 使用 draft/open-coding taxonomy；尽管任务文件要求 `minimum_independent_annotations=2`，多数模型实际只有一位标注者。因此这里只能给描述性 count，不能声称 inter-annotator agreement。
- 开源/闭源比较对 N/A 覆盖非常敏感。未控制 review 完成度前，较低 raw label rate 不能解释成模型更好。
- Annotation rationale 的 `Action #N` 可能是 1-based 展示编号；导出的证据表统一使用 `trajectory.jsonl` 内 0-based `action_index_global`。

## 可复现审计

详见 `audit.json` 与 `trajectory_labels.csv`。关键状态审计：{json.dumps(audit['classification'], sort_keys=True)}。
"""
    path.write_text(report, encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--taxonomy", type=Path, default=Path("configs/synthesis/failure_taxonomy_v0.1.yaml"),
                        help="repo-relative failure-taxonomy config (primary-type priority)")
    parser.add_argument("--distribution", choices=("root_cause", "label_share"), default="root_cause",
                        help="basis of error_category_distribution.csv and the report headline")
    parser.add_argument("--judge-config", type=Path, default=Path("configs/judges/default.yaml"),
                        help="repo-relative; its aggregation section sets repeats / missing policy")
    parser.add_argument("--exclude-na", action="store_true",
                        help="former success rate perfect_pass / (n − n/a) instead of n/a as failure")
    args = parser.parse_args()
    import yaml

    aggregation = yaml.safe_load((args.repo / args.judge_config).read_text(encoding="utf-8"))["aggregation"]
    priority = load_category_priority(args.repo / args.taxonomy)
    build_root = args.repo / "artifacts" / "derail_builds"
    catalog, warnings = catalogue(build_root)
    annotations, annotation_audit, excluded_annotations = load_owned_records(build_root / "human_labels", "annotator_id", catalog)
    rubrics, rubric_audit, excluded_rubrics = load_owned_records(build_root / "human_labels" / "rubric_scores", "reviewer_id", catalog)
    flags, flag_audit = load_flags(build_root / "human_labels" / "rollout_flags", catalog)
    inventory_rows, inventory_classification_audit = classify(catalog, annotations, rubrics, flags)
    rows = [row for row in inventory_rows if row["model"] in INCLUDED_MODELS]
    for row in rows:
        raw = list(annotations.get(row["trajectory_id"], {}).get("error_types") or [])
        primary = (
            primary_label(raw, row["error_types"].split("|"), priority)
            if row["state"] == "failure" else ""
        )
        row["primary_error_type"] = primary
        row["primary_error_category"] = CATEGORY_OF.get(primary, "")
    classification_audit: Counter = Counter()
    for row in rows:
        classification_audit[f"state_{row['state']}"] += 1
        if row["n_a_reason"]:
            classification_audit[f"n_a_{row['n_a_reason']}"] += 1
        classification_audit.update(
            f"dropped_{label}" for label in row["dropped_labels"].split("|") if label
        )
        classification_audit.update(
            f"unknown_{label}" for label in row["unknown_labels"].split("|") if label
        )
    counts = {
        "overall": count_rows(rows, lambda row: True),
        "open": count_rows(rows, lambda row: row["source_group"] == "open"),
        "closed": count_rows(rows, lambda row: row["source_group"] == "closed"),
    }
    pilot = balanced_pilot(
        [row for row in rows if row["model"] in PILOT_MODELS], size=PILOT_SIZE
    )
    assert len({spec.model for spec in BUILDS}) == 6 and len(catalog) == 1104, (
        len(BUILDS), len(catalog)
    )
    assert len(rows) == ROLLOUTS_PER_MODEL * len(INCLUDED_MODELS), len(rows)
    assert counts["overall"][1] == counts["open"][1] + counts["closed"][1]
    assert all(value <= counts["overall"][1] for value in counts["overall"][0].values())
    for counter, raw_n, failures in counts.values():
        assert failures + counter["perfect_pass"] == raw_n - counter["n/a"]
    for row in rows:
        if row["state"] == "perfect_pass":
            assert float(row["weighted_rubric_score"]) == 1.0
        elif row["state"] == "failure":
            assert 0.0 <= float(row["weighted_rubric_score"]) < 1.0
    pilot_raw_labels = sorted({
        label
        for row in pilot
        for label in row["raw_error_types"].split("|")
        if label
    })
    pilot_final_labels = sorted({
        label
        for row in pilot
        for label in row["error_types"].split("|")
        if label
    })
    assert len(pilot) == PILOT_SIZE
    assert PILOT_SIZE % len(PILOT_MODELS) == 0, PILOT_SIZE
    assert Counter(row["model"] for row in pilot) == Counter(
        {model: PILOT_SIZE // len(PILOT_MODELS) for model in PILOT_MODELS}
    )
    assert set(pilot_final_labels) <= set(FINAL_LABELS), (
        sorted(set(pilot_final_labels) - set(FINAL_LABELS))
    )
    pilot_unobserved = sorted(set(FINAL_LABELS) - set(pilot_final_labels))
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "trajectory_labels.csv", rows)
    weighted_means = {
        "overall": mean_weighted_rubric_score(rows),
        "open": mean_weighted_rubric_score(rows, lambda row: row["source_group"] == "open"),
        "closed": mean_weighted_rubric_score(rows, lambda row: row["source_group"] == "closed"),
    }
    passes = {
        "overall": clean_start_pass(rows, aggregation, args.exclude_na),
        "open": clean_start_pass([r for r in rows if r["source_group"] == "open"], aggregation, args.exclude_na),
        "closed": clean_start_pass([r for r in rows if r["source_group"] == "closed"], aggregation, args.exclude_na),
        **{model: clean_start_pass([r for r in rows if r["model"] == model], aggregation, args.exclude_na)
           for model in sorted({row["model"] for row in rows})},
    }
    count_rows_csv = []
    for label in list(FINAL_LABELS) + ["perfect_pass"]:
        row = {"error_type_or_state": label}
        for group in ("overall", "open", "closed"):
            counter, raw_n, failures = counts[group]
            n1 = counter["n/a"]
            denominator = raw_n - n1
            n2 = counter["perfect_pass"]
            row[f"{group}_count"] = counter[label]
            row[f"{group}_raw_canonical_n"] = raw_n
            row[f"{group}_n1_filtered_n_a"] = n1
            row[f"{group}_denominator"] = denominator
            row[f"{group}_rate"] = f"{counter[label] / denominator:.6f}"
            row[f"{group}_rate_of_typed_failure"] = (
                f"{counter[label] / failures:.6f}" if failures else ""
            )
            row[f"{group}_n2_perfect_pass"] = n2
            row[f"{group}_success_rate"] = f"{passes[group].pass_at_k:.6f}"
            row[f"{group}_mean_weighted_rubric_score"] = f"{weighted_means[group]:.6f}"
            row[f"{group}_typed_failure_denominator"] = failures
        count_rows_csv.append(row)
    write_csv(args.out / "error_type_counts.csv", count_rows_csv)
    by_model_csv = []
    for model in sorted({row["model"] for row in rows}):
        counter, raw_n, failures = count_rows(rows, lambda row, model=model: row["model"] == model)
        n1 = counter["n/a"]
        denominator = raw_n - n1
        n2 = counter["perfect_pass"]
        weighted_mean = mean_weighted_rubric_score(rows, lambda row, model=model: row["model"] == model)
        for label in list(FINAL_LABELS) + ["perfect_pass"]:
            by_model_csv.append({
                "model": model,
                "error_type_or_state": label,
                "count": counter[label],
                "raw_canonical_n": raw_n,
                "n1_filtered_n_a": n1,
                "denominator": denominator,
                "rate": f"{counter[label] / denominator:.6f}",
                "n2_perfect_pass": n2,
                "success_rate": f"{passes[model].pass_at_k:.6f}",
                "mean_weighted_rubric_score": f"{weighted_mean:.6f}",
                "typed_failure_denominator": failures,
                "conditional_failure_rate": (
                    f"{counter[label] / failures:.6f}"
                    if failures and label != "perfect_pass" else ""
                ),
            })
    write_csv(args.out / "error_type_counts_by_model.csv", by_model_csv)
    groups = [("overall", rows),
              ("open", [row for row in rows if row["source_group"] == "open"]),
              ("closed", [row for row in rows if row["source_group"] == "closed"])]
    groups += [(model, [row for row in rows if row["model"] == model]) for model in INCLUDED_MODELS]
    write_csv(args.out / "root_cause_distribution.csv",
              [item for name, sub in groups for item in distribution_rows(sub, name, "root_cause")])
    write_csv(args.out / "error_category_distribution.csv",
              [item for name, sub in groups for item in distribution_rows(sub, name, args.distribution)])
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "warnings": warnings,
        "canonical_trajectories_inventory": len(catalog),
        "analysis_trajectories": len(rows),
        "included_models": list(INCLUDED_MODELS),
        "excluded_incomplete_models": list(EXCLUDED_INCOMPLETE_MODELS),
        "annotation_records": dict(annotation_audit),
        "rubric_records": dict(rubric_audit),
        "flag_records": dict(flag_audit),
        "classification": dict(classification_audit),
        "inventory_classification": dict(inventory_classification_audit),
        "denominator_policy": {
            group: {
                "raw_n": summary[1],
                "n1_filtered_n_a": summary[0]["n/a"],
                "denominator_n_minus_n1": summary[1] - summary[0]["n/a"],
                "n2_perfect_pass": summary[0]["perfect_pass"],
                "success_rate": passes[group].pass_at_k,
                "success_denominator": passes[group].unit_count,
                "mean_weighted_rubric_score": weighted_means[group],
            }
            for group, summary in counts.items()
        },
        "success_policy": {
            "rule": (
                "perfect_pass / (n − n/a)（--exclude-na）" if args.exclude_na else
                f"Pass@{aggregation['repeats']} over every task; n/a counts as a failure"
                if aggregation["missing_counts_as_failure"] else
                f"Pass@{aggregation['repeats']}; tasks without a valid verdict excluded"
            ),
            "repeats": aggregation["repeats"],
            "missing_counts_as_failure": aggregation["missing_counts_as_failure"],
            "exclude_na": args.exclude_na,
        },
        "excluded_annotations": excluded_annotations,
        "excluded_rubrics": excluded_rubrics,
        "pilot_trajectory_ids": [row["trajectory_id"] for row in pilot],
        "pilot_summary": {
            "sample_size": len(pilot),
            "sampling_rule": "round-robin over trajectory_id-sorted annotated trajectories",
            "model_counts": dict(Counter(row["model"] for row in pilot)),
            "raw_label_count": len(pilot_raw_labels),
            "raw_labels": pilot_raw_labels,
            "observed_final_label_count": len(pilot_final_labels),
            "observed_final_labels": pilot_final_labels,
            "codebook_labels_not_observed_in_pilot": pilot_unobserved,
            "observed_dropped_labels": sorted(set(pilot_raw_labels) & DROP_AS_FAILURE),
            "observed_rename_rules": {
                source: target for source, target in RENAMES.items() if source in pilot_raw_labels
            },
        },
        "inventory_by_build": [
            {
                "build": spec.name,
                "model": spec.model,
                "status": "included" if spec.model in INCLUDED_MODELS else "excluded_incomplete",
                "canonical_count": sum(1 for item in catalog.values() if item["build"] == spec.name),
            }
            for spec in BUILDS
        ],
    }
    digest_payload = {
        "catalog": catalog,
        "annotations": annotations,
        "rubrics": rubrics,
        "flags": flags,
    }
    audit["input_digest_sha256"] = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.out / "report.md", args.repo.resolve(), rows, counts, pilot, audit, passes)
    headline = []
    for basis in ("root_cause", "label_share"):
        body = [
            [item["value"], item["count"], item["denominator"],
             percent(item["count"], item["denominator"])]
            for item in distribution_rows(rows, "overall", basis) if item["level"] == "category"
        ]
        title = ("每条失败取一个 primary type（论文口径：% of root causes）" if basis == "root_cause"
                 else "多标签标签占比（仅供参考）")
        marker = "（默认）" if basis == args.distribution else ""
        rule = f"primary 规则：{' > '.join(priority)}。\n\n" if basis == "root_cause" else ""
        headline.append(f"### {title}{marker}\n\n{rule}"
                        + md_table(["Category", "Count", "Denominator", "Share"], body))
    with (args.out / "report.md").open("a", encoding="utf-8") as handle:
        handle.write("\n## 错误类别分布\n\n" + "\n\n".join(headline) + "\n")
    print(args.out / "report.md")


if __name__ == "__main__":
    main()
