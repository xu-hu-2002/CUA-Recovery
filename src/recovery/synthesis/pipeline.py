"""Deterministic, verifier-first long-horizon task synthesis pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from recovery.synthesis.compatibility import TypeSystem, build_compatibility_edges
from recovery.synthesis.graph import (
    SynthesisValidationError,
    compose_modules,
    compute_complexity,
    validate_grounded_module,
)
from recovery.synthesis.sampling import compute_skeleton_weights


REJECTION_CODES = frozenset(
    {
        "UNBOUND_PORT",
        "INVALID_GROUNDING",
        "TYPE_MISMATCH",
        "GLOBAL_CONSTRAINT_UNSAT",
        "WORLD_NOT_MATERIALIZABLE",
        "FIXTURE_RESET_FAILED",
        "VERIFIER_INVALID",
        "REALIZATION_DRIFT",
        "INSUFFICIENT_DEPENDENCY",
        "SHORTCUT_DETECTED",
        "DUPLICATE_TASK",
        "NEEDS_HUMAN_REVIEW",
    }
)


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _range(raw: Sequence[Any], field: str) -> Tuple[float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("%s must contain [minimum, maximum]" % field)
    lower, upper = float(raw[0]), float(raw[1])
    if lower > upper:
        raise ValueError("%s minimum exceeds maximum" % field)
    return lower, upper


@dataclass(frozen=True)
class SynthesisConfig:
    generation_version: str
    target_count: int
    target_environment: str
    target_snapshot: str
    source_split: str
    sampling_mode: str
    seed: int
    max_attempts: int
    beam_width: int
    min_modules: int
    max_modules: int
    dependency_depth: Tuple[float, float]
    cross_app_dependencies: Tuple[float, float]
    max_independent_component_ratio: float
    required_delayed_reuse: bool
    max_irreversible_actions: int
    min_verifier_coverage: float
    failure_reweighting: Mapping[str, Any]
    type_system: TypeSystem

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SynthesisConfig":
        mode = str(raw.get("sampling_mode", "empirical_only"))
        if mode not in {"empirical_only", "failure_informed_bounded"}:
            raise ValueError("unknown sampling_mode %s" % mode)
        config = cls(
            generation_version=str(raw.get("generation_version", "v0.1")),
            target_count=int(raw.get("target_count", 1)),
            target_environment=str(raw["target_environment"]),
            target_snapshot=str(raw["target_snapshot"]),
            source_split=str(raw.get("source_split", "source-development")),
            sampling_mode=mode,
            seed=int(raw.get("seed", 0)),
            max_attempts=int(raw.get("max_attempts", 100)),
            beam_width=int(raw.get("beam_width", 16)),
            min_modules=int(raw.get("min_modules", 2)),
            max_modules=int(raw.get("max_modules", 4)),
            dependency_depth=_range(raw.get("dependency_depth", (2, 12)), "dependency_depth"),
            cross_app_dependencies=_range(
                raw.get("cross_app_dependencies", (1, 12)), "cross_app_dependencies"
            ),
            max_independent_component_ratio=float(
                raw.get("max_independent_component_ratio", 0.25)
            ),
            required_delayed_reuse=bool(raw.get("required_delayed_reuse", False)),
            max_irreversible_actions=int(raw.get("max_irreversible_actions", 1)),
            min_verifier_coverage=float(raw.get("min_verifier_coverage", 1.0)),
            failure_reweighting=dict(raw.get("failure_reweighting", {})),
            type_system=TypeSystem.from_config(raw.get("type_system", {})),
        )
        if config.target_count <= 0 or config.max_attempts <= 0 or config.beam_width <= 0:
            raise ValueError("target_count, max_attempts, and beam_width must be positive")
        if config.min_modules < 2 or config.max_modules < config.min_modules:
            raise ValueError("module bounds are invalid")
        if not 0 <= config.max_independent_component_ratio <= 1:
            raise ValueError("max_independent_component_ratio must be in [0, 1]")
        if not 0 <= config.min_verifier_coverage <= 1:
            raise ValueError("min_verifier_coverage must be in [0, 1]")
        return config


@dataclass(frozen=True)
class _SearchState:
    module_ids: Tuple[str, ...]
    compatibility_edge_ids: Tuple[str, ...]


@dataclass(frozen=True)
class SynthesisResult:
    accepted: Tuple[Dict[str, Any], ...]
    rejected: Tuple[Dict[str, Any], ...]
    compatibility_edges: Tuple[Dict[str, Any], ...]
    skeleton_weights: Mapping[str, Mapping[str, float]]


def _rule_accepts_binding(rule: Mapping[str, Any], binding: Mapping[str, Any]) -> bool:
    fields = {
        "producer_op": "source_op",
        "consumer_op": "target_op",
        "value_type": "unified_type",
    }
    return all(
        rule.get(rule_field) in {None, "*", binding.get(binding_field)}
        for rule_field, binding_field in fields.items()
    )


def _filter_edge_for_skeleton(
    edge: Mapping[str, Any], skeleton: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    rules = list(skeleton.get("composition_rules", ()))
    allowed_rules = [
        rule
        for rule in rules
        if int(rule.get("support_task_count", 0)) >= 3 or bool(rule.get("approved"))
    ]
    if not allowed_rules:
        return None
    bindings = [
        dict(binding)
        for binding in edge.get("bindings", ())
        if any(_rule_accepts_binding(rule, binding) for rule in allowed_rules)
    ]
    if not bindings:
        return None
    filtered = dict(edge)
    filtered["bindings"] = bindings
    return filtered


def _weighted_choice(
    rng: random.Random,
    values: Sequence[Mapping[str, Any]],
    weights: Mapping[str, Mapping[str, float]],
) -> Mapping[str, Any]:
    threshold = rng.random()
    cumulative = 0.0
    for value in values:
        cumulative += weights[str(value["skeleton_id"])]["sampling_probability"]
        if threshold <= cumulative:
            return value
    return values[-1]


def _state_score(
    state: _SearchState,
    edge_by_id: Mapping[str, Mapping[str, Any]],
) -> float:
    edge_scores = [
        float(edge_by_id[edge_id]["compatibility_score"])
        for edge_id in state.compatibility_edge_ids
    ]
    return len(state.module_ids) + sum(edge_scores) / max(1, len(edge_scores))


def _required_accepts_bound(
    modules: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]
) -> bool:
    bound = {
        (str(edge["to_module"]), str(binding["input_node_id"]), str(binding["input_port"]))
        for edge in edges
        for binding in edge.get("bindings", ())
    }
    anchor_id = str(modules[0]["module_id"])
    for module in modules:
        module_id = str(module["module_id"])
        if module_id == anchor_id:
            continue
        for port in module["interface"].get("accepts", ()):
            if port.get("required") and (
                module_id,
                str(port["node_id"]),
                str(port["port_id"]),
            ) not in bound:
                return False
    return True


def _verifier_coverage(fragment: Mapping[str, Any]) -> float:
    weights = []
    verified = []
    for node in fragment["nodes"]:
        if not bool(node.get("critical", True)):
            continue
        weight = (
            2.0
            if node["op"] in {"decide", "create", "modify", "delete", "communicate"}
            else 1.0
        )
        weights.append(weight)
        if isinstance(node.get("verifier"), Mapping) and node["verifier"].get("verifier_id"):
            verified.append(weight)
    return sum(verified) / sum(weights) if weights else 1.0


def _derive_rubric(fragment: Mapping[str, Any]) -> Dict[str, Any]:
    incoming: Dict[str, List[str]] = {str(node["node_id"]): [] for node in fragment["nodes"]}
    outgoing: Dict[str, List[str]] = {str(node["node_id"]): [] for node in fragment["nodes"]}
    for edge in fragment["edges"]:
        source = str(edge["from"]["node_id"])
        target = str(edge["to"]["node_id"])
        incoming[target].append(source)
        outgoing[source].append(target)
    indegree = {node_id: len(predecessors) for node_id, predecessors in incoming.items()}
    ready = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
    topological = []
    while ready:
        node_id = ready.pop(0)
        topological.append(node_id)
        for target in sorted(outgoing[node_id]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    node_by_id = {str(node["node_id"]): node for node in fragment["nodes"]}
    milestones = []
    milestone_by_node = {}
    for index, node_id in enumerate(topological, start=1):
        node = node_by_id[node_id]
        milestone_id = "m%d" % index
        milestone_by_node[str(node["node_id"])] = milestone_id
        verifier = node.get("verifier", {})
        milestones.append(
            {
                "milestone_id": milestone_id,
                "node_id": node["node_id"],
                "semantic_goal": node.get("semantic_goal", "%s in %s" % (node["op"], node["app"])),
                "depends_on_nodes": sorted(set(incoming[str(node["node_id"])])),
                "observability": verifier.get("observability", "unobservable"),
                "verifier_id": verifier.get("verifier_id"),
                "weight": 2.0
                if node["op"] in {"decide", "create", "modify", "delete", "communicate"}
                else 1.0,
                "critical": bool(node.get("critical", True)),
                "credit_policy": "no_credit_if_prerequisite_value_is_wrong",
            }
        )
    for milestone in milestones:
        milestone["depends_on"] = [
            milestone_by_node[node_id] for node_id in milestone.pop("depends_on_nodes")
        ]
    return {
        "schema_version": "rubric/0.1",
        "milestones": milestones,
        "scoring": "dependency_consistent_weighted_completion",
    }


class SynthesisPipeline:
    """Search compatible module grafts and retain only hard-valid candidates."""

    def __init__(
        self,
        modules: Iterable[Mapping[str, Any]],
        skeletons: Iterable[Mapping[str, Any]],
        config: SynthesisConfig,
        human_failure_stats: Iterable[Mapping[str, Any]] = (),
        excluded_model_id: Optional[str] = None,
    ):
        self.config = config
        self.modules = tuple(dict(module) for module in modules)
        supplied_skeletons = tuple(dict(skeleton) for skeleton in skeletons)
        self.skeletons = tuple(
            skeleton
            for skeleton in supplied_skeletons
            if skeleton.get("review_status") == "human_approved"
        )
        if not self.modules or not self.skeletons:
            raise ValueError("modules and human-approved skeletons must be non-empty")
        for module in self.modules:
            validate_grounded_module(module)
        self.module_by_id = {str(module["module_id"]): module for module in self.modules}
        if len(self.module_by_id) != len(self.modules):
            raise ValueError("module_id values must be unique")
        if len({str(item["skeleton_id"]) for item in self.skeletons}) != len(self.skeletons):
            raise ValueError("skeleton_id values must be unique")
        self.compatibility_edges = tuple(
            build_compatibility_edges(self.modules, config.type_system)
        )
        reweight = config.failure_reweighting
        self.skeleton_weights = compute_skeleton_weights(
            self.skeletons,
            human_failure_stats,
            enabled=config.sampling_mode == "failure_informed_bounded",
            alpha=float(reweight.get("alpha", 1.0)),
            beta=float(reweight.get("beta", 1.0)),
            min_exposures=int(reweight.get("min_exposures", 10)),
            adjustment_min=float(reweight.get("adjustment_range", (0.5, 2.0))[0]),
            adjustment_max=float(reweight.get("adjustment_range", (0.5, 2.0))[1]),
            excluded_model_id=excluded_model_id,
        )

    def _candidate_modules(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(
            module
            for module in self.modules
            if module["environment_id"] == self.config.target_environment
            and module["snapshot_id"] == self.config.target_snapshot
            and module["source_split"] == self.config.source_split
            and module.get("provenance", {}).get("review_status") == "human_verified"
        )

    def _search(
        self,
        skeleton: Mapping[str, Any],
        anchor: Mapping[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        usable_edges = []
        for edge in self.compatibility_edges:
            if edge["environment_status"] != "direct" or edge["conflicts"]:
                continue
            filtered = _filter_edge_for_skeleton(edge, skeleton)
            if filtered is not None:
                usable_edges.append(filtered)
        edge_by_id = {str(edge["edge_id"]): edge for edge in usable_edges}
        frontier = [_SearchState((str(anchor["module_id"]),), ())]
        seen: Set[Tuple[Tuple[str, ...], Tuple[str, ...]]] = set()
        best_reason = "no compatible composition-rule edge"
        max_rounds = self.config.max_modules * 3

        for _ in range(max_rounds):
            expanded: List[_SearchState] = []
            for state in frontier:
                modules = [self.module_by_id[module_id] for module_id in state.module_ids]
                selected_edges = [edge_by_id[edge_id] for edge_id in state.compatibility_edge_ids]
                if len(modules) >= self.config.min_modules and _required_accepts_bound(
                    modules, selected_edges
                ):
                    try:
                        fragment = compose_modules(modules, selected_edges)
                        complexity = compute_complexity(fragment, str(anchor["module_id"]))
                        accepted, reason = self._passes_targets(fragment, complexity)
                        best_reason = reason
                        if accepted:
                            return self._package_candidate(
                                skeleton, anchor, modules, selected_edges, fragment, complexity
                            ), "accepted"
                    except SynthesisValidationError as exc:
                        best_reason = "%s: %s" % (exc.code, exc.message)

                selected = set(state.module_ids)
                selected_sources = {
                    str(self.module_by_id[module_id]["source_task_id"])
                    for module_id in state.module_ids
                }
                used_edges = set(state.compatibility_edge_ids)
                occupied_targets = {
                    (
                        str(edge_by_id[edge_id]["to_module"]),
                        str(binding["input_node_id"]),
                        str(binding["input_port"]),
                    )
                    for edge_id in state.compatibility_edge_ids
                    for binding in edge_by_id[edge_id]["bindings"]
                }
                for edge in usable_edges:
                    edge_id = str(edge["edge_id"])
                    if edge_id in used_edges:
                        continue
                    source = str(edge["from_module"])
                    target = str(edge["to_module"])
                    if source not in selected and target not in selected:
                        continue
                    new_modules = list(state.module_ids)
                    for module_id in (source, target):
                        if module_id in selected:
                            continue
                        module = self.module_by_id[module_id]
                        if len(new_modules) >= self.config.max_modules:
                            break
                        if (
                            module["environment_id"] != self.config.target_environment
                            or module["snapshot_id"] != self.config.target_snapshot
                            or module["source_split"] != self.config.source_split
                            or module.get("provenance", {}).get("review_status")
                            != "human_verified"
                        ):
                            break
                        if str(module["source_task_id"]) in selected_sources:
                            break
                        new_modules.append(module_id)
                    else:
                        target_keys = {
                            (target, str(binding["input_node_id"]), str(binding["input_port"]))
                            for binding in edge["bindings"]
                        }
                        if target_keys & occupied_targets:
                            continue
                        new_state = _SearchState(
                            tuple(new_modules),
                            tuple(sorted((*state.compatibility_edge_ids, edge_id))),
                        )
                        key = (
                            tuple(sorted(new_state.module_ids)),
                            new_state.compatibility_edge_ids,
                        )
                        if key not in seen:
                            seen.add(key)
                            expanded.append(new_state)
            if not expanded:
                break
            expanded.sort(key=lambda state: _state_score(state, edge_by_id), reverse=True)
            frontier = expanded[: self.config.beam_width]
        return None, best_reason

    def _passes_targets(
        self, fragment: Mapping[str, Any], complexity: Mapping[str, Any]
    ) -> Tuple[bool, str]:
        depth_min, depth_max = self.config.dependency_depth
        cross_min, cross_max = self.config.cross_app_dependencies
        if not depth_min <= complexity["dependency_depth"] <= depth_max:
            return False, "dependency_depth outside configured range"
        if not cross_min <= complexity["cross_app_dependency_count"] <= cross_max:
            return False, "cross_app_dependency_count outside configured range"
        if (
            complexity["independent_component_ratio"]
            > self.config.max_independent_component_ratio
        ):
            return False, "independent_component_ratio exceeds configured maximum"
        if self.config.required_delayed_reuse and complexity[
            "max_information_carry_distance"
        ] < 2:
            return False, "required delayed reuse is absent"
        if complexity["irreversible_action_count"] > self.config.max_irreversible_actions:
            return False, "too many irreversible actions"
        if _verifier_coverage(fragment) < self.config.min_verifier_coverage:
            return False, "critical verifier coverage is insufficient"
        return True, "all intrinsic targets passed"

    def _package_candidate(
        self,
        skeleton: Mapping[str, Any],
        anchor: Mapping[str, Any],
        modules: Sequence[Mapping[str, Any]],
        selected_edges: Sequence[Mapping[str, Any]],
        fragment: Mapping[str, Any],
        complexity: Mapping[str, Any],
    ) -> Dict[str, Any]:
        identity = {
            "generation_version": self.config.generation_version,
            "skeleton_id": skeleton["skeleton_id"],
            "module_ids": sorted(str(module["module_id"]) for module in modules),
            "edge_ids": sorted(str(edge["edge_id"]) for edge in selected_edges),
        }
        candidate_id = "gen_%s" % _stable_hash(identity)[:16]
        empirical_terms = [
            float(rule.get("probability", 1.0))
            for rule in skeleton.get("composition_rules", ())
            if float(rule.get("probability", 1.0)) > 0
        ]
        empirical_likelihood = math.exp(
            sum(math.log(value) for value in empirical_terms) / max(1, len(empirical_terms))
        )
        return {
            "schema_version": "generation-record/0.1",
            "candidate_id": candidate_id,
            "status": "symbolic_accepted",
            "generation_version": self.config.generation_version,
            "sampling_mode": self.config.sampling_mode,
            "source_task_ids": [module["source_task_id"] for module in modules],
            "source_module_ids": [module["module_id"] for module in modules],
            "source_skeleton_ids": [skeleton["skeleton_id"]],
            "anchor_module_id": anchor["module_id"],
            "environment_id": self.config.target_environment,
            "snapshot_id": self.config.target_snapshot,
            "instruction": None,
            "realization_status": "pending_round_trip_validation",
            "task_ir": dict(fragment),
            "world_bindings": {
                "world_subgraph_ids": [module["world_subgraph_id"] for module in modules],
                "compatibility_edge_ids": [edge["edge_id"] for edge in selected_edges],
                "bindings": [
                    binding for edge in selected_edges for binding in edge["bindings"]
                ],
            },
            "complexity": dict(complexity),
            "rubric": _derive_rubric(fragment),
            "validation": {
                "static_pass": True,
                "environment_status": "direct",
                "verifier_coverage": _verifier_coverage(fragment),
                "empirical_likelihood": empirical_likelihood,
                "gold_execution_pass": False,
                "fixture_reset_pass": False,
                "human_review_status": "pending",
            },
            "release_eligible": False,
            "provenance": {
                "candidate_fingerprint": _stable_hash(identity),
                "source_module_fingerprints": {
                    str(module["module_id"]): _stable_hash(module) for module in modules
                },
                "sampling_probability": self.skeleton_weights[str(skeleton["skeleton_id"])][
                    "sampling_probability"
                ],
                "llm_calls": [],
            },
        }

    def run(self) -> SynthesisResult:
        rng = random.Random(self.config.seed)
        candidate_modules = list(self._candidate_modules())
        if not candidate_modules:
            rejection = self._rejection(
                "WORLD_NOT_MATERIALIZABLE",
                "no grounded modules match the configured environment, snapshot, and split",
                attempt=0,
            )
            return SynthesisResult(
                (), (rejection,), self.compatibility_edges, self.skeleton_weights
            )
        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        candidate_fingerprints: Set[str] = set()
        for attempt in range(1, self.config.max_attempts + 1):
            if len(accepted) >= self.config.target_count:
                break
            skeleton = _weighted_choice(rng, self.skeletons, self.skeleton_weights)
            eligible_anchors = [
                module
                for module in candidate_modules
                if not skeleton.get("anchor_ops")
                or any(
                    node["op"] in set(skeleton["anchor_ops"])
                    for node in module["fragment"]["nodes"]
                )
            ]
            if not eligible_anchors:
                rejected.append(
                    self._rejection(
                        "GLOBAL_CONSTRAINT_UNSAT",
                        "no anchor module matches skeleton %s" % skeleton["skeleton_id"],
                        attempt,
                        skeleton_id=str(skeleton["skeleton_id"]),
                    )
                )
                continue
            anchor = eligible_anchors[rng.randrange(len(eligible_anchors))]
            candidate, reason = self._search(skeleton, anchor)
            if candidate is None:
                rejected.append(
                    self._rejection(
                        "INSUFFICIENT_DEPENDENCY",
                        reason,
                        attempt,
                        skeleton_id=str(skeleton["skeleton_id"]),
                        anchor_module_id=str(anchor["module_id"]),
                    )
                )
                continue
            fingerprint = str(candidate["provenance"]["candidate_fingerprint"])
            if fingerprint in candidate_fingerprints:
                rejected.append(
                    self._rejection(
                        "DUPLICATE_TASK",
                        "candidate repeats a prior module/binding composition",
                        attempt,
                        skeleton_id=str(skeleton["skeleton_id"]),
                        anchor_module_id=str(anchor["module_id"]),
                    )
                )
                continue
            candidate_fingerprints.add(fingerprint)
            accepted.append(candidate)
        return SynthesisResult(
            tuple(accepted),
            tuple(rejected),
            self.compatibility_edges,
            self.skeleton_weights,
        )

    @staticmethod
    def _rejection(
        code: str,
        detail: str,
        attempt: int,
        *,
        skeleton_id: Optional[str] = None,
        anchor_module_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if code not in REJECTION_CODES:
            raise ValueError("unknown rejection code %s" % code)
        identity = {
            "attempt": attempt,
            "skeleton_id": skeleton_id,
            "anchor_module_id": anchor_module_id,
            "detail": detail,
        }
        return {
            "schema_version": "generation-record/0.1",
            "candidate_id": "rejected_%s" % _stable_hash(identity)[:16],
            "status": "rejected",
            "rejection_code": code,
            "rejection_detail": detail,
            "attempt": attempt,
            "source_skeleton_ids": [skeleton_id] if skeleton_id else [],
            "anchor_module_id": anchor_module_id,
            "release_eligible": False,
            "provenance": {"candidate_fingerprint": _stable_hash(identity), "llm_calls": []},
        }
