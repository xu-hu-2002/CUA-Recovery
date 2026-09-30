"""Task-IR validation, composition, and intrinsic long-horizon metrics.

The synthesis layer operates on semantic operations rather than GUI actions.  It
therefore stays independent of the replay/action stack used by the DERAIL
takeover benchmark.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


OPERATIONS = frozenset(
    {
        "retrieve",
        "resolve",
        "filter",
        "compare",
        "aggregate",
        "verify",
        "decide",
        "create",
        "modify",
        "delete",
        "communicate",
        "confirm",
    }
)
EDGE_KINDS = frozenset({"data_dependency", "state_dependency", "control_dependency"})
TERMINAL_OPERATIONS = frozenset({"create", "modify", "delete", "communicate", "confirm"})
FAN_IN_OPERATIONS = frozenset({"compare", "aggregate", "decide"})


class SynthesisValidationError(ValueError):
    """A synthesis input or composed graph violates a hard invariant."""

    def __init__(self, code: str, message: str):
        super().__init__("%s: %s" % (code, message))
        self.code = code
        self.message = message


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SynthesisValidationError("INVALID_GROUNDING", "%s must be non-empty" % field)
    return value


def _port_map(nodes: Sequence[Mapping[str, Any]], direction: str) -> Dict[Tuple[str, str], dict]:
    result: Dict[Tuple[str, str], dict] = {}
    for node in nodes:
        node_id = str(node["node_id"])
        for raw_port in node.get(direction, ()):
            port = dict(raw_port)
            key = (node_id, _require_nonempty_string(port.get("port_id"), "port_id"))
            if key in result:
                raise SynthesisValidationError(
                    "INVALID_GROUNDING", "duplicate %s port %s.%s" % (direction, *key)
                )
            _require_nonempty_string(port.get("type"), "%s.type" % direction)
            result[key] = port
    return result


def _edge_end(edge: Mapping[str, Any], side: str) -> Tuple[str, str]:
    raw = edge.get(side)
    if not isinstance(raw, Mapping):
        raise SynthesisValidationError("INVALID_GROUNDING", "edge.%s must be an object" % side)
    return (
        _require_nonempty_string(raw.get("node_id"), "edge.%s.node_id" % side),
        _require_nonempty_string(raw.get("port_id"), "edge.%s.port_id" % side),
    )


def _topological_order(node_ids: Iterable[str], edges: Sequence[Mapping[str, Any]]) -> List[str]:
    nodes = tuple(node_ids)
    outgoing: Dict[str, List[str]] = {node_id: [] for node_id in nodes}
    indegree = {node_id: 0 for node_id in nodes}
    for edge in edges:
        source, _ = _edge_end(edge, "from")
        target, _ = _edge_end(edge, "to")
        if source not in outgoing or target not in outgoing:
            raise SynthesisValidationError("INVALID_GROUNDING", "edge references unknown node")
        outgoing[source].append(target)
        indegree[target] += 1
    ready = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
    order: List[str] = []
    while ready:
        node_id = ready.popleft()
        order.append(node_id)
        for target in sorted(outgoing[node_id]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if len(order) != len(nodes):
        raise SynthesisValidationError("GLOBAL_CONSTRAINT_UNSAT", "task fragment is not a DAG")
    return order


def validate_task_fragment(fragment: Mapping[str, Any]) -> None:
    """Validate semantic nodes, ports, bindings, and DAG structure."""

    nodes = fragment.get("nodes")
    edges = fragment.get("edges")
    if not isinstance(nodes, list) or not nodes:
        raise SynthesisValidationError("INVALID_GROUNDING", "fragment.nodes must be non-empty")
    if not isinstance(edges, list):
        raise SynthesisValidationError("INVALID_GROUNDING", "fragment.edges must be a list")

    node_ids: Set[str] = set()
    for node in nodes:
        if not isinstance(node, Mapping):
            raise SynthesisValidationError("INVALID_GROUNDING", "node must be an object")
        node_id = _require_nonempty_string(node.get("node_id"), "node_id")
        if node_id in node_ids:
            raise SynthesisValidationError("INVALID_GROUNDING", "duplicate node_id %s" % node_id)
        node_ids.add(node_id)
        if node.get("op") not in OPERATIONS:
            raise SynthesisValidationError(
                "INVALID_GROUNDING", "unknown operation %r" % node.get("op")
            )
        _require_nonempty_string(node.get("app"), "%s.app" % node_id)
        if not isinstance(node.get("side_effects", []), list):
            raise SynthesisValidationError(
                "INVALID_GROUNDING", "%s.side_effects must be a list" % node_id
            )

    input_ports = _port_map(nodes, "inputs")
    output_ports = _port_map(nodes, "outputs")
    edge_ids: Set[str] = set()
    bound_inputs: Set[Tuple[str, str]] = set()
    for edge in edges:
        edge_id = _require_nonempty_string(edge.get("edge_id"), "edge_id")
        if edge_id in edge_ids:
            raise SynthesisValidationError("INVALID_GROUNDING", "duplicate edge_id %s" % edge_id)
        edge_ids.add(edge_id)
        if edge.get("kind") not in EDGE_KINDS:
            raise SynthesisValidationError("INVALID_GROUNDING", "unknown dependency kind")
        source = _edge_end(edge, "from")
        target = _edge_end(edge, "to")
        if source not in output_ports:
            raise SynthesisValidationError(
                "UNBOUND_PORT", "edge source %s.%s is not an output" % source
            )
        if target not in input_ports:
            raise SynthesisValidationError(
                "UNBOUND_PORT", "edge target %s.%s is not an input" % target
            )
        if target in bound_inputs:
            raise SynthesisValidationError(
                "UNBOUND_PORT", "input %s.%s has multiple producers" % target
            )
        bound_inputs.add(target)

    for key, port in input_ports.items():
        source = port.get("source")
        if source == "upstream" and key not in bound_inputs:
            raise SynthesisValidationError("UNBOUND_PORT", "input %s.%s lacks an edge" % key)
        if source in {"world", "literal"} and not port.get("grounding"):
            raise SynthesisValidationError(
                "INVALID_GROUNDING", "input %s.%s lacks stable grounding" % key
            )
        if source not in {"world", "literal", "upstream"}:
            raise SynthesisValidationError(
                "INVALID_GROUNDING", "input %s.%s has invalid source" % key
            )

    for key, port in output_ports.items():
        if not port.get("grounding"):
            raise SynthesisValidationError(
                "INVALID_GROUNDING", "output %s.%s lacks world/derived grounding" % key
            )

    _topological_order(node_ids, edges)


def validate_grounded_module(module: Mapping[str, Any]) -> None:
    """Validate a GroundedTaskModule and all public interface references."""

    for field in (
        "module_id",
        "source_task_id",
        "source_split",
        "environment_id",
        "snapshot_id",
        "world_subgraph_id",
    ):
        _require_nonempty_string(module.get(field), field)
    fragment = module.get("fragment")
    interface = module.get("interface")
    if not isinstance(fragment, Mapping) or not isinstance(interface, Mapping):
        raise SynthesisValidationError(
            "INVALID_GROUNDING", "module requires fragment and interface objects"
        )
    validate_task_fragment(fragment)
    nodes = list(fragment["nodes"])
    inputs = _port_map(nodes, "inputs")
    outputs = _port_map(nodes, "outputs")

    seen_public: Set[Tuple[str, str, str]] = set()
    for direction, available in (("accepts", inputs), ("produces", outputs)):
        ports = interface.get(direction, ())
        if not isinstance(ports, list):
            raise SynthesisValidationError(
                "INVALID_GROUNDING", "interface.%s must be a list" % direction
            )
        for port in ports:
            node_id = _require_nonempty_string(port.get("node_id"), "%s.node_id" % direction)
            port_id = _require_nonempty_string(port.get("port_id"), "%s.port_id" % direction)
            key = (node_id, port_id)
            if key not in available:
                raise SynthesisValidationError(
                    "UNBOUND_PORT", "public port %s.%s does not exist" % key
                )
            if port.get("type") != available[key].get("type"):
                raise SynthesisValidationError(
                    "TYPE_MISMATCH", "public port %s.%s changes its fragment type" % key
                )
            public_key = (direction, node_id, port_id)
            if public_key in seen_public:
                raise SynthesisValidationError(
                    "INVALID_GROUNDING", "duplicate public port %s.%s" % key
                )
            seen_public.add(public_key)
            if port.get("grounding") != available[key].get("grounding"):
                raise SynthesisValidationError(
                    "INVALID_GROUNDING", "public port %s.%s changes its grounding" % key
                )

    if not interface.get("produces"):
        raise SynthesisValidationError("UNBOUND_PORT", "module exposes no output port")
    if not isinstance(interface.get("identity_scope", []), list):
        raise SynthesisValidationError("INVALID_GROUNDING", "identity_scope must be a list")
    if not isinstance(interface.get("reads", []), list):
        raise SynthesisValidationError("INVALID_GROUNDING", "reads must be a list")
    if not isinstance(interface.get("side_effects", []), list):
        raise SynthesisValidationError("INVALID_GROUNDING", "side_effects must be a list")


def _namespace(module_id: str, value: str) -> str:
    return "%s::%s" % (module_id, value)


def compose_modules(
    modules: Sequence[Mapping[str, Any]],
    compatibility_edges: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Compose modules and selected bindings into one validated TaskFragment."""

    if not modules:
        raise SynthesisValidationError("GLOBAL_CONSTRAINT_UNSAT", "no modules selected")
    module_by_id = {str(module["module_id"]): module for module in modules}
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    for module in modules:
        module_id = str(module["module_id"])
        for original in module["fragment"]["nodes"]:
            node = copy.deepcopy(original)
            node["node_id"] = _namespace(module_id, str(original["node_id"]))
            node["source_module_id"] = module_id
            for input_port in node.get("inputs", ()):
                bound = input_port.get("bound_from")
                if bound:
                    bound["node_id"] = _namespace(module_id, str(bound["node_id"]))
            nodes.append(node)
        for original in module["fragment"].get("edges", ()):
            edge = copy.deepcopy(original)
            edge["edge_id"] = _namespace(module_id, str(original["edge_id"]))
            edge["from"]["node_id"] = _namespace(
                module_id, str(original["from"]["node_id"])
            )
            edge["to"]["node_id"] = _namespace(module_id, str(original["to"]["node_id"]))
            edges.append(edge)

    node_by_id = {node["node_id"]: node for node in nodes}
    bound_targets: Set[Tuple[str, str]] = set()
    for compat_index, compat in enumerate(compatibility_edges):
        source_module = str(compat["from_module"])
        target_module = str(compat["to_module"])
        if source_module not in module_by_id or target_module not in module_by_id:
            raise SynthesisValidationError(
                "GLOBAL_CONSTRAINT_UNSAT", "binding references an unselected module"
            )
        for binding_index, binding in enumerate(compat.get("bindings", ())):
            source_node = _namespace(source_module, str(binding["output_node_id"]))
            target_node = _namespace(target_module, str(binding["input_node_id"]))
            target_key = (target_node, str(binding["input_port"]))
            if target_key in bound_targets:
                raise SynthesisValidationError(
                    "GLOBAL_CONSTRAINT_UNSAT",
                    "multiple grafts bind %s.%s" % target_key,
                )
            bound_targets.add(target_key)
            target_port = next(
                (
                    port
                    for port in node_by_id[target_node].get("inputs", ())
                    if port["port_id"] == binding["input_port"]
                ),
                None,
            )
            if target_port is None:
                raise SynthesisValidationError("UNBOUND_PORT", "graft target disappeared")
            target_port["source"] = "upstream"
            target_port["bound_from"] = {
                "node_id": source_node,
                "port_id": binding["output_port"],
            }
            edges.append(
                {
                    "edge_id": "graft::%d::%d" % (compat_index, binding_index),
                    "from": {
                        "node_id": source_node,
                        "port_id": binding["output_port"],
                    },
                    "to": {"node_id": target_node, "port_id": binding["input_port"]},
                    "kind": binding.get("dependency_kind", "data_dependency"),
                    "binding_id": binding["binding_id"],
                    "converter_id": binding.get("converter_id"),
                }
            )

    fragment = {
        "schema_version": "task-fragment/0.1",
        "fragment_id": "composed",
        "nodes": nodes,
        "edges": edges,
    }
    validate_task_fragment(fragment)
    return fragment


def _adjacency(
    nodes: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    outgoing = {str(node["node_id"]): [] for node in nodes}
    incoming = {str(node["node_id"]): [] for node in nodes}
    for edge in edges:
        source, _ = _edge_end(edge, "from")
        target, _ = _edge_end(edge, "to")
        outgoing[source].append(target)
        incoming[target].append(source)
    return outgoing, incoming


def _longest_distances_from(
    source: str, order: Sequence[str], outgoing: Mapping[str, Sequence[str]]
) -> Dict[str, int]:
    distances = {node_id: -1 for node_id in order}
    distances[source] = 0
    for node_id in order:
        if distances[node_id] < 0:
            continue
        for target in outgoing[node_id]:
            distances[target] = max(distances[target], distances[node_id] + 1)
    return distances


def _independent_component_ratio(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    anchor_node_ids: Set[str],
) -> float:
    kept = {
        str(node["node_id"])
        for node in nodes
        if node.get("op") not in TERMINAL_OPERATIONS
    }
    if not kept:
        return 0.0
    neighbors: Dict[str, Set[str]] = {node_id: set() for node_id in kept}
    for edge in edges:
        source, _ = _edge_end(edge, "from")
        target, _ = _edge_end(edge, "to")
        if source in kept and target in kept:
            neighbors[source].add(target)
            neighbors[target].add(source)
    components: List[Set[str]] = []
    unseen = set(kept)
    while unseen:
        start = min(unseen)
        stack = [start]
        component: Set[str] = set()
        while stack:
            node_id = stack.pop()
            if node_id in component:
                continue
            component.add(node_id)
            unseen.discard(node_id)
            stack.extend(neighbors[node_id] - component)
        components.append(component)
    anchor_components = [component for component in components if component & anchor_node_ids]
    anchor_component = max(anchor_components or components, key=len)
    return (len(kept) - len(anchor_component)) / float(len(kept))


def compute_complexity(
    fragment: Mapping[str, Any], anchor_module_id: Optional[str] = None
) -> Dict[str, Any]:
    """Compute the intrinsic long-horizon metrics defined by the design document."""

    validate_task_fragment(fragment)
    nodes = list(fragment["nodes"])
    edges = list(fragment["edges"])
    node_by_id = {str(node["node_id"]): node for node in nodes}
    order = _topological_order(node_by_id, edges)
    outgoing, incoming = _adjacency(nodes, edges)

    longest_to = {node_id: 1 for node_id in order}
    for node_id in order:
        for target in outgoing[node_id]:
            longest_to[target] = max(longest_to[target], longest_to[node_id] + 1)
    dependency_depth = max(longest_to.values())

    cross_app = 0
    information_distances: List[int] = []
    distance_cache: Dict[str, Dict[str, int]] = {}
    for edge in edges:
        source, _ = _edge_end(edge, "from")
        target, _ = _edge_end(edge, "to")
        if (
            edge["kind"] != "control_dependency"
            and node_by_id[source]["app"] != node_by_id[target]["app"]
        ):
            cross_app += 1
        if edge["kind"] in {"data_dependency", "state_dependency"}:
            if source not in distance_cache:
                distance_cache[source] = _longest_distances_from(source, order, outgoing)
            information_distances.append(distance_cache[source][target])

    sorted_distances = sorted(information_distances)
    if sorted_distances:
        middle = len(sorted_distances) // 2
        if len(sorted_distances) % 2:
            median_distance = float(sorted_distances[middle])
        else:
            median_distance = (sorted_distances[middle - 1] + sorted_distances[middle]) / 2.0
    else:
        median_distance = 0.0

    fan_ins = [
        len(set(incoming[node_id]))
        for node_id, node in node_by_id.items()
        if node.get("op") in FAN_IN_OPERATIONS
    ]
    branch_count = sum(
        1
        for node_id in order
        if len(
            [
                edge
                for edge in edges
                if _edge_end(edge, "from")[0] == node_id
                and edge["kind"] == "control_dependency"
                and edge.get("predicate")
            ]
        )
        >= 2
    )

    irreversible_nodes = {
        node_id
        for node_id, node in node_by_id.items()
        if any(bool(effect.get("irreversible")) for effect in node.get("side_effects", ()))
    }
    sources = [node_id for node_id in order if not incoming[node_id]]
    shortest = {node_id: None for node_id in order}  # type: Dict[str, Optional[int]]
    queue: deque[str] = deque()
    for source in sources:
        shortest[source] = 1
        queue.append(source)
    while queue:
        node_id = queue.popleft()
        assert shortest[node_id] is not None
        for target in outgoing[node_id]:
            candidate = int(shortest[node_id]) + 1
            if shortest[target] is None or candidate < int(shortest[target]):
                shortest[target] = candidate
                queue.append(target)
    irreversible_depths = [shortest[node_id] for node_id in irreversible_nodes]
    irreversible_depth = min(irreversible_depths) if irreversible_depths else None

    anchor_nodes = {
        node_id
        for node_id, node in node_by_id.items()
        if anchor_module_id is not None and node.get("source_module_id") == anchor_module_id
    }
    return {
        "dependency_depth": dependency_depth,
        "cross_app_dependency_count": cross_app,
        "max_information_carry_distance": max(information_distances, default=0),
        "median_information_carry_distance": median_distance,
        "max_fan_in": max(fan_ins, default=0),
        "mean_fan_in": (sum(fan_ins) / float(len(fan_ins))) if fan_ins else 0.0,
        "branch_count": branch_count,
        "irreversible_action_depth": irreversible_depth,
        "irreversible_action_count": len(irreversible_nodes),
        "independent_component_ratio": _independent_component_ratio(
            nodes, edges, anchor_nodes
        ),
    }
