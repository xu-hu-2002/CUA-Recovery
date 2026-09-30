"""Restricted expression evaluator for Task IR derivations and derived verifiers."""

from __future__ import annotations

import ast
import datetime as _dt
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


class ExpressionError(ValueError):
    pass


def _parse_dt(value: Any) -> _dt.datetime:
    if isinstance(value, _dt.datetime):
        return value
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return _dt.datetime.fromisoformat(text)
    except ValueError as exc:
        raise ExpressionError("not an ISO date-time: %r" % value) from exc


def _format_dt(value: _dt.datetime, template: Optional[str]) -> str:
    if template is None:
        template = "%Y-%m-%dT%H:%M:%S" if value.tzinfo is None else "%Y-%m-%dT%H:%M:%S%z"
    text = value.strftime(template)
    if value.tzinfo is not None and text.endswith("+0000"):
        text = text[:-5] + "+00:00"
    return text


def date_add(value: Any, days: int = 0, hours: int = 0, minutes: int = 0) -> str:
    stamp = _parse_dt(value)
    shifted = stamp + _dt.timedelta(days=days, hours=hours, minutes=minutes)
    template = "%Y-%m-%d" if len(str(value).strip()) == 10 else None
    return _format_dt(shifted, template)


def date_of(value: Any) -> str:
    return _parse_dt(value).strftime("%Y-%m-%d")


def time_of(value: Any) -> str:
    return _parse_dt(value).strftime("%H:%M")


def weekday_of(value: Any) -> str:
    return _parse_dt(value).strftime("%A")


def at_time(date_value: Any, clock: str) -> str:
    day = _parse_dt(date_value).date()
    hour, _, minute = clock.partition(":")
    return _dt.datetime(day.year, day.month, day.day, int(hour), int(minute or 0)).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )


def fmt(template: str, *args: Any, **kwargs: Any) -> str:
    return str(template).format(*args, **kwargs)


def pick(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    return default


def pluck(rows: Iterable[Any], key: str) -> List[Any]:
    return [pick(row, key) for row in rows]


def count_by(rows: Iterable[Any], key: str) -> Dict[Any, int]:
    counts: Dict[Any, int] = {}
    for row in rows:
        value = pick(row, key)
        counts[value] = counts.get(value, 0) + 1
    return counts


def argmax(mapping: Mapping[Any, Any]) -> Any:
    if not mapping:
        return None
    return sorted(mapping.items(), key=lambda item: (-item[1], str(item[0])))[0][0]


def argmin(mapping: Mapping[Any, Any]) -> Any:
    if not mapping:
        return None
    return sorted(mapping.items(), key=lambda item: (item[1], str(item[0])))[0][0]


def first(values: Sequence[Any], default: Any = None) -> Any:
    return values[0] if values else default


def unique(values: Iterable[Any]) -> List[Any]:
    seen: List[Any] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def contains(haystack: Any, needle: Any) -> bool:
    if haystack is None:
        return False
    if isinstance(haystack, str):
        return str(needle) in haystack
    return needle in haystack


DEFAULT_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "len": len,
    "max": max,
    "min": min,
    "sum": sum,
    "sorted": sorted,
    "str": str,
    "int": int,
    "float": float,
    "round": round,
    "abs": abs,
    "any": any,
    "all": all,
    "list": list,
    "bool": bool,
    "tuple": tuple,
    "set": set,
    "dict": dict,
    "lower": lambda value: str(value).lower(),
    "upper": lambda value: str(value).upper(),
    "strip": lambda value: str(value).strip(),
    "date_add": date_add,
    "date_of": date_of,
    "time_of": time_of,
    "weekday_of": weekday_of,
    "at_time": at_time,
    "fmt": fmt,
    "pick": pick,
    "pluck": pluck,
    "count_by": count_by,
    "argmax": argmax,
    "argmin": argmin,
    "first": first,
    "unique": unique,
    "contains": contains,
}

_ALLOWED = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.keyword,
    ast.IfExp,
    ast.Subscript,
    ast.Index,
    ast.Slice,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
)

_BIN = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}
_CMP = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: lambda a, b: a is b,
    ast.IsNot: lambda a, b: a is not b,
}


class _Evaluator:
    def __init__(self, env: Mapping[str, Any], functions: Mapping[str, Callable[..., Any]]):
        self.env = env
        self.functions = functions

    def visit(self, node: ast.AST) -> Any:
        if not isinstance(node, _ALLOWED):
            raise ExpressionError("disallowed syntax: %s" % type(node).__name__)
        method = getattr(self, "visit_" + type(node).__name__, None)
        if method is None:
            raise ExpressionError("unsupported node: %s" % type(node).__name__)
        return method(node)

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.env:
            return self.env[node.id]
        if node.id in ("None", "True", "False", "null", "true", "false"):
            return {
                "None": None,
                "True": True,
                "False": False,
                "null": None,
                "true": True,
                "false": False,
            }[node.id]
        raise ExpressionError("unknown name %r" % node.id)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        return _BIN[type(node.op)](self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return not operand
        return -operand if isinstance(node.op, ast.USub) else +operand

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        values = [self.visit(value) for value in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            if not _CMP[type(op)](left, right):
                return False
            left = right
        return True

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name) or node.func.id not in self.functions:
            raise ExpressionError("call to non-whitelisted function")
        args = [self.visit(arg) for arg in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords if kw.arg}
        return self.functions[node.func.id](*args, **kwargs)

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        return self.visit(node.body) if self.visit(node.test) else self.visit(node.orelse)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        value = self.visit(node.value)
        index = node.slice
        if isinstance(index, ast.Slice):
            lower = self.visit(index.lower) if index.lower else None
            upper = self.visit(index.upper) if index.upper else None
            return value[lower:upper]
        if hasattr(ast, "Index") and isinstance(index, ast.Index):
            index = index.value  # type: ignore[attr-defined]
        return value[self.visit(index)]

    def visit_List(self, node: ast.List) -> Any:
        return [self.visit(item) for item in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        return tuple(self.visit(item) for item in node.elts)

    def visit_Set(self, node: ast.Set) -> Any:
        return {self.visit(item) for item in node.elts}

    def visit_Dict(self, node: ast.Dict) -> Any:
        return {self.visit(key): self.visit(value) for key, value in zip(node.keys, node.values)}

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> Any:
        return self.visit_ListComp(node)  # type: ignore[arg-type]

    def visit_ListComp(self, node: ast.ListComp) -> Any:
        if len(node.generators) != 1:
            raise ExpressionError("only single-generator comprehensions are supported")
        generator = node.generators[0]
        if not isinstance(generator.target, ast.Name):
            raise ExpressionError("comprehension target must be a name")
        out = []
        for item in self.visit(generator.iter):
            inner = _Evaluator(dict(self.env, **{generator.target.id: item}), self.functions)
            if all(inner.visit(cond) for cond in generator.ifs):
                out.append(inner.visit(node.elt))
        return out


def evaluate(
    expression: str,
    env: Mapping[str, Any],
    functions: Optional[Mapping[str, Callable[..., Any]]] = None,
) -> Any:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError("cannot parse %r: %s" % (expression, exc)) from exc
    table = dict(DEFAULT_FUNCTIONS)
    if functions:
        table.update(functions)
    try:
        return _Evaluator(env, table).visit(tree)
    except ExpressionError:
        raise
    except Exception as exc:
        raise ExpressionError("%r: %s: %s" % (expression, type(exc).__name__, exc)) from exc
