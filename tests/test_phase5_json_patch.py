from __future__ import annotations

import pytest

from derail.phase5.json_patch import PatchError, apply_patch


def test_patch_does_not_mutate_input_and_supports_escaped_keys() -> None:
    source = {"contacts": [], "a/b": {"tilde~key": 1}}
    result = apply_patch(
        source,
        [
            {"op": "add", "path": "/contacts/-", "value": {"name": "Jim"}},
            {"op": "replace", "path": "/a~1b/tilde~0key", "value": 2},
        ],
    )
    assert source["contacts"] == []
    assert result == {"contacts": [{"name": "Jim"}], "a/b": {"tilde~key": 2}}


def test_patch_supports_array_insert_and_remove() -> None:
    assert apply_patch({"items": [1, 3]}, [{"op": "add", "path": "/items/1", "value": 2}]) == {
        "items": [1, 2, 3]
    }
    assert apply_patch({"items": [1, 2]}, [{"op": "remove", "path": "/items/0"}]) == {
        "items": [2]
    }


@pytest.mark.parametrize("operation", [{"op": "move", "path": "/x"}, {"op": "remove", "path": "/x"}])
def test_invalid_operation_or_path_is_rejected(operation: dict[str, str]) -> None:
    with pytest.raises(PatchError):
        apply_patch({}, [operation])
