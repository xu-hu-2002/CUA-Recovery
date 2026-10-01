"""Deterministic public action summaries from canonical actions."""

from __future__ import annotations

from .actions import (
    Action,
    ClickAction,
    DragAction,
    HotkeyAction,
    HorizontalScrollAction,
    KeyTransitionAction,
    MouseButtonTransitionAction,
    MoveAction,
    NoOpAction,
    ScrollAction,
    SequenceAction,
    ShellAction,
    TerminateAction,
    TypeAction,
    WaitAction,
)


def summarize_action(action: Action, language: str = "en") -> str:
    """Return an auditable summary of an executed action."""

    if language != "en":
        raise ValueError("language must be en")

    if isinstance(action, ClickAction):
        verb_en = "Double-click" if action.kind == "double_click" else "Click"
        target_en = " %s" % action.target if action.target else ""
        return "%s%s at (%d, %d)." % (verb_en, target_en, action.x_px, action.y_px)

    if isinstance(action, TypeAction):
        suffix = " and press Enter" if action.press_enter else ""
        return "Type %r%s." % (action.text, suffix)

    if isinstance(action, HotkeyAction):
        keys = "+".join(action.keys)
        return "Press hotkey %s." % keys

    if isinstance(action, KeyTransitionAction):
        down = action.kind == "key_down"
        return ("Hold key %s." if down else "Release key %s.") % action.key

    if isinstance(action, MouseButtonTransitionAction):
        down = action.kind == "mouse_down"
        return ("Hold mouse button %s." if down else "Release mouse button %s.") % (
            action.button
        )

    if isinstance(action, NoOpAction):
        return "No desktop action: %s." % action.reason

    if isinstance(action, ScrollAction):
        direction_en = "up" if action.delta_y > 0 else "down"
        amount = abs(action.delta_y)
        return "Scroll %s by %d units at (%d, %d)." % (
            direction_en,
            amount,
            action.x_px,
            action.y_px,
        )

    if isinstance(action, HorizontalScrollAction):
        direction_en = "right" if action.delta_x > 0 else "left"
        amount = abs(action.delta_x)
        return "Scroll horizontally %s by %d units at (%d, %d)." % (
            direction_en,
            amount,
            action.x_px,
            action.y_px,
        )

    if isinstance(action, MoveAction):
        return "Move the pointer to (%d, %d)." % (action.x_px, action.y_px)

    if isinstance(action, DragAction):
        return "Drag from (%d, %d) to (%d, %d)." % (
            action.start_x_px,
            action.start_y_px,
            action.end_x_px,
            action.end_y_px,
        )

    if isinstance(action, SequenceAction):
        summaries = [summarize_action(item, language).rstrip(".") for item in action.actions]
        return "Execute in sequence: %s." % "; ".join(summaries)

    if isinstance(action, WaitAction):
        return "Wait for %.2f seconds." % action.seconds

    if isinstance(action, ShellAction):
        commands = " && ".join(action.commands)
        return "Run %r in the VM shell." % commands

    if isinstance(action, TerminateAction):
        return "Terminate the task with status %s." % action.status

    raise TypeError("unhandled canonical action type: %s" % type(action).__name__)
