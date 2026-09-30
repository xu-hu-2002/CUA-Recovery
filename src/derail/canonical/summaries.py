"""从 canonical action 确定性生成公开 action summary。"""

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
    """返回可审计摘要。

    摘要不包含模型的 hidden reasoning，也不声称是原 agent 的原始思维。它只陈述
    已执行动作的公开语义，因此 repaired step 可以安全重新生成。
    """

    if language not in {"en", "zh"}:
        raise ValueError("language 必须是 en 或 zh")

    if isinstance(action, ClickAction):
        verb_en = "Double-click" if action.kind == "double_click" else "Click"
        verb_zh = "双击" if action.kind == "double_click" else "点击"
        target_en = " %s" % action.target if action.target else ""
        target_zh = "“%s”" % action.target if action.target else "目标位置"
        if language == "zh":
            return "%s%s，坐标 (%d, %d)。" % (verb_zh, target_zh, action.x_px, action.y_px)
        return "%s%s at (%d, %d)." % (verb_en, target_en, action.x_px, action.y_px)

    if isinstance(action, TypeAction):
        suffix = " and press Enter" if action.press_enter else ""
        if language == "zh":
            return "输入文本 %r%s。" % (action.text, "并按回车" if action.press_enter else "")
        return "Type %r%s." % (action.text, suffix)

    if isinstance(action, HotkeyAction):
        keys = "+".join(action.keys)
        return ("按快捷键 %s。" if language == "zh" else "Press hotkey %s.") % keys

    if isinstance(action, KeyTransitionAction):
        down = action.kind == "key_down"
        if language == "zh":
            return ("按住按键 %s。" if down else "释放按键 %s。") % action.key
        return ("Hold key %s." if down else "Release key %s.") % action.key

    if isinstance(action, MouseButtonTransitionAction):
        down = action.kind == "mouse_down"
        if language == "zh":
            return ("按住鼠标%s键。" if down else "释放鼠标%s键。") % action.button
        return ("Hold mouse button %s." if down else "Release mouse button %s.") % (
            action.button
        )

    if isinstance(action, NoOpAction):
        return ("未执行桌面动作：%s。" if language == "zh" else "No desktop action: %s.") % action.reason

    if isinstance(action, ScrollAction):
        direction_en = "up" if action.delta_y > 0 else "down"
        direction_zh = "向上" if action.delta_y > 0 else "向下"
        amount = abs(action.delta_y)
        if language == "zh":
            return "在 (%d, %d) %s滚动 %d 单位。" % (
                action.x_px,
                action.y_px,
                direction_zh,
                amount,
            )
        return "Scroll %s by %d units at (%d, %d)." % (
            direction_en,
            amount,
            action.x_px,
            action.y_px,
        )

    if isinstance(action, HorizontalScrollAction):
        direction_en = "right" if action.delta_x > 0 else "left"
        direction_zh = "向右" if action.delta_x > 0 else "向左"
        amount = abs(action.delta_x)
        if language == "zh":
            return "在 (%d, %d) %s横向滚动 %d 单位。" % (
                action.x_px,
                action.y_px,
                direction_zh,
                amount,
            )
        return "Scroll horizontally %s by %d units at (%d, %d)." % (
            direction_en,
            amount,
            action.x_px,
            action.y_px,
        )

    if isinstance(action, MoveAction):
        if language == "zh":
            return "将鼠标移动到 (%d, %d)。" % (action.x_px, action.y_px)
        return "Move the pointer to (%d, %d)." % (action.x_px, action.y_px)

    if isinstance(action, DragAction):
        if language == "zh":
            return "从 (%d, %d) 拖动到 (%d, %d)。" % (
                action.start_x_px,
                action.start_y_px,
                action.end_x_px,
                action.end_y_px,
            )
        return "Drag from (%d, %d) to (%d, %d)." % (
            action.start_x_px,
            action.start_y_px,
            action.end_x_px,
            action.end_y_px,
        )

    if isinstance(action, SequenceAction):
        summaries = [summarize_action(item, language).rstrip("。.") for item in action.actions]
        if language == "zh":
            return "依次执行：%s。" % "；".join(summaries)
        return "Execute in sequence: %s." % "; ".join(summaries)

    if isinstance(action, WaitAction):
        return ("等待 %.2f 秒。" if language == "zh" else "Wait for %.2f seconds.") % action.seconds

    if isinstance(action, ShellAction):
        commands = " && ".join(action.commands)
        return "Run %r in the VM shell." % commands

    if isinstance(action, TerminateAction):
        if language == "zh":
            return "以 %s 状态结束任务。" % action.status
        return "Terminate the task with status %s." % action.status

    raise TypeError("未处理的 canonical action 类型: %s" % type(action).__name__)
