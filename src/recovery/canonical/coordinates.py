"""Transforms between canonical pixel coordinates and model-specific coordinate protocols."""

from __future__ import annotations

from typing import Tuple

from .actions import CanonicalActionError


def pixel_to_normalized(value_px: int, extent_px: int, maximum: int) -> int:
    """Convert real pixels to ``[0, maximum]``."""

    if extent_px <= 0 or maximum <= 0:
        raise CanonicalActionError("extent_px 和 maximum 必须为正数")
    if not 0 <= value_px < extent_px:
        raise CanonicalActionError("像素坐标 %d 超出 [0, %d)" % (value_px, extent_px))
    return max(0, min(maximum, round(value_px * maximum / extent_px)))


def normalized_to_pixel(value: int, extent_px: int, maximum: int) -> int:
    """Map normalized coordinates back to pixels, clamped to the screen."""

    if extent_px <= 0 or maximum <= 0:
        raise CanonicalActionError("extent_px 和 maximum 必须为正数")
    if not 0 <= value <= maximum:
        raise CanonicalActionError("归一化坐标 %d 超出 [0, %d]" % (value, maximum))
    return min(extent_px - 1, int(value * extent_px / maximum))


def point_to_normalized(
    x_px: int,
    y_px: int,
    width_px: int,
    height_px: int,
    maximum: int,
) -> Tuple[int, int]:
    return (
        pixel_to_normalized(x_px, width_px, maximum),
        pixel_to_normalized(y_px, height_px, maximum),
    )


def rescale_pixel(value_px: int, source_extent: int, target_extent: int) -> int:
    """Deterministically rescale absolute-pixel coordinates."""

    if source_extent <= 0 or target_extent <= 0:
        raise CanonicalActionError("source_extent 和 target_extent 必须为正数")
    if not 0 <= value_px < source_extent:
        raise CanonicalActionError("像素坐标超出源图边界")
    return min(target_extent - 1, round(value_px * target_extent / source_extent))
