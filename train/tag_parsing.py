"""
Parse `<risk_factors_with_boxes>` spans and normalized [x_min, y_min, x_max, y_max] in [0,1].

Supports:
  - Integer grid: values in 0..1000 with at least one corner > 1 (mapped by /1000), e.g. [0123, 0456, 0789, 0999]
  - Legacy floats: [0.123, 0.456, 0.789, 0.987]
  - Legacy xyxy without decimals: [0, 0, 1, 1] (all values <= 1) is read as normalized floats, not the grid

Shared by bounding-box SFT eval, reward code, and future VG-fDPO.
"""

from __future__ import annotations

import re

from train.bounding_box_sft_schema import BOX_COORD_SCALE, TAG_RISK_WITH_BOXES

# Decimal floats (legacy / teacher traces)
_BOX_PATTERN_FLOAT = re.compile(
    r"\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]"
)

# Integer grid 0..1000, 3–4 digit tokens (we also accept 1–4 digits for robustness)
_BOX_PATTERN_INT = re.compile(
    r"\[\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]"
)


def _norm_from_int_quad(a: int, b: int, c: int, d: int) -> tuple[float, float, float, float] | None:
    if max(a, b, c, d) > BOX_COORD_SCALE:
        return None
    return (
        a / float(BOX_COORD_SCALE),
        b / float(BOX_COORD_SCALE),
        c / float(BOX_COORD_SCALE),
        d / float(BOX_COORD_SCALE),
    )


def _quad_from_int_match_groups(a: int, b: int, c: int, d: int) -> tuple[float, float, float, float] | None:
    """
    Disambiguate Shikra-style integers in [0, 1000] from legacy normalized xyxy written without decimals.

    If any coordinate is > 1, treat as discrete grid and divide by BOX_COORD_SCALE; otherwise treat
    the four numbers as floats in [0, 1] (e.g. [0, 0, 1, 1] meaning full image in xyxy).
    """
    if max(a, b, c, d) > 1:
        return _norm_from_int_quad(a, b, c, d)
    return (float(a), float(b), float(c), float(d))


def _parse_first_quad_in_string(hay: str) -> tuple[float, float, float, float] | None:
    """Prefer explicit decimals; then integer-like brackets (grid vs legacy xyxy); then float tokens."""
    for m in _BOX_PATTERN_FLOAT.finditer(hay):
        inner = m.group(0)
        if "." not in inner:
            continue
        return tuple(float(m.group(i)) for i in range(1, 5))
    for m in _BOX_PATTERN_INT.finditer(hay):
        inner = m.group(0)
        if "." in inner:
            continue
        a, b, c, d = (int(m.group(i)) for i in range(1, 5))
        out = _quad_from_int_match_groups(a, b, c, d)
        if out is not None:
            return out
    m = _BOX_PATTERN_FLOAT.search(hay)
    if m:
        return tuple(float(m.group(i)) for i in range(1, 5))
    return None


def extract_risk_factors_with_boxes_block(text: str) -> str | None:
    """Return inner content of the first `<risk_factors_with_boxes>...</risk_factors_with_boxes>` block."""
    open_t = f"<{TAG_RISK_WITH_BOXES}>"
    close_t = f"</{TAG_RISK_WITH_BOXES}>"
    i = text.find(open_t)
    if i < 0:
        return None
    j = text.find(close_t, i)
    if j < 0:
        return None
    return text[i + len(open_t) : j]


def parse_first_norm_box(text: str) -> tuple[float, float, float, float] | None:
    """Parse the first box as normalized xyxy in [0, 1]."""
    block = extract_risk_factors_with_boxes_block(text)
    hay = block if block is not None else text
    return _parse_first_quad_in_string(hay)


def parse_all_norm_boxes(text: str) -> list[tuple[float, float, float, float]]:
    block = extract_risk_factors_with_boxes_block(text)
    hay = block if block is not None else text
    out: list[tuple[float, float, float, float]] = []
    for m in _BOX_PATTERN_INT.finditer(hay):
        if "." in m.group(0):
            continue
        a, b, c, d = (int(m.group(i)) for i in range(1, 5))
        q = _quad_from_int_match_groups(a, b, c, d)
        if q is not None:
            out.append(q)
    for m in _BOX_PATTERN_FLOAT.finditer(hay):
        if "." not in m.group(0):
            continue
        out.append(tuple(float(m.group(i)) for i in range(1, 5)))
    return out


def iou_xyxy_norm(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU for axis-aligned boxes in the same normalized coordinate system."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    aa = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    bb = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = aa + bb - inter
    if union <= 0:
        return 0.0
    return inter / union
