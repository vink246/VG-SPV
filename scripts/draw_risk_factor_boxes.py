"""
Draw phrase-labeled boxes from Method-2 style lines on an image.

Input lines (one per risk factor), e.g.::

  phrase: "camera" | box: [0494, 0753, 0539, 0800]
  phrase: "documents" | box: [0317, 0861, 0421, 0972]
  phrase: "briefcase" | box: [no_box]

Normalized coordinates use the same rules as ``train.tag_parsing`` (integer grid 0..1000, etc.).
Lines with ``no_box`` are skipped (no rectangle). Optional ``no risk`` lines are ignored.

Example::

  python scripts/draw_risk_factor_boxes.py --image path/to/img.jpg --text_file boxes.txt --output out.png

  python scripts/draw_risk_factor_boxes.py --image img.jpg --text 'phrase: "hand" | box: [0.1, 0.2, 0.5, 0.6]' -o out.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PIL import Image, ImageDraw, ImageFont

from train.tag_parsing import parse_first_norm_box

_LINE_RE = re.compile(
    r'^\s*phrase:\s*"([^"]*)"\s*\|\s*box:\s*(.+?)\s*$',
    re.IGNORECASE,
)


def parse_phrase_box_lines(text: str) -> list[tuple[str, tuple[float, float, float, float] | None]]:
    """Return (phrase, xyxy_norm or None for no_box) for each matching line."""
    out: list[tuple[str, tuple[float, float, float, float] | None]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower() in {"no risk", "no_risk"}:
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        phrase = m.group(1)
        box_tok = m.group(2).strip()
        box_lower = box_tok.lower().replace(" ", "")
        if re.fullmatch(r"no_box", box_tok, flags=re.IGNORECASE) or box_lower in {"[no_box]", "no_box"}:
            out.append((phrase, None))
            continue
        q = parse_first_norm_box(box_tok)
        if q is None:
            continue
        out.append((phrase, q))
    return out


def _norm_xyxy_to_pixels(
    box: tuple[float, float, float, float], w: int, h: int
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    px0 = int(round(x0 * w))
    py0 = int(round(y0 * h))
    px1 = int(round(x1 * w))
    py1 = int(round(y1 * h))
    return (max(0, px0), max(0, py0), min(w - 1, px1), min(h - 1, py1))


def draw_on_image(
    image: Image.Image,
    entries: list[tuple[str, tuple[float, float, float, float] | None]],
    *,
    colors: list[str] | None = None,
) -> Image.Image:
    """Return a copy of ``image`` with rectangles and phrase labels for non-None boxes."""
    # Default to green outlines/labels unless custom colors are provided.
    colors = colors or ["#22c55e"]
    im = image.copy().convert("RGB")
    draw = ImageDraw.Draw(im)
    w, h = im.size
    try:
        font = ImageFont.truetype("arial.ttf", max(12, min(w, h) // 40))
    except OSError:
        font = ImageFont.load_default()
    for i, (phrase, q) in enumerate(entries):
        if q is None:
            continue
        color = colors[i % len(colors)]
        rect = _norm_xyxy_to_pixels(q, w, h)
        draw.rectangle(rect, outline=color, width=max(2, min(w, h) // 256))
        label = phrase[:80] + ("…" if len(phrase) > 80 else "")
        tx, ty = rect[0], max(0, rect[1] - 2)
        draw.text((tx, ty), label, fill=color, font=font)
    return im


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Draw Method-2 phrase|box lines on an image.")
    p.add_argument("--image", type=str, required=True, help="Input image path.")
    p.add_argument("--text", type=str, default=None, help="Inline multi-line text (use \\n in shell carefully).")
    p.add_argument("--text_file", type=str, default=None, help="File containing phrase|box lines.")
    p.add_argument("-o", "--output", type=str, required=True, help="Output image path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.text and not args.text_file:
        raise SystemExit("Provide --text and/or --text_file.")
    chunks: list[str] = []
    if args.text_file:
        tf = Path(args.text_file).expanduser().resolve()
        if not tf.is_file():
            raise SystemExit(f"text_file not found: {tf}")
        chunks.append(tf.read_text(encoding="utf-8"))
    if args.text:
        chunks.append(args.text)
    text = "\n".join(chunks)
    entries = parse_phrase_box_lines(text)
    img_path = Path(args.image).expanduser().resolve()
    if not img_path.is_file():
        raise SystemExit(f"Image not found: {img_path}")
    im = Image.open(img_path).convert("RGB")
    out = draw_on_image(im, entries)
    outp = Path(args.output).expanduser().resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.save(outp)
    print(f"Wrote {outp} ({len([e for e in entries if e[1] is not None])} boxes drawn).")


if __name__ == "__main__":
    main()
