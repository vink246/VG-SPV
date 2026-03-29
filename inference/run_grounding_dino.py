"""
Run Grounding DINO inference to generate grounding boxes for an image.

Usage examples (Linux / bash):
    # Config is optional; it is inferred from the checkpoint if omitted.
    python inference/run_grounding_dino.py \
        --checkpoint path/to/groundingdino_swint_ogc.pth \
        --image path/to/image.jpg \
        --text-prompt "dog . person ." \
        --output-json outputs/dino_boxes.json \
        --output-viz outputs/dino_boxes_viz.jpg

    # Or pass config explicitly:
    python inference/run_grounding_dino.py \
        --config path/to/GroundingDINO_SwinT_OGC.py \
        --checkpoint path/to/groundingdino_swint_ogc.pth \
        --image path/to/image.jpg \
        --text-prompt "dog . person ."

Relies on the official GroundingDINO package:
    pip install groundingdino opencv-python pillow
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run GroundingDINO text-conditioned detection on a single image."
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to GroundingDINO config (.py) file. If omitted, inferred from --checkpoint using the installed groundingdino package configs.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to GroundingDINO checkpoint (.pth).",
    )
    p.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image.",
    )
    p.add_argument(
        "--text-prompt",
        type=str,
        required=True,
        help='Text prompt with class names, e.g. "dog . person ."',
    )
    p.add_argument(
        "--box-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold on box logits.",
    )
    p.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="Threshold on text scores.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Device to run on (e.g. "cuda", "cuda:0", "cpu").',
    )
    p.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save detections as JSON.",
    )
    p.add_argument(
        "--output-viz",
        type=str,
        default=None,
        help="Optional path to save visualization image with boxes drawn.",
    )
    return p.parse_args()


# Known checkpoint basename (lowercase, no .pth) -> config filename in groundingdino package.
CHECKPOINT_TO_CONFIG: Dict[str, str] = {
    "groundingdino_swint_ogc": "GroundingDINO_SwinT_OGC.py",
    "groundingdino_swinb_cogcoor": "GroundingDINO_SwinB.cfg.py",
}


def get_groundingdino_config_dir() -> Path:
    """Return the config directory of the installed groundingdino package."""
    try:
        import groundingdino
    except ImportError as exc:
        raise ImportError(
            "groundingdino is not installed. Install it with `pip install groundingdino` "
            "or follow the official installation instructions."
        ) from exc
    pkg_dir = Path(groundingdino.__file__).resolve().parent
    config_dir = pkg_dir / "config"
    if not config_dir.is_dir():
        raise FileNotFoundError(
            f"groundingdino config directory not found: {config_dir}"
        )
    return config_dir


def resolve_config_from_checkpoint(checkpoint_path: str) -> str:
    """
    Resolve the GroundingDINO config path from a checkpoint path.
    Uses a known checkpoint->config mapping, then falls back to matching
    config filenames in the installed package by checkpoint stem.
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    stem = checkpoint_path.stem.lower()
    config_dir = get_groundingdino_config_dir()

    # Explicit mapping first.
    if stem in CHECKPOINT_TO_CONFIG:
        config_file = config_dir / CHECKPOINT_TO_CONFIG[stem]
        if config_file.is_file():
            return str(config_file)
        raise FileNotFoundError(
            f"Config {CHECKPOINT_TO_CONFIG[stem]} not found under {config_dir}"
        )

    # Fallback: list .py configs and pick one whose name is derived from stem.
    # e.g. groundingdino_swint_ogc -> look for GroundingDINO_SwinT_OGC.py
    candidates = list(config_dir.glob("GroundingDINO*.py"))
    if not candidates:
        raise FileNotFoundError(
            f"No GroundingDINO*.py configs found in {config_dir}. "
            "Use --config to specify the config path explicitly."
        )
    # Normalize stem for fuzzy match: remove "groundingdino_" prefix, then compare
    # config name without "GroundingDINO" prefix and .py (e.g. SwinT_OGC).
    stem_core = stem.replace("groundingdino_", "") if stem.startswith("groundingdino_") else stem
    for c in candidates:
        name_core = c.stem.replace("GroundingDINO", "").lstrip("_").lower().replace(".", "_")
        if stem_core in name_core or name_core in stem_core:
            return str(c)
    # Last resort: if only one config, use it; otherwise use the first alphabetically
    # (often SwinT_OGC) and warn.
    if len(candidates) == 1:
        return str(candidates[0])
    default = min(candidates, key=lambda p: p.name)
    print(
        f"Warning: could not infer config from checkpoint {checkpoint_path.name}; "
        f"using {default.name}. Pass --config to override."
    )
    return str(default)


def _patch_bert_get_head_mask_for_grounding_dino() -> None:
    """
    GroundingDINO's bertwarper does ``self.get_head_mask = bert_model.get_head_mask``.
    Newer ``transformers`` removed ``get_head_mask`` from ``BertModel`` / ``PreTrainedModel``,
    which raises AttributeError at model construction. Patch it before loading GroundingDINO.
    """
    from transformers.models.bert.modeling_bert import BertModel

    if hasattr(BertModel, "get_head_mask"):
        return

    try:
        from transformers.modeling_utils import PreTrainedModel
    except ImportError:
        PreTrainedModel = None  # type: ignore

    if PreTrainedModel is not None and hasattr(PreTrainedModel, "get_head_mask"):
        BertModel.get_head_mask = PreTrainedModel.get_head_mask
        return

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            convert = getattr(self, "_convert_head_mask_to_5d", None)
            if callable(convert):
                head_mask = convert(head_mask, num_hidden_layers)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    BertModel.get_head_mask = get_head_mask


def load_grounding_dino_model(
    config_path: str,
    checkpoint_path: str,
    device: str,
) -> Any:
    """
    Load GroundingDINO model using the official groundingdino API.
    Requires `groundingdino` to be installed.
    """
    _patch_bert_get_head_mask_for_grounding_dino()
    try:
        from groundingdino.util.inference import Model
    except ImportError as exc:
        raise ImportError(
            "groundingdino is not installed. Install it with `pip install groundingdino` "
            "or follow the official installation instructions."
        ) from exc

    model = Model(
        model_config_path=config_path,
        model_checkpoint_path=checkpoint_path,
        device=device,
    )
    return model


def run_grounding_dino(
    model: Any,
    image_path: str,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Run GroundingDINO on an image and return detections.

    Returns:
        boxes_xyxy: (N, 4) float32 array in absolute XYXY format.
        scores: (N,) float32 confidence scores.
        labels: list of N strings (class names / phrases).
        logits: (N,) float32 raw logits or aggregated logits (if available).
    """
    pil_image = Image.open(image_path).convert("RGB")
    image_array = np.asarray(pil_image)

    # The groundingdino Model wrapper exposes a predict method with this signature:
    #   predict_with_caption(
    #       image: np.ndarray,
    #       caption: str,
    #       box_threshold: float,
    #       text_threshold: float,
    #   )
    #
    # Some versions expose `predict` instead; we handle both.
    if hasattr(model, "predict_with_caption"):
        boxes, logits, phrases = model.predict_with_caption(
            image=image_array,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
    elif hasattr(model, "predict"):
        boxes, logits, phrases = model.predict(
            image=image_array,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
    else:
        raise AttributeError(
            "GroundingDINO model does not expose `predict_with_caption` or `predict`."
        )

    # boxes are typically in cxcywh normalized format in [0, 1]; convert to XYXY absolute.
    boxes = np.asarray(boxes, dtype=np.float32)
    h, w = image_array.shape[:2]
    boxes_abs = box_convert(
        torch.from_numpy(boxes),
        in_fmt="cxcywh",
        out_fmt="xyxy",
    )
    boxes_abs[:, [0, 2]] *= w
    boxes_abs[:, [1, 3]] *= h
    boxes_abs = boxes_abs.numpy()

    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    scores = 1.0 / (1.0 + np.exp(-logits))
    labels = [str(p) for p in phrases]

    return boxes_abs, scores, labels, logits


def to_serializable(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    labels: List[str],
    logits: np.ndarray,
) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    for i in range(len(labels)):
        x1, y1, x2, y2 = boxes_xyxy[i].tolist()
        detections.append(
            {
                "box_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(scores[i]),
                "logit": float(logits[i]),
                "label": labels[i],
            }
        )
    return detections


def draw_boxes(
    image_path: str,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    labels: List[str],
) -> np.ndarray:
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image with OpenCV: {image_path}")

    for i in range(len(labels)):
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
        score = float(scores[i])
        label = labels[i]
        color = (0, 255, 0)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{label}: {score:.2f}"
        ((tw, th), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image_bgr, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(
            image_bgr,
            text,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return image_bgr


def main() -> None:
    args = parse_args()

    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not Path(args.image).is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    config_path = args.config
    if config_path is None:
        config_path = resolve_config_from_checkpoint(args.checkpoint)
        print(f"Using config: {config_path}")
    elif not Path(config_path).is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    os.makedirs("outputs", exist_ok=True)

    print(f"Loading GroundingDINO from {config_path} and {args.checkpoint} on {args.device}...")
    model = load_grounding_dino_model(
        config_path=config_path,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    print(f"Running GroundingDINO on image {args.image} with prompt: {args.text_prompt!r}")
    boxes_xyxy, scores, labels, logits = run_grounding_dino(
        model=model,
        image_path=args.image,
        text_prompt=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    if len(labels) == 0:
        print("No detections above thresholds.")
    else:
        print(f"Detected {len(labels)} boxes:")
        for i, label in enumerate(labels):
            x1, y1, x2, y2 = boxes_xyxy[i]
            print(
                f"  {i}: {label} | score={scores[i]:.3f} | box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
            )

    detections = to_serializable(boxes_xyxy, scores, labels, logits)

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(detections, indent=2), encoding="utf-8")
        print(f"Saved detections JSON to {out_json}")

    if args.output_viz:
        if len(labels) == 0:
            print("Skipping visualization because there are no detections.")
        else:
            viz = draw_boxes(args.image, boxes_xyxy, scores, labels)
            out_img = Path(args.output_viz)
            out_img.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_img), viz)
            print(f"Saved visualization image to {out_img}")


if __name__ == "__main__":
    main()

