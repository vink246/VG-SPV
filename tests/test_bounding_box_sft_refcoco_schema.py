"""PaDT / conversation-style RefCOCO rows for bbox SFT."""

from __future__ import annotations

from PIL import Image


def test_padt_style_conversation_phrase_and_objects_bbox() -> None:
    from train.bounding_box_sft_dataset import hf_row_to_bbox_sft_sample

    img = Image.new("RGB", (100, 100))
    row = {
        "image": img,
        "conversations": [
            {
                "from": "human",
                "value": (
                    'Please carefully check the image and detect the object this sentence describes: '
                    '"black chair by table".'
                ),
            },
        ],
        "objects": {
            "kind": "list like",
            "value": [
                {
                    "bbox": [0.1, 0.2, 0.5, 0.6],
                    "label": "black chair by table",
                }
            ],
        },
    }
    sample = hf_row_to_bbox_sft_sample(row)
    assert "black chair by table" in sample.messages[1]["content"][0]["text"]


def test_flat_utterance_column() -> None:
    from train.bounding_box_sft_dataset import hf_row_to_bbox_sft_sample

    img = Image.new("RGB", (50, 50))
    row = {"image": img, "utterance": "the red ball", "bbox": [0.0, 0.0, 0.2, 0.2]}
    sample = hf_row_to_bbox_sft_sample(row)
    assert "red ball" in sample.messages[1]["content"][0]["text"]
