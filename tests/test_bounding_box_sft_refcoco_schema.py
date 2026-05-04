"""PaDT / conversation-style RefCOCO rows for bbox SFT."""

from __future__ import annotations

from PIL import Image


def test_padt_hub_objects_as_top_level_list() -> None:
    """PaDT-MLLM/RefCOCO uses ``objects`` as a list of instances (not only ``{value: [...]}``)."""
    from train.bounding_box_sft_dataset import hf_row_to_bbox_sft_sample

    img = Image.new("RGB", (640, 424))
    row = {
        "image": img,
        "conversations": [
            {
                "from": "human",
                "value": (
                    "Please carefully check the image and detect the object this sentence describes: "
                    '"black chair by table".'
                ),
            },
        ],
        "objects": [
            {
                "bbox": [0.7, 0.54, 0.98, 0.88],
                "label": "black chair by table",
                "iscrowd": 0,
            }
        ],
    }
    sample = hf_row_to_bbox_sft_sample(row)
    assert "black chair by table" in sample.messages[1]["content"][0]["text"]


def test_padt_style_conversation_phrase_and_objects_bbox() -> None:
    from train.bounding_box_sft_dataset import hf_row_to_bbox_sft_sample

    img = Image.new("RGB", (100, 100))
    row = {
        "image": img,
        "conversations": [
            {
                "from": "human",
                "value": (
                    "Please carefully check the image and detect the object this sentence describes: "
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


def test_image_as_filesystem_path_string(tmp_path) -> None:
    """PaDT-style hubs often store ``image`` as a path string, not a decoded PIL."""
    from train.bounding_box_sft_dataset import hf_row_to_bbox_sft_sample

    p = tmp_path / "im.png"
    Image.new("RGB", (40, 30), color=(1, 2, 3)).save(p)
    row = {"image": str(p), "utterance": "the red ball", "bbox": [0.0, 0.0, 0.2, 0.2]}
    sample = hf_row_to_bbox_sft_sample(row)
    assert sample.image.size == (40, 30)
    assert "red ball" in sample.messages[1]["content"][0]["text"]


def test_coco_train2014_basename_resolves_under_train2014(tmp_path) -> None:
    """Official MSCOCO layout: ``COCO_train2014_*.jpg`` under ``<root>/train2014/``."""
    from train.bounding_box_sft_dataset import (
        hf_row_to_bbox_sft_sample,
        reset_bbox_hf_coco_root_for_tests,
        set_bbox_hf_coco_root_for_tests,
    )

    reset_bbox_hf_coco_root_for_tests()
    try:
        coco = tmp_path / "coco"
        (coco / "train2014").mkdir(parents=True)
        name = "COCO_train2014_000000000001.jpg"
        Image.new("RGB", (80, 60)).save(coco / "train2014" / name)
        set_bbox_hf_coco_root_for_tests(coco)
        row = {"image": name, "utterance": "a cat", "bbox": [0.0, 0.0, 0.5, 0.5]}
        sample = hf_row_to_bbox_sft_sample(row)
        assert sample.image.size == (80, 60)
    finally:
        reset_bbox_hf_coco_root_for_tests()


def test_coco_train2014_basename_falls_back_to_train2017(tmp_path) -> None:
    """If ``train2014/`` is missing, same basename may still resolve under ``train2017/``."""
    from train.bounding_box_sft_dataset import (
        hf_row_to_bbox_sft_sample,
        reset_bbox_hf_coco_root_for_tests,
        set_bbox_hf_coco_root_for_tests,
    )

    reset_bbox_hf_coco_root_for_tests()
    try:
        coco = tmp_path / "coco"
        (coco / "train2017").mkdir(parents=True)
        name = "COCO_train2014_000000000099.jpg"
        Image.new("RGB", (10, 11)).save(coco / "train2017" / name)
        set_bbox_hf_coco_root_for_tests(coco)
        row = {"image": name, "utterance": "a cat", "bbox": [0.0, 0.0, 0.5, 0.5]}
        sample = hf_row_to_bbox_sft_sample(row)
        assert sample.image.size == (10, 11)
    finally:
        reset_bbox_hf_coco_root_for_tests()


def test_coco_basename_resolves_via_bbox_sft_image_root_env(tmp_path, monkeypatch) -> None:
    from train.bounding_box_sft_dataset import hf_row_to_bbox_sft_sample, reset_bbox_hf_coco_root_for_tests

    monkeypatch.setenv("BBOX_SFT_IMAGE_ROOT", str(tmp_path))
    reset_bbox_hf_coco_root_for_tests()
    try:
        (tmp_path / "train2017").mkdir()
        name = "COCO_train2017_000000000002.jpg"
        Image.new("RGB", (10, 12)).save(tmp_path / "train2017" / name)
        row = {"image": name, "utterance": "dog", "bbox": [0.1, 0.1, 0.9, 0.9]}
        sample = hf_row_to_bbox_sft_sample(row)
        assert sample.image.size == (10, 12)
    finally:
        reset_bbox_hf_coco_root_for_tests()
        monkeypatch.delenv("BBOX_SFT_IMAGE_ROOT", raising=False)
