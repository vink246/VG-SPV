import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import yaml

from train.dpo_yaml import DPOTrainConfig, default_dpo_config_path, load_dpo_train_config, merge_dpo_train_config


class TestDPOYaml(unittest.TestCase):
    def test_default_yaml_loads(self):
        p = default_dpo_config_path()
        self.assertTrue(p.is_file(), f"Missing default config: {p}")
        cfg = load_dpo_train_config(p)
        self.assertIsInstance(cfg, DPOTrainConfig)
        self.assertTrue(cfg.use_vgfdpo)
        self.assertTrue(cfg.use_vdpo)
        self.assertEqual(cfg.grounding_mode, "semantic")
        self.assertAlmostEqual(cfg.alpha_format, 13.0)
        self.assertAlmostEqual(cfg.beta, 0.1)

    def test_merge_overrides(self):
        base = load_dpo_train_config(default_dpo_config_path())
        merged = merge_dpo_train_config(base, {"use_vgfdpo": False, "learning_rate": 1e-6})
        self.assertFalse(merged.use_vgfdpo)
        self.assertTrue(merged.use_vdpo)
        self.assertEqual(merged.learning_rate, 1e-6)

    def test_from_dict_rejects_unknown_key(self):
        with self.assertRaises(ValueError):
            DPOTrainConfig.from_dict({"model_name": "x", "not_a_field": 1})

    def test_roundtrip_yaml(self):
        cfg = load_dpo_train_config(default_dpo_config_path())
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "t.yaml"
            path.write_text(yaml.safe_dump(asdict(cfg), default_flow_style=False), encoding="utf-8")
            cfg2 = load_dpo_train_config(path)
            self.assertEqual(asdict(cfg2), asdict(cfg))


if __name__ == "__main__":
    unittest.main()
