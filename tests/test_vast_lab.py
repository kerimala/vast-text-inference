import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "vast_lab.py"
SPEC = importlib.util.spec_from_file_location("vast_lab", MODULE_PATH)
vast_lab = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(vast_lab)


class OfferValidationTests(unittest.TestCase):
    def setUp(self):
        self.config = vast_lab.load_config(vast_lab.DEFAULT_CONFIG)
        self.offer = {
            "id": 42,
            "gpu_name": "A100 SXM4",
            "gpu_ram": 81920,
            "num_gpus": 1,
            "cpu_arch": "amd64",
            "reliability": 0.98,
            "inet_down": 1200,
            "direct_port_count": 2,
            "disk_space": 500,
            "dph_total": 0.62,
            "verified": False,
        }

    def test_valid_unverified_offer_is_accepted(self):
        self.assertEqual(vast_lab.validate_offer(self.config, self.offer), [])

    def test_expensive_offer_is_rejected(self):
        self.offer["dph_total"] = 0.76
        self.assertIn("Hourly price exceeds the cap", vast_lab.validate_offer(self.config, self.offer))

    def test_wrong_cpu_architecture_is_rejected(self):
        self.offer["cpu_arch"] = "arm64"
        self.assertIn("CPU architecture differs", vast_lab.validate_offer(self.config, self.offer))

    def test_wrong_gpu_is_rejected(self):
        self.offer["gpu_name"] = "RTX 5090"
        self.assertIn("GPU model is outside the allowlist", vast_lab.validate_offer(self.config, self.offer))


if __name__ == "__main__":
    unittest.main()
