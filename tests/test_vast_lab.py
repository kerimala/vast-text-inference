import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "vast_lab.py"
REPO_ROOT = MODULE_PATH.parents[1]
FP8_CONFIG_PATH = REPO_ROOT / "config" / "orcarouter-qwen38-fp8-openwebui.json"
FP8_PROVISIONING_PATH = REPO_ROOT / "provisioning" / "orcarouter-qwen38-fp8-openwebui.sh"
SPEC = importlib.util.spec_from_file_location("vast_lab", MODULE_PATH)
vast_lab = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(vast_lab)


class OfferValidationTests(unittest.TestCase):
    def setUp(self):
        self.config = vast_lab.load_config(vast_lab.DEFAULT_CONFIG)
        self.offer = {
            "id": 42,
            "machine_id": 7,
            "gpu_name": "A100 SXM4",
            "gpu_ram": 81920,
            "num_gpus": 1,
            "cpu_arch": "amd64",
            "reliability": 0.98,
            "inet_down": 1200,
            "inet_down_cost": 0.01,
            "direct_port_count": 2,
            "disk_space": 500,
            "dph_total": 0.62,
            "verified": False,
        }

    def test_valid_unverified_offer_is_accepted(self):
        self.assertEqual(vast_lab.validate_offer(self.config, self.offer), [])

    def test_expensive_offer_is_rejected(self):
        self.offer["dph_total"] = self.config["max_hourly_usd"] + 0.01
        self.assertIn("Hourly price exceeds the cap", vast_lab.validate_offer(self.config, self.offer))

    def test_wrong_cpu_architecture_is_rejected(self):
        self.offer["cpu_arch"] = "arm64"
        self.assertIn("CPU architecture differs", vast_lab.validate_offer(self.config, self.offer))

    def test_wrong_gpu_is_rejected(self):
        self.offer["gpu_name"] = "RTX 5090"
        self.assertIn("GPU model is outside the allowlist", vast_lab.validate_offer(self.config, self.offer))

    def test_quote_includes_first_model_download(self):
        quote = vast_lab.cost_quote(self.config, self.offer, 120)
        self.assertEqual(quote["estimated_rental_cost_usd"], 1.24)
        self.assertEqual(quote["estimated_first_model_download_cost_usd"], 0.75)
        self.assertEqual(quote["estimated_total_cost_usd"], 1.99)


class OrcaRouterFp8ProfileTests(unittest.TestCase):
    def setUp(self):
        self.config = vast_lab.load_config(FP8_CONFIG_PATH)
        self.offer = {
            "id": 84,
            "machine_id": 9,
            "gpu_name": "RTX 6000Ada",
            "gpu_ram": 49140,
            "num_gpus": 1,
            "cpu_arch": "amd64",
            "compute_cap": 890,
            "reliability": 0.999,
            "inet_down": 600,
            "inet_down_cost": 0.0026041666666666665,
            "direct_port_count": 16,
            "disk_space": 500,
            "dph_total": 0.63,
            "verified": True,
        }

    def test_native_fp8_ada_offer_is_accepted(self):
        self.assertEqual(vast_lab.validate_offer(self.config, self.offer), [])

    def test_live_verification_shape_is_accepted(self):
        self.offer.pop("verified")
        self.offer["verification"] = "verified"
        self.offer["vericode"] = 1
        self.assertEqual(vast_lab.validate_offer(self.config, self.offer), [])

    def test_ampere_compute_capability_is_rejected(self):
        self.offer["compute_cap"] = 860
        self.assertIn(
            "GPU compute capability is too low",
            vast_lab.validate_offer(self.config, self.offer),
        )

    def test_four_hour_quote_includes_model_download(self):
        quote = vast_lab.cost_quote(self.config, self.offer, 240)
        self.assertEqual(quote["estimated_rental_cost_usd"], 2.52)
        self.assertEqual(quote["estimated_first_model_download_cost_usd"], 0.09)
        self.assertEqual(quote["estimated_total_cost_usd"], 2.61)

    def test_provisioning_is_pinned_and_loopback_only(self):
        script = FP8_PROVISIONING_PATH.read_text(encoding="utf-8")
        self.assertIn(self.config["model_revision"], script)
        self.assertIn(f'OPENWEBUI_VERSION="{self.config["openwebui_version"]}"', script)
        self.assertIn("--language-model-only", script)
        self.assertIn("--kv-cache-dtype fp8", script)
        self.assertNotIn("--speculative-config", script)
        self.assertIn("serve --host 127.0.0.1 --port 3000", script)
        self.assertNotIn("--host 0.0.0.0", script)

    def test_openwebui_web_search_is_keyless_and_smoke_tested(self):
        script = FP8_PROVISIONING_PATH.read_text(encoding="utf-8")
        self.assertIn("export ENABLE_WEB_SEARCH=true", script)
        self.assertIn('readonly OPENWEBUI_WEB_SEARCH_ENGINE="duckduckgo"', script)
        self.assertIn(
            'export WEB_SEARCH_ENGINE="${OPENWEBUI_WEB_SEARCH_ENGINE}"',
            script,
        )
        self.assertIn("export DDGS_BACKEND=auto", script)
        self.assertIn("export WEB_SEARCH_RESULT_COUNT=", script)
        self.assertIn("export WEB_SEARCH_CONCURRENT_REQUESTS=", script)
        self.assertIn("export BYPASS_WEB_SEARCH_WEB_LOADER=false", script)
        self.assertIn("export ENABLE_WEB_LOADER_SSL_VERIFICATION=true", script)
        self.assertIn("from ddgs import DDGS", script)
        self.assertIn("for _ in $(seq 1 3)", script)
        self.assertIn("if web_search_ok; then", script)
        self.assertIn("testing-web-search", script)
        self.assertNotIn("SEARCH_API_KEY", script)

    def test_remote_failed_marker_aborts_wait_immediately(self):
        helper = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("state/failed; then exit 42", helper)
        self.assertIn('probe.returncode == 42', helper)
        self.assertIn('Remote provisioning reported failure.', helper)


if __name__ == "__main__":
    unittest.main()
