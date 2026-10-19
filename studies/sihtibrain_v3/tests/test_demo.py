import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class DemoTests(unittest.TestCase):
    def test_demo_records_continuous_contrast_and_equal_budget_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, "demo.py", "--output", tmp,
                            "--trials", "18"], check=True, capture_output=True, text=True)
            data = json.loads((Path(tmp) / "metrics.json").read_text())
            self.assertGreater(data["adapted"]["late_flux_contrast_median"], 0.45)
            self.assertEqual(data["no_adaptation"]["late_flux_contrast_median"], 0.)
            self.assertEqual(len(data["signatures"]["adaptive_two"]), 3)
            self.assertIn("static_two", data["heldout_accuracy"])
            self.assertIn("symmetric_two", data["heldout_accuracy"])
            self.assertTrue((Path(tmp) / "gate1.svg").exists())
            self.assertTrue((Path(tmp) / "gate1.png").exists())


if __name__ == "__main__":
    unittest.main()
