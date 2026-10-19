import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TrackerDemoTests(unittest.TestCase):
    def test_demo_records_causal_equal_budget_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, "demo_tracker.py", "--output", tmp,
                            "--seeds", "41", "97"], check=True, capture_output=True, text=True)
            result = json.loads((Path(tmp) / "metrics.json").read_text())
            self.assertEqual(result["protocol"]["columns"], 16)
            self.assertEqual(result["protocol"]["frames"], 20)
            self.assertEqual(result["metrics"]["adaptive"], result["metrics"]["preset"])
            self.assertIn("symmetric", result["metrics"])
            self.assertIn("velocity", result["metrics"])
            self.assertIn("persistence", result["metrics"])
            self.assertLess(result["phase_average_vs_symmetric_max_abs"], 1e-12)
            self.assertIn("velocity_minus_adaptive_paired_mean", result["paired_comparison"])
            self.assertTrue((Path(tmp) / "tracker.svg").exists())


if __name__ == "__main__":
    unittest.main()
