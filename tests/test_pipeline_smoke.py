from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PipelineSmokeTest(unittest.TestCase):
    def test_pipeline_regenerates_expected_outputs(self):
        subprocess.run([sys.executable, "scripts/run_pipeline.py"], cwd=ROOT, check=True)
        metrics = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(metrics["metrics"]), 3)
        self.assertTrue((ROOT / "docs" / "index.html").exists())
        self.assertGreaterEqual(len(list((ROOT / "docs" / "figures").glob("*.svg"))), 3)
        self.assertTrue((ROOT / "data" / "public_safe_results.csv").exists())


if __name__ == "__main__":
    unittest.main()
