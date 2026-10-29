"""Small scientific integrity checks with answers known independently of models."""
from pathlib import Path  # Build paths inside the project for temporary fixtures.
import sys  # Make the shared implementation importable without installing it globally.
import tempfile  # Keep synthetic Parquet fixtures separate from the real dataset.
import unittest  # Run assertions using Python's standard testing framework.
import numpy as np  # Construct a trajectory with a known geometric length.
import pandas as pd  # Build small labelled source fixtures.

ROOT = Path(__file__).resolve().parents[1]  # Locate this project from the test file.
sys.path.insert(0, str(ROOT / "src"))  # Import the exact code used by notebooks.
from dota_data import latest_rows  # Check the temporal feature extraction boundary.
from dota_movement import trajectory_pairs  # Check matched geometry and death boundaries.


class ScientificIntegrity(unittest.TestCase):
    def test_future_state_is_excluded(self):
        folder = ROOT / ".cache/tests"  # Keep test fixtures inside the authorised workspace.
        folder.mkdir(parents=True, exist_ok=True)  # Create the local temporary parent if needed.
        with tempfile.TemporaryDirectory(dir=folder) as directory:  # Automatically remove only this known local fixture directory.
            path = Path(directory) / "states.parquet"  # Store a tiny synthetic source table.
            frame = pd.DataFrame({"match_id": [1, 1, 1], "slot": [0, 0, 0], "tick": [70, 100, 130], "game_time": [0, 1, 2], "value": [7, 999, 9999]})  # Make future values obviously different.
            frame.to_parquet(path, index=False)  # Exercise the real Parquet filtering path.
            anchors = pd.DataFrame({"match_id": [1], "horizon_seconds": [600], "cutoff_tick": [100]})  # Declare that the safe anchor occurs at tick one hundred.
            result = latest_rows(path, frame.columns.tolist(), ["match_id", "slot"], anchors=anchors)  # Use the real extraction helper with an independent clock fixture.
            self.assertEqual(result.value.tolist(), [7])  # The value at the anchor and every later value must be excluded.

    def test_matched_corner_path(self):
        t = np.arange(61, dtype=float)  # Observe a simple path once a second for one minute.
        frame = pd.DataFrame({"game_time": t, "x": np.minimum(t, 30), "y": np.maximum(t - 30, 0), "is_alive": True, "move_speed": 10})  # Walk two perpendicular legs of known length.
        pairs, _, _ = trajectory_pairs(frame)  # Compare the same endpoints using the actual measurement implementation.
        pair = next(row for row in pairs if row["offset_seconds"] == 0)  # Inspect the grid containing the path's exact endpoints.
        self.assertAlmostEqual(pair["native_distance"], 60)  # The two straight legs total sixty coordinate units.
        self.assertAlmostEqual(pair["coarse_distance"], np.sqrt(30 ** 2 + 30 ** 2))  # The minute sample sees the diagonal, independently known from geometry.
        frame.loc[30, "is_alive"] = False  # Insert a death between the same endpoints.
        pairs, _, _ = trajectory_pairs(frame)  # Recalculate without changing endpoint positions.
        pair = next(row for row in pairs if row["offset_seconds"] == 0)  # Select the same coarse sampling phase.
        self.assertEqual(pair["matched_intervals"], 0)  # A path must not connect across the death boundary.


if __name__ == "__main__":  # Allow direct test execution from a terminal.
    unittest.main()  # Print failures clearly rather than producing a result table anyway.
