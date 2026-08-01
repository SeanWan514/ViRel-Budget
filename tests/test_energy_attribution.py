from __future__ import annotations

import unittest

from scripts.measurement.attribute_nvidia_energy import (
    PowerSamples,
    _attribute_call,
    _nearest_sample,
)


class EnergyAttributionIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        rows = [
            {"epoch": 1.0, "power_w": 100.0},
            {"epoch": 1.2, "power_w": 120.0},
            {"epoch": 1.4, "power_w": 140.0},
        ]
        self.samples = PowerSamples(rows=rows, epochs=[row["epoch"] for row in rows])

    def test_window_mean_matches_timestamp_rule(self) -> None:
        result = _attribute_call(
            {"start_epoch": 1.0, "end_epoch": 1.4},
            self.samples,
        )
        self.assertAlmostEqual(result["mean_power_w"], 120.0)
        self.assertAlmostEqual(result["measured_energy_joule"], 48.0)

    def test_short_call_uses_nearest_midpoint_sample(self) -> None:
        result = _attribute_call(
            {"start_epoch": 1.25, "end_epoch": 1.27},
            self.samples,
        )
        self.assertAlmostEqual(result["mean_power_w"], 120.0)
        self.assertAlmostEqual(result["measured_energy_joule"], 2.4)

    def test_nearest_lookup_checks_both_sides(self) -> None:
        self.assertEqual(_nearest_sample(self.samples, 1.31)["power_w"], 140.0)
        self.assertEqual(_nearest_sample(self.samples, 0.7)["power_w"], 100.0)


if __name__ == "__main__":
    unittest.main()
