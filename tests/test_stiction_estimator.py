import os
import sys
import unittest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.video_predictor import FramePrediction, estimate_stiction


def _results(values):
    return [
        FramePrediction(index, index * 100.0, float(value), "test")
        for index, value in enumerate(values)
    ]


class StictionEstimatorTest(unittest.TestCase):
    def test_level_1_for_smooth_motion(self):
        values = [index * 0.8 for index in range(101)]
        self.assertEqual(estimate_stiction(_results(values)).level, 1)

    def test_level_2_for_intermittent_stalls(self):
        values = []
        angle = 0.0
        for index in range(101):
            if index % 15 in (5, 6, 7):
                pass
            elif index % 15 == 8:
                angle += 3.2
            else:
                angle += 0.8
            values.append(min(angle, 80.0))
        self.assertEqual(estimate_stiction(_results(values)).level, 2)

    def test_level_3_for_stick_slip_motion(self):
        values = [min((index // 10) * 8.0, 80.0) for index in range(101)]
        self.assertEqual(estimate_stiction(_results(values)).level, 3)


if __name__ == "__main__":
    unittest.main()
