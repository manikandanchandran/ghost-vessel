"""Offline tests with synthetic coastlines; no proprietary or external AIS input needed."""
import sys
from pathlib import Path
import unittest
import numpy as np
from shapely.geometry import box
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gvlib as g

class SegmentAwareCoastlineTest(unittest.TestCase):
    def setUp(self):
        # Land is lon [0.5, 1.5] and lat [0.5, 1.5].
        self.coast = box(.5, .5, 1.5, 1.5)

    @staticmethod
    def w(coords):
        return np.array([[lat, lon, 10, 90] for lat, lon in coords], dtype=float)

    def test_crossing_between_water_only_reports(self):
        # Neither endpoint lies on land, but the connecting segment crosses.
        trajectory = self.w([(1., 0.), (1., 2.)])[None]
        self.assertTrue(g.coastline_intersections(trajectory, self.coast)[0])

    def test_no_intersection(self):
        trajectory = self.w([(0., 0.), (0., 2.)])[None]
        self.assertFalse(g.coastline_intersections(trajectory, self.coast)[0])

    def test_point_on_land(self):
        trajectory = self.w([(1., 1.), (0., 2.)])[None]
        self.assertTrue(g.coastline_intersections(trajectory, self.coast)[0])

    def test_boundary_touching_is_flagged(self):
        trajectory = self.w([(.5, 0.), (.5, 2.)])[None]
        self.assertTrue(g.coastline_intersections(trajectory, self.coast)[0])

    def test_multiple_windows_and_batches(self):
        arr = np.array([self.w([(1., 0.), (1., 2.)]),
                        self.w([(0., 0.), (0., 2.)]),
                        self.w([(1., 1.), (2., 2.)])])
        np.testing.assert_array_equal(g.coastline_intersections(arr, self.coast, batch_windows=1),
                                      np.array([True, False, True]))

    def test_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            g.coastline_intersections(np.ones((2, 4)), self.coast)
        with self.assertRaises(ValueError):
            g.coastline_intersections(np.ones((2, 1, 4)), self.coast)

if __name__ == '__main__':
    unittest.main()
