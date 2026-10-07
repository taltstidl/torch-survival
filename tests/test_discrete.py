import unittest

import torch

from torch_survival.discrete import linear_time_points, time_to_idx


class DiscreteUnitTest(unittest.TestCase):
    def test_deephit(self):
        n_times = 21
        time = torch.tensor([0, 10, 10.5, 11, 20, 20])
        time_points = linear_time_points(time, n_times)
        # Event times should move to the right cut points
        event = torch.tensor([True, True, True, True, True, True])
        idx = time_to_idx(event, time, time_points)
        self.assertListEqual(idx.tolist(), [0, 10, 11, 11, 20, 20])
        # Censoring times should move to the left cut points
        event = torch.tensor([False, False, False, False, False, True])
        idx = time_to_idx(event, time, time_points)
        self.assertListEqual(idx.tolist(), [0, 10, 10, 11, 20, 20])
