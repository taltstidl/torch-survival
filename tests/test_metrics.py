import unittest

import torch

from torch_survival.discrete import linear_time_points, time_to_idx
from torch_survival.metrics import c_index_antolini


class AntoliniUnitTest(unittest.TestCase):
    @staticmethod
    def _create_random_datasets(num_datasets, num_samples, num_times=6):
        torch.manual_seed(42)
        estimate = torch.rand((num_datasets, num_samples, num_times)).softmax(dim=-1)
        estimate = 1. - estimate.cumsum(dim=-1)
        event = torch.randint(0, 2, (num_datasets, num_samples)).bool()
        time = torch.rand((num_datasets, num_samples))
        return estimate, event, time

    def test_simple(self):
        event = torch.tensor([False, False, True, True])
        time = torch.tensor([0.3, 0.7, 0.5, 1.0])
        estimate = torch.tensor([
            [1.0, 0.5, 0.2, 0.1, 0.0, 0.0],
            [1.0, 0.8, 0.7, 0.3, 0.2, 0.0],
            [1.0, 0.9, 0.6, 0.3, 0.2, 0.0],
            [1.0, 1.0, 0.9, 0.8, 0.7, 0.0],
        ])
        time_points = linear_time_points(time, 6)
        time_idx = time_to_idx(event, time, time_points, event_mode='floor')
        metric = c_index_antolini(estimate, event, time, time_idx).item()
        self.assertAlmostEqual(metric, 1.0000, places=4)

    def test_random(self):
        import pandas as pd
        from pycox.evaluation import EvalSurv
        # Create a random batch of 8 datasets with 64 samples each
        estimate, event, time = AntoliniUnitTest._create_random_datasets(num_datasets=8, num_samples=64)
        # Verify that pycox and our implementation match
        for i in range(8):
            # Our implementation
            time_points = linear_time_points(time[i], 6)
            time_idx = time_to_idx(event[i], time[i], time_points, event_mode='floor')
            metric_ours = c_index_antolini(estimate[i], event[i], time[i], time_idx).item()
            # pycox implementation
            surv_df = pd.DataFrame(estimate[i].numpy().T, index=time_points.numpy())
            ev = EvalSurv(surv_df, time[i].numpy(), event[i].numpy(), censor_surv='km')
            metric_pycox = ev.concordance_td(method='antolini')
            # Equality check
            self.assertAlmostEqual(metric_ours, metric_pycox, places=4)

    def test_batched(self):
        import pandas as pd
        from pycox.evaluation import EvalSurv
        # Create a random batch of 8 datasets with 64 samples each
        estimate, event, time = AntoliniUnitTest._create_random_datasets(num_datasets=8, num_samples=64)
        # Our implementation
        time_points = linear_time_points(time, 6)
        time_idx = time_to_idx(event, time, time_points, event_mode='floor')
        metric_ours = c_index_antolini(estimate, event, time, time_idx).tolist()
        # Verify that pycox and our implementation match
        for i in range(8):
            # pycox implementation
            surv_df = pd.DataFrame(estimate[i].numpy().T, index=time_points.numpy())
            ev = EvalSurv(surv_df, time[i].numpy(), event[i].numpy(), censor_surv='km')
            metric_pycox = ev.concordance_td(method='antolini')
            # Equality check
            self.assertAlmostEqual(metric_ours[i], metric_pycox, places=4)
