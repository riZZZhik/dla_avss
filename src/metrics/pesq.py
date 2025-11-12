from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_pesq


class PESQMetric(BaseMetric):
    def __init__(self, fs, keep_same_device, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert fs == 8000 or fs == 16000, "fs must be 8000 or 16000"
        self.fs = fs
        self.mode = "nb" if fs == 8000 else "wb"
        self.keep_same_device = keep_same_device

    def __call__(self, preds: Tensor, target: Tensor, **kwargs):
        return calc_pesq(preds, target, self.fs, self.mode, self.keep_same_device)
