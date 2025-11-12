from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_sdr


class SDRiMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, preds: Tensor, target: Tensor, original_mix: Tensor, **kwargs):
        return calc_sdr(preds, target) - calc_sdr(original_mix, target)
