from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_si_snr


class SI_SNRiMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, preds: Tensor, target: Tensor, original_mix: Tensor, **kwargs):
        return calc_si_snr(preds, target) - calc_si_snr(original_mix, target)
