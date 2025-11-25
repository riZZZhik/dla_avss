from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_sdr


class SDRiMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, preds: Tensor, speakers: Tensor, mix: Tensor, **kwargs):
        # Not pit, because sdrI

        preds_s1 = preds[..., 0, :]
        preds_s2 = preds[..., 1, :]
        speaker1 = speakers[..., 0, :]
        speaker2 = speakers[..., 1, :]

        val_per1 = (
            calc_sdr(preds_s1, speaker1)
            - calc_sdr(mix, speaker1)
            + calc_sdr(preds_s2, speaker2)
            - calc_sdr(mix, speaker2)
        ) * 0.5
        val_per2 = (
            calc_sdr(preds_s1, speaker2)
            - calc_sdr(mix, speaker2)
            + calc_sdr(preds_s2, speaker1)
            - calc_sdr(mix, speaker1)
        ) * 0.5

        return torch.maximum(val_per1, val_per2).mean()
