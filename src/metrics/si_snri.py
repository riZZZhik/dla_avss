from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_si_snr


class SI_SNRiMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, preds: Tensor, speakers: Tensor, mix: Tensor, **kwargs):
        # Not pit, because snrI

        preds_s1 = preds[..., 0, :]
        preds_s2 = preds[..., 1, :]
        speaker1 = speakers[..., 0, :]
        speaker2 = speakers[..., 1, :]

        val_per1 = (
            calc_si_snr(preds_s1, speaker1)
            - calc_si_snr(mix, speaker1)
            + calc_si_snr(preds_s2, speaker2)
            - calc_si_snr(mix, speaker2)
        ).mean()
        val_per2 = (
            calc_si_snr(preds_s1, speaker2)
            - calc_si_snr(mix, speaker2)
            + calc_si_snr(preds_s2, speaker1)
            - calc_si_snr(mix, speaker1)
        ).mean()

        return max(val_per1, val_per2)
