from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_pesq


class PESQMetric(BaseMetric):
    def __init__(self, fs, keep_same_device, upit=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert fs == 8000 or fs == 16000, "fs must be 8000 or 16000"
        self.fs = fs
        self.mode = "nb" if fs == 8000 else "wb"
        self.keep_same_device = keep_same_device
        self.upit = upit

    def __call__(self, preds: Tensor, speakers: Tensor, **kwargs):
        # Not pit, because of the fs, mode, keep_same_device parameters
        preds_s1 = preds[..., 0, :]
        preds_s2 = preds[..., 1, :]
        speaker1 = speakers[..., 0, :]
        speaker2 = speakers[..., 1, :]

        if self.upit:
            val_per1 = (
                calc_pesq(
                    preds_s1,
                    speaker1,
                    fs=self.fs,
                    mode=self.mode,
                    keep_same_device=self.keep_same_device,
                )
                + calc_pesq(
                    preds_s2,
                    speaker2,
                    fs=self.fs,
                    mode=self.mode,
                    keep_same_device=self.keep_same_device,
                )
            ) * 0.5
            val_per2 = (
                calc_pesq(
                    preds_s1,
                    speaker2,
                    fs=self.fs,
                    mode=self.mode,
                    keep_same_device=self.keep_same_device,
                )
                + calc_pesq(
                    preds_s2,
                    speaker1,
                    fs=self.fs,
                    mode=self.mode,
                    keep_same_device=self.keep_same_device,
                )
            ) * 0.5

            val = torch.maximum(val_per1, val_per2).mean()
        else:
            val = (
                (
                    calc_pesq(
                        preds_s1,
                        speaker1,
                        fs=self.fs,
                        mode=self.mode,
                        keep_same_device=self.keep_same_device,
                    )
                    + calc_pesq(
                        preds_s2,
                        speaker2,
                        fs=self.fs,
                        mode=self.mode,
                        keep_same_device=self.keep_same_device,
                    )
                )
                * 0.5
            ).mean()

        return val
