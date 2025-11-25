import itertools

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_si_snr


class SI_SNRiMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, preds: Tensor, speakers: Tensor, mix: Tensor, **kwargs):
        preds_s1 = preds[..., 0, :]
        preds_s2 = preds[..., 1, :]
        speaker1 = speakers[..., 0, :]
        speaker2 = speakers[..., 1, :]

        data = {
            "preds_s1": preds_s1,
            "preds_s2": preds_s2,
            "speaker_1": speaker1,
            "speaker_2": speaker2,
        }

        batch_metrics = []
        for val_ind in range(preds_s1.shape[0]):
            metrics = []
            for perm in itertools.permutations(range(2)):
                curr_metric = 0
                for ind_target, ind_pred in enumerate(perm):
                    curr_metric += calc_si_snr(data[f"preds_s{ind_pred+1}"][val_ind], data[f"speaker_{ind_target+1}"][val_ind]) - calc_si_snr(mix[val_ind], data[f"speaker_{ind_target+1}"][val_ind])
                metrics.append(curr_metric / 2)
            batch_metrics.append(max(metrics))

        return sum(batch_metrics) / len(batch_metrics)
