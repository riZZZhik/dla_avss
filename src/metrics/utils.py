import torch
from torchmetrics.functional.audio import (
    scale_invariant_signal_noise_ratio,
    signal_distortion_ratio,
)
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility


def calc_si_snr(preds, target) -> float:
    return scale_invariant_signal_noise_ratio(preds, target)


def calc_sdr(preds, target) -> float:
    return signal_distortion_ratio(preds, target)


def calc_pesq(preds, target, fs, mode, keep_same_device) -> float:
    return perceptual_evaluation_speech_quality(
        preds, target, fs, mode, keep_same_device=keep_same_device
    )


def calc_stoi(preds, target, fs, keep_same_device) -> float:
    return short_time_objective_intelligibility(
        preds, target, fs, keep_same_device=keep_same_device
    ).float()
