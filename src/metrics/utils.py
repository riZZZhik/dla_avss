import warnings

warnings.filterwarnings(
    "ignore", message=".*pkg_resources.*deprecated.*", category=UserWarning
)

import torch  # noqa: E402
from torchmetrics.functional.audio import (  # noqa: E402
    scale_invariant_signal_noise_ratio,
    signal_distortion_ratio,
)
from torchmetrics.functional.audio.pesq import (  # noqa: E402
    perceptual_evaluation_speech_quality,
)
from torchmetrics.functional.audio.stoi import (  # noqa: E402
    short_time_objective_intelligibility,
)


def calc_si_snr(preds, target, **kwargs) -> float:
    return scale_invariant_signal_noise_ratio(preds, target)


def calc_sdr(preds, target, **kwargs) -> float:
    val = signal_distortion_ratio(preds, target)
    val = torch.where(torch.isfinite(val), val, torch.zeros_like(val))
    return val


def calc_pesq(
    preds, target, fs=16000, mode="wb", keep_same_device=True, **kwargs
) -> float:
    return perceptual_evaluation_speech_quality(
        preds, target, fs, mode, keep_same_device=keep_same_device
    )


def calc_stoi(preds, target, fs=16000, keep_same_device=True, **kwargs) -> float:
    return short_time_objective_intelligibility(
        preds, target, fs, keep_same_device=keep_same_device
    ).float()
