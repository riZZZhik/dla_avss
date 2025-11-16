import torch
from torch import Tensor, nn
from torchmetrics.audio import PermutationInvariantTraining

from src.metrics import calc_si_snr


class SI_SNR_loss(nn.Module):
    def __init__(self, upit=True) -> None:
        super().__init__()

        self.upit = (
            PermutationInvariantTraining(
                calc_si_snr, mode="speaker-wise", eval_func="max"
            )
            if upit
            else None
        )

    def forward(self, preds, speakers, **batch) -> Tensor:
        # BxSpeakersxL
        if self.upit is None:
            preds_s1 = preds[..., 0, :]
            preds_s2 = preds[..., 1, :]
            speaker1 = speakers[..., 0, :]
            speaker2 = speakers[..., 1, :]
            loss = (
                -(
                    calc_si_snr(preds_s1, speaker1) + calc_si_snr(preds_s2, speaker2)
                ).sum()
                / 2
                / speakers.shape[0]
            )
        else:
            loss = -self.upit(preds, speakers)

        return {"loss": loss}
