from src.metrics.utils import calc_si_snr


def get_true_predictions(pred_s1, pred_s2, mix, speaker1, speaker2):
    """
    Returns preds for speaker1 and speaker2 with the biggest si_snri

    Args:
        pred_s1 (Tensor): prediction 1.
        pred_s2 (Tensor): prediction 2.
        mix (Tensor): original mixed data.
        speaker1 (Tensor): Ground-truth speaker1 record.
        speaker2 (Tensor): Ground-truth speaker2 record.
    Returns:
        Prediction_for_s1 (Tensor): the most appropriate prediction for speaker1.
        Prediction_for_s2 (Tensor): the most appropriate prediction for speaker2.
    """

    permutation_11_22 = (
        calc_si_snr(pred_s1, speaker1)
        - calc_si_snr(mix, speaker1)
        + calc_si_snr(pred_s2, speaker2)
        - calc_si_snr(mix, speaker2)
    ).mean()
    permutation_12_21 = (
        calc_si_snr(pred_s1, speaker2)
        - calc_si_snr(mix, speaker2)
        + calc_si_snr(pred_s2, speaker1)
        - calc_si_snr(mix, speaker1)
    ).mean()

    if permutation_11_22 < permutation_12_21:
        return pred_s2, pred_s1
    return pred_s1, pred_s2
