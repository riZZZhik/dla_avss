import json
import os
import sys
import urllib.parse as ul
from pathlib import Path

import torchaudio
from huggingface_hub import hf_hub_download
from torch import tensor
from tqdm.auto import tqdm

from src.metrics.utils import calc_pesq, calc_sdr, calc_si_snr, calc_stoi
from src.utils.utils import get_true_predictions


def download_custom_dataset():
    """
    Downloads custom dataset from url sys.argv[2] into current directory
    Reference: https://gist.github.com/Yegorov/dc61c42aa4e89e139cd8248f59af6b3e
    """
    base_url = (
        "https://cloud-api.yandex.net:443/v1/disk/public/resources/download?public_key="
    )
    url = ul.quote_plus(sys.argv[2])
    res = os.popen("wget -qO - {}{}".format(base_url, url)).read()
    json_res = json.loads(res)
    filename = ul.parse_qs(ul.urlparse(json_res["href"]).query)["filename"][0]
    os.system("wget '{}' -O '{}'".format(json_res["href"], filename))


def download_checkpoint():
    """
    Downloads best model checkpoint into folder = sys.argv[2]
    from repo = = sys.argv[3]
    and model_file = sys.argv[4]
    """
    hf_hub_download(
        repo_id=sys.argv[3],
        filename=sys.argv[4],
        local_dir=sys.argv[2],
    )


def calc_metrics():
    """
    Calculates metrics on model output from directory = sys.argv[2]
    and ground-truth records from dirs s1 and s2 from directory = sys.argv[3]
    and mix record from directory = sys.argv[4]
    """
    output_s1_dir = Path(sys.argv[2]) / "s1"
    output_s2_dir = Path(sys.argv[2]) / "s2"
    groundtruth_s1_dir = Path(sys.argv[3]) / "s1"
    groundtruth_s2_dir = Path(sys.argv[3]) / "s2"

    with_mix = False
    if len(sys.argv) > 4:
        with_mix = True

    output_s1_filenames = os.listdir(output_s1_dir)
    metrics = {
        "SI-SNR": [],
        "PESQ": [],
        "STOI": [],
    }

    if with_mix:
        mix_dir = Path(sys.argv[4])
        metrics["SI-SNRi"] = []
        metrics["SDRi"] = []

    for output_s1_filename in tqdm(output_s1_filenames):
        predicted_s1, _ = torchaudio.load(str(output_s1_dir / output_s1_filename))
        predicted_s2, _ = torchaudio.load(str(output_s2_dir / output_s1_filename))
        groundtruth_s1, _ = torchaudio.load(
            str(groundtruth_s1_dir / output_s1_filename)
        )
        groundtruth_s2, _ = torchaudio.load(
            str(groundtruth_s2_dir / output_s1_filename)
        )

        predicted_s1 = predicted_s1[0:1, :]  # remove all channels but the first
        predicted_s2 = predicted_s2[0:1, :]
        groundtruth_s1 = groundtruth_s1[0:1, :]
        groundtruth_s2 = groundtruth_s2[0:1, :]

        predicted_s1, predicted_s2 = get_true_predictions(
            predicted_s1, predicted_s2, groundtruth_s1, groundtruth_s2
        )

        metrics["SI-SNR"].append(
            (
                calc_si_snr(predicted_s1, groundtruth_s1)
                + calc_si_snr(predicted_s2, groundtruth_s2)
            )
            / 2
        )
        metrics["PESQ"].append(
            (
                calc_pesq(predicted_s1, groundtruth_s1)
                + calc_pesq(predicted_s2, groundtruth_s2)
            )
            / 2
        )
        metrics["STOI"].append(
            (
                calc_stoi(predicted_s1, groundtruth_s1)
                + calc_stoi(predicted_s2, groundtruth_s2)
            )
            / 2
        )

        if with_mix:
            mix_audio, _ = torchaudio.load(str(mix_dir / output_s1_filename))
            mix_audio = mix_audio[0:1, :]

            metrics["SI-SNRi"].append(
                (
                    calc_si_snr(predicted_s1, groundtruth_s1)
                    - calc_si_snr(mix_audio, groundtruth_s1)
                    + calc_si_snr(predicted_s2, groundtruth_s2)
                    - calc_si_snr(mix_audio, groundtruth_s2)
                )
                / 2
            )
            metrics["SDRi"].append(
                (
                    calc_sdr(predicted_s1, groundtruth_s1)
                    - calc_sdr(mix_audio, groundtruth_s1)
                    + calc_sdr(predicted_s2, groundtruth_s2)
                    - calc_sdr(mix_audio, groundtruth_s2)
                )
                / 2
            )

    for metric_name in metrics.keys():
        print(f"Mean {metric_name}: {tensor(metrics[metric_name]).mean()}")


if __name__ == "__main__":
    func_name = sys.argv[1]
    globals()[func_name]()
