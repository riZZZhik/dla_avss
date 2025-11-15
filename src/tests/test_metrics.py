import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch  # noqa: E402
from torch.testing import assert_close  # noqa: E402

from src.metrics.utils import calc_pesq, calc_sdr, calc_si_snr, calc_stoi  # noqa: E402


def test_si_snr():
    preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
    target = torch.tensor([3.0, -0.5, 2.0, 7.0])
    assert_close(calc_si_snr(preds, target), torch.tensor(15.0918), atol=1e-4, rtol=0)


def test_sdr():
    preds = torch.tensor([2.5, 0.0, 1.9, 8.0])
    target = torch.tensor([3.0, -0.5, 2.0, 7.0])
    assert_close(calc_sdr(preds, target), torch.tensor(153.5253), atol=1e-4, rtol=0)


def test_pesq():
    preds = torch.tensor([2.5, 0.0, 1.9, 8.0] * 4000)
    target = torch.tensor([3.0, -0.5, 2.0, 7.0] * 4000)
    assert_close(
        calc_pesq(preds, target, fs=16000, mode="wb", keep_same_device=True),
        torch.tensor(4.3343),
        atol=1e-4,
        rtol=0,
    )


def test_stoi():
    preds = torch.tensor([2.5, 0.0, 1.9, 8.0] * 4000, dtype=torch.float32)
    target = torch.tensor([3.0, -0.5, 2.0, 7.0] * 4000, dtype=torch.float32)
    assert_close(
        calc_stoi(preds, target, fs=16000, keep_same_device=True),
        torch.tensor(0.9960),
        atol=1e-4,
        rtol=0,
    )


def test():
    test_si_snr()
    test_sdr()
    test_pesq()
    test_stoi()
    print("Success")


if __name__ == "__main__":
    test()
