import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset


class AVSSDataset(BaseDataset):
  
    def __init__(
        self,
        data_dir,
        index_dir,
        split,
        *args,
        visual_dim: int = 96,
        **kwargs,
    ):
        self._data_dir = Path(data_dir)
        self._index_dir = Path(index_dir)
        self.visual_dim = visual_dim

        index = self._get_or_load_index(split)
        self._index = index

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, split):
        index_path = self._index_dir / f"{split}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = []
            audio_path = self._data_dir / "audio"
            video_path = self._data_dir / "mouths"

            assert (
                audio_path / split / "mix"
            ).exists(), f"Path {str(audio_path / split / 'mix')} doesn't exist"

            mix_files = list((audio_path / split / "mix").iterdir())
            for mix_file in tqdm(mix_files):
                mix_name = mix_file.name
                speaker1_id = mix_name[: mix_name.find("_")]
                speaker2_id = mix_name[mix_name.find("_") + 1 : mix_name.find(".")]
                speaker1_audio_path = audio_path / split / "s1" / mix_name
                speaker2_audio_path = audio_path / split / "s2" / mix_name
                speaker1_video_path = video_path / f"{speaker1_id}.npz"
                speaker2_video_path = video_path / f"{speaker2_id}.npz"

                t_info = torchaudio.info(mix_file)
                audio_len = t_info.num_frames / t_info.sample_rate

                row = {
                    "audio_path": str(mix_file),
                    "speaker1_video_path": str(speaker1_video_path),
                    "speaker2_video_path": str(speaker2_video_path),
                    "audio_len": audio_len,
                }

                if speaker1_audio_path.exists() and speaker2_audio_path.exists():
                    row["speaker1_audio_path"] = str(speaker1_audio_path)
                    row["speaker2_audio_path"] = str(speaker2_audio_path)

                index.append(row)

            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    @staticmethod
    def _load_vis_npz(path: Path) -> torch.Tensor:
        """
        npz files shaped to tensor
        """
        npz = np.load(str(path))
        arr = npz[npz.files[0]]

        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        elif arr.ndim == 3:
            arr = arr.mean(axis=(0, 1))
        elif arr.ndim == 4:
            arr = arr.mean(axis=(0, 2, 3))
        else:
            arr = arr.reshape(-1)

        return torch.from_numpy(arr).float()

    def __getitem__(self, idx: int) -> dict:
        """
        npz added to paths
        """
        sample = super().__getitem__(idx)

        # берём строку из исходного индекса, где лежат пути к видео
        row = self._index[idx]
        v1_path = Path(row["speaker1_video_path"])
        v2_path = Path(row["speaker2_video_path"])

        if v1_path.exists():
            sample["vis_s1"] = self._load_vis_npz(v1_path)
        if v2_path.exists():
            sample["vis_s2"] = self._load_vis_npz(v2_path)

        if self.visual_dim is not None and "vis_s1" in sample:
            assert sample["vis_s1"].shape[-1] == self.visual_dim, \
                f"Expected visual_dim={self.visual_dim}, got {sample['vis_s1'].shape[-1]}"

        return sample
