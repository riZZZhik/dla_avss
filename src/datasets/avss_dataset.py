import json
from pathlib import Path

import torchaudio
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset


class AVSSDataset(BaseDataset):
    def __init__(self, data_dir, index_dir, split, *args, **kwargs):
        self._data_dir = Path(data_dir)
        self._index_dir = Path(index_dir)
        index = self._get_or_load_index(split)
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
                speaker1ID = mix_name[: mix_name.find("_")]
                speaker2ID = mix_name[mix_name.find("_") + 1 : mix_name.find(".")]
                speaker1_audio_path = audio_path / split / "s1" / mix_name
                speaker2_audio_path = audio_path / split / "s2" / mix_name
                speaker1_video_path = video_path / (speaker1ID + ".npz")
                speaker2_video_path = video_path / (speaker2ID + ".npz")

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
