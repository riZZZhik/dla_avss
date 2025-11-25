from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirAVDataset(BaseDataset):

    def __init__(self, data_dir: str, *args, **kwargs):
        root_dir = Path(data_dir)
        audio_root = root_dir / "audio"
        mix_dir = audio_root / "mix"
        s1_dir = audio_root / "s1"
        s2_dir = audio_root / "s2"
        mouths_root = root_dir / "mouths"

        exts = {".wav", ".flac", ".mp3", ".m4a"}

        data = []
        for mix_path in mix_dir.iterdir():
            if mix_path.suffix.lower() not in exts:
                continue

            mix_name = mix_path.name
            stem = mix_path.stem

            if "_" in stem:
                s1_id, s2_id = stem.split("_", 1)
            else:
                s1_id = stem
                s2_id = stem

            t_info = torchaudio.info(str(mix_path))
            audio_len = t_info.num_frames / t_info.sample_rate

            row = {
                "audio_path": str(mix_path),
                "audio_len": audio_len,
            }

            s1_audio = s1_dir / mix_name
            s2_audio = s2_dir / mix_name

            if s1_audio.exists() and s2_audio.exists():
                row["speaker1_audio_path"] = str(s1_audio)
                row["speaker2_audio_path"] = str(s2_audio)

            s1_video = mouths_root / f"{s1_id}.npz"
            s2_video = mouths_root / f"{s2_id}.npz"

            if s1_video.exists():
                row["speaker1_video_path"] = str(s1_video)
            if s2_video.exists():
                row["speaker2_video_path"] = str(s2_video)

            data.append(row)

        super().__init__(data, *args, **kwargs)
