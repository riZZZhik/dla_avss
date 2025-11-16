import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    mixs = [item["mix"] for item in dataset_items]  # L -> BxL
    audio_paths = [item["audio_path"] for item in dataset_items]
    speakerss = [item["speakers"] for item in dataset_items]  # SxL -> BxSxL

    assert torch.tensor(
        [len(mix) == len(mixs[0]) for mix in mixs]
    ).all(), "Mix records must have the same length"
    assert torch.tensor(
        [len(speakers) == 2 for speakers in speakerss]
    ).all(), "Not all records sonsists of 2 speakers"
    assert torch.tensor(
        [
            len(speaker1_rec) == len(speakerss[0][0])
            for speaker1_rec in [speakerss_row[0] for speakerss_row in speakerss]
        ]
    ).all(), "Speaker's records must have the same length"
    assert torch.tensor(
        [
            len(speaker2_rec) == len(speakerss[0][0])
            for speaker2_rec in [speakerss_row[1] for speakerss_row in speakerss]
        ]
    ).all(), "Speaker's records must have the same length"

    res = {
        "mix": torch.stack(mixs, dim=0),
        "audio_path": audio_paths,
        "speakers": torch.stack(speakerss, dim=0),
    }

    return res
