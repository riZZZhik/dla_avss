import torch


def collate_fn(dataset_items: list[dict]):
    mixs = [item["mix"] for item in dataset_items]
    audio_paths = [item["audio_path"] for item in dataset_items]
    speakerss = [item["speakers"] for item in dataset_items]

    assert torch.tensor(
        [len(mix) == len(mixs[0]) for mix in mixs]
    ).all(), "Mix records must have the same length"
    assert torch.tensor(
        [len(speakers) == 2 for speakers in speakerss]
    ).all(), "Not all records sonsists of 2 speakers"
    assert torch.tensor(
        [
            len(speaker1_rec) == len(speakerss[0][0])
            for speaker1_rec in (s[0] for s in speakerss)
        ]
    ).all(), "Speaker1 records must have the same length"
    assert torch.tensor(
        [
            len(speaker2_rec) == len(speakerss[0][1])
            for speaker2_rec in (s[1] for s in speakerss)
        ]
    ).all(), "Speaker2 records must have the same length"

    res = {
        "mix": torch.stack(mixs, dim=0),             
        "audio_path": audio_paths,
        "speakers": torch.stack(speakerss, dim=0),   
    }
    
    if "vis_s1" in dataset_items[0]:
        vis1_list = [item["vis_s1"] for item in dataset_items] 
        vis2_list = [item["vis_s2"] for item in dataset_items]  
        res["vis_s1"] = torch.stack(vis1_list, dim=0)         
        res["vis_s2"] = torch.stack(vis2_list, dim=0)          

    return res
