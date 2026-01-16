import io
import numpy as np
import torch
import webdataset as wds
import glob

def has_latent(sample):
    return ("latent.npy" in sample) and ("cls.txt" in sample)

def decode_npy_fp32(data: bytes) -> torch.Tensor:
    arr = np.load(io.BytesIO(data), allow_pickle=False)   # np.float32
    return torch.from_numpy(arr).float()                  # torch.float32 CPU

def make_latent_loader(urls: str, batch_size: int, num_workers: int):
    shards = sorted(glob.glob(urls))
    dataset = (
        wds.WebDataset(shards, shardshuffle=True,                
                handler=wds.handlers.ignore_and_continue,
                nodesplitter=wds.shardlists.split_by_node, 
                workersplitter=wds.shardlists.split_by_worker,)
        .shuffle(10000)
        .select(has_latent)  # 关键：先过滤
        .to_tuple("latent.npy", "cls.txt")
        .map_tuple(
            decode_npy_fp32,
            lambda s: torch.tensor(int(s.decode("utf-8")), dtype=torch.long),
        )
        .batched(batch_size, partial=False)
    )
    return wds.WebLoader(dataset, num_workers=num_workers, batch_size=None)