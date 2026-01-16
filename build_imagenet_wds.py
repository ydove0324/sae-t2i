import os
import io
import yaml
import argparse
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import webdataset as wds


# ---------------------------
# must match your training
# ---------------------------
def normalize_sae(tensor: torch.Tensor) -> torch.Tensor:
    ema_shift_factor = 0.0019670347683131695
    ema_scale_factor = 0.247765451669693
    return (tensor - ema_shift_factor) / ema_scale_factor


def center_crop_arr(pil_image, image_size: int):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def build_transform(image_size: int):
    import torchvision.transforms as T
    return T.Compose([
        T.Lambda(lambda pil: center_crop_arr(pil, image_size)),
        T.ToTensor(),                      # [0,1]
        T.Lambda(lambda t: t * 2.0 - 1.0), # [-1,1]
    ])


def load_index_synset_map(index_synset_path: str):
    with open(index_synset_path, "r") as f:
        data = yaml.safe_load(f)

    index2synset = {}
    if isinstance(data, dict):
        for k, v in data.items():
            index2synset[int(k)] = str(v)
    elif isinstance(data, list):
        for idx, v in enumerate(data):
            index2synset[idx] = str(v)
    else:
        raise ValueError(f"Unsupported index_synset format: {type(data)}")

    synset2index = {syn: idx for idx, syn in index2synset.items()}
    return index2synset, synset2index


IMG_EXTS = (".jpeg", ".jpg", ".png", ".bmp", ".webp")


class ImageNetPathDataset(Dataset):
    """
    return: x_img[-1,1] float32 CPU, label int, path str
    """
    def __init__(self, root: str, index_synset_path: str, image_size: int):
        self.root = root
        self.tfm = build_transform(image_size)
        _, synset2index = load_index_synset_map(index_synset_path)

        samples: List[Tuple[str, int]] = []
        for entry in os.scandir(root):
            if not entry.is_dir():
                continue
            synset = entry.name
            if not synset.startswith("n"):
                continue
            if synset not in synset2index:
                continue
            label = synset2index[synset]
            for fname in os.listdir(entry.path):
                if fname.lower().endswith(IMG_EXTS):
                    samples.append((os.path.join(entry.path, fname), label))

        if not samples:
            raise RuntimeError(f"No images under {root}")

        self.samples = samples
        print(f"[ImageNetPathDataset] {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.tfm(img)  # torch float32 CPU [-1,1]
        return x, int(label), path


def tensor_to_npy_bytes_fp32(x: torch.Tensor) -> bytes:
    """
    x: torch tensor CPU -> store as np.float32 .npy bytes
    """
    arr = x.detach().cpu().numpy().astype(np.float32, copy=False)
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def init_ddp() -> tuple[int, int, int]:
    """
    return (rank, world_size, local_rank)
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed not available")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world, local_rank


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--imagenet_root", type=str, default="/share/project/datasets/ImageNet/train")
    parser.add_argument("--index_synset_path", type=str, default="/share/project/datasets/ImageNet/train/index_synset.yaml")
    parser.add_argument("--image_size", type=int, default=256)

    # output
    parser.add_argument(
        "--out_prefix",
        type=str,
        required=True,
        help="Output prefix without %d. Actual pattern: <out_prefix>-r{rank:03d}-%06d.tar",
    )
    parser.add_argument("--maxcount", type=int, default=2000, help="samples per shard")
    parser.add_argument("--maxsize_gb", type=float, default=2.0, help="soft cap per shard")
    parser.add_argument("--include_path", action="store_true", help="store path.txt in each sample")

    # compute
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true", help="enable pin_memory (recommended)")

    # subset control (global slicing, then rank split)
    parser.add_argument("--start", type=int, default=0, help="skip first N samples globally (resume)")
    parser.add_argument("--limit", type=int, default=-1, help="only process N samples globally")

    # VAE
    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--decoder_type", type=str, choices=["diffusion_decoder", "cnn_decoder"], default="diffusion_decoder")

    args = parser.parse_args()

    rank, world, local_rank = init_ddp()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # IMPORTANT: reuse the exact VAE loader you use in training
    from projects.rae.train import load_dinov3_vae  # keep your path

    vae = load_dinov3_vae(args.vae_ckpt, device, decoder_type=args.decoder_type)
    vae.eval()

    # dataset (global order is deterministic-ish by scandir order; if you want strict order, sort samples in dataset)
    ds = ImageNetPathDataset(args.imagenet_root, args.index_synset_path, args.image_size)

    # global slice first
    all_indices = list(range(len(ds)))
    if args.start > 0:
        all_indices = all_indices[args.start:]
    if args.limit and args.limit > 0:
        all_indices = all_indices[:args.limit]

    # rank split
    my_indices = all_indices[rank::world]

    # small wrapper subset
    class Subset(Dataset):
        def __init__(self, base, idxs):
            self.base = base
            self.idxs = idxs
        def __len__(self):
            return len(self.idxs)
        def __getitem__(self, i):
            return self.base[self.idxs[i]]

    ds2 = Subset(ds, my_indices)

    dl = DataLoader(
        ds2,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    # output pattern per rank
    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_pattern = f"{args.out_prefix}-r{rank:03d}-%06d.tar"
    # sanity: ShardWriter uses printf-style formatting internally
    if "%d" not in out_pattern and "%0" not in out_pattern:
        raise ValueError(f"Bad out_pattern={out_pattern} (must contain %d or %0Nd)")

    maxsize = int(args.maxsize_gb * (1024 ** 3))

    if rank == 0:
        print(f"[ddp] world={world} out_pattern example: {out_pattern % 0}")
    print(f"[rank {rank}] processing {len(my_indices)} samples, writing: {out_pattern}")

    # write shards (each rank independent)
    written = 0
    with wds.ShardWriter(out_pattern, maxcount=args.maxcount, maxsize=maxsize) as sink:
        for x_img, y, paths in dl:
            x_img = x_img.to(device, non_blocking=True)          # [B,3,H,W]
            latent, _p = vae.encode(x_img)                       # [B,1280,16,16]
            latent = normalize_sae(latent).float().cpu()         # FORCE fp32 CPU

            B = latent.shape[0]
            for i in range(B):
                # global id: original dataset index (after start/limit slicing, before rank split)
                # we stored the real dataset index in my_indices list
                global_ds_index = my_indices[written + i]  # safe because sequential iterate
                key = f"{rank:03d}-{global_ds_index:012d}"  # unique across ranks

                sample = {
                    "__key__": key,
                    "latent.npy": tensor_to_npy_bytes_fp32(latent[i]),  # fp32 .npy
                    "cls.txt": str(int(y[i].item())).encode("utf-8"),
                }
                if args.include_path:
                    sample["path.txt"] = str(paths[i]).encode("utf-8")

                sink.write(sample)

            written += B
            if written % 5000 == 0:
                print(f"[rank {rank}] wrote {written}/{len(my_indices)}")

    dist.barrier()
    if rank == 0:
        print("done (all ranks).")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
